import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.admin_audit_repository import AdminAuditRepository
from app.db.database import Base
from app.db.model_admission_operation_repository import ModelAdmissionOperationRepository
from app.db.model_catalog_control_repository import ModelCatalogControlRepository
from app.db.models import AdminAuditEvent, ModelAdmissionOperation, ModelCatalogControl
from app.schemas.response import ApiException
from app.services.admin_audit_service import AdminAuditService
from app.services.model_management_service import ModelManagementConfig, ModelManagementService


def _sha256(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_governance_run(root: Path, started_at: datetime, *, status: str, queue=None):
    run_id = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.astimezone(UTC).isoformat(),
        "status": status,
    }
    artifacts = {"cycle-summary.json": summary}
    if queue is not None:
        artifacts["candidate-queue.json"] = queue
    manifest = {
        "schema_version": 1,
        "artifacts": {
            name: {"sha256": _sha256(payload), "record_count": len(payload)} for name, payload in artifacts.items()
        },
    }
    for name, payload in artifacts.items():
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "run_id": run_id,
        "run_path": f"runs/{run_id}",
        "summary_sha256": _sha256(summary),
        "manifest_sha256": _sha256(manifest),
    }
    pointer_name = "latest-success.json" if status == "success" else "latest-failure.json"
    (root / pointer_name).write_text(json.dumps(pointer), encoding="utf-8")
    return run_id, run_dir


class FakeCatalog:
    def __init__(self):
        self.generations = 0
        self.entries = {
            "qwen-max-latest": {
                "db_model": True,
                "underlying": "openai/qwen-max",
                "metadata": {"display_name": "Qwen Max", "provider_key": "qwen"},
            }
        }

    def list_aliases(self):
        return self.entries

    def get_model_entry(self, model_id):
        return self.entries.get(model_id)

    def bump_generation(self):
        self.generations += 1
        return str(self.generations)


class ModelManagementServiceTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.control_repository = ModelCatalogControlRepository(self.db)
        self.operation_repository = ModelAdmissionOperationRepository(self.db)
        self.audit = AdminAuditService(AdminAuditRepository(self.db))
        self.catalog = FakeCatalog()
        self.admin = SimpleNamespace(
            id="admin-1",
            username="root",
            email="root@example.com",
            is_superuser=True,
        )
        self.now = datetime(2026, 8, 4, 1, 2, 3, 456789, tzinfo=UTC)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.governance_root = Path(self.temp_dir.name)
        self.fingerprint = "a" * 64
        self.queue = [
            {
                "provider_key": "moonshot",
                "model_id": "kimi-k3",
                "state": "admission_ready",
                "reasons": [],
                "candidate_fingerprint": self.fingerprint,
                "candidate": {
                    "model_id": "kimi-k3",
                    "metadata": {
                        "display_name": "Kimi K3",
                        "provider_display": "Moonshot",
                        "capabilities": {"functionCalling": True},
                    },
                    "registration": {"api_key_env": "MOONSHOT_API_KEY"},
                },
            }
        ]
        self.run_id, self.run_dir = _write_governance_run(
            self.governance_root,
            self.now - timedelta(minutes=1),
            status="success",
            queue=self.queue,
        )

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def build_service(self, **overrides):
        config_values = {
            "governance_root": self.governance_root,
            "management_enabled": True,
            "governance_max_age_seconds": 3600,
            "worker_enabled": True,
            "worker_token": "worker-secret",
            "lease_seconds": 600,
        }
        config_values.update(overrides.pop("config", {}))
        plan_loader = overrides.pop(
            "plan_loader",
            Mock(return_value={"run_id": self.run_id, "eligible": [{"model_id": "kimi-k3"}]}),
        )
        return ModelManagementService(
            control_repository=self.control_repository,
            operation_repository=self.operation_repository,
            audit_service=overrides.pop("audit_service", self.audit),
            config=ModelManagementConfig(**config_values),
            catalog=overrides.pop("catalog", self.catalog),
            plan_loader=plan_loader,
            clock=overrides.pop("clock", lambda: self.now),
            **overrides,
        )

    def _request_admission(self, service, *, request_id="req-admit"):
        return service.request_admission(
            fingerprint=self.fingerprint,
            model_id="kimi-k3",
            expected_run_id=self.run_id,
            reason="完成验收后准入",
            admin=self.admin,
            request_id=request_id,
        )

    def test_snapshot_exact_contract_and_safe_candidate_projection(self):
        snapshot = self.build_service().get_snapshot()

        self.assertEqual(
            set(snapshot),
            {"generated_at", "governance", "capabilities", "models", "candidates", "operations"},
        )
        self.assertEqual(snapshot["governance"], {"available": True, "run_id": self.run_id})
        self.assertEqual(
            set(snapshot["models"][0]),
            {
                "model_id",
                "name",
                "provider",
                "provider_display",
                "health",
                "selectable",
                "routable",
                "state",
                "revision",
                "reason",
                "updated_at",
            },
        )
        candidate = snapshot["candidates"][0]
        self.assertEqual(
            set(candidate),
            {
                "candidate_fingerprint",
                "model_id",
                "name",
                "provider_key",
                "provider_display",
                "state",
                "reasons",
                "capabilities",
            },
        )
        self.assertEqual(candidate["candidate_fingerprint"], self.fingerprint)
        self.assertEqual(candidate["model_id"], "kimi-k3")
        self.assertEqual(candidate["provider_key"], "moonshot")
        serialized = json.dumps(snapshot, default=str)
        self.assertNotIn(self.temp_dir.name, serialized)
        self.assertNotIn("MOONSHOT_API_KEY", serialized)

    def test_governance_missing_tampered_stale_or_newer_verified_failure_degrades(self):
        (self.run_dir / "cycle-summary.json").write_text("{}", encoding="utf-8")
        self.assertFalse(self.build_service().get_snapshot()["governance"]["available"])

        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.governance_root = Path(self.temp_dir.name)
        self.run_id, self.run_dir = _write_governance_run(
            self.governance_root,
            self.now - timedelta(hours=2),
            status="success",
            queue=self.queue,
        )
        self.assertFalse(self.build_service().get_snapshot()["governance"]["available"])

        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.governance_root = Path(self.temp_dir.name)
        self.run_id, self.run_dir = _write_governance_run(
            self.governance_root,
            self.now - timedelta(minutes=2),
            status="success",
            queue=self.queue,
        )
        _write_governance_run(
            self.governance_root,
            self.now - timedelta(minutes=1),
            status="failed",
        )
        self.assertFalse(self.build_service().get_snapshot()["governance"]["available"])

    def test_recent_operations_remain_visible_when_governance_degrades(self):
        service = self.build_service()
        operation = self._request_admission(service)
        (self.run_dir / "cycle-summary.json").write_text("{}", encoding="utf-8")

        snapshot = service.get_snapshot()

        self.assertFalse(snapshot["governance"]["available"])
        self.assertEqual([item["operation_id"] for item in snapshot["operations"]], [operation.id])

    def test_visibility_uses_cas_and_rolls_back_with_audit(self):
        service = self.build_service()
        hidden = service.set_visibility(
            model_id="qwen-max-latest",
            selectable=False,
            expected_revision=None,
            reason="暂时隐藏",
            admin=self.admin,
            request_id="req-1",
        )
        self.assertEqual((hidden.selectable, hidden.routable, hidden.revision), (False, True, 1))

        with self.assertRaises(ApiException) as stale:
            service.set_visibility(
                model_id="qwen-max-latest",
                selectable=True,
                expected_revision=None,
                reason="恢复展示",
                admin=self.admin,
                request_id="req-2",
            )
        self.assertEqual(stale.exception.status_code, 409)

        failing_audit = Mock()
        failing_audit._record.side_effect = RuntimeError("audit unavailable")
        service = self.build_service(audit_service=failing_audit)
        with self.assertRaises(RuntimeError):
            service.set_visibility(
                model_id="qwen-max-latest",
                selectable=True,
                expected_revision=1,
                reason="恢复展示",
                admin=self.admin,
                request_id="req-3",
            )
        self.db.expire_all()
        self.assertFalse(self.db.query(ModelCatalogControl).one().selectable)
        self.assertEqual(self.db.query(AdminAuditEvent).count(), 1)

    def test_admission_request_is_persistent_idempotent_and_failed_retry_resets_same_row(self):
        service = self.build_service()
        created = self._request_admission(service)
        duplicate = self._request_admission(service, request_id="req-duplicate")

        self.assertEqual(created.id, duplicate.id)
        self.assertEqual(created.status, "pending")
        self.assertEqual(self.db.query(AdminAuditEvent).count(), 1)

        created.status = "failed"
        created.attempts = 2
        created.lease_token_hash = "old"
        created.lease_expires_at = self.now
        created.catalog_invalidated_at = self.now
        created.result = {"phase": "failed", "secret": "do-not-project"}
        created.terminal_at = self.now
        self.db.commit()

        retried = self._request_admission(service, request_id="req-retry")
        self.assertEqual(retried.id, created.id)
        self.assertEqual(retried.status, "pending")
        self.assertEqual(retried.attempts, 2)
        self.assertIsNone(retried.lease_token_hash)
        self.assertEqual(retried.result, {})
        self.assertEqual(self.db.query(AdminAuditEvent).count(), 2)

    def test_admission_fails_closed_before_persisting(self):
        disabled = self.build_service(config={"management_enabled": False})
        with self.assertRaises(ApiException) as raised:
            self._request_admission(disabled)
        self.assertEqual(raised.exception.status_code, 503)

        (self.run_dir / "candidate-queue.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(ApiException) as tampered:
            self._request_admission(self.build_service())
        self.assertEqual(tampered.exception.status_code, 503)
        self.assertEqual(self.db.query(ModelAdmissionOperation).count(), 0)

    def test_claim_global_mutex_reclaim_and_minimum_lease(self):
        service = self.build_service(config={"lease_seconds": 30})
        first = self._request_admission(service)
        claimed, first_token = service.claim_operation()
        self.assertEqual(claimed.id, first.id)
        lease_expires_at = service._as_utc(claimed.lease_expires_at)
        self.assertGreaterEqual((lease_expires_at - self.now).total_seconds(), 600)

        second = ModelAdmissionOperation(
            id="operation-2",
            model_id="other-model",
            candidate_fingerprint="b" * 64,
            governance_run_id=self.run_id,
            status="pending",
            requested_by="admin-1",
            request_id="req-2",
            reason="第二个任务",
            attempts=0,
            result={},
            created_at=self.now - timedelta(days=1),
            updated_at=self.now - timedelta(days=1),
        )
        self.operation_repository.add(second)
        self.db.commit()
        self.assertIsNone(service.claim_operation())

        claimed.lease_expires_at = self.now - timedelta(seconds=1)
        self.db.commit()
        reclaimed, second_token = service.claim_operation()
        self.assertEqual(reclaimed.id, first.id)
        self.assertNotEqual(first_token, second_token)
        self.assertEqual(reclaimed.attempts, 2)

    def test_worker_callbacks_enforce_lease_invalidation_and_idempotent_terminal_audit(self):
        service = self.build_service()
        self._request_admission(service)
        operation, lease_token = service.claim_operation()

        with self.assertRaises(ApiException):
            service.complete_operation(
                operation_id=operation.id,
                lease_token=lease_token,
                status="succeeded",
                phase="complete",
                error_code=None,
                completed_phases=["register"],
                writes_performed=True,
                compensation={},
            )

        generation = service.invalidate_catalog(operation_id=operation.id, lease_token=lease_token)
        self.assertEqual(generation, "1")
        completed = service.complete_operation(
            operation_id=operation.id,
            lease_token=lease_token,
            status="succeeded",
            phase="complete",
            error_code=None,
            completed_phases=["register", "allowlist", "readback"],
            writes_performed=True,
            compensation={"attempted": False, "private_detail": "drop-me"},
        )
        repeated = service.complete_operation(
            operation_id=operation.id,
            lease_token=lease_token,
            status="succeeded",
            phase="complete",
            error_code=None,
            completed_phases=[],
            writes_performed=True,
            compensation={},
        )
        self.assertEqual(completed.id, repeated.id)
        self.assertNotIn("private_detail", completed.result["compensation"])
        self.assertEqual(self.db.query(AdminAuditEvent).count(), 2)

    def test_expired_lease_can_complete_until_operation_is_reclaimed(self):
        service = self.build_service()
        self._request_admission(service)
        operation, old_token = service.claim_operation()
        operation.lease_expires_at = self.now - timedelta(seconds=1)
        self.db.commit()

        completed = service.complete_operation(
            operation_id=operation.id,
            lease_token=old_token,
            status="failed",
            phase="readback",
            error_code="timeout",
            completed_phases=["register"],
            writes_performed=True,
            compensation={"attempted": True},
        )
        self.assertEqual(completed.status, "failed")

        completed.status = "running"
        completed.terminal_at = None
        completed.lease_expires_at = self.now - timedelta(seconds=1)
        self.db.commit()
        reclaimed, new_token = service.claim_operation()
        self.assertNotEqual(old_token, new_token)
        with self.assertRaises(ApiException) as stale:
            service.complete_operation(
                operation_id=reclaimed.id,
                lease_token=old_token,
                status="failed",
                phase="readback",
                error_code="timeout",
                completed_phases=[],
                writes_performed=True,
                compensation={},
            )
        self.assertEqual(stale.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
