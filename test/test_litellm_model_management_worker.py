import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from unittest.mock import patch

from scripts import run_litellm_model_management_worker as worker
from scripts.check_litellm_candidate_preflight import (
    CaseResult,
    PreflightReport,
    candidate_contract_fingerprint,
    parse_candidate,
)
from scripts.verify_litellm_governance_snapshot import (
    VerifiedGovernanceSnapshot,
    canonical_sha256,
)


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class RecordingClient:
    def __init__(self, claim_payload):
        self.claim_payload = claim_payload
        self.calls = []
        self.complete_failures = 0
        self.stale_complete_operation_ids: set[str] = set()
        self.renew_calls = 0
        self.renew_failures = 0
        self.renewed_twice = Event()

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/claim"):
            if self.claim_payload is None:
                return FakeResponse(status_code=204)
            return FakeResponse(self.claim_payload)
        if url.endswith("/invalidate-catalog"):
            return FakeResponse({"generation": "9"})
        if url.endswith("/renew"):
            self.renew_calls += 1
            if self.renew_failures > 0:
                self.renew_failures -= 1
                return FakeResponse({"detail": "temporary"}, status_code=503)
            if self.renew_calls >= 2:
                self.renewed_twice.set()
            return FakeResponse({"lease_expires_at": "2026-08-04T01:12:03+00:00"})
        if url.endswith("/complete"):
            if any(f"/{operation_id}/complete" in url for operation_id in self.stale_complete_operation_ids):
                return FakeResponse({"detail": "stale lease"}, status_code=409)
            if self.complete_failures > 0:
                self.complete_failures -= 1
                return FakeResponse({"detail": "temporary"}, status_code=503)
            return FakeResponse({"status": "accepted"})
        raise AssertionError(f"unexpected POST {url}")


def claim_payload(*, fingerprint="a" * 64, model_id="qwen3.8-max", operation_id="op-123"):
    return {
        "operation_id": operation_id,
        "lease_token": "lease-secret",
        "run_id": "20260804T010000000000Z",
        "candidate_fingerprint": fingerprint,
        "model_id": model_id,
    }


def candidate_contract():
    capabilities = {
        "imageGen": False,
        "deepThinking": False,
        "fileSupport": False,
        "functionCalling": False,
        "vision": False,
        "webSearch": False,
    }
    return {
        "provider_key": "moonshot",
        "provider_display": "Moonshot",
        "model_id": "kimi-k3",
        "litellm_model": "moonshot/kimi-k3",
        "preflight_model": "candidate/moonshot/kimi-k3",
        "preflight_route": {
            "status": "ready",
            "route_model_name": "candidate/moonshot/*",
            "route_litellm_model": "moonshot/*",
            "api_base": "https://api.moonshot.cn/v1",
            "api_key_env": "MOONSHOT_API_KEY",
            "credential_generation": "test-v1",
            "reasons": [],
        },
        "isolation_status": "candidate",
        "metadata": {
            "display_name": "Kimi K3",
            "provider_key": "moonshot",
            "provider_display": "Moonshot",
            "capabilities": capabilities,
            "pricing": {"input": 0.6, "output": 2.5, "unit": "USD/1M tokens"},
        },
        "metadata_evidence": {
            "cost_map_matched": True,
            "reviewed_override_applied": False,
        },
        "registration": {
            "api_base": "https://api.moonshot.cn/v1",
            "api_key_env": "MOONSHOT_API_KEY",
            "endpoint_status": "verified",
        },
    }


def candidate_record(*, state="preflight_required"):
    candidate = candidate_contract()
    fingerprint = candidate_contract_fingerprint(candidate)
    return {
        "provider_key": "moonshot",
        "model_id": "kimi-k3",
        "state": state,
        "reasons": [],
        "candidate_snapshot_sha256": canonical_sha256(candidate),
        "candidate_fingerprint": fingerprint,
        "acceptance_file": f"{fingerprint}.json",
        "candidate": candidate,
    }


def successful_preflight(candidate):
    return PreflightReport(
        healthy=True,
        dry_run=False,
        candidate=parse_candidate(candidate),
        cases=[
            CaseResult(
                name="text",
                status="passed",
                usage_present=True,
                cost_present=True,
                output_present=True,
            ),
            CaseResult(
                name="stream",
                status="passed",
                usage_present=True,
                cost_present=True,
                output_present=True,
                stream_done=True,
            ),
            CaseResult(name="tool_calling", status="skipped"),
            CaseResult(name="vision", status="skipped"),
            CaseResult(name="reasoning", status="skipped"),
            CaseResult(name="preserved_tool_round", status="skipped"),
        ],
    )


def succeeded_result():
    return {
        "status": "succeeded",
        "phase": "complete",
        "completed_phases": ["model_new", "verify", "key_update", "catalog_invalidation", "fusion_readback"],
        "writes_performed": True,
        "error": None,
        "compensation": {
            "attempted": False,
            "key_restored": False,
            "model_deleted": False,
            "catalog_invalidated": False,
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


class ModelManagementWorkerTests(unittest.TestCase):
    def test_initial_lease_renewal_failure_spools_known_no_write_result(self):
        client = RecordingClient(claim_payload())
        client.renew_failures = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)

            with self.assertRaisesRegex(worker.WorkerProtocolError, "operation_lease_renewal_failed"):
                worker.process_once(
                    client=client,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                    governance_root=Path("/governance"),
                    governance_max_age_seconds=86400,
                    litellm_base_url="http://127.0.0.1:4000",
                    master_key="master-secret",
                    virtual_key="virtual-secret",
                    environ={},
                    state_dir=state_dir,
                )

            payload = json.loads(next(state_dir.glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(payload["result"]["error_code"], "operation_lease_renewal_failed")
        self.assertFalse(payload["result"]["writes_performed"])
        self.assertFalse(payload["result"]["compensation"]["manual_cleanup_required"])

    def test_lease_heartbeat_renews_during_long_running_operation(self):
        client = RecordingClient(claim_payload())
        heartbeat = worker.OperationLeaseHeartbeat(
            client,
            fusion_base_url="http://127.0.0.1:8002",
            worker_token="worker-secret",
            operation_id="op-123",
            lease_token="lease-secret",
            interval_seconds=0.01,
        )

        heartbeat.start()
        try:
            self.assertTrue(client.renewed_twice.wait(timeout=0.5))
            heartbeat.ensure_healthy()
        finally:
            heartbeat.stop()

        self.assertGreaterEqual(client.renew_calls, 2)

    def test_preflight_required_runs_real_preflight_writes_redacted_acceptance_then_admits(self):
        record = candidate_record()
        claim = claim_payload(
            fingerprint=record["candidate_fingerprint"],
            model_id=record["model_id"],
        )
        client = RecordingClient(claim)
        with tempfile.TemporaryDirectory() as temp_dir:
            acceptance_dir = Path(temp_dir)
            os.chmod(acceptance_dir, 0o700)
            with (
                patch.object(worker, "_load_verified_candidate", return_value=record),
                patch.object(
                    worker,
                    "run_preflight",
                    return_value=successful_preflight(record["candidate"]),
                ) as preflight,
                patch.object(worker, "execute_admission", return_value=succeeded_result()) as execute,
            ):
                result = worker.process_once(
                    client=client,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                    governance_root=Path("/governance"),
                    governance_max_age_seconds=86400,
                    litellm_base_url="http://127.0.0.1:4000",
                    master_key="master-secret",
                    virtual_key="virtual-secret",
                    candidate_key="candidate-secret",
                    acceptance_dir=acceptance_dir,
                    environ={"MOONSHOT_API_KEY": "provider-secret"},
                )

            self.assertEqual(result["status"], "succeeded")
            preflight.assert_called_once()
            self.assertEqual(preflight.call_args.kwargs["api_key"], "candidate-secret")
            execute.assert_called_once()
            plan = execute.call_args.kwargs["plan"]
            self.assertEqual(plan["run_id"], claim["run_id"])
            self.assertEqual([item["model_id"] for item in plan["eligible"]], ["kimi-k3"])
            acceptance_path = acceptance_dir / f"{record['candidate_fingerprint']}.json"
            acceptance_text = acceptance_path.read_text(encoding="utf-8")
            self.assertNotIn("candidate-secret", acceptance_text)
            self.assertNotIn("master-secret", acceptance_text)
            self.assertNotIn("provider-secret", acceptance_text)
            self.assertEqual(os.stat(acceptance_path).st_mode & 0o777, 0o600)
            self.assertEqual(list(acceptance_dir.glob(f".{acceptance_path.name}.*")), [])

    def test_preflight_failure_performs_no_admission_writes(self):
        record = candidate_record()
        claim = claim_payload(
            fingerprint=record["candidate_fingerprint"],
            model_id=record["model_id"],
        )
        failed = PreflightReport(
            healthy=False,
            dry_run=False,
            candidate=parse_candidate(record["candidate"]),
            cases=[CaseResult(name="text", status="failed", issues=("missing_output",))],
        )
        client = RecordingClient(claim)
        with tempfile.TemporaryDirectory() as temp_dir:
            acceptance_dir = Path(temp_dir)
            os.chmod(acceptance_dir, 0o700)
            with (
                patch.object(worker, "_load_verified_candidate", return_value=record),
                patch.object(worker, "run_preflight", return_value=failed),
                patch.object(worker, "execute_admission") as execute,
            ):
                result = worker.process_once(
                    client=client,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                    governance_root=Path("/governance"),
                    governance_max_age_seconds=86400,
                    litellm_base_url="http://127.0.0.1:4000",
                    master_key="master-secret",
                    virtual_key="virtual-secret",
                    candidate_key="candidate-secret",
                    acceptance_dir=acceptance_dir,
                    environ={},
                )

            execute.assert_not_called()
            self.assertEqual(result["error"]["code"], "candidate_preflight_failed")
            self.assertFalse(result["writes_performed"])
            self.assertEqual(list(acceptance_dir.iterdir()), [])

    def test_verified_candidate_rejects_fingerprint_model_and_state_mismatch(self):
        record = candidate_record()
        snapshot = VerifiedGovernanceSnapshot(
            run_id=claim_payload()["run_id"],
            started_at=datetime.now(UTC),
            queue=[record],
        )
        with patch.object(worker, "load_verified_governance_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(worker.VerifiedPlanError, "candidate_claim_mismatch"):
                worker._load_verified_candidate(
                    governance_root=Path("/governance"),
                    expected_run_id=snapshot.run_id,
                    candidate_fingerprint="b" * 64,
                    model_id=record["model_id"],
                    max_age_seconds=86400,
                )

            with self.assertRaisesRegex(worker.VerifiedPlanError, "candidate_claim_mismatch"):
                worker._load_verified_candidate(
                    governance_root=Path("/governance"),
                    expected_run_id=snapshot.run_id,
                    candidate_fingerprint=record["candidate_fingerprint"],
                    model_id="other-model",
                    max_age_seconds=86400,
                )

            invalid_state = {**record, "state": "quarantined"}
            snapshot = VerifiedGovernanceSnapshot(
                run_id=claim_payload()["run_id"],
                started_at=datetime.now(UTC),
                queue=[invalid_state],
            )
            with (
                patch.object(worker, "load_verified_governance_snapshot", return_value=snapshot),
                self.assertRaisesRegex(worker.VerifiedPlanError, "candidate_state_not_admissible"),
            ):
                worker._load_verified_candidate(
                    governance_root=Path("/governance"),
                    expected_run_id=snapshot.run_id,
                    candidate_fingerprint=record["candidate_fingerprint"],
                    model_id=record["model_id"],
                    max_age_seconds=86400,
                )

    def test_acceptance_atomic_failure_leaves_no_partial_file_and_insecure_directory_is_rejected(self):
        record = candidate_record()
        acceptance = {"healthy": True, "candidate_fingerprint": record["candidate_fingerprint"]}
        with tempfile.TemporaryDirectory() as temp_dir:
            acceptance_dir = Path(temp_dir)
            os.chmod(acceptance_dir, 0o700)
            with (
                patch.object(worker.os, "replace", side_effect=OSError("disk failure")),
                self.assertRaisesRegex(worker.VerifiedPlanError, "candidate_acceptance_write_failed"),
            ):
                worker._write_acceptance_atomic(
                    acceptance_dir,
                    candidate_fingerprint=record["candidate_fingerprint"],
                    acceptance=acceptance,
                )
            self.assertEqual(list(acceptance_dir.iterdir()), [])

            os.chmod(acceptance_dir, 0o755)
            with self.assertRaisesRegex(worker.VerifiedPlanError, "candidate_acceptance_directory_insecure"):
                worker._write_acceptance_atomic(
                    acceptance_dir,
                    candidate_fingerprint=record["candidate_fingerprint"],
                    acceptance=acceptance,
                )

    def test_idle_claim_exits_without_executor(self):
        client = RecordingClient(None)

        with patch.object(worker, "execute_admission") as execute:
            result = worker.process_once(
                client=client,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
                governance_root=Path("/governance"),
                governance_max_age_seconds=86400,
                litellm_base_url="http://127.0.0.1:4000",
                master_key="master-secret",
                virtual_key="virtual-secret",
                environ={},
            )

        self.assertEqual(result, {"status": "idle"})
        execute.assert_not_called()
        self.assertEqual(len(client.calls), 1)

    def test_success_invalidates_catalog_before_completion(self):
        client = RecordingClient(claim_payload())

        def execute(**kwargs):
            kwargs["catalog_invalidation_fn"]()
            return succeeded_result()

        with (
            patch.object(worker, "_load_verified_candidate", return_value={"state": "admission_ready"}),
            patch.object(worker, "load_verified_admission_plan", return_value={"run_id": claim_payload()["run_id"]}),
            patch.object(worker, "execute_admission", side_effect=execute) as executor,
        ):
            result = worker.process_once(
                client=client,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
                governance_root=Path("/governance"),
                governance_max_age_seconds=86400,
                litellm_base_url="http://127.0.0.1:4000",
                master_key="master-secret",
                virtual_key="virtual-secret",
                environ={"DASHSCOPE_API_KEY": "provider-secret"},
            )

        self.assertEqual(result["status"], "succeeded")
        executor.assert_called_once()
        urls = [url for url, _ in client.calls]
        self.assertLess(
            urls.index("http://127.0.0.1:8002/api/internal/model-management/admissions/op-123/invalidate-catalog"),
            urls.index("http://127.0.0.1:8002/api/internal/model-management/admissions/op-123/complete"),
        )
        complete_payload = client.calls[-1][1]["json"]
        self.assertEqual(complete_payload["status"], "succeeded")
        self.assertNotIn("master-secret", json.dumps(complete_payload))
        self.assertNotIn("provider-secret", json.dumps(complete_payload))
        self.assertNotIn("lease-secret", json.dumps(complete_payload))

    def test_verification_failure_is_persisted_without_external_execution(self):
        client = RecordingClient(claim_payload())

        with (
            patch.object(worker, "_load_verified_candidate", return_value={"state": "admission_ready"}),
            patch.object(
                worker,
                "load_verified_admission_plan",
                side_effect=worker.VerifiedPlanError("manifest_sha256_mismatch"),
            ),
            patch.object(worker, "execute_admission") as execute,
        ):
            result = worker.process_once(
                client=client,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
                governance_root=Path("/governance"),
                governance_max_age_seconds=86400,
                litellm_base_url="http://127.0.0.1:4000",
                master_key="master-secret",
                virtual_key="virtual-secret",
                environ={},
            )

        execute.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "manifest_sha256_mismatch")
        self.assertEqual(client.calls[-1][1]["json"]["error_code"], "manifest_sha256_mismatch")

    def test_complete_retries_transient_failure(self):
        client = RecordingClient(claim_payload())
        client.complete_failures = 2

        with (
            patch.object(worker, "_load_verified_candidate", return_value={"state": "admission_ready"}),
            patch.object(worker, "load_verified_admission_plan", return_value={"run_id": claim_payload()["run_id"]}),
            patch.object(worker, "execute_admission", return_value=succeeded_result()),
            patch.object(worker.time, "sleep") as sleep,
        ):
            result = worker.process_once(
                client=client,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
                governance_root=Path("/governance"),
                governance_max_age_seconds=86400,
                litellm_base_url="http://127.0.0.1:4000",
                master_key="master-secret",
                virtual_key="virtual-secret",
                environ={},
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(sleep.call_count, 2)

    def test_interrupted_spool_is_completed_as_manual_recovery_without_reexecution(self):
        client = RecordingClient(None)
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            worker._write_spool(state_dir, claim=claim_payload(), result=None)

            completed = worker.flush_spooled_results(
                client,
                state_dir=state_dir,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
            )

            self.assertEqual(completed, 1)
            self.assertEqual(list(state_dir.glob("*.json")), [])
        payload = client.calls[-1][1]["json"]
        self.assertEqual(payload["error_code"], "worker_execution_interrupted")
        self.assertTrue(payload["writes_performed"])
        self.assertTrue(payload["compensation"]["manual_cleanup_required"])

    def test_spool_fsyncs_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            original_fsync = os.fsync
            with patch.object(worker.os, "fsync", wraps=original_fsync) as fsync:
                worker._write_spool(state_dir, claim=claim_payload(), result=None)

        self.assertEqual(fsync.call_count, 2)

    def test_stale_spool_is_quarantined_without_blocking_later_results(self):
        client = RecordingClient(None)
        client.stale_complete_operation_ids.add("op-stale")
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            worker._write_spool(
                state_dir,
                claim=claim_payload(operation_id="op-stale"),
                result=worker._verification_failure("stale"),
            )
            worker._write_spool(
                state_dir,
                claim=claim_payload(operation_id="op-current"),
                result=worker._verification_failure("current"),
            )

            with patch.object(worker, "print") as journal_warning:
                completed = worker.flush_spooled_results(
                    client,
                    state_dir=state_dir,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                )

            self.assertEqual(completed, 1)
            self.assertEqual(list(state_dir.glob("*.json")), [])
            quarantined = list((state_dir / "quarantine").glob("*.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertIn("op-stale", quarantined[0].read_text(encoding="utf-8"))
            warning = json.loads(journal_warning.call_args.args[0])
            self.assertEqual(warning["error"]["code"], "stale_spool_quarantined")
            self.assertNotIn("op-stale", journal_warning.call_args.args[0])
            self.assertIs(journal_warning.call_args.kwargs["file"], worker.sys.stderr)

    def test_governance_is_revalidated_after_preflight_before_admission(self):
        record = candidate_record()
        claim = claim_payload(
            fingerprint=record["candidate_fingerprint"],
            model_id=record["model_id"],
        )
        client = RecordingClient(claim)
        with tempfile.TemporaryDirectory() as temp_dir:
            acceptance_dir = Path(temp_dir)
            os.chmod(acceptance_dir, 0o700)
            with (
                patch.object(
                    worker,
                    "_load_verified_candidate",
                    side_effect=[record, worker.VerifiedPlanError("governance_run_changed")],
                ) as load_candidate,
                patch.object(
                    worker,
                    "run_preflight",
                    return_value=successful_preflight(record["candidate"]),
                ),
                patch.object(worker, "execute_admission") as execute,
            ):
                result = worker.process_once(
                    client=client,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                    governance_root=Path("/governance"),
                    governance_max_age_seconds=86400,
                    litellm_base_url="http://127.0.0.1:4000",
                    master_key="master-secret",
                    virtual_key="virtual-secret",
                    candidate_key="candidate-secret",
                    acceptance_dir=acceptance_dir,
                    environ={"MOONSHOT_API_KEY": "provider-secret"},
                )

        self.assertEqual(load_candidate.call_count, 2)
        execute.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "governance_run_changed")

    def test_terminal_result_stays_spooled_until_api_accepts_it(self):
        client = RecordingClient(claim_payload())
        client.complete_failures = worker.COMPLETE_ATTEMPTS
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            with (
                patch.object(
                    worker,
                    "load_verified_admission_plan",
                    return_value={"run_id": claim_payload()["run_id"]},
                ),
                patch.object(worker, "_load_verified_candidate", return_value={"state": "admission_ready"}),
                patch.object(worker, "execute_admission", return_value=succeeded_result()),
                patch.object(worker.time, "sleep"),
                self.assertRaisesRegex(worker.WorkerProtocolError, "operation_complete_failed"),
            ):
                worker.process_once(
                    client=client,
                    fusion_base_url="http://127.0.0.1:8002",
                    worker_token="worker-secret",
                    governance_root=Path("/governance"),
                    governance_max_age_seconds=86400,
                    litellm_base_url="http://127.0.0.1:4000",
                    master_key="master-secret",
                    virtual_key="virtual-secret",
                    environ={},
                    state_dir=state_dir,
                )

            self.assertEqual(len(list(state_dir.glob("*.json"))), 1)
            client.complete_failures = 0
            completed = worker.flush_spooled_results(
                client,
                state_dir=state_dir,
                fusion_base_url="http://127.0.0.1:8002",
                worker_token="worker-secret",
            )
            self.assertEqual(completed, 1)
            self.assertEqual(list(state_dir.glob("*.json")), [])

    def test_freshness_rejects_newer_verified_failure(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            success_time = now - timedelta(hours=1)
            failure_time = now - timedelta(minutes=5)
            success_run_id = success_time.strftime("%Y%m%dT%H%M%S%fZ")
            failure_run_id = failure_time.strftime("%Y%m%dT%H%M%S%fZ")
            self._write_summary(root, success_run_id, "success", success_time, "latest-success.json")
            self._write_summary(root, failure_run_id, "failed", failure_time, "latest-failure.json")

            with self.assertRaisesRegex(worker.VerifiedPlanError, "newer_governance_failure"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id=success_run_id,
                    max_age_seconds=86400,
                    now=now,
                )

    def test_freshness_ignores_verified_failure_from_future(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            success_time = now - timedelta(minutes=5)
            failure_time = now + timedelta(hours=1)
            success_run_id = success_time.strftime("%Y%m%dT%H%M%S%fZ")
            failure_run_id = failure_time.strftime("%Y%m%dT%H%M%S%fZ")
            self._write_summary(root, success_run_id, "success", success_time, "latest-success.json")
            self._write_summary(root, failure_run_id, "failed", failure_time, "latest-failure.json")

            worker.validate_governance_freshness(
                governance_root=root,
                expected_run_id=success_run_id,
                max_age_seconds=86400,
                now=now,
            )

    def test_freshness_rejects_stale_success(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            success_time = now - timedelta(days=2)
            success_run_id = success_time.strftime("%Y%m%dT%H%M%S%fZ")
            self._write_summary(root, success_run_id, "success", success_time, "latest-success.json")

            with self.assertRaisesRegex(worker.VerifiedPlanError, "governance_run_stale"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id=success_run_id,
                    max_age_seconds=86400,
                    now=now,
                )

    def test_freshness_rejects_run_id_and_started_at_mismatch_like_api(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        run_time = now - timedelta(minutes=2)
        run_id = run_time.strftime("%Y%m%dT%H%M%S%fZ")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_summary(
                root,
                run_id,
                "success",
                run_time + timedelta(seconds=1),
                "latest-success.json",
            )

            with self.assertRaisesRegex(worker.VerifiedPlanError, "governance_start_time_mismatch"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id=run_id,
                    max_age_seconds=86400,
                    now=now,
                )

    def test_freshness_rejects_schema_api_would_reject(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        run_time = now - timedelta(minutes=2)
        run_id = run_time.strftime("%Y%m%dT%H%M%S%fZ")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_summary(root, run_id, "success", run_time, "latest-success.json")
            pointer_path = root / "latest-success.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["schema_version"] = 2
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            with self.assertRaisesRegex(worker.VerifiedPlanError, "governance_pointer_schema_invalid"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id=run_id,
                    max_age_seconds=86400,
                    now=now,
                )

    def test_worker_max_age_uses_same_required_environment_name_as_api(self):
        with patch.dict(
            "os.environ",
            {"LITELLM_GOVERNANCE_MAX_AGE_SECONDS": "7200"},
            clear=True,
        ):
            args = worker._parse_args(["--governance-root", "/governance", "--acceptance-dir", "/acceptance"])
        self.assertEqual(args.governance_max_age_seconds, 7200)

        with patch.dict("os.environ", {}, clear=True), self.assertRaises(SystemExit):
            worker._parse_args(["--governance-root", "/governance", "--acceptance-dir", "/acceptance"])

    @staticmethod
    def _write_summary(root, run_id, status, started_at, pointer_name):
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "status": status,
        }
        manifest = {
            "schema_version": 1,
            "artifacts": {
                "cycle-summary.json": {
                    "sha256": canonical_sha256(summary),
                    "record_count": 1,
                }
            },
        }
        (run_dir / "cycle-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        pointer = {
            "schema_version": 1,
            "run_id": run_id,
            "run_path": f"runs/{run_id}",
            "summary_sha256": canonical_sha256(summary),
            "manifest_sha256": canonical_sha256(manifest),
        }
        (root / pointer_name).write_text(json.dumps(pointer), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
