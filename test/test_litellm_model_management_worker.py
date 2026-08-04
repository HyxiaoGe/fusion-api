import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts import run_litellm_model_management_worker as worker


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

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/claim"):
            if self.claim_payload is None:
                return FakeResponse(status_code=204)
            return FakeResponse(self.claim_payload)
        if url.endswith("/invalidate-catalog"):
            return FakeResponse({"generation": "9"})
        if url.endswith("/complete"):
            if self.complete_failures > 0:
                self.complete_failures -= 1
                return FakeResponse({"detail": "temporary"}, status_code=503)
            return FakeResponse({"status": "accepted"})
        raise AssertionError(f"unexpected POST {url}")


def claim_payload():
    return {
        "operation_id": "op-123",
        "lease_token": "lease-secret",
        "run_id": "20260804T010000000000Z",
        "candidate_fingerprint": "a" * 64,
        "model_id": "qwen3.8-max",
    }


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
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


class ModelManagementWorkerTests(unittest.TestCase):
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
            patch.object(worker, "load_verified_admission_plan", return_value={"run_id": claim_payload()["run_id"]}),
            patch.object(worker, "validate_governance_freshness"),
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
            patch.object(worker, "load_verified_admission_plan", return_value={"run_id": claim_payload()["run_id"]}),
            patch.object(worker, "validate_governance_freshness"),
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
                patch.object(worker, "validate_governance_freshness"),
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
            self._write_summary(root, "success-run", "success", now - timedelta(hours=1), "latest-success.json")
            self._write_summary(root, "failure-run", "failed", now - timedelta(minutes=5), "latest-failure.json")

            with self.assertRaisesRegex(worker.VerifiedPlanError, "newer_governance_failure"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id="success-run",
                    max_age_seconds=86400,
                    now=now,
                )

    def test_freshness_rejects_stale_success(self):
        now = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_summary(root, "success-run", "success", now - timedelta(days=2), "latest-success.json")

            with self.assertRaisesRegex(worker.VerifiedPlanError, "governance_run_stale"):
                worker.validate_governance_freshness(
                    governance_root=root,
                    expected_run_id="success-run",
                    max_age_seconds=86400,
                    now=now,
                )

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
                    "sha256": worker.canonical_sha256(summary),
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
            "summary_sha256": worker.canonical_sha256(summary),
            "manifest_sha256": worker.canonical_sha256(manifest),
        }
        (root / pointer_name).write_text(json.dumps(pointer), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
