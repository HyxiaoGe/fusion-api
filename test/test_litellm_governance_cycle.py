import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from scripts import check_litellm_candidate_preflight as preflight
from scripts import run_litellm_governance_cycle as cycle


class FakeResponse:
    headers = {"etag": '"test-etag"'}


def registry():
    return {
        "litellm": {
            "base_url": "http://litellm.internal:4000",
            "api_key_env": "LITELLM_MASTER_KEY",
        },
        "providers": [
            {
                "adapter": "moonshot",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key_env": "MOONSHOT_API_KEY",
            }
        ],
    }


def candidate_report(*, status="ok", complete_metadata=True):
    candidate = {
        "provider_key": "moonshot",
        "provider_display": "Moonshot",
        "model_id": "kimi-k3",
        "litellm_model": "moonshot/kimi-k3",
        "isolation_status": "candidate",
        "eligible_for_registration": False,
    }
    if complete_metadata:
        candidate.update(
            {
                "metadata": {
                    "display_name": "Kimi K3",
                    "provider_key": "moonshot",
                    "provider_display": "Moonshot",
                    "capabilities": {"functionCalling": True, "vision": True},
                    "pricing": {"input": 3.0, "output": 15.0, "unit": "USD/1M tokens"},
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
            }
        )
    return {
        "mode": "read_only",
        "candidate_preflight": {
            "status": "ready",
            "transport": "litellm_provider_wildcard",
            "credential_source": "LITELLM_CANDIDATE_KEY",
            "reasons": [],
            "providers": {
                "moonshot": {
                    "status": "ready",
                    "route_model_name": "candidate/moonshot/*",
                    "route_litellm_model": "moonshot/*",
                    "api_base": "https://api.moonshot.cn/v1",
                    "api_key_env": "MOONSHOT_API_KEY",
                    "credential_generation": "test-v1",
                    "reasons": [],
                }
            },
        },
        "summary": {
            "providers_total": 1,
            "providers_ok": 1 if status == "ok" else 0,
            "providers_failed": 0 if status == "ok" else 1,
        },
        "providers": {
            "moonshot": {
                "status": status,
                "error": None if status == "ok" else {"code": "upstream_request_failed"},
                "report": {
                    "provider": {"key": "moonshot", "display": "Moonshot"},
                    "new": [candidate] if status == "ok" else [],
                    "existing": [],
                    "removed": [],
                    "unknown": [],
                },
            }
        },
    }


def cost_map():
    return {
        "moonshot/kimi-k3": {
            "litellm_provider": "moonshot",
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_reasoning": True,
        }
    }


def acceptance(candidate):
    report = preflight.PreflightReport(
        healthy=True,
        dry_run=False,
        candidate=preflight.parse_candidate(candidate),
        cases=[
            preflight.CaseResult(
                name=name,
                status="passed",
                usage_present=True,
                cost_present=True,
                output_present=True,
                stream_done=True if name == "stream" else None,
                reasoning_present=True if name == "reasoning" else None,
            )
            for name in (
                "text",
                "stream",
                "tool_calling",
                "vision",
                "reasoning",
                "preserved_tool_round",
            )
        ],
    )
    return preflight.serialize_report(report, context={})


class LiteLLMGovernanceCycleTests(unittest.TestCase):
    def test_pointer_never_regresses_to_an_older_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pointer_path = Path(temp_dir) / "latest-success.json"
            newer = {"run_id": "20260728T160000000002Z"}
            older = {"run_id": "20260728T160000000001Z"}

            self.assertTrue(cycle.write_pointer_monotonic(pointer_path, newer))
            self.assertFalse(cycle.write_pointer_monotonic(pointer_path, older))
            self.assertEqual(json.loads(pointer_path.read_text(encoding="utf-8")), newer)

    def test_pointer_rejects_malformed_existing_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pointer_path = Path(temp_dir) / "latest-success.json"
            pointer_path.write_text(json.dumps({"run_id": "corrupted"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "现有治理指针"):
                cycle.write_pointer_monotonic(pointer_path, {"run_id": "20260728T160000000002Z"})

    def _run(self, output_dir, report, *, acceptance_dir=None, cost_payload=None):
        return cycle.run_governance_cycle(
            registry=registry(),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "MOONSHOT_API_KEY": "moonshot-secret",
            },
            output_dir=output_dir,
            acceptance_dir=acceptance_dir,
            now=datetime(2026, 7, 28, 16, 0, tzinfo=UTC),
            cost_map_fetcher=lambda **_: (cost_payload or cost_map(), FakeResponse()),
            candidate_coordinator=lambda **_: report,
        )

    def test_successful_cycle_writes_auditable_artifacts_and_preflight_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            summary = self._run(output_dir, candidate_report())

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["preflight_required"], 1)
            pointer = json.loads((output_dir / "latest-success.json").read_text(encoding="utf-8"))
            run_dir = output_dir / pointer["run_path"]
            queue = json.loads((run_dir / "candidate-queue.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(queue[0]["state"], "preflight_required")
            self.assertEqual(len(queue[0]["candidate_fingerprint"]), 64)
            self.assertTrue((run_dir / "cost-map.json").is_file())
            self.assertIn("candidate-queue.json", manifest["artifacts"])
            self.assertFalse(summary["external_writes_performed"])
            self.assertFalse(summary["paid_requests_performed"])

    def test_incomplete_metadata_is_quarantined_without_acceptance_filename(self):
        report = candidate_report(complete_metadata=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            summary = self._run(
                output_dir,
                report,
                cost_payload={
                    "other/model": {
                        "litellm_provider": "other",
                        "input_cost_per_token": 0.1,
                        "output_cost_per_token": 0.2,
                    }
                },
            )

            pointer = json.loads((output_dir / "latest-success.json").read_text(encoding="utf-8"))
            queue = json.loads((output_dir / pointer["run_path"] / "candidate-queue.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["quarantined"], 1)
            self.assertEqual(queue[0]["state"], "quarantined")
            self.assertIsNone(queue[0]["candidate_fingerprint"])
            self.assertIn("metadata_pricing_incomplete", queue[0]["reasons"])

    def test_fingerprint_named_acceptance_advances_candidate_to_admission_ready(self):
        report = candidate_report()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root / "baseline", report)
            baseline_pointer = json.loads((root / "baseline" / "latest-success.json").read_text(encoding="utf-8"))
            baseline_queue = json.loads(
                (root / "baseline" / baseline_pointer["run_path"] / "candidate-queue.json").read_text(encoding="utf-8")
            )
            candidate = baseline_queue[0]["candidate"]
            fingerprint = baseline_queue[0]["candidate_fingerprint"]
            acceptance_dir = root / "acceptance"
            acceptance_dir.mkdir()
            (acceptance_dir / f"{fingerprint}.json").write_text(
                json.dumps(acceptance(candidate)),
                encoding="utf-8",
            )

            summary = self._run(root / "reports", report, acceptance_dir=acceptance_dir)

            pointer = json.loads((root / "reports" / "latest-success.json").read_text(encoding="utf-8"))
            run_dir = root / "reports" / pointer["run_path"]
            queue = json.loads((run_dir / "candidate-queue.json").read_text(encoding="utf-8"))
            plans = json.loads((run_dir / "admission-plans.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["admission_ready"], 1)
            self.assertEqual(queue[0]["state"], "admission_ready")
            self.assertEqual(plans[fingerprint]["run_id"], pointer["run_id"])
            self.assertEqual(plans[fingerprint]["allowlist_plan"]["add"], ["kimi-k3"])

    def test_provider_failure_does_not_advance_latest_success_pointer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            previous = {"run_id": "previous"}
            (output_dir / "latest-success.json").write_text(json.dumps(previous), encoding="utf-8")

            summary = self._run(output_dir, candidate_report(status="error"))

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(
                json.loads((output_dir / "latest-success.json").read_text(encoding="utf-8")),
                previous,
            )

    def test_cost_fetch_exception_is_persisted_without_advancing_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            previous = {"run_id": "previous"}
            output_dir.mkdir(exist_ok=True)
            (output_dir / "latest-success.json").write_text(json.dumps(previous), encoding="utf-8")

            summary = cycle.run_governance_cycle(
                registry=registry(),
                environ={},
                output_dir=output_dir,
                now=datetime(2026, 7, 28, 16, 0, tzinfo=UTC),
                cost_map_fetcher=lambda **_: (_ for _ in ()).throw(OSError("secret upstream URL")),
            )

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["issues"][0]["code"], "governance_cycle_exception")
            self.assertNotIn("secret upstream URL", json.dumps(summary))
            self.assertEqual(
                json.loads((output_dir / "latest-success.json").read_text(encoding="utf-8")),
                previous,
            )
            failure = json.loads((output_dir / "latest-failure.json").read_text(encoding="utf-8"))
            run_dir = output_dir / failure["run_path"]
            self.assertTrue((run_dir / "cycle-summary.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "latest-failure.json").is_file())

    def test_removed_models_enter_manual_retirement_review_without_delete_plan(self):
        report = candidate_report()
        report["providers"]["moonshot"]["report"]["removed"] = [
            {
                "provider_key": "moonshot",
                "model_id": "kimi-old",
                "litellm_model": "moonshot/kimi-old",
                "isolation_status": "retirement_review",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            summary = self._run(output_dir, report)

            pointer = json.loads((output_dir / "latest-success.json").read_text(encoding="utf-8"))
            review = json.loads(
                (output_dir / pointer["run_path"] / "retirement-review.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["retirement_review"], 1)
            self.assertFalse(review[0]["automatic_deletion_allowed"])
            self.assertNotIn("delete", json.dumps(review).lower())

    def test_invalid_acceptance_is_a_failed_cycle_and_never_advances_success(self):
        report = candidate_report()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._run(root / "baseline", report)
            baseline_pointer = json.loads((root / "baseline" / "latest-success.json").read_text(encoding="utf-8"))
            baseline_queue = json.loads(
                (root / "baseline" / baseline_pointer["run_path"] / "candidate-queue.json").read_text(encoding="utf-8")
            )
            fingerprint = baseline_queue[0]["candidate_fingerprint"]
            acceptance_dir = root / "acceptance"
            acceptance_dir.mkdir()
            (acceptance_dir / f"{fingerprint}.json").write_text("{invalid", encoding="utf-8")

            summary = self._run(root / "reports", report, acceptance_dir=acceptance_dir)

            self.assertEqual(summary["status"], "failed")
            self.assertFalse((root / "reports" / "latest-success.json").exists())
            self.assertTrue((root / "reports" / "latest-failure.json").exists())

    def test_cli_has_no_apply_mode(self):
        with self.assertRaises(SystemExit):
            cycle._parse_args(
                [
                    "--registry",
                    "registry.json",
                    "--output-dir",
                    "reports",
                    "--apply",
                ]
            )


if __name__ == "__main__":
    unittest.main()
