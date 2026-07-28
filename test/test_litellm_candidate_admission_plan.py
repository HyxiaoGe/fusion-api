import contextlib
import io
import json
import unittest

from scripts import check_litellm_candidate_preflight as preflight
from scripts import plan_litellm_candidate_admission as admission


def kimi_candidate(*, pricing=True, endpoint_status="verified", extra=None):
    metadata = {
        "display_name": "Kimi K3",
        "provider_key": "moonshot",
        "provider_display": "Moonshot",
        "capabilities": {
            "deepThinking": True,
            "functionCalling": True,
            "vision": True,
        },
        "pricing": {"input": 0.6, "output": 2.5, "unit": "USD"} if pricing else {},
    }
    candidate = {
        "provider_key": "moonshot",
        "provider_display": "Moonshot",
        "model_id": "kimi-k3",
        "litellm_model": "moonshot/kimi-k3",
        "isolation_status": "candidate",
        "eligible_for_registration": False,
        "metadata": metadata,
        "metadata_evidence": {
            "cost_map_matched": True,
            "reviewed_override_applied": False,
        },
        "registration": {
            "api_base": "https://api.moonshot.cn/v1",
            "api_key_env": "MOONSHOT_API_KEY",
            "endpoint_status": endpoint_status,
        },
    }
    candidate.update(extra or {})
    return candidate


def candidate_report(candidate=None, *, provider_status="ok"):
    return {
        "mode": "read_only",
        "providers": {
            "moonshot": {
                "status": provider_status,
                "report": {
                    "provider": {"key": "moonshot", "display": "Moonshot"},
                    "new": [candidate or kimi_candidate()],
                    "existing": [],
                    "removed": [],
                    "unknown": [],
                },
            }
        },
    }


def candidate_acceptance(*, model_id="kimi-k3", failed=False, high_risk=False):
    candidate = kimi_candidate()
    if model_id != candidate["model_id"]:
        candidate["model_id"] = model_id
    summary = {
        "acceptance_stage": "candidate_pre_registration",
        "transport": "litellm_wildcard",
        "healthy": not failed,
        "dry_run": False,
        "candidate_fingerprint": preflight.candidate_contract_fingerprint(candidate),
        "total": 5,
        "success_count": 4 if failed else 5,
        "failure_count": 1 if failed else 0,
        "skipped_count": 0,
        "by_model": {
            model_id: {
                "total": 5,
                "success_count": 4 if failed else 5,
                "failure_count": 1 if failed else 0,
                "skipped_count": 0,
            }
        },
        "candidate_checks_by_model": {
            model_id: {
                "text": True,
                "stream": True,
                "tool": True,
                "usage": True,
                "cost": True,
            }
        },
        "quality_issues": [],
        "tool_expectation_mismatch_count": 0,
    }
    if high_risk:
        summary["quality_issues"] = [
            {
                "model_id": model_id,
                "scenario_id": "tool_round",
                "severity": "high",
                "flags": ["reasoning_tag_leak"],
            }
        ]
    return summary


class CandidateAdmissionPlanTests(unittest.TestCase):
    def test_kimi_k3_is_eligible_with_complete_metadata_and_candidate_acceptance(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["mode"], "dry_run")
        self.assertEqual(plan["status"], "complete")
        self.assertFalse(plan["writes_performed"])
        self.assertEqual([item["model_id"] for item in plan["eligible"]], ["kimi-k3"])
        model_new = plan["eligible"][0]["model_new_plan"]
        self.assertEqual(model_new["endpoint"], "/model/new")
        self.assertEqual(model_new["minimum_litellm_version"], "1.93.0")
        self.assertEqual(model_new["payload"]["litellm_params"]["model"], "moonshot/kimi-k3")
        self.assertEqual(
            model_new["payload"]["litellm_params"]["api_key"],
            "os.environ/MOONSHOT_API_KEY",
        )
        self.assertNotIn("api_key_env", model_new["payload"]["litellm_params"])
        self.assertEqual(plan["allowlist_plan"], {"add": ["kimi-k3"], "remove": []})
        self.assertEqual(plan["post_registration_fusion_gate"]["status"], "required")

    def test_applied_preflight_serialization_is_directly_consumable(self):
        report = preflight.PreflightReport(
            healthy=True,
            dry_run=False,
            candidate=preflight.parse_candidate(kimi_candidate()),
            cases=[
                preflight.CaseResult(
                    name=name,
                    status="passed",
                    usage_present=True,
                    cost_present=True,
                    output_present=True,
                    stream_done=True if name == "stream" else None,
                )
                for name in ("text", "stream", "tool_calling")
            ],
        )

        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=preflight.serialize_report(report, context={}),
        )

        self.assertEqual([item["model_id"] for item in plan["eligible"]], ["kimi-k3"])

    def test_failed_candidate_acceptance_is_isolated(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=candidate_acceptance(failed=True),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("candidate_acceptance_failed", plan["isolated"][0]["reasons"])

    def test_provider_failure_is_reported_as_pipeline_failure(self):
        report = candidate_report(provider_status="error")
        report["providers"]["moonshot"]["error"] = {"code": "upstream_request_failed"}
        report["providers"]["moonshot"]["report"]["new"] = []

        plan = admission.build_admission_plan(
            candidate_report=report,
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["status"], "failed")
        self.assertEqual(plan["candidates_evaluated"], 0)
        self.assertEqual(
            plan["pipeline_issues"],
            [{"provider_key": "moonshot", "code": "upstream_request_failed"}],
        )

    def test_high_risk_quality_flag_is_isolated(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=candidate_acceptance(high_risk=True),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("high_quality_risk", plan["isolated"][0]["reasons"])

    def test_missing_pricing_is_isolated(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(kimi_candidate(pricing=False)),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("metadata_pricing_incomplete", plan["isolated"][0]["reasons"])

    def test_metadata_without_provenance_is_isolated(self):
        candidate = kimi_candidate()
        candidate.pop("metadata_evidence")

        plan = admission.build_admission_plan(
            candidate_report=candidate_report(candidate),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("metadata_evidence_missing", plan["isolated"][0]["reasons"])

    def test_unknown_endpoint_is_isolated(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(kimi_candidate(endpoint_status="unknown")),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("registration_endpoint_unverified", plan["isolated"][0]["reasons"])

    def test_acceptance_model_mismatch_is_isolated(self):
        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=candidate_acceptance(model_id="kimi-k2.6"),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("candidate_acceptance_model_missing", plan["isolated"][0]["reasons"])

    def test_acceptance_contract_change_is_isolated(self):
        candidate = kimi_candidate()
        candidate["litellm_model"] = "moonshot/kimi-k3-revised"

        plan = admission.build_admission_plan(
            candidate_report=candidate_report(candidate),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("candidate_acceptance_contract_mismatch", plan["isolated"][0]["reasons"])

    def test_raw_api_key_is_rejected_and_never_leaks_to_plan(self):
        secret = "sk-moonshot-secret"
        candidate = kimi_candidate(extra={"registration": {**kimi_candidate()["registration"], "api_key": secret}})

        plan = admission.build_admission_plan(
            candidate_report=candidate_report(candidate),
            candidate_acceptance_summary=candidate_acceptance(),
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("raw_api_key_forbidden", plan["isolated"][0]["reasons"])
        self.assertNotIn(secret, json.dumps(plan, ensure_ascii=False))

    def test_fusion_summary_cannot_replace_candidate_pre_registration_acceptance(self):
        fusion_summary = {
            "by_model": {"kimi-k3": {"total": 5, "success_count": 5, "failure_count": 0, "skipped_count": 0}},
            "quality_issues": [],
            "tool_expectation_mismatch_count": 0,
        }
        wrong_stage = {**candidate_acceptance(), "acceptance_stage": "fusion_post_registration"}

        plan = admission.build_admission_plan(
            candidate_report=candidate_report(),
            candidate_acceptance_summary=wrong_stage,
            fusion_acceptance_summary=fusion_summary,
        )

        self.assertEqual(plan["eligible"], [])
        self.assertIn("candidate_acceptance_stage_invalid", plan["isolated"][0]["reasons"])
        self.assertEqual(plan["post_registration_fusion_gate"]["status"], "observed")

    def test_cli_rejects_apply_argument(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            admission._parse_args(
                [
                    "--candidate-report",
                    "candidates.json",
                    "--candidate-acceptance-summary",
                    "candidate-acceptance.json",
                    "--output",
                    "plan.json",
                    "--apply",
                ]
            )


if __name__ == "__main__":
    unittest.main()
