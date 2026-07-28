import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_litellm_candidate_preflight as preflight
from scripts import execute_litellm_candidate_admission as executor
from scripts import plan_litellm_candidate_admission as planner


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def model_entry(alias, underlying, model_uuid, *, api_base=None, metadata=None):
    return {
        "model_name": alias,
        "litellm_params": {"model": underlying, "api_base": api_base},
        "model_info": {
            "id": model_uuid,
            "db_model": True,
            "metadata": metadata
            or {
                "provider_key": "moonshot",
                "provider_display": "Moonshot",
                "capabilities": {"functionCalling": True},
                "pricing": {"input": 0.6, "output": 2.5, "unit": "USD"},
            },
        },
    }


def candidate_contract():
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
            "reasons": [],
        },
        "isolation_status": "candidate",
        "metadata": {
            "display_name": "Kimi K3",
            "provider_key": "moonshot",
            "provider_display": "Moonshot",
            "capabilities": {"functionCalling": True},
            "pricing": {"input": 0.6, "output": 2.5, "unit": "USD"},
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


def canonical_sha256(payload):
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def write_verified_governance_root(root):
    candidate = candidate_contract()
    fingerprint = preflight.candidate_contract_fingerprint(candidate)
    eligible = planner.build_eligible_entry(candidate)
    run_id = "20260728T160000000000Z"
    plan = {
        "run_id": run_id,
        "status": "complete",
        "mode": "dry_run",
        "writes_performed": False,
        "http_requests_performed": False,
        "eligible": [eligible],
        "isolated": [],
        "pipeline_issues": [],
        "candidates_evaluated": 1,
        "allowlist_plan": {"add": ["kimi-k3"], "remove": []},
        "post_registration_fusion_gate": {
            "status": "required",
            "stage": "fusion_post_registration",
            "affects_pre_registration_eligibility": False,
        },
    }
    queue = [
        {
            "provider_key": "moonshot",
            "model_id": "kimi-k3",
            "state": "admission_ready",
            "reasons": [],
            "candidate_snapshot_sha256": canonical_sha256(candidate),
            "candidate_fingerprint": fingerprint,
            "acceptance_file": f"{fingerprint}.json",
            "candidate": candidate,
        }
    ]
    plans = {fingerprint: plan}
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "candidate-queue.json").write_text(json.dumps(queue), encoding="utf-8")
    (run_dir / "admission-plans.json").write_text(json.dumps(plans), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifacts": {
            "candidate-queue.json": {
                "sha256": canonical_sha256(queue),
                "record_count": 1,
            },
            "admission-plans.json": {
                "sha256": canonical_sha256(plans),
                "record_count": 1,
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "run_id": run_id,
        "run_path": f"runs/{run_id}",
        "manifest_sha256": canonical_sha256(manifest),
    }
    (root / "latest-success.json").write_text(json.dumps(pointer), encoding="utf-8")
    return {
        "fingerprint": fingerprint,
        "run_id": run_id,
        "run_dir": run_dir,
        "plan": plan,
        "queue": queue,
        "plans": plans,
        "manifest": manifest,
        "pointer": pointer,
    }


def rewrite_hash_chain(root, artifacts):
    run_dir = artifacts["run_dir"]
    queue = json.loads((run_dir / "candidate-queue.json").read_text(encoding="utf-8"))
    plans = json.loads((run_dir / "admission-plans.json").read_text(encoding="utf-8"))
    manifest = artifacts["manifest"]
    manifest["artifacts"]["candidate-queue.json"]["sha256"] = canonical_sha256(queue)
    manifest["artifacts"]["admission-plans.json"]["sha256"] = canonical_sha256(plans)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = artifacts["pointer"]
    pointer["manifest_sha256"] = canonical_sha256(manifest)
    (root / "latest-success.json").write_text(json.dumps(pointer), encoding="utf-8")


def admission_plan():
    eligible = {
        "model_id": "kimi-k3",
        "provider_key": "moonshot",
        "model_new_plan": {
            "mode": "dry_run",
            "endpoint": "/model/new",
            "payload": {
                "model_name": "kimi-k3",
                "litellm_params": {
                    "model": "moonshot/kimi-k3",
                    "api_base": "https://api.moonshot.cn/v1",
                    "api_key": "os.environ/MOONSHOT_API_KEY",
                },
                "model_info": {
                    "metadata": {
                        "provider_key": "moonshot",
                        "provider_display": "Moonshot",
                        "capabilities": {"functionCalling": True},
                        "pricing": {"input": 0.6, "output": 2.5, "unit": "USD"},
                        "source": "fusion-governance-v1",
                        "governance": {
                            "candidate_fingerprint": "a" * 64,
                            "metadata_evidence": {"cost_map_matched": True},
                        },
                    }
                },
            },
        },
    }
    return {
        "run_id": "run-20260728-001",
        "mode": "dry_run",
        "eligible": [eligible],
        "isolated": [],
        "allowlist_plan": {"add": ["kimi-k3"], "remove": []},
    }


class StatefulClient:
    def __init__(
        self,
        *,
        fail_stage=None,
        cas_conflict=False,
        concurrent_key_change=False,
        omit_created_uuid=False,
        readback_mismatch=None,
        final_readback_mismatch=False,
    ):
        self.fail_stage = fail_stage
        self.cas_conflict = cas_conflict
        self.concurrent_key_change = concurrent_key_change
        self.omit_created_uuid = omit_created_uuid
        self.readback_mismatch = readback_mismatch
        self.final_readback_mismatch = final_readback_mismatch
        self.entries = [model_entry("deepseek-chat", "deepseek/deepseek-chat", "uuid-deepseek")]
        self.key_models = ["deepseek-chat"]
        self.calls = []
        self.model_info_reads = 0
        self.key_updates = 0
        self.hide_candidate_once = fail_stage == "verify"

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/model/info"):
            self.model_info_reads += 1
            if self.cas_conflict and self.model_info_reads == 2:
                self.entries.append(model_entry("other-new", "moonshot/other-new", "uuid-other"))
            if self.final_readback_mismatch and self.model_info_reads == 4:
                candidate_entry = next(item for item in self.entries if item["model_name"] == "kimi-k3")
                candidate_entry["model_info"]["metadata"]["governance"]["candidate_fingerprint"] = "b" * 64
            entries = list(self.entries)
            if self.hide_candidate_once and any(item["model_name"] == "kimi-k3" for item in entries):
                self.hide_candidate_once = False
                entries = [item for item in entries if item["model_name"] != "kimi-k3"]
            return FakeResponse({"data": entries})
        if "/key/info?" in url:
            return FakeResponse({"info": {"models": list(self.key_models)}})
        if url.endswith("/api/models/"):
            if self.fail_stage == "fusion_readback":
                raise RuntimeError("fusion readback failed with secret")
            models = [{"modelId": item["model_name"]} for item in self.entries if item["model_name"] in self.key_models]
            return FakeResponse({"data": {"models": models}})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/model/new"):
            payload = kwargs["json"]
            metadata = copy.deepcopy(payload["model_info"]["metadata"])
            api_base = payload["litellm_params"]["api_base"]
            if self.readback_mismatch == "api_base":
                api_base = "https://wrong.example/v1"
            elif self.readback_mismatch == "metadata":
                metadata["governance"]["candidate_fingerprint"] = "b" * 64
            self.entries.append(
                model_entry(
                    payload["model_name"],
                    payload["litellm_params"]["model"],
                    "uuid-kimi-k3",
                    api_base=api_base,
                    metadata=metadata,
                )
            )
            if self.fail_stage == "model_new":
                if self.concurrent_key_change:
                    self.key_models.append("external-model")
                raise RuntimeError("model new failed after write")
            if self.omit_created_uuid:
                return FakeResponse({"status": "ok"})
            return FakeResponse({"model_info": {"id": "uuid-kimi-k3"}})
        if url.endswith("/key/update"):
            self.key_updates += 1
            self.key_models = list(kwargs["json"]["models"])
            if self.fail_stage == "key_update" and self.key_updates == 1:
                raise RuntimeError("key update failed after write")
            return FakeResponse({"status": "ok"})
        if url.endswith("/model/delete"):
            model_uuid = kwargs["json"]["id"]
            self.entries = [item for item in self.entries if item["model_info"]["id"] != model_uuid]
            return FakeResponse({"status": "ok"})
        raise AssertionError(f"unexpected POST {url}")


def execute(
    client,
    *,
    apply=True,
    expected_run_id="run-20260728-001",
    confirm_model_id="kimi-k3",
    fingerprint="a" * 64,
    audit_fn=None,
):
    return executor.execute_admission(
        plan=admission_plan(),
        apply=apply,
        expected_run_id=expected_run_id,
        confirm_model_id=confirm_model_id,
        confirm_fingerprint=fingerprint,
        litellm_base_url="http://litellm.internal:4000",
        fusion_base_url="https://fusion.example",
        master_key="master-super-secret",
        virtual_key="virtual-super-secret",
        environ={"MOONSHOT_API_KEY": "provider-super-secret"},
        client=client,
        audit_fn=audit_fn or (lambda **_: True),
    )


class CandidateAdmissionExecutorTests(unittest.TestCase):
    def test_verified_governance_root_loads_exact_admission_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)

            plan = executor.load_verified_admission_plan(
                governance_root=root,
                candidate_fingerprint=artifacts["fingerprint"],
            )

            self.assertEqual(plan, artifacts["plan"])

    def test_cli_apply_rejects_arbitrary_plan_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "untrusted-plan.json"
            output_path = root / "result.json"
            plan_path.write_text(json.dumps(admission_plan()), encoding="utf-8")

            exit_code = executor.main(
                [
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(output_path),
                    "--apply",
                ]
            )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(result["status"], "verification_failed")
            self.assertEqual(result["error"]["code"], "apply_plan_argument_forbidden")
            self.assertFalse(result["writes_performed"])

    def test_verified_loader_rejects_run_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            pointer = artifacts["pointer"]
            pointer["run_path"] = "../outside"
            (root / "latest-success.json").write_text(json.dumps(pointer), encoding="utf-8")

            with self.assertRaisesRegex(executor.VerifiedPlanError, "run_path_invalid"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_verified_loader_rejects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            manifest_path = artifacts["run_dir"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(executor.VerifiedPlanError, "manifest_sha256_mismatch"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_verified_loader_rejects_tampered_artifact_sha(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            plan_path = artifacts["run_dir"] / "admission-plans.json"
            plans = json.loads(plan_path.read_text(encoding="utf-8"))
            plans[artifacts["fingerprint"]]["allowlist_plan"]["add"] = ["tampered"]
            plan_path.write_text(json.dumps(plans), encoding="utf-8")

            with self.assertRaisesRegex(executor.VerifiedPlanError, "admission_plans_sha256_mismatch"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_verified_loader_rejects_hash_valid_but_tampered_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            queue_path = artifacts["run_dir"] / "candidate-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["state"] = "preflight_required"
            queue_path.write_text(json.dumps(queue), encoding="utf-8")
            rewrite_hash_chain(root, artifacts)

            with self.assertRaisesRegex(executor.VerifiedPlanError, "candidate_not_admission_ready"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_verified_loader_recomputes_queue_candidate_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            queue_path = artifacts["run_dir"] / "candidate-queue.json"
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            queue[0]["candidate"]["registration"]["api_base"] = "https://evil.example/v1"
            queue[0]["candidate_snapshot_sha256"] = canonical_sha256(queue[0]["candidate"])
            queue_path.write_text(json.dumps(queue), encoding="utf-8")
            rewrite_hash_chain(root, artifacts)

            with self.assertRaisesRegex(executor.VerifiedPlanError, "candidate_fingerprint_mismatch"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_verified_loader_rejects_hash_valid_but_tampered_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts = write_verified_governance_root(root)
            plans_path = artifacts["run_dir"] / "admission-plans.json"
            plans = json.loads(plans_path.read_text(encoding="utf-8"))
            plans[artifacts["fingerprint"]]["eligible"][0]["model_new_plan"]["payload"]["litellm_params"][
                "api_base"
            ] = "https://evil.example/v1"
            plans_path.write_text(json.dumps(plans), encoding="utf-8")
            rewrite_hash_chain(root, artifacts)

            with self.assertRaisesRegex(executor.VerifiedPlanError, "eligible_entry_mismatch"):
                executor.load_verified_admission_plan(
                    governance_root=root,
                    candidate_fingerprint=artifacts["fingerprint"],
                )

    def test_default_dry_run_performs_zero_http_requests(self):
        client = StatefulClient()

        result = execute(client, apply=False, expected_run_id="", confirm_model_id="", fingerprint="")

        self.assertEqual(result["status"], "planned")
        self.assertFalse(result["writes_performed"])
        self.assertEqual(client.calls, [])

    def test_apply_requires_matching_run_model_and_fingerprint_confirmations(self):
        cases = [
            ("wrong-run", "kimi-k3", "a" * 64, "run_id_mismatch"),
            ("run-20260728-001", "wrong-model", "a" * 64, "model_id_mismatch"),
            ("run-20260728-001", "kimi-k3", "b" * 64, "fingerprint_mismatch"),
        ]
        for run_id, model_id, fingerprint, error_code in cases:
            with self.subTest(error_code=error_code):
                client = StatefulClient()
                result = execute(
                    client,
                    expected_run_id=run_id,
                    confirm_model_id=model_id,
                    fingerprint=fingerprint,
                )
                self.assertEqual(result["status"], "authorization_failed")
                self.assertEqual(result["error"]["code"], error_code)
                self.assertEqual(client.calls, [])

    def test_apply_rejects_plan_without_run_id(self):
        client = StatefulClient()
        plan = admission_plan()
        plan.pop("run_id")

        result = executor.execute_admission(
            plan=plan,
            apply=True,
            expected_run_id="",
            confirm_model_id="kimi-k3",
            confirm_fingerprint="a" * 64,
            litellm_base_url="http://litellm.internal:4000",
            fusion_base_url="https://fusion.example",
            master_key="master-super-secret",
            virtual_key="virtual-super-secret",
            environ={"MOONSHOT_API_KEY": "provider-super-secret"},
            client=client,
        )

        self.assertEqual(result["status"], "authorization_failed")
        self.assertEqual(result["error"]["code"], "run_id_missing")
        self.assertEqual(client.calls, [])

    def test_cas_conflict_stops_before_any_write(self):
        client = StatefulClient(cas_conflict=True)

        result = execute(client)

        self.assertEqual(result["status"], "cas_conflict")
        self.assertEqual(result["error"]["code"], "before_state_changed")
        self.assertFalse([call for call in client.calls if call[0] == "POST"])

    def test_full_state_machine_succeeds(self):
        client = StatefulClient()

        result = execute(client)

        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["writes_performed"])
        self.assertEqual(result["completed_phases"], ["model_new", "verify", "key_update", "fusion_readback", "audit"])
        self.assertIn("kimi-k3", client.key_models)
        self.assertTrue(any(item["model_name"] == "kimi-k3" for item in client.entries))
        self.assertFalse(result["compensation"]["attempted"])

    def test_model_new_keeps_environment_reference_out_of_database_payload(self):
        client = StatefulClient()

        result = execute(client)

        self.assertEqual(result["status"], "succeeded")
        model_new_call = next(call for call in client.calls if call[0] == "POST" and call[1].endswith("/model/new"))
        payload = model_new_call[2]["json"]
        self.assertEqual(payload["litellm_params"]["api_key"], "os.environ/MOONSHOT_API_KEY")
        self.assertNotIn("provider-super-secret", json.dumps(payload))

    def test_each_mutating_stage_failure_rolls_back_model_and_allowlist(self):
        for stage in ("verify", "key_update", "fusion_readback", "audit"):
            with self.subTest(stage=stage):
                client = StatefulClient(fail_stage=None if stage == "audit" else stage)
                audit_fn = (lambda **_: False) if stage == "audit" else None

                result = execute(client, audit_fn=audit_fn)

                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["compensation"]["attempted"])
                self.assertTrue(result["compensation"]["key_restored"])
                self.assertTrue(result["compensation"]["model_deleted"])
                self.assertEqual(client.key_models, ["deepseek-chat"])
                self.assertFalse(any(item["model_name"] == "kimi-k3" for item in client.entries))

    def test_uncertain_model_new_failure_never_deletes_by_alias(self):
        client = StatefulClient(fail_stage="model_new")

        result = execute(client)

        model_delete_calls = [call for call in client.calls if call[0] == "POST" and call[1].endswith("/model/delete")]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["compensation"]["model_ownership_unverified"])
        self.assertTrue(result["compensation"]["manual_cleanup_required"])
        self.assertEqual(model_delete_calls, [])
        self.assertTrue(any(item["model_name"] == "kimi-k3" for item in client.entries))

    def test_compensation_does_not_overwrite_concurrent_key_change(self):
        client = StatefulClient(fail_stage="model_new", concurrent_key_change=True)

        result = execute(client)

        key_update_calls = [call for call in client.calls if call[0] == "POST" and call[1].endswith("/key/update")]
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["compensation"]["key_restored"])
        self.assertIn("rollback_key_cas_conflict", result["compensation"]["errors"])
        self.assertEqual(key_update_calls, [])
        self.assertEqual(client.key_models, ["deepseek-chat", "external-model"])
        self.assertTrue(result["compensation"]["manual_cleanup_required"])
        self.assertTrue(any(item["model_name"] == "kimi-k3" for item in client.entries))

    def test_success_without_response_uuid_blocks_allowlist_and_requires_manual_cleanup(self):
        client = StatefulClient(omit_created_uuid=True)

        result = execute(client)

        key_update_calls = [call for call in client.calls if call[0] == "POST" and call[1].endswith("/key/update")]
        model_delete_calls = [call for call in client.calls if call[0] == "POST" and call[1].endswith("/model/delete")]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "model_ownership_unverified")
        self.assertEqual(result["owned_model_uuid"], "")
        self.assertEqual(key_update_calls, [])
        self.assertEqual(model_delete_calls, [])
        self.assertTrue(result["compensation"]["manual_cleanup_required"])

    def test_readback_api_base_or_metadata_mismatch_blocks_allowlist(self):
        for mismatch in ("api_base", "metadata"):
            with self.subTest(mismatch=mismatch):
                client = StatefulClient(readback_mismatch=mismatch)

                result = execute(client)

                key_update_calls = [
                    call for call in client.calls if call[0] == "POST" and call[1].endswith("/key/update")
                ]
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["phase"], "verify")
                self.assertEqual(key_update_calls, [])
                self.assertTrue(result["compensation"]["model_deleted"])

    def test_final_pre_audit_readback_mismatch_rolls_back(self):
        client = StatefulClient(final_readback_mismatch=True)

        result = execute(client)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["phase"], "audit")
        self.assertEqual(result["error"]["code"], "model_verify_failed")
        self.assertEqual(client.key_models, ["deepseek-chat"])
        self.assertTrue(result["compensation"]["model_deleted"])

    def test_secrets_never_appear_in_transaction_result(self):
        client = StatefulClient(fail_stage="model_new")

        result = execute(client)
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertNotIn("master-super-secret", serialized)
        self.assertNotIn("virtual-super-secret", serialized)
        self.assertNotIn("provider-super-secret", serialized)
        self.assertNotIn("os.environ/MOONSHOT_API_KEY", serialized)


if __name__ == "__main__":
    unittest.main()
