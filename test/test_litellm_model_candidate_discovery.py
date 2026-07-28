import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import discover_litellm_model_candidates as discovery


def litellm_entry(
    alias: str,
    underlying_model: str | None,
    *,
    provider_key: str | None = None,
) -> dict:
    metadata = {"provider_key": provider_key} if provider_key else {}
    return {
        "model_name": alias,
        "litellm_params": {"model": underlying_model} if underlying_model else {},
        "model_info": {"db_model": True, "metadata": metadata},
    }


class ModelCandidateDiscoveryTests(unittest.TestCase):
    def test_moonshot_adapter_classifies_new_existing_removed_and_unknown_candidates(self):
        upstream_snapshot = {
            "object": "list",
            "data": [
                {"id": "kimi-k3", "object": "model"},
                {"id": "kimi-k2.6", "object": "model"},
                {"id": "embedding-v1", "object": "model"},
                {"object": "model"},
            ],
        }
        litellm_snapshot = {
            "data": [
                litellm_entry("kimi-k2.5", "moonshot/kimi-k2.6"),
                litellm_entry("kimi-k1-retired", "moonshot/kimi-k1"),
                litellm_entry("deepseek-chat", "deepseek/deepseek-chat"),
            ]
        }

        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot=upstream_snapshot,
            litellm_snapshot=litellm_snapshot,
        )

        self.assertEqual([item.model_id for item in report.new], ["kimi-k3"])
        self.assertEqual([item.model_id for item in report.existing], ["kimi-k2.6"])
        self.assertEqual(report.existing[0].aliases, ("kimi-k2.5",))
        self.assertEqual([item.model_id for item in report.removed], ["kimi-k1"])
        self.assertEqual(
            [(item.source, item.reason) for item in report.unknown],
            [
                ("upstream", "Moonshot 适配器不接纳非 Kimi 模型"),
                ("upstream", "模型条目缺少非空 id"),
            ],
        )

    def test_openai_compatible_adapter_is_configurable_and_isolates_unmappable_litellm_entries(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="acme",
            provider_display="Acme AI",
            litellm_prefix="openai",
            api_model_prefix="acme-",
        )
        upstream_snapshot = {"data": [{"id": "acme-chat"}, {"id": "other-chat"}]}
        litellm_snapshot = {
            "data": [
                litellm_entry("acme-production", "openai/acme-chat", provider_key="acme"),
                litellm_entry("acme-broken", None, provider_key="acme"),
            ]
        }

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot=upstream_snapshot,
            litellm_snapshot=litellm_snapshot,
        )

        self.assertEqual([item.model_id for item in report.existing], ["acme-chat"])
        self.assertEqual(report.new, [])
        self.assertEqual(report.removed, [])
        self.assertEqual(len(report.unknown), 2)
        self.assertEqual(report.unknown[0].source, "upstream")
        self.assertEqual(report.unknown[0].model_id, "other-chat")
        self.assertEqual(report.unknown[1].source, "litellm")
        self.assertEqual(report.unknown[1].model_id, "acme-broken")

    def test_shared_openai_prefix_does_not_claim_other_provider_entries_without_evidence(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="acme",
            provider_display="Acme AI",
            litellm_prefix="openai",
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={"data": [{"id": "acme-chat"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry("acme-chat", "openai/acme-chat", provider_key="acme"),
                    litellm_entry("other-chat", "openai/other-chat"),
                ]
            },
        )

        self.assertEqual([item.model_id for item in report.existing], ["acme-chat"])
        self.assertEqual(report.removed, [])

    def test_duplicate_upstream_model_is_reported_once_and_candidate_is_deduplicated(self):
        adapter = discovery.MoonshotProviderAdapter()

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot=[{"id": "kimi-k3"}, {"id": "kimi-k3"}],
            litellm_snapshot=[],
        )

        self.assertEqual([item.model_id for item in report.new], ["kimi-k3"])
        self.assertEqual(len(report.unknown), 1)
        self.assertEqual(report.unknown[0].reason, "上游快照包含重复模型 id")

    def test_empty_upstream_does_not_mark_all_existing_models_removed(self):
        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot={"data": []},
            litellm_snapshot={
                "data": [
                    litellm_entry("kimi-k2.5", "moonshot/kimi-k2.6"),
                    litellm_entry("kimi-k1", "moonshot/kimi-k1"),
                ]
            },
        )

        self.assertEqual(report.removed, [])
        self.assertEqual(len(report.unknown), 1)
        self.assertEqual(report.unknown[0].source, "upstream")
        self.assertIn("空", report.unknown[0].reason)

    def test_serialized_report_explicitly_declares_read_only_quarantine(self):
        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot={"data": [{"id": "kimi-k3"}]},
            litellm_snapshot={"data": []},
        )

        payload = discovery.serialize_report(report)

        self.assertEqual(payload["mode"], "read_only")
        self.assertFalse(payload["writes_performed"])
        self.assertEqual(payload["write_actions"], [])
        self.assertEqual(payload["new"][0]["isolation_status"], "candidate")
        self.assertFalse(payload["new"][0]["eligible_for_registration"])
        self.assertEqual(
            payload["summary"],
            {"new": 1, "existing": 0, "removed": 0, "unknown": 0},
        )

    def test_cli_only_reads_snapshots_and_outputs_json_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream_path = root / "moonshot-models.json"
            litellm_path = root / "litellm-model-info.json"
            upstream_path.write_text(json.dumps({"data": [{"id": "kimi-k3"}]}), encoding="utf-8")
            litellm_path.write_text(json.dumps({"data": []}), encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = discovery.main(
                    [
                        "--provider",
                        "moonshot",
                        "--upstream-snapshot",
                        str(upstream_path),
                        "--litellm-snapshot",
                        str(litellm_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["new"][0]["model_id"], "kimi-k3")
        self.assertEqual(payload["write_actions"], [])

    def test_cli_rejects_apply_argument(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            discovery._parse_args(
                [
                    "--provider",
                    "moonshot",
                    "--upstream-snapshot",
                    "upstream.json",
                    "--litellm-snapshot",
                    "litellm.json",
                    "--apply",
                ]
            )


if __name__ == "__main__":
    unittest.main()
