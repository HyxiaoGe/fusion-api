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
    def test_qwen_new_naming_and_product_policy_do_not_trigger_false_retirement(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="qwen",
            provider_display="通义千问",
            litellm_prefix="openai",
            api_model_prefixes=("qwen", "qwq-", "qvq-", "text-embedding-"),
            candidate_model_patterns=(
                r"qwen3\.\d+-(?:max|plus|flash)",
                r"qwen-(?:max|plus|flash)",
            ),
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={
                "data": [
                    {"id": "qwen3.8-max"},
                    {"id": "qwen3.7-max"},
                    {"id": "qwen3.6-plus"},
                    {"id": "qwen-image-2.0"},
                    {"id": "qwen3-tts-instruct-flash"},
                    {"id": "qwen3.7-text-embedding"},
                    {"id": "text-embedding-v4"},
                    {"id": "qwen3.8-max-image"},
                    {"id": "qwen3.7-max-preview"},
                    {"id": "deepseek-v4-pro"},
                    {"id": "kimi/kimi-k3"},
                ]
            },
            litellm_snapshot={
                "data": [
                    litellm_entry("qwen-max-latest", "openai/qwen3.7-max", provider_key="qwen"),
                    litellm_entry("qwen3.6-plus", "openai/qwen3.6-plus", provider_key="qwen"),
                    litellm_entry("qwen-image", "openai/qwen-image-2.0", provider_key="qwen"),
                    litellm_entry(
                        "text-embedding-v4",
                        "openai/text-embedding-v4",
                        provider_key="qwen",
                    ),
                ]
            },
        )

        self.assertEqual([item.model_id for item in report.new], ["qwen3.8-max"])
        self.assertEqual(
            [item.model_id for item in report.existing],
            ["qwen-image-2.0", "qwen3.6-plus", "qwen3.7-max", "text-embedding-v4"],
        )
        self.assertEqual(report.removed, [])
        self.assertEqual(
            [(item.source, item.model_id) for item in report.unknown],
            [
                ("product_policy", "qwen-image-2.0"),
                ("product_policy", "qwen3-tts-instruct-flash"),
                ("product_policy", "qwen3.7-text-embedding"),
                ("product_policy", "text-embedding-v4"),
                ("product_policy", "qwen3.8-max-image"),
                ("product_policy", "qwen3.7-max-preview"),
                ("upstream", "deepseek-v4-pro"),
                ("upstream", "kimi/kimi-k3"),
            ],
        )

    def test_unknown_upstream_model_still_proves_existing_model_has_not_been_removed(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="qwen",
            provider_display="通义千问",
            litellm_prefix="openai",
            api_model_prefix="qwen-",
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={"data": [{"id": "qwen3.7-max"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry("qwen-max-latest", "openai/qwen3.7-max", provider_key="qwen"),
                ]
            },
        )

        self.assertEqual(report.new, [])
        self.assertEqual(report.existing, [])
        self.assertEqual(report.removed, [])
        self.assertEqual(report.unknown[0].model_id, "qwen3.7-max")
        self.assertIn("qwen-", report.unknown[0].reason)

    def test_product_policy_uses_stripped_upstream_id_without_false_retirement(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="xai",
            provider_display="xAI",
            litellm_prefix="openrouter/x-ai",
            upstream_strip_prefix="x-ai/",
            candidate_model_patterns=(r"grok-4\.5",),
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={"data": [{"id": "x-ai/grok-4.20-beta"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry(
                        "grok-4.20-beta",
                        "openrouter/x-ai/grok-4.20-beta",
                        provider_key="xai",
                    )
                ]
            },
        )

        self.assertEqual(report.new, [])
        self.assertEqual([item.model_id for item in report.existing], ["grok-4.20-beta"])
        self.assertEqual(report.removed, [])
        self.assertEqual(
            [(item.source, item.model_id) for item in report.unknown],
            [("product_policy", "grok-4.20-beta")],
        )

    def test_third_party_model_mislabeled_as_qwen_is_not_claimed_or_retired(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="qwen",
            provider_display="通义千问",
            litellm_prefix="openai",
            api_model_prefixes=("qwen", "qwq-", "qvq-"),
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={
                "data": [
                    {"id": "qwen3.8-max"},
                    {"id": "deepseek-v4-pro"},
                ]
            },
            litellm_snapshot={
                "data": [
                    litellm_entry(
                        "wrong-provider-model",
                        "openai/deepseek-v4-pro",
                        provider_key="qwen",
                    )
                ]
            },
        )

        self.assertEqual([item.model_id for item in report.new], ["qwen3.8-max"])
        self.assertEqual(report.existing, [])
        self.assertEqual(report.removed, [])
        self.assertEqual(report.unknown[0].model_id, "deepseek-v4-pro")

    def test_legacy_and_multi_prefixes_are_merged_and_invalid_policy_regex_fails_closed(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="acme",
            provider_display="Acme",
            litellm_prefix="openai",
            api_model_prefix="acme-",
            api_model_prefixes=("acme3.", "acme-"),
        )

        self.assertIsNotNone(adapter.adapt_upstream_model({"id": "acme-chat"})[0])
        self.assertIsNotNone(adapter.adapt_upstream_model({"id": "acme3.1-chat"})[0])
        self.assertEqual(adapter.api_model_prefixes, ("acme-", "acme3."))
        with self.assertRaisesRegex(ValueError, "无效正则"):
            discovery.OpenAICompatibleProviderAdapter(
                provider_key="acme",
                provider_display="Acme",
                litellm_prefix="openai",
                candidate_model_patterns=("[",),
            )

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

    def test_configurable_upstream_fields_strip_prefix_and_filter_generation_method(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="google",
            provider_display="Google Gemini",
            litellm_prefix="gemini",
            upstream_id_field="name",
            upstream_strip_prefix="models/",
            required_generation_method="generateContent",
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={
                "data": [
                    {
                        "name": "models/gemini-3.6-flash",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            },
            litellm_snapshot={"data": []},
        )

        self.assertEqual([item.model_id for item in report.new], ["gemini-3.6-flash"])
        self.assertEqual(report.new[0].litellm_model, "gemini/gemini-3.6-flash")
        self.assertEqual(len(report.unknown), 1)
        self.assertIn("generateContent", report.unknown[0].reason)

    def test_nested_litellm_prefix_round_trips_openrouter_xai_model(self):
        adapter = discovery.OpenAICompatibleProviderAdapter(
            provider_key="xai",
            provider_display="xAI",
            litellm_prefix="openrouter/x-ai",
            upstream_strip_prefix="x-ai/",
        )

        report = discovery.discover_candidates(
            adapter=adapter,
            upstream_snapshot={"data": [{"id": "x-ai/grok-4.5"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry(
                        "grok-4.20-beta",
                        "openrouter/x-ai/grok-4.20-beta",
                        provider_key="xai",
                    )
                ]
            },
        )

        self.assertEqual([item.model_id for item in report.new], ["grok-4.5"])
        self.assertEqual(report.new[0].litellm_model, "openrouter/x-ai/grok-4.5")
        self.assertEqual([item.model_id for item in report.removed], ["grok-4.20-beta"])
        self.assertTrue(
            adapter.owns_litellm_entry(
                litellm_entry(
                    "grok-4.5",
                    "openrouter/x-ai/grok-4.5",
                    provider_key="xai",
                )
            )
        )

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

    def test_exact_non_db_candidate_pricing_route_does_not_hide_new_model(self):
        route = litellm_entry(
            "candidate/moonshot/kimi-k3",
            "moonshot/kimi-k3",
            provider_key="moonshot",
        )
        route["model_info"]["db_model"] = False
        route["model_info"]["metadata"]["purpose"] = "candidate_pricing_override"

        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot={"data": [{"id": "kimi-k3"}]},
            litellm_snapshot={"data": [route]},
        )

        self.assertEqual([item.model_id for item in report.new], ["kimi-k3"])
        self.assertEqual(report.existing, [])

    def test_non_db_business_wildcard_does_not_hide_model_from_fusion_candidate_queue(self):
        wildcard_entry = litellm_entry(
            "gemini/gemini-3.6-flash",
            "gemini/gemini-3.6-flash",
            provider_key="google",
        )
        wildcard_entry["model_info"]["db_model"] = False

        report = discovery.discover_candidates(
            adapter=discovery.OpenAICompatibleProviderAdapter(
                provider_key="google",
                provider_display="Google Gemini",
                litellm_prefix="gemini",
                upstream_id_field="name",
                upstream_strip_prefix="models/",
                required_generation_method="generateContent",
            ),
            upstream_snapshot={
                "data": [
                    {
                        "name": "models/gemini-3.6-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
            litellm_snapshot={"data": [wildcard_entry]},
        )

        self.assertEqual([item.model_id for item in report.new], ["gemini-3.6-flash"])
        self.assertEqual(report.existing, [])

    def test_candidate_is_quarantined_when_business_alias_is_owned_by_other_provider(self):
        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot={"data": [{"id": "kimi-k3"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry(
                        "kimi-k3",
                        "openai/other-provider-model",
                        provider_key="other",
                    )
                ]
            },
        )

        self.assertEqual(report.new, [])
        self.assertEqual(report.existing, [])
        self.assertEqual(len(report.unknown), 1)
        self.assertIn("别名", report.unknown[0].reason)

    def test_exact_underlying_alias_without_provider_ownership_is_still_quarantined(self):
        report = discovery.discover_candidates(
            adapter=discovery.MoonshotProviderAdapter(),
            upstream_snapshot={"data": [{"id": "kimi-k3"}]},
            litellm_snapshot={
                "data": [
                    litellm_entry(
                        "kimi-k3",
                        "moonshot/kimi-k3",
                        provider_key="other",
                    )
                ]
            },
        )

        self.assertEqual(report.new, [])
        self.assertEqual(report.existing, [])
        self.assertEqual(len(report.unknown), 1)
        self.assertIn("未归属", report.unknown[0].reason)

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
