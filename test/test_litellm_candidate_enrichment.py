import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import enrich_litellm_model_candidates as enrichment
from scripts.fetch_litellm_cost_map import cost_map_sha256


def candidate_report():
    return {
        "candidate_preflight": {
            "status": "ready",
            "transport": "litellm_provider_wildcard",
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
        "providers": {
            "moonshot": {
                "status": "ok",
                "report": {
                    "provider": {"key": "moonshot", "display": "Moonshot"},
                    "new": [
                        {
                            "provider_key": "moonshot",
                            "model_id": "kimi-k3",
                            "litellm_model": "moonshot/kimi-k3",
                            "isolation_status": "candidate",
                        }
                    ],
                },
            }
        },
    }


def registry():
    return {
        "providers": [
            {
                "adapter": "moonshot",
                "base_url": "https://api.moonshot.cn/v1",
                "api_key_env": "MOONSHOT_API_KEY",
            }
        ]
    }


def enrich(**kwargs):
    cost_map = kwargs["cost_map"]
    kwargs["cost_map_status"] = {
        "status": "success",
        "model_count": len(cost_map),
        "sha256": cost_map_sha256(cost_map),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://example.test/model-cost.json",
    }
    return enrichment.enrich_candidate_report(**kwargs)


class CandidateEnrichmentTests(unittest.TestCase):
    def test_candidate_is_bound_to_its_verified_provider_route(self):
        report = enrich(
            candidate_report=candidate_report(),
            registry=registry(),
            cost_map={
                "moonshot/kimi-k3": {
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                    "supports_function_calling": True,
                }
            },
        )

        candidate = report["providers"]["moonshot"]["report"]["new"][0]
        self.assertEqual(candidate["preflight_model"], "candidate/moonshot/kimi-k3")
        self.assertEqual(
            candidate["preflight_route"]["api_base"],
            candidate["registration"]["api_base"],
        )

    def test_versioned_override_example_has_valid_approval_hash(self):
        path = Path(__file__).resolve().parents[1] / "ops/litellm/candidate-overrides.example.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        approval = enrichment.validate_override_approval(payload)

        self.assertEqual(approval["policy_version"], "fusion-model-override/v1")

    def test_production_override_has_full_document_hash_and_domestic_pricing_provenance(self):
        path = Path(__file__).resolve().parents[1] / "ops/litellm/candidate-overrides.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        approval = enrichment.validate_override_approval(payload)
        models = payload["providers"]["moonshot"]["models"]
        model = models["kimi-k3"]

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(len(approval["document_sha256"]), 64)
        self.assertEqual(model["pricing"]["input"], 3.0)
        self.assertEqual(model["pricing_provenance"]["billing_rates"]["input"], 20.0)
        self.assertEqual(
            set(model["capabilities"]),
            set(enrichment.FUSION_CAPABILITY_FIELDS),
        )
        self.assertEqual(
            set(models),
            {"kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed"},
        )
        self.assertEqual(models["kimi-k2.7-code"]["pricing"]["input"], 0.95)
        self.assertEqual(models["kimi-k2.7-code"]["pricing"]["output"], 4.0)
        self.assertEqual(models["kimi-k2.7-code-highspeed"]["pricing"]["input"], 1.9)
        self.assertEqual(models["kimi-k2.7-code-highspeed"]["pricing"]["output"], 8.0)

    def test_published_regional_price_pair_allows_minor_provider_rounding(self):
        providers = {
            "moonshot": {
                "models": {
                    "kimi-k2.7-code": {
                        "capabilities": {"functionCalling": True},
                        "pricing": {
                            "input": 0.95,
                            "output": 4.0,
                            "cache_read_input": 0.19,
                            "unit": "USD/1M tokens",
                        },
                        "pricing_provenance": {
                            "api_base": "https://api.moonshot.cn/v1",
                            "billing_region": "CN",
                            "billing_currency": "CNY",
                            "canonical_currency": "USD",
                            "canonicalization_method": "provider_published_regional_price_pair",
                            "input_basis": "cache_miss",
                            "billing_rates": {
                                "input": 6.5,
                                "output": 27.0,
                                "cache_read_input": 1.3,
                                "unit": "CNY/1M tokens",
                            },
                            "canonical_rates": {
                                "input": 0.95,
                                "output": 4.0,
                                "cache_read_input": 0.19,
                                "unit": "USD/1M tokens",
                            },
                            "source_evidence": [
                                {
                                    "region": "CN",
                                    "url": "https://platform.kimi.com/docs/pricing/chat-k27-code.md",
                                },
                                {
                                    "region": "GLOBAL",
                                    "url": "https://platform.kimi.ai/docs/pricing/chat-k27-code.md",
                                },
                            ],
                        },
                    }
                }
            }
        }
        approval = {
            "policy_version": "fusion-model-override/v2",
            "reviewed_by": "model-governance",
            "reviewed_at": "2026-07-29T12:00:00+08:00",
            "valid_until": "2026-10-29T12:00:00+08:00",
            "source_urls": [
                "https://platform.kimi.com/docs/pricing/chat-k27-code.md",
                "https://platform.kimi.ai/docs/pricing/chat-k27-code.md",
            ],
            "providers_sha256": cost_map_sha256(providers),
        }
        document = {
            "schema_version": 2,
            "approval": approval,
            "providers": providers,
        }
        approval["document_sha256"] = cost_map_sha256(document)

        report = candidate_report()
        report["providers"]["moonshot"]["report"]["new"][0].update(
            {
                "model_id": "kimi-k2.7-code",
                "litellm_model": "moonshot/kimi-k2.7-code",
            }
        )
        result = enrich(
            candidate_report=report,
            registry=registry(),
            cost_map={},
            overrides={
                "schema_version": 2,
                "approval": approval,
                "providers": providers,
            },
        )

        candidate = result["providers"]["moonshot"]["report"]["new"][0]
        self.assertEqual(candidate["metadata"]["pricing"]["input"], 0.95)

        invalid_providers = copy.deepcopy(providers)
        invalid_providers["moonshot"]["models"]["kimi-k2.7-code"]["pricing_provenance"][
            "billing_rates"
        ]["output"] = 35.0
        invalid_approval = copy.deepcopy(approval)
        invalid_approval["providers_sha256"] = cost_map_sha256(invalid_providers)
        invalid_document = {
            "schema_version": 2,
            "approval": {
                key: value
                for key, value in invalid_approval.items()
                if key != "document_sha256"
            },
            "providers": invalid_providers,
        }
        invalid_approval["document_sha256"] = cost_map_sha256(invalid_document)

        with self.assertRaisesRegex(ValueError, "区域价格比例"):
            enrichment.validate_override_approval(
                {
                    "schema_version": 2,
                    "approval": invalid_approval,
                    "providers": invalid_providers,
                }
            )

    def test_cost_map_status_must_match_and_be_fresh(self):
        cost_map = {"model": {"input_cost_per_token": 0.1}}
        now = datetime.now(timezone.utc)
        valid = {
            "status": "success",
            "model_count": 1,
            "sha256": cost_map_sha256(cost_map),
            "fetched_at": now.isoformat(),
            "source_url": "https://example.test/model-cost.json",
        }

        enrichment.verify_cost_map_status(cost_map=cost_map, status=valid, now=now)

        with self.assertRaisesRegex(ValueError, "不匹配"):
            enrichment.verify_cost_map_status(
                cost_map=cost_map,
                status={**valid, "sha256": "bad"},
                now=now,
            )
        with self.assertRaisesRegex(ValueError, "时间"):
            enrichment.verify_cost_map_status(
                cost_map=cost_map,
                status={**valid, "fetched_at": (now - timedelta(days=2)).isoformat()},
                now=now,
            )

    def test_cost_map_match_populates_registration_pricing_and_capabilities(self):
        source = candidate_report()
        original = copy.deepcopy(source)

        result = enrich(
            candidate_report=source,
            registry=registry(),
            cost_map={
                "moonshot/kimi-k3": {
                    "input_cost_per_token": 0.0000006,
                    "output_cost_per_token": 0.0000025,
                    "supports_function_calling": True,
                    "supports_vision": True,
                    "supports_reasoning": True,
                }
            },
        )

        candidate = result["providers"]["moonshot"]["report"]["new"][0]
        self.assertEqual(source, original)
        self.assertEqual(candidate["registration"]["endpoint_status"], "verified")
        self.assertEqual(candidate["registration"]["api_key_env"], "MOONSHOT_API_KEY")
        self.assertEqual(
            candidate["metadata"]["pricing"],
            {"input": 0.6, "output": 2.5, "unit": "USD/1M tokens"},
        )
        self.assertTrue(candidate["metadata"]["capabilities"]["functionCalling"])
        self.assertTrue(candidate["metadata"]["capabilities"]["vision"])
        self.assertEqual(candidate["metadata_evidence"]["cost_map_key"], "moonshot/kimi-k3")

    def test_missing_cost_map_evidence_keeps_candidate_incomplete_for_fail_closed_admission(self):
        result = enrich(
            candidate_report=candidate_report(),
            registry=registry(),
            cost_map={},
        )

        candidate = result["providers"]["moonshot"]["report"]["new"][0]
        self.assertNotIn("pricing", candidate["metadata"])
        self.assertNotIn("capabilities", candidate["metadata"])
        self.assertFalse(candidate["metadata_evidence"]["cost_map_matched"])

    def test_reviewed_override_can_fill_cost_map_gap_without_secret_material(self):
        providers = {
            "moonshot": {
                "models": {
                    "kimi-k3": {
                        "display_name": "Kimi K3",
                        "capabilities": {"functionCalling": True, "vision": True},
                        "pricing": {
                            "input": 3.0,
                            "output": 15.0,
                            "cache_read_input": 0.3,
                            "unit": "USD/1M tokens",
                        },
                        "pricing_provenance": {
                            "billing_region": "CN",
                            "billing_currency": "CNY",
                            "canonical_currency": "USD",
                            "canonicalization_method": "provider_published_regional_price_pair",
                            "billing_rates": {
                                "input": 20.0,
                                "output": 100.0,
                                "cache_read_input": 2.0,
                                "unit": "CNY/1M tokens",
                            },
                            "canonical_rates": {
                                "input": 3.0,
                                "output": 15.0,
                                "cache_read_input": 0.3,
                                "unit": "USD/1M tokens",
                            },
                        },
                    }
                }
            }
        }
        result = enrich(
            candidate_report=candidate_report(),
            registry=registry(),
            cost_map={},
            overrides={
                "schema_version": 1,
                "approval": {
                    "policy_version": "fusion-model-override/v1",
                    "reviewed_by": "model-governance",
                    "reviewed_at": "2026-07-28T12:00:00+08:00",
                    "source_urls": ["https://example.test/kimi-k3"],
                    "providers_sha256": cost_map_sha256(providers),
                },
                "providers": providers,
            },
        )

        candidate = result["providers"]["moonshot"]["report"]["new"][0]
        self.assertEqual(candidate["metadata"]["display_name"], "Kimi K3")
        self.assertEqual(candidate["metadata"]["pricing"]["cache_read_input"], 0.3)
        self.assertEqual(candidate["metadata"]["pricing_provenance"]["billing_currency"], "CNY")
        self.assertTrue(candidate["metadata_evidence"]["reviewed_override_applied"])
        self.assertEqual(
            candidate["metadata_evidence"]["override_approval"]["policy_version"],
            "fusion-model-override/v1",
        )
        self.assertNotIn("api_key", candidate["registration"])

    def test_override_pricing_provenance_must_match_canonical_pricing(self):
        providers = {
            "moonshot": {
                "models": {
                    "kimi-k3": {
                        "capabilities": {"functionCalling": True},
                        "pricing": {"input": 3.0, "output": 15.0, "unit": "USD/1M tokens"},
                        "pricing_provenance": {
                            "billing_region": "CN",
                            "billing_currency": "CNY",
                            "canonical_currency": "USD",
                            "canonicalization_method": "provider_published_regional_price_pair",
                            "billing_rates": {
                                "input": 20.0,
                                "output": 100.0,
                                "unit": "CNY/1M tokens",
                            },
                            "canonical_rates": {
                                "input": 2.0,
                                "output": 15.0,
                                "unit": "USD/1M tokens",
                            },
                        },
                    }
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "canonical_rates"):
            enrich(
                candidate_report=candidate_report(),
                registry=registry(),
                cost_map={},
                overrides={
                    "schema_version": 1,
                    "approval": {
                        "policy_version": "fusion-model-override/v1",
                        "reviewed_by": "model-governance",
                        "reviewed_at": "2026-07-29T09:00:00+08:00",
                        "source_urls": [
                            "https://platform.kimi.com/docs/pricing/chat-k3",
                            "https://platform.kimi.ai/docs/pricing/chat-k3",
                        ],
                        "providers_sha256": cost_map_sha256(providers),
                    },
                    "providers": providers,
                },
            )

    def test_override_without_matching_approval_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "providers_sha256"):
            enrich(
                candidate_report=candidate_report(),
                registry=registry(),
                cost_map={},
                overrides={
                    "schema_version": 1,
                    "approval": {
                        "policy_version": "fusion-model-override/v1",
                        "reviewed_by": "model-governance",
                        "reviewed_at": "2026-07-28T12:00:00+08:00",
                        "source_urls": ["https://example.test/kimi-k3"],
                        "providers_sha256": "bad",
                    },
                    "providers": {"moonshot": {"models": {}}},
                },
            )

    def test_provider_specific_cost_prefix_supports_shared_openai_adapter(self):
        report = candidate_report()
        report["providers"] = {
            "qwen": {
                "status": "ok",
                "report": {
                    "provider": {"key": "qwen", "display": "通义千问"},
                    "new": [
                        {
                            "provider_key": "qwen",
                            "model_id": "qwen-new",
                            "litellm_model": "openai/qwen-new",
                            "isolation_status": "candidate",
                        }
                    ],
                },
            }
        }
        provider_registry = {
            "providers": [
                {
                    "adapter": "openai-compatible",
                    "provider_key": "qwen",
                    "provider_display": "通义千问",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key_env": "QWEN_API_KEY",
                    "cost_map_prefix": "qwen",
                }
            ]
        }

        result = enrich(
            candidate_report=report,
            registry=provider_registry,
            cost_map={
                "qwen/qwen-new": {
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000003,
                    "supports_function_calling": True,
                }
            },
        )

        candidate = result["providers"]["qwen"]["report"]["new"][0]
        self.assertEqual(candidate["metadata_evidence"]["cost_map_key"], "qwen/qwen-new")
        self.assertEqual(candidate["metadata"]["pricing"]["input"], 1.0)

    def test_shared_openai_cost_entry_cannot_cross_provider_boundary(self):
        report = candidate_report()
        report["providers"] = {
            "qwen": {
                "status": "ok",
                "report": {
                    "provider": {"key": "qwen", "display": "通义千问"},
                    "new": [
                        {
                            "provider_key": "qwen",
                            "model_id": "qwen-new",
                            "litellm_model": "openai/qwen-new",
                            "isolation_status": "candidate",
                        }
                    ],
                },
            }
        }
        provider_registry = {
            "providers": [
                {
                    "adapter": "openai-compatible",
                    "provider_key": "qwen",
                    "provider_display": "通义千问",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key_env": "QWEN_API_KEY",
                    "cost_map_prefix": "qwen",
                }
            ]
        }

        result = enrich(
            candidate_report=report,
            registry=provider_registry,
            cost_map={
                "openai/qwen-new": {
                    "litellm_provider": "openai",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000003,
                    "supports_function_calling": True,
                }
            },
        )

        candidate = result["providers"]["qwen"]["report"]["new"][0]
        self.assertFalse(candidate["metadata_evidence"]["cost_map_matched"])
        self.assertNotIn("pricing", candidate["metadata"])

    def test_shared_openai_prefix_is_never_treated_as_xiaomi_provider_identity(self):
        candidate = {
            "provider_key": "xiaomi",
            "model_id": "mimo-new",
            "litellm_model": "openai/mimo-new",
        }
        provider = {
            "adapter": "openai-compatible",
            "provider_key": "xiaomi",
            "litellm_prefix": "openai",
        }

        key, entry = enrichment._cost_entry(
            candidate,
            provider,
            {
                "openai/mimo-new": {
                    "litellm_provider": "openai",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000003,
                }
            },
        )

        self.assertEqual(key, "")
        self.assertIsNone(entry)

    def test_cost_entry_without_capability_evidence_does_not_invent_capabilities(self):
        result = enrich(
            candidate_report=candidate_report(),
            registry=registry(),
            cost_map={
                "moonshot/kimi-k3": {
                    "input_cost_per_token": 0.0000006,
                    "output_cost_per_token": 0.0000025,
                }
            },
        )

        candidate = result["providers"]["moonshot"]["report"]["new"][0]
        self.assertIn("pricing", candidate["metadata"])
        self.assertNotIn("capabilities", candidate["metadata"])


if __name__ == "__main__":
    unittest.main()
