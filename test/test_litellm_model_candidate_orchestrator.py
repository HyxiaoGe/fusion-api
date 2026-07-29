import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import orchestrate_litellm_model_candidates as orchestrator


class FakeResponse:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def registry(*providers):
    return {
        "litellm": {
            "base_url": "http://litellm.internal:4000",
            "api_key_env": "LITELLM_MASTER_KEY",
            "candidate_preflight": {
                "transport": "litellm_provider_wildcard",
                "api_key_env": "LITELLM_CANDIDATE_KEY",
            },
        },
        "providers": list(providers),
    }


def moonshot_config():
    return {
        "adapter": "moonshot",
        "base_url": "https://api.moonshot.example",
        "api_key_env": "MOONSHOT_API_KEY",
        "credential_generation": "test-v1",
        "candidate_preflight": {
            "route_model_name": "candidate/moonshot/*",
            "route_litellm_model": "moonshot/*",
        },
    }


def acme_config():
    return {
        "adapter": "openai-compatible",
        "provider_key": "acme",
        "provider_display": "Acme AI",
        "litellm_prefix": "openai",
        "api_model_prefix": "acme-",
        "base_url": "https://api.acme.example/v1",
        "api_key_env": "ACME_API_KEY",
        "credential_generation": "test-v1",
        "candidate_preflight": {
            "route_model_name": "candidate/acme/*",
            "route_litellm_model": "openai/*",
        },
    }


def google_config():
    return {
        "adapter": "openai-compatible",
        "provider_key": "google",
        "provider_display": "Google Gemini",
        "litellm_prefix": "gemini",
        "base_url": "",
        "api_key_env": "GOOGLE_API_KEY",
        "credential_generation": "test-v1",
        "discovery": {
            "url": "https://generativelanguage.googleapis.com/v1beta/models",
            "auth": "x-goog-api-key",
            "response_field": "models",
            "id_field": "name",
            "strip_prefix": "models/",
            "required_generation_method": "generateContent",
            "pagination": {
                "page_size": 1000,
                "page_size_param": "pageSize",
                "page_token_param": "pageToken",
                "next_page_token_field": "nextPageToken",
            },
        },
    }


class ModelCandidateOrchestratorTests(unittest.TestCase):
    def test_minimax_without_discovery_config_uses_standard_v1_models_catalog(self):
        provider = {
            **acme_config(),
            "provider_key": "minimax",
            "provider_display": "MiniMax",
            "litellm_prefix": "minimax",
            "api_model_prefix": "MiniMax-",
            "base_url": "https://api.minimax.io",
            "api_key_env": "MINIMAX_API_KEY",
        }
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "https://api.minimax.io/v1/models": FakeResponse({"data": [{"id": "MiniMax-M2.7"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(provider),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "MINIMAX_API_KEY": "minimax-secret",
            },
            client=client,
        )

        self.assertEqual(report["providers"]["minimax"]["status"], "ok")
        self.assertEqual(
            report["providers"]["minimax"]["report"]["new"][0]["litellm_model"],
            "minimax/MiniMax-M2.7",
        )
        self.assertIn("https://api.minimax.io/v1/models", [call[0] for call in client.calls])

    def test_provider_without_credential_generation_is_blocked(self):
        provider = acme_config()
        provider.pop("credential_generation")

        status = orchestrator._candidate_preflight_status(
            litellm_config=registry(provider)["litellm"],
            providers=[provider],
            litellm_snapshot={
                "data": [
                    {
                        "model_name": "candidate/acme/*",
                        "litellm_params": {
                            "model": "openai/*",
                            "api_base": "https://api.acme.example/v1",
                        },
                        "model_info": {
                            "metadata": {
                                "provider_key": "acme",
                                "purpose": "candidate_preflight",
                            }
                        },
                    }
                ]
            },
            candidate_key_models=["candidate/acme/*"],
            candidate_key_lookup_failed=False,
            environ={
                "LITELLM_MASTER_KEY": "master",
                "LITELLM_CANDIDATE_KEY": "candidate",
            },
        )

        self.assertIn(
            "candidate_preflight_credential_generation_missing",
            status["providers"]["acme"]["reasons"],
        )

    def test_db_backed_candidate_route_is_never_ready(self):
        provider = acme_config()
        status = orchestrator._candidate_preflight_status(
            litellm_config=registry(provider)["litellm"],
            providers=[provider],
            litellm_snapshot={
                "data": [
                    {
                        "model_name": "candidate/acme/*",
                        "litellm_params": {
                            "model": "openai/*",
                            "api_base": "https://api.acme.example/v1",
                        },
                        "model_info": {
                            "db_model": True,
                            "metadata": {
                                "provider_key": "acme",
                                "purpose": "candidate_preflight",
                            },
                        },
                    }
                ]
            },
            candidate_key_models=["candidate/acme/*"],
            candidate_key_lookup_failed=False,
            environ={
                "LITELLM_MASTER_KEY": "master",
                "LITELLM_CANDIDATE_KEY": "candidate",
            },
        )

        self.assertEqual(status["providers"]["acme"]["status"], "blocked")
        self.assertIn(
            "candidate_preflight_route_db_backed",
            status["providers"]["acme"]["reasons"],
        )

    def test_v193_expanded_catalog_is_resolved_to_raw_wildcard_route(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/acme/gpt-4o",
                                "litellm_params": {
                                    "model": "candidate/acme/gpt-4o",
                                    "api_base": "https://api.acme.example/v1",
                                },
                                "model_info": {"id": "route-id"},
                            }
                        ]
                    }
                ),
                "http://litellm.internal:4000/model/info?litellm_model_id=route-id": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/acme/*",
                                "litellm_params": {
                                    "model": "openai/*",
                                    "api_base": "https://api.acme.example/v1",
                                    "api_key": "os.environ/ACME_API_KEY",
                                },
                                "model_info": {
                                    "metadata": {
                                        "provider_key": "acme",
                                        "purpose": "candidate_preflight",
                                    }
                                },
                            }
                        ]
                    }
                ),
                "http://litellm.internal:4000/key/info?key=candidate-secret": FakeResponse(
                    {"info": {"models": ["candidate/acme/*"]}}
                ),
                "https://api.acme.example/v1/models": FakeResponse({"data": [{"id": "acme-new"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(acme_config()),
            environ={
                "LITELLM_MASTER_KEY": "master",
                "LITELLM_CANDIDATE_KEY": "candidate-secret",
                "ACME_API_KEY": "provider",
            },
            client=client,
        )

        self.assertEqual(report["candidate_preflight"]["status"], "ready")
        self.assertIn("litellm_model_id=route-id", " ".join(call[0] for call in client.calls))

    def test_native_provider_flattened_wildcard_is_resolved_by_route_metadata(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/kimi-k2.6",
                                "litellm_params": {
                                    "model": "candidate/kimi-k2.6",
                                    "api_base": "https://api.moonshot.example",
                                },
                                "model_info": {
                                    "id": "moonshot-route-id",
                                    "metadata": {
                                        "provider_key": "moonshot",
                                        "purpose": "candidate_preflight",
                                    },
                                },
                            }
                        ]
                    }
                ),
                "http://litellm.internal:4000/model/info?litellm_model_id=moonshot-route-id": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/moonshot/*",
                                "litellm_params": {
                                    "model": "moonshot/*",
                                    "api_base": "https://api.moonshot.example",
                                    "api_key": "os.environ/MOONSHOT_API_KEY",
                                },
                                "model_info": {
                                    "metadata": {
                                        "provider_key": "moonshot",
                                        "purpose": "candidate_preflight",
                                    }
                                },
                            }
                        ]
                    }
                ),
                "http://litellm.internal:4000/key/info?key=candidate-secret": FakeResponse(
                    {"info": {"models": ["candidate/moonshot/*"]}}
                ),
                "https://api.moonshot.example/v1/models": FakeResponse(
                    {"data": [{"id": "kimi-k2.6"}, {"id": "kimi-k3"}]}
                ),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={
                "LITELLM_MASTER_KEY": "master",
                "LITELLM_CANDIDATE_KEY": "candidate-secret",
                "MOONSHOT_API_KEY": "provider",
            },
            client=client,
        )

        self.assertEqual(report["candidate_preflight"]["status"], "ready")
        self.assertIn("litellm_model_id=moonshot-route-id", " ".join(call[0] for call in client.calls))

    def test_shared_openai_adapter_routes_are_verified_per_provider(self):
        qwen = {
            **acme_config(),
            "provider_key": "qwen",
            "provider_display": "通义千问",
            "api_model_prefix": "qwen-",
            "base_url": "https://qwen.example/v1",
            "api_key_env": "QWEN_API_KEY",
            "candidate_preflight": {
                "route_model_name": "candidate/qwen/*",
                "route_litellm_model": "openai/*",
            },
        }
        xiaomi = {
            **acme_config(),
            "provider_key": "xiaomi",
            "provider_display": "小米",
            "api_model_prefix": "mimo-",
            "base_url": "https://xiaomi.example/v1",
            "api_key_env": "XIAOMI_API_KEY",
            "candidate_preflight": {
                "route_model_name": "candidate/xiaomi/*",
                "route_litellm_model": "openai/*",
            },
        }
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/qwen/*",
                                "litellm_params": {
                                    "model": "openai/*",
                                    "api_base": "https://qwen.example/v1",
                                    "api_key": "os.environ/QWEN_API_KEY",
                                },
                                "model_info": {
                                    "metadata": {
                                        "provider_key": "qwen",
                                        "purpose": "candidate_preflight",
                                    }
                                },
                            },
                            {
                                "model_name": "candidate/xiaomi/*",
                                "litellm_params": {
                                    "model": "openai/*",
                                    "api_base": "https://wrong.example/v1",
                                    "api_key": "os.environ/XIAOMI_API_KEY",
                                },
                                "model_info": {
                                    "metadata": {
                                        "provider_key": "xiaomi",
                                        "purpose": "candidate_preflight",
                                    }
                                },
                            },
                        ]
                    }
                ),
                "http://litellm.internal:4000/key/info?key=candidate-secret": FakeResponse(
                    {
                        "info": {
                            "models": [
                                "candidate/qwen/*",
                                "candidate/xiaomi/*",
                            ]
                        }
                    }
                ),
                "https://qwen.example/v1/models": FakeResponse({"data": [{"id": "qwen-new"}]}),
                "https://xiaomi.example/v1/models": FakeResponse({"data": [{"id": "mimo-new"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(qwen, xiaomi),
            environ={
                "LITELLM_MASTER_KEY": "master",
                "LITELLM_CANDIDATE_KEY": "candidate-secret",
                "QWEN_API_KEY": "qwen",
                "XIAOMI_API_KEY": "xiaomi",
            },
            client=client,
        )

        self.assertEqual(report["candidate_preflight"]["status"], "partial")
        self.assertEqual(report["candidate_preflight"]["providers"]["qwen"]["status"], "ready")
        self.assertEqual(
            report["candidate_preflight"]["providers"]["xiaomi"]["reasons"],
            ["candidate_preflight_route_api_base_mismatch"],
        )

    def test_multiple_providers_are_fetched_and_grouped(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "candidate/moonshot/*",
                                "litellm_params": {
                                    "model": "moonshot/*",
                                    "api_base": "https://api.moonshot.example",
                                    "api_key": "os.environ/MOONSHOT_API_KEY",
                                },
                                "model_info": {
                                    "metadata": {"provider_key": "moonshot", "purpose": "candidate_preflight"}
                                },
                            },
                            {
                                "model_name": "candidate/acme/*",
                                "litellm_params": {
                                    "model": "openai/*",
                                    "api_base": "https://api.acme.example/v1",
                                    "api_key": "os.environ/ACME_API_KEY",
                                },
                                "model_info": {"metadata": {"provider_key": "acme", "purpose": "candidate_preflight"}},
                            },
                            {
                                "model_name": "kimi-k2.5",
                                "litellm_params": {"model": "moonshot/kimi-k2.6"},
                                "model_info": {"metadata": {"provider_key": "moonshot"}},
                            },
                            {
                                "model_name": "acme-chat",
                                "litellm_params": {"model": "openai/acme-chat"},
                                "model_info": {"metadata": {"provider_key": "acme"}},
                            },
                        ]
                    }
                ),
                "http://litellm.internal:4000/key/info?key=candidate-secret": FakeResponse(
                    {"info": {"models": ["candidate/moonshot/*", "candidate/acme/*"]}}
                ),
                "https://api.moonshot.example/v1/models": FakeResponse(
                    {"data": [{"id": "kimi-k2.6"}, {"id": "kimi-k3"}]}
                ),
                "https://api.acme.example/v1/models": FakeResponse(
                    {"data": [{"id": "acme-chat"}, {"id": "acme-reasoner"}]}
                ),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config(), acme_config()),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "MOONSHOT_API_KEY": "moonshot-secret",
                "ACME_API_KEY": "acme-secret",
                "LITELLM_CANDIDATE_KEY": "candidate-secret",
            },
            client=client,
        )

        self.assertEqual(report["summary"], {"providers_total": 2, "providers_ok": 2, "providers_failed": 0})
        self.assertEqual(report["providers"]["moonshot"]["report"]["new"][0]["model_id"], "kimi-k3")
        self.assertEqual(report["providers"]["acme"]["report"]["new"][0]["model_id"], "acme-reasoner")
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(all(call[1].keys() <= {"headers", "timeout"} for call in client.calls))
        self.assertEqual(report["candidate_preflight"]["status"], "ready")

    def test_google_catalog_uses_header_auth_paginates_and_filters_generate_content(self):
        google = google_config()
        catalog_url = google["discovery"]["url"]

        class PaginatedClient(FakeHttpClient):
            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if url == "http://litellm.internal:4000/model/info":
                    return FakeResponse({"data": []})
                if url == catalog_url:
                    params = kwargs.get("params")
                    if params == {"pageSize": 1000}:
                        return FakeResponse(
                            {
                                "models": [
                                    {
                                        "name": "models/gemini-3.6-flash",
                                        "supportedGenerationMethods": ["generateContent"],
                                    },
                                    {
                                        "name": "models/text-embedding-004",
                                        "supportedGenerationMethods": ["embedContent"],
                                    },
                                ],
                                "nextPageToken": "page-2",
                            }
                        )
                    if params == {"pageSize": 1000, "pageToken": "page-2"}:
                        return FakeResponse(
                            {
                                "models": [
                                    {
                                        "name": "models/gemini-3.6-pro",
                                        "supportedGenerationMethods": ["generateContent"],
                                    }
                                ]
                            }
                        )
                raise AssertionError(f"未预期的请求: {url} {kwargs}")

        secret = "google-super-secret"
        client = PaginatedClient({})
        report = orchestrator.coordinate_candidates(
            registry=registry(google),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "GOOGLE_API_KEY": secret,
            },
            client=client,
        )

        provider = report["providers"]["google"]
        self.assertEqual(provider["status"], "ok")
        self.assertEqual(
            [item["model_id"] for item in provider["report"]["new"]],
            ["gemini-3.6-flash", "gemini-3.6-pro"],
        )
        self.assertEqual(provider["report"]["summary"]["unknown"], 1)
        catalog_calls = [call for call in client.calls if call[0] == catalog_url]
        self.assertEqual(len(catalog_calls), 2)
        self.assertTrue(all(call[1]["headers"] == {"x-goog-api-key": secret} for call in catalog_calls))
        self.assertTrue(all(secret not in call[0] for call in catalog_calls))
        self.assertNotIn(secret, json.dumps(report, ensure_ascii=False))

    def test_openrouter_xai_discovery_keeps_nested_litellm_underlying(self):
        xai = {
            "adapter": "openai-compatible",
            "provider_key": "xai",
            "provider_display": "xAI",
            "litellm_prefix": "openrouter/x-ai",
            "base_url": "",
            "api_key_env": "OPENROUTER_API_KEY",
            "credential_generation": "test-v1",
            "discovery": {
                "url": "https://openrouter.ai/api/v1/models",
                "auth": "bearer",
                "response_field": "data",
                "id_field": "id",
                "strip_prefix": "x-ai/",
            },
        }
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "https://openrouter.ai/api/v1/models": FakeResponse(
                    {"data": [{"id": "x-ai/grok-4.5"}, {"id": "openai/gpt-5"}]}
                ),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(xai),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "OPENROUTER_API_KEY": "openrouter-secret",
            },
            client=client,
        )

        provider = report["providers"]["xai"]["report"]
        self.assertEqual([item["model_id"] for item in provider["new"]], ["grok-4.5"])
        self.assertEqual(provider["new"][0]["litellm_model"], "openrouter/x-ai/grok-4.5")
        self.assertEqual(provider["summary"]["unknown"], 1)

    def test_candidate_preflight_requires_candidate_key_wildcard_allowlist(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "http://litellm.internal:4000/key/info?key=candidate-secret": FakeResponse(
                    {"info": {"models": ["kimi-k2.5"]}}
                ),
                "https://api.moonshot.example/v1/models": FakeResponse({"data": [{"id": "kimi-k3"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "LITELLM_CANDIDATE_KEY": "candidate-secret",
                "MOONSHOT_API_KEY": "moonshot-secret",
            },
            client=client,
        )

        self.assertEqual(report["candidate_preflight"]["status"], "blocked")
        self.assertEqual(
            report["candidate_preflight"]["reasons"],
            ["candidate_preflight_key_allowlist_mismatch"],
        )
        self.assertNotIn("candidate-secret", json.dumps(report))

    def test_candidate_preflight_rejects_master_key_reuse_without_key_lookup(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "https://api.moonshot.example/v1/models": FakeResponse({"data": [{"id": "kimi-k3"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={
                "LITELLM_MASTER_KEY": "shared-secret",
                "LITELLM_CANDIDATE_KEY": "shared-secret",
                "MOONSHOT_API_KEY": "moonshot-secret",
            },
            client=client,
        )

        self.assertEqual(
            report["candidate_preflight"]["reasons"],
            ["candidate_preflight_key_not_dedicated"],
        )
        self.assertFalse(any("/key/info?" in call[0] for call in client.calls))

    def test_candidate_preflight_is_blocked_without_wildcard_route_or_dedicated_key(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "https://api.moonshot.example/v1/models": FakeResponse({"data": [{"id": "kimi-k3"}]}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={
                "LITELLM_MASTER_KEY": "litellm-secret",
                "MOONSHOT_API_KEY": "moonshot-secret",
            },
            client=client,
        )

        self.assertEqual(report["candidate_preflight"]["status"], "blocked")
        self.assertEqual(
            set(report["candidate_preflight"]["reasons"]),
            {"candidate_preflight_key_missing"},
        )
        self.assertIn(
            "candidate_preflight_route_missing",
            report["candidate_preflight"]["providers"]["moonshot"]["reasons"],
        )

    def test_missing_provider_key_fails_closed_without_request_or_removed_candidates(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "kimi-old",
                                "litellm_params": {"model": "moonshot/kimi-old"},
                                "model_info": {"metadata": {"provider_key": "moonshot"}},
                            }
                        ]
                    }
                )
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={"LITELLM_MASTER_KEY": "litellm-secret"},
            client=client,
        )

        provider = report["providers"]["moonshot"]
        self.assertEqual(provider["status"], "error")
        self.assertEqual(provider["error"]["code"], "missing_api_key")
        self.assertEqual(provider["report"]["removed"], [])
        self.assertEqual([call[0] for call in client.calls], ["http://litellm.internal:4000/model/info"])

    def test_provider_http_failure_is_redacted_and_does_not_produce_removed_candidates(self):
        secret = "moonshot-super-secret"
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse({"data": []}),
                "https://api.moonshot.example/v1/models": RuntimeError(f"401 token={secret}"),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={"LITELLM_MASTER_KEY": "litellm-secret", "MOONSHOT_API_KEY": secret},
            client=client,
        )

        provider = report["providers"]["moonshot"]
        self.assertEqual(provider["error"]["code"], "upstream_request_failed")
        self.assertEqual(provider["report"]["removed"], [])
        self.assertNotIn(secret, json.dumps(report, ensure_ascii=False))

    def test_empty_provider_result_is_error_and_never_marks_litellm_models_removed(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
                            {
                                "model_name": "kimi-k2.5",
                                "litellm_params": {"model": "moonshot/kimi-k2.6"},
                                "model_info": {"metadata": {"provider_key": "moonshot"}},
                            }
                        ]
                    }
                ),
                "https://api.moonshot.example/v1/models": FakeResponse({"data": []}),
            }
        )

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config()),
            environ={"LITELLM_MASTER_KEY": "litellm-secret", "MOONSHOT_API_KEY": "moonshot-secret"},
            client=client,
        )

        provider = report["providers"]["moonshot"]
        self.assertEqual(provider["error"]["code"], "upstream_empty")
        self.assertEqual(provider["report"]["removed"], [])

    def test_missing_litellm_key_fails_all_providers_without_network_requests(self):
        client = FakeHttpClient({})

        report = orchestrator.coordinate_candidates(
            registry=registry(moonshot_config(), acme_config()),
            environ={"MOONSHOT_API_KEY": "moonshot-secret", "ACME_API_KEY": "acme-secret"},
            client=client,
        )

        self.assertEqual(report["summary"]["providers_failed"], 2)
        self.assertEqual(client.calls, [])
        self.assertEqual(
            {item["error"]["code"] for item in report["providers"].values()},
            {"missing_litellm_api_key"},
        )

    def test_atomic_output_replaces_target_in_same_directory(self):
        payload = {"mode": "read_only", "providers": {}}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "candidate-report.json"
            output.write_text('{"old": true}', encoding="utf-8")

            with patch.object(os, "replace", wraps=os.replace) as replace:
                orchestrator.write_report_atomic(output, payload)

            source, target = replace.call_args.args
            self.assertEqual(Path(source).parent, output.parent)
            self.assertEqual(Path(target), output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_cli_rejects_apply_argument(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            orchestrator._parse_args(
                [
                    "--registry",
                    "providers.json",
                    "--output",
                    "report.json",
                    "--apply",
                ]
            )


if __name__ == "__main__":
    unittest.main()
