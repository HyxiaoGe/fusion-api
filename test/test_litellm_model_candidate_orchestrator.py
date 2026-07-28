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
        },
        "providers": list(providers),
    }


def moonshot_config():
    return {
        "adapter": "moonshot",
        "base_url": "https://api.moonshot.example",
        "api_key_env": "MOONSHOT_API_KEY",
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
    }


class ModelCandidateOrchestratorTests(unittest.TestCase):
    def test_multiple_providers_are_fetched_and_grouped(self):
        client = FakeHttpClient(
            {
                "http://litellm.internal:4000/model/info": FakeResponse(
                    {
                        "data": [
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
            },
            client=client,
        )

        self.assertEqual(report["summary"], {"providers_total": 2, "providers_ok": 2, "providers_failed": 0})
        self.assertEqual(report["providers"]["moonshot"]["report"]["new"][0]["model_id"], "kimi-k3")
        self.assertEqual(report["providers"]["acme"]["report"]["new"][0]["model_id"], "acme-reasoner")
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(call[1].keys() <= {"headers", "timeout"} for call in client.calls))

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
