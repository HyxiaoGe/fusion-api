import json
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

import httpx

from scripts import check_litellm_candidate_preflight as preflight

CANDIDATE = {
    "model_id": "kimi-k3",
    "litellm_model": "moonshot/kimi-k3",
    "capabilities": {"toolCalling": True},
    "pricing": {"input": 0.001, "output": 0.003, "unit": "USD"},
}


class FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        headers: dict | None = None,
        lines: list[str] | None = None,
        status_code: int = 200,
    ):
        self._payload = payload or {}
        self.headers = headers or {}
        self._lines = lines or []
        self.status_code = status_code

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://litellm/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


class FakeClient:
    def __init__(self, *, responses: list[FakeResponse], stream_response: FakeResponse):
        self.responses = list(responses)
        self.stream_response = stream_response
        self.post_calls: list[tuple[str, dict]] = []
        self.stream_calls: list[tuple[str, str, dict]] = []

    def post(self, url: str, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.responses.pop(0)

    @contextmanager
    def stream(self, method: str, url: str, **kwargs):
        self.stream_calls.append((method, url, kwargs))
        yield self.stream_response


def text_response(*, usage: bool = True, cost: bool = True, content: str = "ok") -> FakeResponse:
    payload = {
        "choices": [{"message": {"content": content}}],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    headers = {"x-litellm-response-cost": "0.0001"} if cost else {}
    return FakeResponse(payload, headers=headers)


def tool_response() -> FakeResponse:
    return FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        headers={"x-litellm-response-cost": "0.0002"},
    )


def stream_response(*, usage: bool = True, cost: bool = True) -> FakeResponse:
    usage_payload = ', "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}' if usage else ""
    return FakeResponse(
        headers={"x-litellm-response-cost": "0.0001"} if cost else {},
        lines=[
            'data: {"choices":[{"delta":{"content":"o"}}]}',
            f'data: {{"choices":[{{"delta":{{"content":"k"}}}}]{usage_payload}}}',
            "data: [DONE]",
        ],
    )


class LiteLLMCandidatePreflightTests(unittest.TestCase):
    def test_enriched_candidate_metadata_is_accepted_directly(self):
        candidate = preflight.parse_candidate(
            {
                "model_id": "kimi-k3",
                "litellm_model": "moonshot/kimi-k3",
                "metadata": {
                    "capabilities": {"functionCalling": True},
                    "pricing": {"input": 0.6, "output": 2.5, "unit": "USD/1M tokens"},
                },
            }
        )

        self.assertTrue(candidate.supports_tool_calling)
        self.assertEqual(candidate.pricing["input"], 0.6)

    def test_dry_run_builds_matrix_without_requests(self):
        client = Mock()

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key=None,
            apply=False,
            client=client,
        )

        self.assertTrue(report.healthy)
        self.assertTrue(report.dry_run)
        self.assertEqual([case.name for case in report.cases], ["text", "stream", "tool_calling"])
        self.assertTrue(all(case.status == "planned" for case in report.cases))
        client.post.assert_not_called()
        client.stream.assert_not_called()

    def test_apply_checks_text_stream_and_tool_contracts(self):
        client = FakeClient(
            responses=[text_response(), tool_response()],
            stream_response=stream_response(),
        )

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        self.assertTrue(report.healthy)
        self.assertEqual([case.status for case in report.cases], ["passed", "passed", "passed"])
        self.assertEqual(len(client.post_calls), 2)
        self.assertEqual(len(client.stream_calls), 1)
        stream_body = client.stream_calls[0][2]["json"]
        self.assertTrue(stream_body["stream"])
        self.assertEqual(stream_body["stream_options"], {"include_usage": True})
        tool_body = client.post_calls[1][1]["json"]
        self.assertEqual(tool_body["tool_choice"]["function"]["name"], "get_weather")

    def test_tool_case_is_skipped_when_capability_is_not_declared(self):
        candidate = dict(CANDIDATE)
        candidate["capabilities"] = {"toolCalling": False}
        client = FakeClient(responses=[text_response()], stream_response=stream_response())

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(candidate),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        self.assertTrue(report.healthy)
        tool_case = [case for case in report.cases if case.name == "tool_calling"][0]
        self.assertEqual(tool_case.status, "skipped")
        self.assertEqual(len(client.post_calls), 1)

    def test_missing_usage_or_cost_fails_preflight(self):
        client = FakeClient(
            responses=[text_response(usage=False, cost=False), tool_response()],
            stream_response=stream_response(),
        )

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        self.assertFalse(report.healthy)
        text_case = report.cases[0]
        self.assertFalse(text_case.usage_present)
        self.assertFalse(text_case.cost_present)
        self.assertIn("missing_usage", text_case.issues)
        self.assertIn("missing_cost", text_case.issues)

    def test_http_error_is_sanitized_and_fails_preflight(self):
        client = FakeClient(
            responses=[FakeResponse(status_code=500), tool_response()],
            stream_response=stream_response(),
        )

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        self.assertFalse(report.healthy)
        self.assertEqual(report.cases[0].status, "failed")
        self.assertIn("HTTPStatusError", report.cases[0].issues)

    def test_stream_requires_text_done_usage_and_cost(self):
        client = FakeClient(
            responses=[text_response(), tool_response()],
            stream_response=stream_response(usage=False, cost=False),
        )

        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        stream_case = report.cases[1]
        self.assertEqual(stream_case.status, "failed")
        self.assertTrue(stream_case.output_present)
        self.assertTrue(stream_case.stream_done)
        self.assertIn("missing_usage", stream_case.issues)
        self.assertIn("missing_cost", stream_case.issues)

    def test_serialized_summary_never_contains_keys(self):
        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key=None,
            apply=False,
            client=Mock(),
        )

        serialized = preflight.serialize_report(
            report,
            context={
                "litellm_base_url": "http://litellm:4000",
                "master_key": "sk-master-secret",
                "candidate_key": "sk-candidate-secret",
            },
        )
        output = json.dumps(serialized, ensure_ascii=False)

        self.assertNotIn("sk-master-secret", output)
        self.assertNotIn("sk-candidate-secret", output)
        self.assertNotIn("master_key", serialized["context"])
        self.assertNotIn("candidate_key", serialized["context"])
        self.assertEqual(serialized["acceptance_stage"], "candidate_pre_registration")
        self.assertEqual(serialized["transport"], "litellm_wildcard")

    def test_serialized_applied_report_matches_admission_summary_contract(self):
        client = FakeClient(
            responses=[text_response(), tool_response()],
            stream_response=stream_response(),
        )
        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        serialized = preflight.serialize_report(report, context={})

        self.assertFalse(serialized["dry_run"])
        self.assertEqual(serialized["by_model"]["kimi-k3"]["failure_count"], 0)
        self.assertTrue(all(serialized["candidate_checks_by_model"]["kimi-k3"].values()))
        self.assertEqual(serialized["quality_issues"], [])

    def test_reasoning_tag_leak_becomes_high_quality_issue(self):
        client = FakeClient(
            responses=[text_response(content="<think>secret</think> ok"), tool_response()],
            stream_response=stream_response(),
        )
        report = preflight.run_preflight(
            candidate=preflight.parse_candidate(CANDIDATE),
            base_url="http://litellm:4000",
            api_key="sk-candidate-secret",
            apply=True,
            client=client,
        )

        serialized = preflight.serialize_report(report, context={})

        self.assertEqual(serialized["quality_issues"][0]["severity"], "high")
        self.assertIn("reasoning_tag_leak", serialized["quality_issues"][0]["flags"])

    def test_main_returns_nonzero_when_contract_fails(self):
        failed_report = preflight.PreflightReport(
            healthy=False,
            dry_run=False,
            candidate=preflight.parse_candidate(CANDIDATE),
            cases=[preflight.CaseResult(name="text", status="failed", issues=("HTTPStatusError",))],
        )

        with (
            patch.dict(
                "os.environ",
                {"LITELLM_CANDIDATE_KEY": "sk-candidate-secret"},
                clear=True,
            ),
            patch.object(preflight, "run_preflight", return_value=failed_report),
            patch("builtins.print") as print_mock,
        ):
            exit_code = preflight.main([json.dumps(CANDIDATE), "--apply"])

        self.assertEqual(exit_code, 1)
        self.assertNotIn("sk-candidate-secret", print_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
