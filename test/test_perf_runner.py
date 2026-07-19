import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from scripts.perf.core import (
    CleanupManifest,
    RequestSample,
    SSEParser,
    StopPolicy,
    build_safe_result,
    extract_agent_trace_ids,
    summarize_samples,
)
from scripts.perf.runner import (
    HttpClient,
    JsonResponse,
    RunnerError,
    _FailClosedRedirectHandler,
    authenticate,
    build_parser,
    cleanup_run,
    execute,
)

VALID_INTERNAL_TOKEN = "test-internal-auth-token-0123456789abcdef"


class FakeClient:
    def __init__(self):
        self.calls = []

    def request_json(self, method, url, *, payload=None, token=None, extra_headers=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "token": token,
                "extra_headers": extra_headers,
            }
        )
        if url.endswith("/auth/register"):
            return JsonResponse(201, {"refresh_token": "registration-refresh"})
        if url.endswith("/auth/login"):
            return JsonResponse(
                200,
                {"access_token": "scoped-access", "refresh_token": "fusion-refresh"},
            )
        return JsonResponse(200, {})


class SSEParserTest(unittest.TestCase):
    def test_parses_multiline_data_and_cursor(self):
        parser = SSEParser()

        self.assertIsNone(parser.feed_line("id: 42-1\n"))
        self.assertIsNone(parser.feed_line('data: {"chunk_type":"answering",\n'))
        self.assertIsNone(parser.feed_line('data: "data":{"delta":"你好"}}\n'))
        event = parser.feed_line("\n")

        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, "42-1")
        self.assertEqual(event.payload["chunk_type"], "answering")
        self.assertEqual(event.payload["data"]["delta"], "你好")
        self.assertFalse(event.done)

    def test_parses_done_marker_without_json_error(self):
        parser = SSEParser()
        parser.feed_line("data: [DONE]\n")

        event = parser.feed_line("\n")

        self.assertTrue(event.done)
        self.assertIsNone(event.payload)

    def test_ignores_comments_and_empty_events(self):
        parser = SSEParser()

        self.assertIsNone(parser.feed_line(": keepalive\n"))
        self.assertIsNone(parser.feed_line("\n"))

    def test_extracts_run_started_ids_from_supported_envelope_positions(self):
        envelopes = [
            {
                "chunk_type": "agent_event",
                "data": {"type": "run_started", "run_id": "run-direct", "trace_id": "trace-direct"},
            },
            {
                "chunk_type": "agent_event",
                "data": {"event": {"type": "run_started", "run_id": "run-nested", "trace_id": "trace-nested"}},
            },
            {
                "chunk_type": "agent_event",
                "type": "run_started",
                "run_id": "run-top",
                "trace_id": "trace-top",
                "data": {},
            },
        ]

        extracted = [extract_agent_trace_ids(envelope) for envelope in envelopes]

        self.assertEqual(
            extracted,
            [
                ("run-direct", "trace-direct"),
                ("run-nested", "trace-nested"),
                ("run-top", "trace-top"),
            ],
        )

    def test_does_not_extract_ids_from_non_start_or_non_agent_events(self):
        self.assertEqual(
            extract_agent_trace_ids(
                {"chunk_type": "agent_event", "data": {"type": "step_started", "trace_id": "trace-secret"}}
            ),
            (None, None),
        )
        self.assertEqual(
            extract_agent_trace_ids(
                {"chunk_type": "error", "data": {"type": "run_started", "trace_id": "error-content"}}
            ),
            (None, None),
        )


class StatisticsTest(unittest.TestCase):
    def test_summarizes_latency_and_failures(self):
        samples = [
            RequestSample(latency_ms=10, status=200),
            RequestSample(latency_ms=20, status=200),
            RequestSample(latency_ms=30, status=503, error="http_503"),
            RequestSample(latency_ms=40, status=None, error="timeout", timed_out=True),
        ]

        summary = summarize_samples(samples)

        self.assertEqual(summary["requests"], 4)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["p50_ms"], 20)
        self.assertEqual(summary["p95_ms"], 40)
        self.assertEqual(summary["error_rate"], 0.5)
        self.assertEqual(summary["timeout_rate"], 0.25)

    def test_hard_stop_policy_waits_for_minimum_sample_count(self):
        policy = StopPolicy(min_samples=20, max_error_rate=0.05, max_timeout_rate=0.05)
        short_summary = summarize_samples(
            [RequestSample(latency_ms=10, status=500, error="http_500") for _ in range(10)]
        )
        full_summary = summarize_samples(
            [RequestSample(latency_ms=10, status=500, error="http_500")]
            + [RequestSample(latency_ms=10, status=200) for _ in range(19)]
        )

        self.assertEqual(policy.evaluate(short_summary), [])
        self.assertIn("error_rate", policy.evaluate(full_summary))

    def test_hard_stop_policy_checks_consecutive_failures_and_latency_ceiling(self):
        policy = StopPolicy(max_consecutive_failures=3, max_p95_ms=500)

        reasons = policy.evaluate(
            {"requests": 4, "error_rate": 0, "timeout_rate": 0, "p95_ms": 800},
            consecutive_failures=3,
        )

        self.assertEqual(reasons, ["consecutive_failures", "p95_ms"])


class CleanupManifestTest(unittest.TestCase):
    def test_cleanup_plan_is_exact_and_deduplicated(self):
        manifest = CleanupManifest(run_id="perf-20260711-abcd", email="fusion-perf+perf-20260711-abcd@example.invalid")
        manifest.add_conversation("conv-b")
        manifest.add_conversation("conv-a")
        manifest.add_conversation("conv-b")
        manifest.add_refresh_token("registration", "secret-registration")
        manifest.add_refresh_token("fusion_login", "secret-login")
        manifest.add_agent_trace("run-2", "trace-2")
        manifest.add_agent_trace("run-1", "trace-1")
        manifest.add_agent_trace("run-1", "trace-1")

        plan = manifest.cleanup_plan()

        self.assertEqual(plan["conversation_ids"], ["conv-a", "conv-b"])
        self.assertEqual(plan["refresh_token_labels"], ["fusion_login", "registration"])
        self.assertEqual(plan["agent_run_ids"], ["run-1", "run-2"])
        self.assertEqual(plan["agent_trace_ids"], ["trace-1", "trace-2"])
        self.assertNotIn("secret", json.dumps(plan))
        self.assertNotIn("@", json.dumps(plan))
        self.assertEqual(plan["run_id"], "perf-20260711-abcd")

    def test_safe_result_never_contains_credentials_or_exact_email(self):
        manifest = CleanupManifest(run_id="perf-20260711-abcd", email="fusion-perf+perf-20260711-abcd@example.invalid")
        manifest.add_refresh_token("registration", "refresh-secret")
        manifest.add_conversation("conv-secret")
        manifest.add_agent_trace("run-public", "trace-public")

        result = build_safe_result(
            manifest=manifest,
            stages=[{"kind": "sse", "concurrency": 1}],
            cleanup={"conversations_deleted": 1, "tokens_revoked": 1, "errors": []},
            stopped=False,
            stop_reasons=[],
        )
        serialized = json.dumps(result)

        self.assertNotIn("refresh-secret", serialized)
        self.assertNotIn("@example.invalid", serialized)
        self.assertNotIn("conv-secret", serialized)
        self.assertNotIn("password", serialized)
        self.assertEqual(result["run_id"], "perf-20260711-abcd")
        self.assertEqual(result["agent_run_ids"], ["run-public"])
        self.assertEqual(result["agent_trace_ids"], ["trace-public"])
        self.assertRegex(result["account_fingerprint"], r"^[0-9a-f]{12}$")

    def test_authentication_logs_in_again_with_configured_client_id(self):
        client = FakeClient()
        manifest = CleanupManifest(run_id="perf-abcd", email="fusion-perf+perf-abcd@seanfield.org")

        access_token = authenticate(
            client,
            "https://auth.example",
            "app-public-id",
            manifest,
            "password-secret",
            internal_auth_token=VALID_INTERNAL_TOKEN,
        )

        self.assertEqual(access_token, "scoped-access")
        self.assertEqual(client.calls[1]["payload"]["client_id"], "app-public-id")
        self.assertEqual(
            [call["extra_headers"] for call in client.calls],
            [
                {"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
                {"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
            ],
        )
        self.assertEqual(
            [label for label, _ in manifest.refresh_tokens()],
            ["fusion_login", "registration"],
        )

    def test_authentication_fails_before_network_when_internal_token_is_missing(self):
        client = FakeClient()
        manifest = CleanupManifest(run_id="perf-abcd", email="fusion-perf+perf-abcd@seanfield.org")

        for invalid_token in (None, "", "x" * 31, f" {VALID_INTERNAL_TOKEN}", f"{VALID_INTERNAL_TOKEN} "):
            with self.subTest(invalid_token=invalid_token):
                with self.assertRaisesRegex(RunnerError, "FUSION_PERF_INTERNAL_AUTH_TOKEN"):
                    authenticate(
                        client,
                        "https://auth.example",
                        "app-public-id",
                        manifest,
                        "password-secret",
                        internal_auth_token=invalid_token,
                    )

        self.assertEqual(client.calls, [])

    def test_http_client_sends_only_explicit_extra_headers(self):
        class FakeHttpResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"{}"

        client = HttpClient(timeout_seconds=1)
        with (
            patch.object(client._no_redirect_opener, "open", return_value=FakeHttpResponse()) as restricted_open,
            patch("scripts.perf.runner.urllib.request.urlopen", return_value=FakeHttpResponse()) as urlopen,
        ):
            client.request_json(
                "POST",
                "https://auth.example/auth/login",
                payload={"email": "private@example.com"},
                extra_headers={"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
            )
            client.request_json("GET", "https://fusion.example/api/models/")

        first_headers = {key.lower(): value for key, value in restricted_open.call_args.args[0].header_items()}
        second_headers = {key.lower(): value for key, value in urlopen.call_args.args[0].header_items()}
        self.assertEqual(first_headers["x-fusion-internal-auth"], VALID_INTERNAL_TOKEN)
        self.assertNotIn("x-fusion-internal-auth", second_headers)
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                self.assertIsNone(
                    _FailClosedRedirectHandler().redirect_request(
                        restricted_open.call_args.args[0],
                        None,
                        status,
                        "Redirect",
                        {},
                        "https://other.example/capture",
                    )
                )

    def test_http_client_turns_auth_redirect_into_sanitized_failure(self):
        client = HttpClient(timeout_seconds=1)
        redirect = urllib.error.HTTPError(
            "https://auth.example/auth/login",
            302,
            "Found",
            {},
            io.BytesIO(b"redirect body"),
        )

        with patch.object(client._no_redirect_opener, "open", side_effect=redirect) as restricted_open:
            with self.assertRaisesRegex(RunnerError, "HTTP 302: 请求失败") as raised:
                client.request_json(
                    "POST",
                    "https://auth.example/auth/login",
                    payload={"email": "private@example.com"},
                    extra_headers={"X-Fusion-Internal-Auth": VALID_INTERNAL_TOKEN},
                )

        self.assertEqual(restricted_open.call_count, 1)
        self.assertNotIn(VALID_INTERNAL_TOKEN, str(raised.exception))

    def test_runner_authentication_reads_internal_token_from_environment(self):
        args = build_parser().parse_args(
            [
                "--mode",
                "sse",
                "--target-url",
                "https://fusion.example",
                "--auth-url",
                "https://auth.example",
                "--model-id",
                "model-a",
            ]
        )

        with (
            patch.dict("os.environ", {"FUSION_PERF_INTERNAL_AUTH_TOKEN": VALID_INTERNAL_TOKEN}),
            patch("scripts.perf.runner.HttpClient", return_value=object()),
            patch("scripts.perf.runner.authenticate", side_effect=RunnerError("expected stop")) as auth_mock,
            patch(
                "scripts.perf.runner.cleanup_run",
                return_value={"conversations_deleted": 0, "tokens_revoked": 0, "errors": []},
            ),
        ):
            with self.assertRaisesRegex(RunnerError, "expected stop"):
                execute(args)

        self.assertEqual(auth_mock.call_args.kwargs["internal_auth_token"], VALID_INTERNAL_TOKEN)

    def test_cleanup_deletes_only_manifest_conversations_and_revokes_both_tokens(self):
        client = FakeClient()
        manifest = CleanupManifest(run_id="perf-abcd", email="fusion-perf+perf-abcd@seanfield.org")
        manifest.add_conversation("conv-b")
        manifest.add_conversation("conv-a")
        manifest.add_refresh_token("registration", "registration-refresh")
        manifest.add_refresh_token("fusion_login", "fusion-refresh")

        result = cleanup_run(
            client,
            "https://fusion.example",
            "https://auth.example",
            "scoped-access",
            manifest,
        )

        delete_urls = [call["url"] for call in client.calls if call["method"] == "DELETE"]
        self.assertEqual(
            delete_urls,
            [
                "https://fusion.example/api/chat/conversations/conv-a",
                "https://fusion.example/api/chat/conversations/conv-b",
            ],
        )
        self.assertEqual(result, {"conversations_deleted": 2, "tokens_revoked": 2, "errors": []})


if __name__ == "__main__":
    unittest.main()
