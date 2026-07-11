import argparse
import json
import unittest

from scripts.perf.core import CleanupManifest
from scripts.perf.full_runner import (
    CapturingLoginClient,
    _cleanup_full,
    build_import_payload,
    extract_run_started_message_id,
    normalize_http_stage,
    validate_args,
)
from scripts.perf.runner import JsonResponse, RunnerError


class FakeClient:
    def request_json(self, method, url, *, payload=None, token=None):
        return JsonResponse(
            200,
            {
                "access_token": "access-secret",
                "refresh_token": "refresh-secret",
                "email": "private@example.com",
            },
        )


class FakeCleanupClient:
    def __init__(self, fail_revoke=False, missing_conversations=False):
        self.fail_revoke = fail_revoke
        self.missing_conversations = missing_conversations
        self.calls = []

    def request_json(self, method, url, *, payload=None, token=None):
        self.calls.append((method, url, payload, token))
        if self.fail_revoke and url.endswith("/auth/token/revoke"):
            raise RuntimeError("private exception")
        if self.missing_conversations and method == "DELETE":
            raise RunnerError("HTTP 404: 请求失败")
        return JsonResponse(200, {})


class FullRunnerTests(unittest.TestCase):
    def test_extracts_message_id_only_from_run_started_agent_event(self):
        self.assertEqual(
            extract_run_started_message_id(
                {
                    "chunk_type": "agent_event",
                    "data": {"type": "run_started", "message_id": "message-1"},
                }
            ),
            "message-1",
        )
        self.assertIsNone(
            extract_run_started_message_id(
                {"chunk_type": "agent_event", "data": {"type": "step_started", "message_id": "message-1"}}
            )
        )
        self.assertIsNone(extract_run_started_message_id({"chunk_type": "answering", "data": {}}))

    def test_login_capture_keeps_refresh_tokens_in_manifest_only(self):
        manifest = CleanupManifest(run_id="perf-safe", email="private@example.com")
        client = CapturingLoginClient(FakeClient(), manifest)

        response = client.request_json(
            "POST",
            "https://auth.example/auth/login",
            payload={"email": "private@example.com", "password": "password-secret"},
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(len(manifest.refresh_tokens()), 1)
        serialized = json.dumps(manifest.cleanup_plan())
        self.assertNotIn("refresh-secret", serialized)
        self.assertNotIn("private@example.com", serialized)
        self.assertNotIn("password-secret", repr(client))

    def test_parallel_cleanup_is_exact_and_deduplicates_errors(self):
        manifest = CleanupManifest(run_id="perf-safe", email="private@example.com")
        manifest.add_conversation("conv-b")
        manifest.add_conversation("conv-a")
        manifest.add_refresh_token("one", "refresh-one")
        manifest.add_refresh_token("two", "refresh-two")

        ok = _cleanup_full(
            FakeCleanupClient(),
            "https://fusion.example",
            "https://auth.example",
            "access-secret",
            manifest,
            workers=2,
        )
        failed = _cleanup_full(
            FakeCleanupClient(fail_revoke=True),
            "https://fusion.example",
            "https://auth.example",
            "access-secret",
            manifest,
            workers=2,
        )
        already_missing = _cleanup_full(
            FakeCleanupClient(missing_conversations=True),
            "https://fusion.example",
            "https://auth.example",
            "access-secret",
            manifest,
            workers=2,
        )

        self.assertEqual(ok, {"conversations_deleted": 2, "tokens_revoked": 2, "errors": []})
        self.assertEqual(failed["errors"], ["token_revoke_failed"])
        self.assertEqual(failed["conversations_deleted"], 2)
        self.assertEqual(already_missing["conversations_deleted"], 2)
        self.assertEqual(already_missing["errors"], [])

    def test_normalizes_http_stage_to_strict_admin_shape(self):
        stage = normalize_http_stage(
            {
                "scenario": "conversation_list",
                "kind": "http",
                "method": "GET",
                "authenticated": True,
                "concurrency": 25,
                "requests": 50,
                "successful": 50,
                "failed": 0,
                "p50_ms": 100,
                "p95_ms": 300,
                "max_ms": 400,
                "error_rate": 0,
                "timeout_rate": 0,
                "requests_per_second": 80,
                "url": "https://private.example",
            }
        )

        self.assertEqual(stage["scenario"], "conversation_list")
        self.assertNotIn("method", stage)
        self.assertNotIn("authenticated", stage)
        self.assertNotIn("url", stage)

    def test_import_payload_contains_only_safe_summary(self):
        payload = build_import_payload(
            run_id="perf-20260712-safe",
            model_id="deepseek-chat",
            stages=[
                {
                    "scenario": "sse_short",
                    "kind": "sse",
                    "concurrency": 1,
                    "flows": 1,
                    "successful": 1,
                    "failed": 0,
                    "p95_ttft_ms": 900,
                }
            ],
            stopped=False,
            stop_reasons=[],
            cleanup={"conversations_deleted": 1, "tokens_revoked": 2, "errors": []},
            resources={"api": {"cpu_percent": 20, "memory_mib": 250, "restarts": 0, "oom": False}},
            started_at="2026-07-12T00:00:00Z",
            finished_at="2026-07-12T00:30:00Z",
        )

        self.assertEqual(payload["environment"], "production")
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["status"], "completed")
        serialized = json.dumps(payload)
        for forbidden in (
            "account_fingerprint",
            "conversation_id",
            "agent_run_id",
            "email",
            "access_token",
            "refresh_token",
            "content",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_production_and_soak_arguments_are_guarded(self):
        base = {
            "target_url": "https://fusion.seanfield.org",
            "auth_url": "https://auth.seanfield.org",
            "prometheus_url": "http://127.0.0.1:19999",
            "confirm_production": False,
            "model_id": "deepseek-chat",
            "l1_concurrency": [10, 25, 50],
            "short_sse_concurrency": [1, 3, 5, 10],
            "long_sse_concurrency": [1, 3, 5],
            "l3_concurrency": [1, 3, 5],
            "requests_per_stage": 50,
            "timeout_seconds": 120,
            "short_max_tokens": 64,
            "long_max_tokens": 512,
            "soak_duration_seconds": 1800,
            "soak_cadence_seconds": 60,
            "soak_concurrency": 2,
        }

        with self.assertRaisesRegex(Exception, "显式确认"):
            validate_args(argparse.Namespace(**base))
        base["confirm_production"] = True
        validate_args(argparse.Namespace(**base))
        base["soak_duration_seconds"] = 60
        with self.assertRaisesRegex(Exception, "1800"):
            validate_args(argparse.Namespace(**base))

    def test_production_confirmation_cannot_be_bypassed_by_case_trailing_dot_or_auth_only(self):
        base = {
            "target_url": "https://FUSION.SEANFIELD.ORG./",
            "auth_url": "https://auth.seanfield.org",
            "prometheus_url": "http://127.0.0.1:19999",
            "confirm_production": False,
            "model_id": "deepseek-chat",
            "l1_concurrency": [10, 25, 50],
            "short_sse_concurrency": [1, 3, 5, 10],
            "long_sse_concurrency": [1, 3, 5],
            "l3_concurrency": [1, 3, 5],
            "requests_per_stage": 50,
            "timeout_seconds": 120,
            "short_max_tokens": 64,
            "long_max_tokens": 512,
            "soak_duration_seconds": 1800,
            "soak_cadence_seconds": 60,
            "soak_concurrency": 2,
        }
        with self.assertRaisesRegex(Exception, "显式确认"):
            validate_args(argparse.Namespace(**base))

        base["target_url"] = "http://127.0.0.1:8000"
        with self.assertRaisesRegex(Exception, "显式确认"):
            validate_args(argparse.Namespace(**base))

    def test_production_concurrency_is_capped_to_reviewed_matrix(self):
        args = argparse.Namespace(
            target_url="https://fusion.seanfield.org",
            auth_url="https://auth.seanfield.org",
            prometheus_url="http://127.0.0.1:19999",
            confirm_production=True,
            model_id="deepseek-chat",
            l1_concurrency=[100],
            short_sse_concurrency=[1, 3, 5, 10],
            long_sse_concurrency=[1, 3, 5],
            l3_concurrency=[1, 3, 5],
            requests_per_stage=50,
            timeout_seconds=120,
            short_max_tokens=64,
            long_max_tokens=512,
            soak_duration_seconds=1800,
            soak_cadence_seconds=60,
            soak_concurrency=2,
        )
        with self.assertRaisesRegex(Exception, "上限"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
