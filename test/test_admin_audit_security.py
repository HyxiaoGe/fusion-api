import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from app.db.models import AgentSession, ToolCallLog
from app.schemas.admin_audit import PerformanceStageSummary
from app.services.admin_audit_sanitizer import mask_email, sanitize_admin_value
from app.services.admin_audit_service import AdminAuditService


class AdminAuditSanitizerTests(unittest.TestCase):
    def test_recursively_redacts_secret_keys_tokens_and_sensitive_url_query(self):
        value = {
            "Authorization": "Bearer top-secret-token",
            "nested": {
                "api_key": "sk-secret",
                "cookie": "sid=secret",
                "url": "https://example.com/path?token=abc&safe=ok&X-Amz-Signature=deadbeef",
            },
            "items": ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature", {"password": "secret"}],
        }

        sanitized, redacted_fields = sanitize_admin_value(value)
        serialized = json.dumps(sanitized)

        self.assertNotIn("top-secret-token", serialized)
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("sid=secret", serialized)
        self.assertNotIn("deadbeef", serialized)
        self.assertNotIn("eyJhbGci", serialized)
        self.assertIn("safe=ok", serialized)
        self.assertIn("nested.api_key", redacted_fields)
        self.assertIn("nested.url.query.token", redacted_fields)

    def test_truncates_oversized_strings_and_keeps_unknown_shapes(self):
        sanitized, redacted_fields = sanitize_admin_value(
            {"unknown": {"payload": "甲" * 5000}},
            max_string_chars=128,
        )

        self.assertTrue(sanitized["unknown"]["payload"].endswith("…"))
        self.assertLessEqual(len(sanitized["unknown"]["payload"]), 129)
        self.assertIn("unknown.payload", redacted_fields)

    def test_redacts_high_confidence_unstructured_credentials_and_bounds_dicts(self):
        value = {
            "error": (
                "openai=sk-abcdefghijklmnopqrstuvwxyz123456 "
                "github=ghp_abcdefghijklmnopqrstuvwxyz1234567890 "
                "aws=AKIAABCDEFGHIJKLMNOP"
            ),
            "large": {f"key-{index}": index for index in range(20)},
        }

        sanitized, redacted_fields = sanitize_admin_value(value, max_dict_items=5)
        serialized = json.dumps(sanitized)

        self.assertNotIn("sk-", serialized)
        self.assertNotIn("ghp_", serialized)
        self.assertNotIn("AKIA", serialized)
        self.assertLessEqual(len(sanitized["large"]), 6)
        self.assertIn("large", redacted_fields)

    def test_invalid_url_port_never_raises_and_session_status_is_not_redacted(self):
        sanitized, redacted_fields = sanitize_admin_value(
            {
                "url": "https://user:pass@example.com:bad/path?token=secret",
                "session_status": "completed",
            }
        )

        self.assertEqual(sanitized["session_status"], "completed")
        self.assertEqual(sanitized["url"], "[REDACTED]")
        self.assertNotIn("session_status", redacted_fields)

    def test_masks_email_without_hiding_domain(self):
        self.assertEqual(mask_email("admin@example.com"), "ad***@example.com")
        self.assertEqual(mask_email("a@example.com"), "a***@example.com")
        self.assertIsNone(mask_email(None))

    def test_tool_projection_never_returns_raw_content_or_unknown_payload(self):
        search_tool = ToolCallLog(
            id="tool-search",
            tool_name="web_search",
            status="success",
            model_id="model",
            provider="provider",
            input_params={"query": "Fusion", "api_key": "secret"},
            output_data={
                "result_count": 1,
                "sources": [
                    {
                        "title": "结果",
                        "url": "https://example.com/?token=secret&safe=1",
                        "content": "抓取的完整正文",
                        "description": "长摘要",
                    }
                ],
            },
        )
        unknown_tool = ToolCallLog(
            id="tool-unknown",
            tool_name="future_private_tool",
            status="success",
            model_id="model",
            provider="provider",
            input_params={"ordinary": "private input"},
            output_data={"ordinary": "private output"},
        )

        search = AdminAuditService._tool_item(search_tool)
        unknown = AdminAuditService._tool_item(unknown_tool)
        serialized = json.dumps(search, ensure_ascii=False)

        self.assertNotIn("完整正文", serialized)
        self.assertNotIn("长摘要", serialized)
        self.assertNotIn("secret", serialized)
        self.assertEqual(unknown["arguments"], {})
        self.assertEqual(unknown["result_preview"], {})
        self.assertIn("arguments", unknown["redacted_fields"])
        self.assertIn("result_preview", unknown["redacted_fields"])

    def test_agent_config_and_progress_use_allowlists_for_future_dirty_fields(self):
        session = AgentSession(
            id="run-1",
            conversation_id="conv-1",
            user_id="user-1",
            model_id="model",
            provider="provider",
            status="completed",
            run_config={
                "max_steps": 3,
                "runtime_config_versions": {"prompt_bundle/fusion": "v1"},
                "system_prompt": "private-system-prompt",
                "provider_request": {"messages": ["private-request"]},
                "storage_key": "private-path",
            },
        )
        snapshot = SimpleNamespace(
            state={
                "status": "completed",
                "progress": {"phase": "answering", "label": "回答", "developer_prompt": "private"},
                "resolved_prompt": "private-resolved-prompt",
                "provider_response": {"content": "private-response"},
            }
        )

        item = AdminAuditService(repository=None)._run_item(
            {"session": session, "steps": [], "snapshot": snapshot, "tool_calls": []}
        )
        serialized = json.dumps(item, ensure_ascii=False)

        self.assertEqual(item["config"]["max_steps"], 3)
        self.assertNotIn("private-system-prompt", serialized)
        self.assertNotIn("private-request", serialized)
        self.assertNotIn("private-path", serialized)
        self.assertNotIn("private-resolved-prompt", serialized)
        self.assertNotIn("private-response", serialized)


class PerformanceStageSummaryTests(unittest.TestCase):
    def test_accepts_safe_l1_to_l4_aggregate_fields(self):
        stage = PerformanceStageSummary.model_validate(
            {
                "scenario": "disconnect_reconnect",
                "kind": "recovery",
                "concurrency": 5,
                "duration_seconds": 1800,
                "success_rate": 0.98,
                "total": 50,
                "successful": 49,
                "failed": 1,
                "duplicate_events": 0,
                "lost_events": 0,
                "ordering_errors": 0,
                "executed_ticks": 1800,
                "skipped_ticks": 2,
                "flows_with_output": 49,
                "output_chunks": 900,
                "reasoning_chunks": 200,
                "answering_chunks": 700,
                "visible_chars": 12000,
                "approx_tokens": 6000,
                "first_output_p50_ms": 300,
                "first_output_p95_ms": 900,
                "chunk_interval_p50_ms": 40,
                "chunk_interval_p95_ms": 120,
                "chunk_interval_max_ms": 400,
                "tokens_per_second": 18.5,
                "tokens_per_second_p50": 17.5,
                "tokens_per_second_p95": 23.5,
                "recovery_latency_ms": 220,
                "recovery_latency_p95_ms": 500,
                "stop_latency_ms": 80,
                "stop_latency_p95_ms": 160,
            }
        )

        self.assertEqual(stage.scenario, "disconnect_reconnect")
        self.assertEqual(stage.kind, "recovery")
        self.assertEqual(stage.success_rate, 0.98)
        self.assertEqual(stage.chunk_interval_p95_ms, 120)
        self.assertEqual(stage.lost_events, 0)

    def test_accepts_all_safe_stage_kinds_and_preserves_old_http_payload(self):
        for kind in ("http", "sse", "recovery", "stop", "soak"):
            with self.subTest(kind=kind):
                self.assertEqual(PerformanceStageSummary(kind=kind, concurrency=1).kind, kind)

        legacy = PerformanceStageSummary.model_validate(
            {
                "kind": "http",
                "concurrency": 10,
                "requests": 100,
                "successful": 100,
                "failed": 0,
                "requests_per_second": 120.5,
                "p50_ms": 20,
                "p95_ms": 60,
                "error_rate": 0,
            }
        )
        self.assertEqual(legacy.requests, 100)
        self.assertIsNone(legacy.scenario)

    def test_rejects_identifiers_content_extra_fields_and_invalid_ranges(self):
        invalid_payloads = [
            {"kind": "recovery", "concurrency": 1, "scenario": "user@example.com"},
            {"kind": "unknown", "concurrency": 1},
            {"kind": "soak", "concurrency": 1, "success_rate": 1.01},
            {"kind": "sse", "concurrency": 1, "duplicate_events": -1},
            {"kind": "stop", "concurrency": 1, "stop_latency_ms": -0.1},
            {"kind": "http", "concurrency": 1, "conversation_id": "private-id"},
            {"kind": "sse", "concurrency": 1, "message": "private body"},
            {"kind": "sse", "concurrency": 1, "content": "private body"},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                PerformanceStageSummary.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
