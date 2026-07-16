import json
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError

from app.db.models import AgentSession, Message, ToolCallLog
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
        projected_url = urlsplit(sanitized["nested"]["url"])
        self.assertEqual(projected_url.path, "/path")
        projected_query = parse_qs(projected_url.query)
        self.assertEqual(set(projected_query), {"token", "safe", "X-Amz-Signature"})
        self.assertTrue(all(values == ["[REDACTED]"] for values in projected_query.values()))
        self.assertIn("nested.api_key", redacted_fields)
        self.assertIn("nested.url.query.token", redacted_fields)
        self.assertIn("nested.url.query.safe", redacted_fields)

    def test_redacts_all_gcs_v4_query_values_but_preserves_object_path_and_parameter_names(self):
        url = (
            "https://storage.googleapis.com/private-bucket/report.pdf"
            "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
            "&X-Goog-Credential=gcs-private-credential"
            "&X-Goog-Date=20260712T120000Z"
            "&X-Goog-Expires=900"
            "&X-Goog-SignedHeaders=host"
            "&X-Goog-Signature=gcs-private-signature"
            "&download=report.pdf"
        )

        sanitized, redacted_fields = sanitize_admin_value({"url": url})

        projected = urlsplit(sanitized["url"])
        self.assertEqual(projected.path, "/private-bucket/report.pdf")
        query = parse_qs(projected.query)
        self.assertEqual(
            set(query),
            {
                "X-Goog-Algorithm",
                "X-Goog-Credential",
                "X-Goog-Date",
                "X-Goog-Expires",
                "X-Goog-SignedHeaders",
                "X-Goog-Signature",
                "download",
            },
        )
        self.assertTrue(all(values == ["[REDACTED]"] for values in query.values()))
        self.assertNotIn("gcs-private-credential", sanitized["url"])
        self.assertNotIn("gcs-private-signature", sanitized["url"])
        self.assertIn("url.query.download", redacted_fields)

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

    def test_redacts_google_and_slack_credentials_in_unstructured_strings(self):
        google_key = "AIza" + "A" * 35
        slack_token = "xox" + "b-123456789012-123456789012-abcdefghijklmnopqrstuvwx"

        sanitized, redacted_fields = sanitize_admin_value({"error": f"google={google_key} slack={slack_token}"})
        serialized = json.dumps(sanitized)

        self.assertNotIn(google_key, serialized)
        self.assertNotIn(slack_token, serialized)
        self.assertIn("error", redacted_fields)

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
            error_message=("429 upstream failed with " + "xox" + "b-123456789012-123456789012-secretsecret"),
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
        self.assertEqual(
            unknown["error"],
            {"type": "rate_limited", "message": "上游服务请求过于频繁"},
        )
        self.assertIn("error", unknown["redacted_fields"])
        self.assertNotIn("xoxb-", json.dumps(unknown))

    def test_mcp_tool_projection_keeps_safe_error_code_without_internal_payload(self):
        mcp_tool = ToolCallLog(
            id="tool-mcp",
            tool_name="mcp_internal_alias",
            status="failed",
            model_id="model",
            provider="llm-provider",
            input_params={
                "mcp_server_id": "server-amap",
                "remote_tool_name": "maps_text_search",
                "provider": "amap",
                "config_version": 7,
                "definition_sha256": "a" * 64,
                "argument_count": 3,
                "endpoint_url": "https://mcp.amap.com/private?key=secret",
                "api_key": "secret-key",
                "keywords": "民治烤肉",
            },
            output_data={
                "mcp_server_id": "server-amap",
                "remote_tool_name": "maps_text_search",
                "provider": "amap",
                "config_version": 7,
                "definition_sha256": "a" * 64,
                "status": "failed",
                "payload_bytes": None,
                "error_code": "rate_limited",
                "raw_response": "quota response with secret-key",
            },
            error_message="MCP 工具暂时不可用",
        )

        item = AdminAuditService._tool_item(mcp_tool)
        serialized = json.dumps(item, ensure_ascii=False)

        self.assertEqual(item["arguments"]["mcp_server_id"], "server-amap")
        self.assertEqual(item["arguments"]["remote_tool_name"], "maps_text_search")
        self.assertEqual(item["result_preview"]["error_code"], "rate_limited")
        self.assertEqual(item["result_preview"]["status"], "failed")
        self.assertNotIn("endpoint_url", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("民治烤肉", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("secret-key", serialized)

    def test_error_projection_uses_boundaries_and_supports_structured_and_chinese_markers(self):
        cases = {
            "HTTP 401 unauthorized": "authentication_failed",
            "HTTP 403 forbidden": "authentication_failed",
            "upstream status 4012": "execution_failed",
            "model gpt-403b failed": "execution_failed",
            "provider rate_limit": "rate_limited",
            "provider rate-limit": "rate_limited",
            "请求频率限制": "rate_limited",
            "读取超时": "timeout",
            "任务中断": "cancelled",
            "用户取消": "cancelled",
            "上游连接失败": "upstream_unavailable",
        }

        for raw_error, expected_type in cases.items():
            with self.subTest(raw_error=raw_error):
                self.assertEqual(
                    AdminAuditService._error_projection(raw_error, "failed")["type"],
                    expected_type,
                )

    def test_message_projection_strictly_allowlists_blocks_usage_and_questions(self):
        slack_question_token = "xox" + "p-123456789012-123456789012-secretsecret"
        message = Message(
            id="message-safe-projection",
            conversation_id="conv-1",
            role="assistant",
            content=[
                {
                    "type": "text",
                    "id": "text-1",
                    "text": "正文",
                    "provider_response": "private-text-extra",
                },
                {
                    "type": "thinking",
                    "id": "thinking-1",
                    "thinking": "思考正文",
                    "raw_request": "private-thinking-extra",
                },
                {
                    "type": "file",
                    "id": "file-1",
                    "file_id": "stored-file-1",
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                    "width": 100,
                    "height": 200,
                    "thumbnail_url": "https://storage.example/thumb?token=private-thumbnail-token",
                    "storage_key": "users/private/report.pdf",
                    "path": "/private/report.pdf",
                },
                {
                    "type": "search",
                    "id": "search-1",
                    "query": "Fusion",
                    "status": "degraded",
                    "source_count": 1,
                    "sources": [
                        {
                            "title": "来源",
                            "url": "https://example.com/?safe=1&token=private-source-token",
                            "content": "private-source-body",
                            "description": "private-source-description",
                        }
                    ],
                    "error_message": "Bearer private-search-error",
                    "provider_response": {"content": "private-search-response"},
                },
                {
                    "type": "url_read",
                    "id": "url-1",
                    "url": "https://reader.example/article?safe=1&token=private-reader-token",
                    "title": "文章",
                    "status": "failed",
                    "error_message": "timeout: private-reader-error",
                    "content": "private-reader-body",
                },
                {
                    "type": "future_private_block",
                    "id": "future-1",
                    "status": "streaming",
                    "payload": "private-unknown-payload",
                    "thumbnail_url": "https://private.example/unknown",
                },
            ],
            usage={
                "input_tokens": 12,
                "output_tokens": 34,
                "total_tokens": 46,
                "cost": 9.99,
                "provider_response": "private-usage-response",
            },
            suggested_questions=[
                "安全问题",
                slack_question_token,
                *[f"问题-{index}" for index in range(20)],
            ],
        )

        item = AdminAuditService._message_item(message)

        self.assertEqual(item["content"][0], {"type": "text", "id": "text-1", "text": "正文"})
        self.assertEqual(
            item["content"][1],
            {"type": "thinking", "id": "thinking-1", "thinking": "思考正文"},
        )
        self.assertEqual(
            item["content"][2],
            {
                "type": "file",
                "id": "file-1",
                "file_id": "stored-file-1",
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "width": 100,
                "height": 200,
            },
        )
        self.assertEqual(item["content"][3]["query"], "Fusion")
        self.assertEqual(item["content"][3]["error_type"], "execution_failed")
        self.assertEqual(item["content"][4]["error_type"], "timeout")
        self.assertEqual(
            item["content"][5],
            {
                "type": "future_private_block",
                "id": "future-1",
                "status": "streaming",
                "content_hidden": True,
            },
        )
        self.assertEqual(item["usage"], {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46})
        self.assertEqual(len(item["suggested_questions"]), 10)
        serialized = json.dumps(item, ensure_ascii=False)
        for private_value in (
            "private-text-extra",
            "private-thinking-extra",
            "private-thumbnail-token",
            "users/private/report.pdf",
            "/private/report.pdf",
            "private-source-token",
            "private-source-body",
            "private-source-description",
            "private-search-error",
            "private-search-response",
            "private-reader-token",
            "private-reader-error",
            "private-reader-body",
            "private-unknown-payload",
            "private-usage-response",
            "xoxp-",
        ):
            self.assertNotIn(private_value, serialized)

    def test_agent_config_and_progress_use_allowlists_for_future_dirty_fields(self):
        session = AgentSession(
            id="run-1",
            conversation_id="conv-1",
            user_id="user-1",
            model_id="model",
            provider="provider",
            status="completed",
            error_message="401 unauthorized with AIza" + "B" * 35,
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
        self.assertEqual(item["error"], {"type": "authentication_failed", "message": "上游服务认证失败"})
        self.assertNotIn("AIza", serialized)


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
