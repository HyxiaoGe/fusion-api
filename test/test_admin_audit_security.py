import json
import unittest
from types import SimpleNamespace

from app.db.models import AgentSession, ToolCallLog
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


if __name__ == "__main__":
    unittest.main()
