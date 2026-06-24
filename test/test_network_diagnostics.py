import os
import unittest
import unittest.mock

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.db.models import ToolCallLog  # noqa: E402
from app.services.network_diagnostics_service import NetworkDiagnosticsService  # noqa: E402


class NetworkDiagnosticsServiceTests(unittest.TestCase):
    def test_empty_response_for_message_without_agent_rows(self):
        db = unittest.mock.MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        service = NetworkDiagnosticsService(db)

        result = service.build_for_message(
            conversation_id="conv-1",
            message_id="assistant-1",
            is_admin=False,
        )

        self.assertTrue(result.is_empty)
        self.assertEqual(result.summary.total_tool_calls, 0)
        self.assertEqual(result.tools, [])

    def test_tool_item_is_sanitized_for_user(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-1",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="web_search",
            status="success",
            duration_ms=123,
            model_id="model",
            provider="provider",
            input_params={"query": "redis stream"},
            output_data={"result_count": 5, "secret": "hidden"},
            trace_id="trace-1",
            step_number=2,
        )

        item = service._tool_item_from_log(log, is_admin=False)

        self.assertEqual(item.target, "redis stream")
        self.assertEqual(item.result_count, 5)
        self.assertIsNone(item.admin)

    def test_tool_item_includes_admin_fields_for_admin(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-1",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="url_read",
            status="failed",
            error_message="timeout",
            duration_ms=5000,
            model_id="model",
            provider="provider",
            input_params={"url": "https://example.com/a"},
            output_data={"content": "must not leak"},
            trace_id="trace-1",
            step_number=1,
        )

        item = service._tool_item_from_log(log, is_admin=True)

        self.assertEqual(item.target, "https://example.com/a")
        self.assertEqual(item.reason, "timeout")
        self.assertEqual(item.admin["trace_id"], "trace-1")
        self.assertEqual(item.admin["input_params"], {"url": "https://example.com/a"})
        self.assertNotIn("output_data", item.admin)
