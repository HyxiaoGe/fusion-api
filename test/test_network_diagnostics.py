import os
import unittest
import unittest.mock

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.db.models import AgentSession, ToolCallLog  # noqa: E402
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

    def test_web_search_tool_item_includes_dynamic_network_metadata(self):
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
            input_params={
                "query": "redis stream",
                "count": 8,
                "intent": "comparison",
                "domains": ["redis.io"],
                "recency_days": 30,
            },
            output_data={
                "requested_count": 8,
                "actual_count": 7,
                "context_source_count": 6,
                "budget_limited": True,
            },
        )

        item = service._tool_item_from_log(log, is_admin=False)

        self.assertEqual(item.requested_count, 8)
        self.assertEqual(item.actual_count, 7)
        self.assertEqual(item.context_count, 6)
        self.assertEqual(item.intent, "comparison")
        self.assertEqual(item.domains, ["redis.io"])
        self.assertEqual(item.recency_days, 30)
        self.assertTrue(item.budget_limited)

    def test_success_url_read_reason_prefers_input_params(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-1",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="url_read",
            status="success",
            duration_ms=123,
            model_id="model",
            provider="provider",
            input_params={"url": "https://example.com/a", "reason": "需要核实官方原文细节"},
            output_data={"reason": "旧原因"},
        )

        item = service._tool_item_from_log(log, is_admin=False)

        self.assertEqual(item.reason, "需要核实官方原文细节")

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

    def test_itinerary_admin_summary_reuses_safe_tool_logs_and_message_block(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        session = AgentSession(
            id="run-trip",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            model_id="model-trip",
            provider="provider-trip",
            status="completed",
            total_duration_ms=4321,
            total_steps=2,
            total_tool_calls=3,
        )
        logs = [
            ToolCallLog(
                id="log-flight",
                conversation_id="conv-1",
                message_id="assistant-1",
                user_id="user-1",
                tool_name="search_flights",
                status="success",
                duration_ms=120,
                model_id="model-trip",
                provider="provider-trip",
                output_data={"status": "success", "private": "must-not-leak"},
            ),
            ToolCallLog(
                id="log-weather",
                conversation_id="conv-1",
                message_id="assistant-1",
                user_id="user-1",
                tool_name="weather_forecast",
                status="failed",
                duration_ms=300,
                model_id="model-trip",
                provider="provider-trip",
                output_data={"error_code": "upstream_error"},
            ),
        ]
        tools = [service._tool_item_from_log(log, is_admin=True) for log in logs]

        summary = service._build_itinerary_admin_summary(
            session,
            tools,
            [{"type": "itinerary_results", "status": "degraded", "origin": "private-city"}],
        )

        self.assertEqual(summary["result"], "partial")
        self.assertEqual(summary["model_id"], "model-trip")
        self.assertEqual(summary["provider"], "provider-trip")
        self.assertEqual(summary["total_duration_ms"], 4321)
        self.assertEqual(summary["upstream_error_count"], 1)
        self.assertNotIn("private-city", str(summary))
        self.assertNotIn("must-not-leak", str(summary))

    def test_itinerary_tool_error_categories_are_admin_only(self):
        service = NetworkDiagnosticsService(unittest.mock.MagicMock())
        log = ToolCallLog(
            id="log-train",
            conversation_id="conv-1",
            message_id="assistant-1",
            user_id="user-1",
            tool_name="search_trains",
            status="failed",
            error_message="出行查询暂时不可用",
            duration_ms=5000,
            model_id="model-trip",
            provider="provider-trip",
            input_params={"secret": "must-not-leak"},
            output_data={
                "error_code": "travel_run_budget_exhausted",
                "validation_errors": ["origin:required"],
            },
        )

        user_item = service._tool_item_from_log(log, is_admin=False)
        admin_item = service._tool_item_from_log(log, is_admin=True)

        self.assertIsNone(user_item.admin)
        self.assertEqual(user_item.reason, "出行查询暂时不可用")
        self.assertEqual(admin_item.admin["outcome"], "failed")
        self.assertEqual(admin_item.admin["error_category"], "travel_budget_exhausted")
        self.assertTrue(admin_item.admin["travel_budget_exhausted"])
        self.assertFalse(admin_item.admin["server_budget_exhausted"])
        self.assertNotIn("validation_errors", str(admin_item.admin))
        self.assertNotIn("must-not-leak", str(admin_item.admin))
