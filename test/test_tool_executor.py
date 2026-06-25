"""tool_executor 重试策略测试"""

import unittest
from unittest.mock import AsyncMock, patch

from app.services.stream.tool_executor import _should_retry_tool_result, execute_tools_parallel
from app.services.tool_handlers.base import ToolResult


class ToolRetryPolicyTests(unittest.TestCase):
    def test_degraded_result_is_terminal_and_not_retried(self):
        """degraded 是已接受的降级结果，不应再触发 backoff 重试/放弃日志"""
        result = ToolResult(status="degraded", error_message="reader-service 暂时未返回内容")

        self.assertFalse(_should_retry_tool_result(result))


class ToolExecutorMessageIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tools_parallel_passes_message_id_to_handler_log(self):
        """工具调用日志必须关联最终 assistant message"""
        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [{"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")

    async def test_execute_tools_parallel_uses_network_budget_normalized_args_for_handler_and_log(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success", data={"sources": []})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [
                    {
                        "id": "call-1",
                        "name": "web_search",
                        "arguments": {
                            "query": "redis",
                            "count": 99,
                            "domains": ["https://Redis.io/docs", "bad domain"],
                            "recency_days": 0,
                        },
                    }
                ],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=NetworkToolBudget(),
            )

        normalized = {
            "query": "redis",
            "count": 10,
            "recency_days": 1,
        }
        handler.execute.assert_awaited_once_with(normalized)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], normalized)

    async def test_execute_tools_parallel_returns_degraded_when_network_budget_exhausted_without_handler_execute(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")
        budget = NetworkToolBudget()
        for i in range(4):
            budget.prepare_web_search_args({"query": f"q{i}"})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [{"id": "call-5", "name": "web_search", "arguments": {"query": "q5", "count": 8}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=budget,
            )

        result = results[0][1]
        self.assertEqual(result.status, "degraded")
        self.assertTrue(result.data["budget_limited"])
        handler.execute.assert_not_awaited()
        handler.log.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
