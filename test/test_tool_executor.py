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


if __name__ == "__main__":
    unittest.main()
