"""tool_executor 重试策略测试"""

import unittest

from app.services.stream.tool_executor import _should_retry_tool_result
from app.services.tool_handlers.base import ToolResult


class ToolRetryPolicyTests(unittest.TestCase):
    def test_degraded_result_is_terminal_and_not_retried(self):
        """degraded 是已接受的降级结果，不应再触发 backoff 重试/放弃日志"""
        result = ToolResult(status="degraded", error_message="reader-service 暂时未返回内容")

        self.assertFalse(_should_retry_tool_result(result))


if __name__ == "__main__":
    unittest.main()
