"""tool_handlers 单元测试"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.tool_handlers.base import ToolResult


class WebSearchHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from app.services.tool_handlers.web_search import WebSearchHandler

        self.handler = WebSearchHandler()

    async def test_execute_success(self):
        """搜索成功返回 sources"""
        from app.schemas.chat import SearchSource

        mock_sources = [
            SearchSource(title="Result", url="https://example.com", description="desc")
        ]
        with patch(
            "app.services.tool_handlers.web_search.search_web",
            new_callable=AsyncMock,
            return_value=mock_sources,
        ):
            result = await self.handler.execute({"query": "test"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["result_count"], 1)

    async def test_execute_empty_query_returns_degraded(self):
        """query 为空返回 degraded"""
        result = await self.handler.execute({"query": ""})
        self.assertEqual(result.status, "degraded")

    async def test_execute_search_failure(self):
        """搜索异常返回 failed"""
        with patch(
            "app.services.tool_handlers.web_search.search_web",
            new_callable=AsyncMock,
            side_effect=Exception("search down"),
        ):
            result = await self.handler.execute({"query": "test"})

        self.assertEqual(result.status, "failed")

    def test_format_llm_context(self):
        """格式化搜索上下文包含来源编号"""
        from app.schemas.chat import SearchSource

        sources = [SearchSource(title="R1", url="https://a.com", description="d1")]
        result = ToolResult(
            status="success",
            data={"sources": sources, "result_count": 1},
        )
        context = self.handler.format_llm_context(result)
        self.assertIn("[1]", context)
        self.assertIn("R1", context)

    def test_build_content_block(self):
        """构造 SearchBlock"""
        from app.schemas.chat import SearchSource

        sources = [SearchSource(title="R1", url="https://a.com", description="d1")]
        result = ToolResult(
            status="success",
            data={"sources": sources, "result_count": 1},
        )
        block = self.handler.build_content_block(result, "blk_123", "log_456")
        self.assertEqual(block.type, "search")
        self.assertEqual(len(block.sources), 1)


if __name__ == "__main__":
    unittest.main()
