"""tool_handlers 单元测试"""

import unittest
from unittest.mock import AsyncMock, patch

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


class UrlReadHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from app.services.tool_handlers.url_read import UrlReadHandler

        self.handler = UrlReadHandler()

    async def test_execute_success(self):
        """读取成功返回内容"""
        from app.services.reader_client import UrlReadResult

        mock_result = UrlReadResult(
            url="https://example.com",
            title="Example",
            content="# Example\n\nHello world content here",
            favicon="https://favicon.example.com",
            content_length=35,
            fetch_ms=500,
        )
        with patch(
            "app.services.tool_handlers.url_read.read_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.handler.execute({"url": "https://example.com"})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["title"], "Example")
        self.assertIn("Hello world", result.data["content"])

    async def test_execute_empty_url_returns_degraded(self):
        """url 为空返回 degraded"""
        result = await self.handler.execute({"url": ""})
        self.assertEqual(result.status, "degraded")

    async def test_execute_failure_returns_failed(self):
        """读取失败返回 failed"""
        with patch(
            "app.services.tool_handlers.url_read.read_url",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await self.handler.execute({"url": "https://example.com"})

        self.assertEqual(result.status, "failed")

    def test_format_llm_context_truncates_long_content(self):
        """超长内容会被截断"""
        long_content = "x" * 20000
        result = ToolResult(
            status="success",
            data={
                "url": "https://example.com",
                "title": "Test",
                "content": long_content,
            },
        )
        context = self.handler.format_llm_context(result)
        self.assertLess(len(context), 15000)
        self.assertIn("内容已截断", context)

    def test_format_llm_context_short_content_not_truncated(self):
        """短内容不会被截断"""
        result = ToolResult(
            status="success",
            data={
                "url": "https://example.com",
                "title": "Test",
                "content": "Short content",
            },
        )
        context = self.handler.format_llm_context(result)
        self.assertIn("Short content", context)
        self.assertNotIn("内容已截断", context)

    def test_build_content_block(self):
        """构造 UrlBlock"""
        result = ToolResult(
            status="success",
            data={
                "url": "https://example.com",
                "title": "Example",
                "favicon": "https://favicon.example.com",
            },
        )
        block = self.handler.build_content_block(result, "blk_123", "log_456")
        self.assertEqual(block.type, "url_read")
        self.assertEqual(block.url, "https://example.com")
        self.assertEqual(block.title, "Example")


if __name__ == "__main__":
    unittest.main()
