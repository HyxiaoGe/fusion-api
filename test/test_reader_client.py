"""reader_client 单元测试"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.external.reader_client import read_url


class ReadUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_url_success(self):
        """成功读取 URL 返回 UrlReadResult"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "url": "https://example.com",
            "title": "Example",
            "content": "# Example\n\nHello world",
            "favicon": "https://www.google.com/s2/favicons?sz=32&domain=example.com",
            "content_length": 23,
            "fetch_ms": 500,
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client) as mock_ctor:
            result = await read_url("https://example.com")

        self.assertIsNotNone(result)
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.title, "Example")
        self.assertIn("Hello world", result.content)
        mock_ctor.assert_called_once_with(timeout=12.0)

    async def test_read_url_failure_returns_none(self):
        """请求失败时返回 None"""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        self.assertIsNone(result)

    async def test_read_url_timeout_returns_none(self):
        """超时时返回 None"""
        import httpx as real_httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        self.assertIsNone(result)

    async def test_read_url_timeout_logs_warning_not_error(self):
        """外部读取超时是工具降级信号，不应打 ERROR 触发告警"""
        import httpx as real_httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            result = await read_url("https://example.com")

        self.assertIsNone(result)
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
