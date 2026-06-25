"""reader_client 单元测试"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as real_httpx

from app.services.external import reader_client
from app.services.external.reader_client import read_url


class ReadUrlTests(unittest.IsolatedAsyncioTestCase):
    def _read_url_with_diagnostics(self):
        self.assertTrue(
            hasattr(reader_client, "read_url_with_diagnostics"),
            "read_url_with_diagnostics 应存在",
        )
        return reader_client.read_url_with_diagnostics

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
        mock_ctor.assert_called_once_with(timeout=20.0)

    async def test_read_url_with_diagnostics_timeout_classifies_failure(self):
        """诊断读取超时时返回 timeout 失败原因"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com")

        self.assertIsNone(response.result)
        self.assertEqual(response.failure.kind, "timeout")
        self.assertIn("超时", response.failure.message)
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()

    async def test_read_url_with_diagnostics_logs_safe_url(self):
        """reader_client 自身日志不应写入原始敏感 URL"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com/page?token=secret&safe=1")

        self.assertEqual(response.failure.kind, "timeout")
        warning_text = str(mock_logger.warning.call_args)
        self.assertIn("https://example.com/page", warning_text)
        self.assertNotIn("token", warning_text)
        self.assertNotIn("secret", warning_text)

    async def test_read_url_with_diagnostics_http_status_classifies_failure(self):
        """诊断读取 HTTP 状态异常时返回 http_status 失败原因"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        request = real_httpx.Request("GET", "https://reader.local/read")
        http_response = real_httpx.Response(503, request=request)
        status_error = real_httpx.HTTPStatusError(
            "Service Unavailable",
            request=request,
            response=http_response,
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(side_effect=status_error)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com")

        self.assertIsNone(response.result)
        self.assertEqual(response.failure.kind, "http_status")
        self.assertIn("503", response.failure.message)
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()

    async def test_read_url_with_diagnostics_http_status_detail_omits_target_url(self):
        """HTTP 状态异常 detail 不应持久化 reader-service 请求 URL 或目标 URL query"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        request = real_httpx.Request(
            "GET",
            "https://reader.local/read?url=https%3A%2F%2Fexample.com%2Fpage%3Ftoken%3Dsecret",
        )
        http_response = real_httpx.Response(502, request=request)
        status_error = real_httpx.HTTPStatusError(
            "Bad Gateway for https://reader.local/read?url=https%3A%2F%2Fexample.com%2Fpage%3Ftoken%3Dsecret",
            request=request,
            response=http_response,
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(side_effect=status_error)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger"),
        ):
            response = await read_url_with_diagnostics("https://example.com/page?token=secret")

        self.assertEqual(response.failure.kind, "http_status")
        self.assertIn("502", response.failure.detail)
        self.assertNotIn("token", response.failure.detail)
        self.assertNotIn("example.com/page", response.failure.detail)

    async def test_read_url_with_diagnostics_request_error_classifies_failure(self):
        """诊断读取请求异常时返回 request_error 失败原因"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        request = real_httpx.Request("GET", "https://reader.local/read")
        request_error = real_httpx.RequestError("connection refused", request=request)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=request_error)

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com")

        self.assertIsNone(response.result)
        self.assertEqual(response.failure.kind, "request_error")
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()

    async def test_read_url_with_diagnostics_parse_error_classifies_failure(self):
        """诊断读取响应字段异常时返回 parse_error 失败原因"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "url": "https://example.com",
            "title": "Example",
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com")

        self.assertIsNone(response.result)
        self.assertEqual(response.failure.kind, "parse_error")
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_not_called()

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
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        self.assertIsNone(result)

    async def test_read_url_timeout_logs_warning_not_error(self):
        """外部读取超时是工具降级信号，不应打 ERROR 触发告警"""
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
