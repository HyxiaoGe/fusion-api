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

    @staticmethod
    def _http_error_response(status_code: int, body=None, *, text: str | None = None):
        request = real_httpx.Request(
            "GET",
            "https://reader.local/read?url=https%3A%2F%2Fexample.com%2Fpage%3Ftoken%3Dsecret",
        )
        if body is not None:
            response = real_httpx.Response(status_code, json=body, request=request)
        else:
            response = real_httpx.Response(status_code, text=text or "", request=request)
        return response

    @staticmethod
    def _mock_client_for_response(response):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=response)
        return mock_client

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
        self.assertEqual(result.attempts, 1)
        mock_ctor.assert_called_once_with(timeout=20.0)

    async def test_read_url_success_accepts_reader_attempts(self):
        response = self._http_error_response(
            200,
            {
                "url": "https://example.com",
                "title": "Example",
                "content": "content",
                "favicon": None,
                "content_length": 7,
                "fetch_ms": 9000,
                "attempts": 2,
            },
        )
        mock_client = self._mock_client_for_response(response)

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        self.assertIsNotNone(result)
        self.assertEqual(result.attempts, 2)

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

    async def test_read_url_with_diagnostics_logs_only_domain_and_safe_failure_metadata(self):
        """reader_client 降级日志只保留域名与安全诊断字段。"""
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=real_httpx.TimeoutException("timeout"))

        with (
            patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
            patch("app.services.external.reader_client.logger") as mock_logger,
        ):
            response = await read_url_with_diagnostics("https://example.com/private/report?token=secret&safe=1")

        self.assertEqual(response.failure.kind, "timeout")
        warning_text = str(mock_logger.warning.call_args)
        self.assertIn("domain=example.com", warning_text)
        self.assertIn("kind=timeout", warning_text)
        self.assertIn("service_status=None", warning_text)
        self.assertIn("upstream_status=None", warning_text)
        self.assertIn("attempts=1", warning_text)
        self.assertIn("reader_duration_ms=None", warning_text)
        self.assertNotIn("https://", warning_text)
        self.assertNotIn("private", warning_text)
        self.assertNotIn("report", warning_text)
        self.assertNotIn("safe=1", warning_text)
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
        self.assertEqual(response.failure.message, "网页暂时无法读取，已跳过该来源")
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
        self.assertEqual(response.failure.kind, "http_status")
        self.assertNotIn("token", str(response.failure))
        self.assertNotIn("example.com/page", str(response.failure))

    async def test_structured_reader_errors_are_parsed_from_allowlist(self):
        cases = (
            ("timeout", True, None),
            ("upstream_auth", False, 401),
            ("rate_limited", True, 429),
            ("upstream_error", True, 503),
            ("request_error", False, None),
        )
        read_url_with_diagnostics = self._read_url_with_diagnostics()

        for kind, retryable, upstream_status in cases:
            with self.subTest(kind=kind):
                response = self._http_error_response(
                    502,
                    {
                        "detail": {
                            "kind": kind,
                            "message": "恶意上游文本 token=secret https://example.com/private",
                            "retryable": retryable,
                            "upstream_status": upstream_status,
                            "attempts": 2,
                            "duration_ms": 1234,
                        }
                    },
                )
                mock_client = self._mock_client_for_response(response)
                with (
                    patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client),
                    patch("app.services.external.reader_client.logger") as mock_logger,
                ):
                    result = await read_url_with_diagnostics("https://example.com/private?token=secret")

                self.assertEqual(result.failure.kind, kind)
                self.assertEqual(result.failure.retryable, retryable)
                self.assertEqual(result.failure.upstream_status, upstream_status)
                self.assertEqual(result.failure.attempts, 2)
                self.assertEqual(result.failure.reader_duration_ms, 1234)
                self.assertNotIn("恶意上游文本", result.failure.message)
                self.assertNotIn("secret", str(result.failure))
                self.assertNotIn("恶意上游文本", str(mock_logger.warning.call_args))
                self.assertNotIn("secret", str(mock_logger.warning.call_args))

    async def test_structured_reader_error_fields_are_type_and_range_checked(self):
        response = self._http_error_response(
            502,
            {
                "detail": {
                    "kind": "timeout",
                    "message": "不能透传",
                    "retryable": "true",
                    "upstream_status": True,
                    "attempts": 999,
                    "duration_ms": -50,
                }
            },
        )
        mock_client = self._mock_client_for_response(response)

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await self._read_url_with_diagnostics()("https://example.com")

        self.assertFalse(result.failure.retryable)
        self.assertIsNone(result.failure.upstream_status)
        self.assertEqual(result.failure.attempts, 10)
        self.assertEqual(result.failure.reader_duration_ms, 0)

    async def test_unknown_structured_kind_falls_back_without_upstream_text(self):
        response = self._http_error_response(
            502,
            {
                "detail": {
                    "kind": "future_secret_error",
                    "message": "token=secret https://example.com/private",
                }
            },
        )
        mock_client = self._mock_client_for_response(response)

        with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
            result = await self._read_url_with_diagnostics()("https://example.com")

        self.assertEqual(result.failure.kind, "http_status")
        self.assertNotIn("secret", str(result.failure))

    async def test_legacy_and_non_json_http_errors_keep_http_status_fallback(self):
        cases = (
            ("legacy_string", {"detail": "Jina Reader 返回 502 token=secret"}, None),
            ("empty", None, ""),
            ("html", None, "<html>token=secret</html>"),
            ("malformed_json", None, '{"detail": token=secret'),
        )
        for name, body, text in cases:
            with self.subTest(name=name):
                response = self._http_error_response(502, body, text=text)
                mock_client = self._mock_client_for_response(response)
                with patch("app.services.external.reader_client.httpx.AsyncClient", return_value=mock_client):
                    result = await self._read_url_with_diagnostics()("https://example.com")

                self.assertEqual(result.failure.kind, "http_status")
                self.assertNotIn("secret", str(result.failure))

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
