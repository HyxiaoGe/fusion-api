import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import httpx

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.example:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "fusion-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.example:8100/.well-known/jwks.json")

from app.core import avatar_proxy


def _make_response(content=b"img", content_type="image/jpeg"):
    resp = MagicMock()
    resp.content = content
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


class IsAllowedAvatarUrlTests(unittest.TestCase):
    def test_allows_google_https(self):
        self.assertTrue(
            avatar_proxy.is_allowed_avatar_url("https://lh3.googleusercontent.com/a/x=s96-c")
        )

    def test_allows_github_https(self):
        self.assertTrue(
            avatar_proxy.is_allowed_avatar_url("https://avatars.githubusercontent.com/u/1?v=4")
        )

    def test_rejects_http_scheme(self):
        self.assertFalse(
            avatar_proxy.is_allowed_avatar_url("http://lh3.googleusercontent.com/a/x")
        )

    def test_rejects_other_host(self):
        self.assertFalse(avatar_proxy.is_allowed_avatar_url("https://evil.example/internal"))

    def test_rejects_ssrf_metadata_host(self):
        # 防 SSRF：白名单之外一律拒绝（含云元数据地址）。
        self.assertFalse(
            avatar_proxy.is_allowed_avatar_url("https://169.254.169.254/latest/meta-data")
        )

    def test_rejects_garbage(self):
        self.assertFalse(avatar_proxy.is_allowed_avatar_url("not a url"))


class FetchAvatarTests(unittest.TestCase):
    def setUp(self):
        avatar_proxy._cache.clear()

    def test_disallowed_url_raises_400(self):
        with self.assertRaises(avatar_proxy.AvatarProxyError) as ctx:
            avatar_proxy.fetch_avatar("https://evil.example/x")
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("app.core.avatar_proxy.httpx.get")
    def test_fetches_and_returns_bytes(self, mock_get):
        mock_get.return_value = _make_response(content=b"PNGDATA", content_type="image/png")
        body, ctype = avatar_proxy.fetch_avatar("https://lh3.googleusercontent.com/a/x=s96-c")
        self.assertEqual(body, b"PNGDATA")
        self.assertEqual(ctype, "image/png")
        mock_get.assert_called_once()

    @patch("app.core.avatar_proxy.httpx.get")
    def test_second_call_hits_cache(self, mock_get):
        mock_get.return_value = _make_response(content=b"X", content_type="image/png")
        url = "https://lh3.googleusercontent.com/a/y=s96-c"
        avatar_proxy.fetch_avatar(url)
        avatar_proxy.fetch_avatar(url)
        mock_get.assert_called_once()  # 第二次走缓存，不再打 Google

    @patch("app.core.avatar_proxy.httpx.get")
    def test_non_image_content_type_raises_502(self, mock_get):
        mock_get.return_value = _make_response(content=b"<html>", content_type="text/html")
        with self.assertRaises(avatar_proxy.AvatarProxyError) as ctx:
            avatar_proxy.fetch_avatar("https://lh3.googleusercontent.com/a/z")
        self.assertEqual(ctx.exception.status_code, 502)

    @patch("app.core.avatar_proxy.httpx.get")
    def test_oversize_raises_502(self, mock_get):
        big = b"x" * (avatar_proxy.MAX_BYTES + 1)
        mock_get.return_value = _make_response(content=big, content_type="image/png")
        with self.assertRaises(avatar_proxy.AvatarProxyError) as ctx:
            avatar_proxy.fetch_avatar("https://lh3.googleusercontent.com/a/big")
        self.assertEqual(ctx.exception.status_code, 502)

    @patch("app.core.avatar_proxy.httpx.get", side_effect=httpx.HTTPError("boom"))
    def test_upstream_error_raises_502(self, _mock_get):
        with self.assertRaises(avatar_proxy.AvatarProxyError) as ctx:
            avatar_proxy.fetch_avatar("https://lh3.googleusercontent.com/a/err")
        self.assertEqual(ctx.exception.status_code, 502)


class AvatarRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib

        from fastapi.testclient import TestClient

        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        avatar_proxy._cache.clear()

    @patch("app.core.avatar_proxy.httpx.get")
    def test_route_serves_image_with_cache_header(self, mock_get):
        mock_get.return_value = _make_response(content=b"IMG", content_type="image/jpeg")
        r = self.client.get(
            "/api/auth/avatar",
            params={"url": "https://lh3.googleusercontent.com/a/x=s96-c"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b"IMG")
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        self.assertIn("max-age", r.headers.get("cache-control", ""))

    def test_route_rejects_disallowed_host(self):
        r = self.client.get("/api/auth/avatar", params={"url": "https://evil.example/x"})
        self.assertEqual(r.status_code, 400)

    def test_route_requires_url_param(self):
        r = self.client.get("/api/auth/avatar")
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
