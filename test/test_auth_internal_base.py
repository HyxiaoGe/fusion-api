"""AUTH_SERVICE_INTERNAL_BASE_URL 的 split-horizon 解析测试。

设置内网 base 后，服务端取数（JWKS 抓取、userinfo）应走内网地址，绕开公网域名经
Cloudflare tunnel 的回环延迟；而 issuer 校验仍须使用公网 AUTH_SERVICE_BASE_URL
（见 app/core/security.py 用其当 JWT issuer）。
"""

import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.example:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "fusion-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.example:8100/.well-known/jwks.json")

from app.core.config import Settings  # noqa: E402


class AuthInternalBaseTests(unittest.TestCase):
    def test_internal_base_redirects_server_fetches(self):
        s = Settings(
            AUTH_SERVICE_BASE_URL="https://auth.seanfield.org",
            AUTH_SERVICE_JWKS_URL="https://auth.seanfield.org/.well-known/jwks.json",
            AUTH_SERVICE_INTERNAL_BASE_URL="http://192.168.1.11:8100",
        )
        # 服务端取数（JWKS / userinfo）走内网
        self.assertEqual(
            s.RESOLVED_AUTH_SERVICE_JWKS_URL,
            "http://192.168.1.11:8100/.well-known/jwks.json",
        )
        self.assertEqual(
            s.AUTH_SERVICE_USERINFO_URL,
            "http://192.168.1.11:8100/auth/userinfo",
        )
        # issuer 仍为公网域名，绝不能被内网 base 改写
        self.assertEqual(s.AUTH_SERVICE_BASE_URL, "https://auth.seanfield.org")

    def test_trailing_slash_internal_base_normalized(self):
        s = Settings(
            AUTH_SERVICE_BASE_URL="https://auth.seanfield.org",
            AUTH_SERVICE_INTERNAL_BASE_URL="http://192.168.1.11:8100/",
        )
        self.assertEqual(
            s.RESOLVED_AUTH_SERVICE_JWKS_URL,
            "http://192.168.1.11:8100/.well-known/jwks.json",
        )
        self.assertEqual(
            s.AUTH_SERVICE_USERINFO_URL,
            "http://192.168.1.11:8100/auth/userinfo",
        )

    def test_without_internal_base_keeps_public_behaviour(self):
        s = Settings(
            AUTH_SERVICE_BASE_URL="https://auth.seanfield.org",
            AUTH_SERVICE_JWKS_URL="https://auth.seanfield.org/.well-known/jwks.json",
            AUTH_SERVICE_INTERNAL_BASE_URL=None,
        )
        # 未设置内网 base 时，行为与改动前完全一致
        self.assertEqual(
            s.RESOLVED_AUTH_SERVICE_JWKS_URL,
            "https://auth.seanfield.org/.well-known/jwks.json",
        )
        self.assertEqual(
            s.AUTH_SERVICE_USERINFO_URL,
            "https://auth.seanfield.org/auth/userinfo",
        )


if __name__ == "__main__":
    unittest.main()
