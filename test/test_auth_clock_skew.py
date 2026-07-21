"""认证 JWT 时钟偏差配置回归。"""

import unittest

from pydantic import ValidationError

from app.core.config import Settings, settings
from app.core.security import jwt_validator


class AuthClockSkewValidationTests(unittest.TestCase):
    def test_auth_clock_skew_defaults_to_five_seconds(self) -> None:
        configured = Settings()

        self.assertEqual(configured.AUTH_SERVICE_JWT_LEEWAY_SECONDS, 5.0)

    def test_fusion_jwt_validator_uses_configured_clock_skew(self) -> None:
        self.assertEqual(
            jwt_validator.leeway_seconds,
            settings.AUTH_SERVICE_JWT_LEEWAY_SECONDS,
        )

    def test_auth_clock_skew_rejects_unsafe_range(self) -> None:
        for value in (-0.1, 60.1):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    Settings(AUTH_SERVICE_JWT_LEEWAY_SECONDS=value)
