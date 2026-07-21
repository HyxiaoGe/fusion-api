"""认证 JWT 时钟偏差配置回归。"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.core.security import jwt_validator


def test_auth_clock_skew_defaults_to_five_seconds() -> None:
    configured = Settings()

    assert configured.AUTH_SERVICE_JWT_LEEWAY_SECONDS == 5.0


def test_fusion_jwt_validator_uses_configured_clock_skew() -> None:
    assert jwt_validator.leeway_seconds == settings.AUTH_SERVICE_JWT_LEEWAY_SECONDS


@pytest.mark.parametrize("value", [-0.1, 60.1])
def test_auth_clock_skew_rejects_unsafe_range(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(AUTH_SERVICE_JWT_LEEWAY_SECONDS=value)
