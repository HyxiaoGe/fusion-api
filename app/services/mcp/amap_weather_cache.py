"""高德天气安全投影的短期 Redis 缓存。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.logger import app_logger as logger
from app.core.redis import get_redis_pool
from app.schemas.chat import WeatherForecastDay

WEATHER_CACHE_TTL_SECONDS = 1800
_ADCODE_PATTERN = re.compile(r"^\d{6}$")


class WeatherCacheBackend(Protocol):
    async def get(self, adcode: str) -> dict[str, Any] | None: ...

    async def set(self, adcode: str, value: dict[str, Any]) -> None: ...


class WeatherCacheRecord(BaseModel):
    """缓存只允许公共天气核心，禁止混入查询文本、坐标和日志标识。"""

    model_config = ConfigDict(extra="forbid")

    resolved_location: str = Field(min_length=1, max_length=120)
    forecast_days: list[WeatherForecastDay] = Field(min_length=1, max_length=4)
    fetched_at: datetime
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("fetched_at")
    @classmethod
    def validate_fetched_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("天气缓存时间必须包含时区")
        return value

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 240 for item in value):
            raise ValueError("天气缓存 limitations 非法")
        return value

    @model_validator(mode="after")
    def validate_forecast_days(self):
        dates = [item.date for item in self.forecast_days]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("天气缓存日期必须升序且唯一")
        return self


class AmapWeatherCache:
    """按服务身份摘要与行政区编码隔离缓存；故障永远旁路。"""

    def __init__(
        self,
        *,
        service_identity: str,
        redis_getter: Callable[[], Any] = get_redis_pool,
        ttl_seconds: int = WEATHER_CACHE_TTL_SECONDS,
    ) -> None:
        self._service_digest = hashlib.sha256(service_identity.encode("utf-8")).hexdigest()[:24]
        self._redis_getter = redis_getter
        self._ttl_seconds = ttl_seconds

    def _key(self, adcode: str) -> str:
        if not _ADCODE_PATTERN.fullmatch(adcode):
            raise ValueError("行政区编码非法")
        return f"mcp:weather:v1:{self._service_digest}:{adcode}"

    async def get(self, adcode: str) -> dict[str, Any] | None:
        redis = self._redis_getter()
        if redis is None:
            return None
        try:
            raw = await redis.get(self._key(adcode))
            if not isinstance(raw, str):
                return None
            record = WeatherCacheRecord.model_validate_json(raw)
            return record.model_dump(mode="json")
        except (ValidationError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("天气缓存值无效，已旁路: adcode=%s", adcode)
            return None
        except Exception as error:  # noqa: BLE001 — 缓存故障不能影响真实查询
            logger.warning("天气缓存读取失败，已旁路: adcode=%s error=%s", adcode, type(error).__name__)
            return None

    async def set(self, adcode: str, value: dict[str, Any]) -> None:
        redis = self._redis_getter()
        if redis is None:
            return
        try:
            record = WeatherCacheRecord.model_validate(value)
            await redis.setex(
                self._key(adcode),
                self._ttl_seconds,
                record.model_dump_json(),
            )
        except (ValidationError, TypeError, ValueError):
            logger.warning("天气缓存写入值无效，已跳过: adcode=%s", adcode)
        except Exception as error:  # noqa: BLE001 — 缓存故障不能影响产品结果
            logger.warning("天气缓存写入失败，已旁路: adcode=%s error=%s", adcode, type(error).__name__)
