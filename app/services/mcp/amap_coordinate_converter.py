"""使用高德官方 Web 服务把设备 WGS84 坐标转换为 GCJ02。"""

from __future__ import annotations

import os

import httpx

from app.services.agent.context_broker import Geolocation

_AMAP_COORDINATE_CONVERT_URL = "https://restapi.amap.com/v3/assistant/coordinate/convert"
_SAFE_ERROR_MESSAGE = "高德坐标转换不可用"


class AmapCoordinateConversionError(Exception):
    def __init__(self) -> None:
        super().__init__(_SAFE_ERROR_MESSAGE)


class _SensitiveQueryTransport(httpx.AsyncBaseTransport):
    """在 HTTPX 日志层之后注入 Key 与坐标，避免完整查询串进入访问日志。"""

    def __init__(
        self,
        query_params: dict[str, str],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._query_params = query_params
        self._transport = transport or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        sensitive_request = httpx.Request(
            request.method,
            request.url.copy_merge_params(self._query_params),
            headers=request.headers,
            stream=request.stream,
            extensions=request.extensions,
        )
        return await self._transport.handle_async_request(sensitive_request)

    async def aclose(self) -> None:
        await self._transport.aclose()


async def convert_wgs84_to_gcj02(
    location: Geolocation,
    *,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """转换单个坐标；异常内容永不携带请求 URL、响应正文或 Key。"""
    resolved_key = api_key or os.getenv("AMAP_MCP_API_KEY")
    if not isinstance(resolved_key, str) or not resolved_key.strip():
        raise AmapCoordinateConversionError

    params = {
        "key": resolved_key.strip(),
        "locations": f"{location.longitude:.6f},{location.latitude:.6f}",
        "coordsys": "gps",
    }
    try:
        async with httpx.AsyncClient(
            transport=_SensitiveQueryTransport(params, transport),
            timeout=httpx.Timeout(5.0, connect=3.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(_AMAP_COORDINATE_CONVERT_URL)
        if response.status_code != 200:
            raise AmapCoordinateConversionError
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "1":
            raise AmapCoordinateConversionError
        raw_coordinate = payload.get("locations")
        if not isinstance(raw_coordinate, str):
            raise AmapCoordinateConversionError
        raw_lon, raw_lat = raw_coordinate.split(",", 1)
        longitude = float(raw_lon)
        latitude = float(raw_lat)
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise AmapCoordinateConversionError
        return f"{longitude:.6f},{latitude:.6f}"
    except AmapCoordinateConversionError:
        raise
    except Exception:
        raise AmapCoordinateConversionError from None
