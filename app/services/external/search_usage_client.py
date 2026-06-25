"""
搜索服务用量客户端。

只做 Fusion API 到内网 search-service 的透传与错误脱敏：
Firecrawl API Key 仍只存在于 search-service，前端和 fusion-api 都不接触。
"""

from typing import Any

import httpx

from app.core.config import settings


class SearchUsageClientError(RuntimeError):
    pass


async def get_firecrawl_usage() -> dict[str, Any]:
    return await _get_json("/usage/firecrawl")


async def get_firecrawl_historical_usage() -> dict[str, Any]:
    return await _get_json("/usage/firecrawl/historical")


async def _get_json(path: str) -> dict[str, Any]:
    url = f"{settings.SEARCH_SERVICE_URL.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUsageClientError("联网用量查询失败") from exc

    if not isinstance(payload, dict):
        raise SearchUsageClientError("联网用量查询失败")
    return payload
