"""
搜索服务客户端 — 封装对私有 search-service 的 HTTP 调用
"""

from typing import List

import httpx

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.schemas.chat import SearchSource


async def search_web(query: str, count: int = 5) -> List[SearchSource]:
    """
    调用 search-service 执行网络搜索。
    返回 SearchSource 列表；失败时返回空列表（不阻断对话）。
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.SEARCH_SERVICE_URL}/search",
                json={
                    "query": query,
                    "type": "web",
                    "count": count,
                    "freshness": "pw",  # 优先返回一周内的结果
                },
            )
            resp.raise_for_status()
            data = resp.json()

        requested_provider = data.get("requested_provider")
        result_provider = data.get("result_provider") or data.get("provider")
        fallback_used = bool(data.get("fallback_used", False))
        provider_chain = data.get("provider_chain") if isinstance(data.get("provider_chain"), list) else []

        return [
            SearchSource(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                content=r.get("content"),
                favicon=r.get("favicon"),
                requested_provider=requested_provider,
                result_provider=result_provider,
                fallback_used=fallback_used,
                provider_chain=provider_chain,
            )
            for r in data.get("results", [])
        ]
    except Exception as e:
        logger.error(f"搜索服务调用失败: {e}")
        return []
