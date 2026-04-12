"""
网页读取服务客户端 — 封装对 reader-service 的 HTTP 调用
"""

from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.config import settings
from app.core.logger import app_logger as logger


@dataclass
class UrlReadResult:
    """reader-service 返回的网页读取结果"""

    url: str
    title: Optional[str]
    content: str
    favicon: Optional[str]
    content_length: int
    fetch_ms: int


async def read_url(url: str, timeout: float = 5.0) -> Optional[UrlReadResult]:
    """
    调用 reader-service 读取网页内容。
    返回 UrlReadResult；失败时返回 None（不阻断对话）。
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{settings.READER_SERVICE_URL}/read",
                params={"url": url},
            )
            resp.raise_for_status()
            data = resp.json()

        return UrlReadResult(
            url=data["url"],
            title=data.get("title"),
            content=data["content"],
            favicon=data.get("favicon"),
            content_length=data.get("content_length", 0),
            fetch_ms=data.get("fetch_ms", 0),
        )
    except Exception as e:
        logger.error(f"reader-service 调用失败: url={url}, error={type(e).__name__}: {e}")
        return None
