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


async def read_url(url: str, timeout: float | None = None) -> Optional[UrlReadResult]:
    """
    调用 reader-service 读取网页内容。
    返回 UrlReadResult；失败时返回 None（不阻断对话）。
    """
    effective_timeout = settings.READER_SERVICE_TIMEOUT if timeout is None else timeout
    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
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
    except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning(
            f"reader-service 暂时未返回内容，已降级跳过: url={url}, "
            f"timeout={effective_timeout}s, error={type(e).__name__}: {e}"
        )
        return None
    except Exception as e:
        logger.error(f"reader-service 响应解析失败: url={url}, error={type(e).__name__}: {e}")
        return None
