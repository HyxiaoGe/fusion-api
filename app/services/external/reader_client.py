"""
网页读取服务客户端 — 封装对 reader-service 的 HTTP 调用
"""

from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.services.security.url_policy import evaluate_url_policy


@dataclass
class UrlReadResult:
    """reader-service 返回的网页读取结果"""

    url: str
    title: Optional[str]
    content: str
    favicon: Optional[str]
    content_length: int
    fetch_ms: int


@dataclass
class UrlReadFailure:
    """reader-service 读取失败的结构化原因"""

    kind: str
    message: str
    detail: str


@dataclass
class UrlReadResponse:
    """reader-service 读取响应，成功和失败原因二选一"""

    result: UrlReadResult | None
    failure: UrlReadFailure | None = None


def _build_result(data: dict) -> UrlReadResult:
    return UrlReadResult(
        url=data["url"],
        title=data.get("title"),
        content=data["content"],
        favicon=data.get("favicon"),
        content_length=data.get("content_length", 0),
        fetch_ms=data.get("fetch_ms", 0),
    )


def _failure(kind: str, message: str, exc: Exception, *, detail: str | None = None) -> UrlReadFailure:
    return UrlReadFailure(
        kind=kind,
        message=message,
        detail=detail or type(exc).__name__,
    )


def _safe_log_url(url: str) -> str:
    try:
        policy = evaluate_url_policy(url)
    except Exception:
        return ""
    return policy.safe_log_url or ""


async def read_url_with_diagnostics(url: str, timeout: float | None = None) -> UrlReadResponse:
    """
    调用 reader-service 读取网页内容。
    返回读取结果或结构化失败原因；失败时不阻断对话。
    """
    effective_timeout = settings.READER_SERVICE_TIMEOUT if timeout is None else timeout
    log_url = _safe_log_url(url)
    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            resp = await client.get(
                f"{settings.READER_SERVICE_URL}/read",
                params={"url": url},
            )
            resp.raise_for_status()
            try:
                data = resp.json()
                return UrlReadResponse(result=_build_result(data))
            except (ValueError, KeyError, TypeError) as e:
                failure = _failure("parse_error", "reader-service 响应解析失败，已降级跳过", e)
                logger.error(
                    f"{failure.message}: url={log_url}, timeout={effective_timeout}s, "
                    f"error={failure.detail}"
                )
                return UrlReadResponse(result=None, failure=failure)
    except httpx.TimeoutException as e:
        failure = _failure("timeout", "reader-service 读取超时，已降级跳过", e)
        logger.warning(
            f"{failure.message}: url={log_url}, timeout={effective_timeout}s, "
            f"error={failure.detail}"
        )
        return UrlReadResponse(result=None, failure=failure)
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        failure = _failure(
            "http_status",
            f"reader-service 返回 HTTP {status_code}，已降级跳过",
            e,
            detail=f"HTTP {status_code}",
        )
        logger.warning(
            f"{failure.message}: url={log_url}, timeout={effective_timeout}s, "
            f"error={failure.detail}"
        )
        return UrlReadResponse(result=None, failure=failure)
    except httpx.RequestError as e:
        failure = _failure("request_error", "reader-service 请求失败，已降级跳过", e)
        logger.warning(
            f"{failure.message}: url={log_url}, timeout={effective_timeout}s, "
            f"error={failure.detail}"
        )
        return UrlReadResponse(result=None, failure=failure)
    except Exception as e:
        failure = _failure("unknown", "reader-service 未知异常，已降级跳过", e)
        logger.error(
            f"{failure.message}: url={log_url}, timeout={effective_timeout}s, "
            f"error={failure.detail}"
        )
        return UrlReadResponse(result=None, failure=failure)


async def read_url(url: str, timeout: float | None = None) -> Optional[UrlReadResult]:
    """
    调用 reader-service 读取网页内容。
    返回 UrlReadResult；失败时返回 None（不阻断对话）。
    """
    response = await read_url_with_diagnostics(url, timeout=timeout)
    return response.result
