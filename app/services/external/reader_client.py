"""
网页读取服务客户端 — 封装对 reader-service 的 HTTP 调用
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

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
    attempts: int = 1


@dataclass
class UrlReadFailure:
    """reader-service 读取失败的结构化原因"""

    kind: str
    message: str
    retryable: bool = False
    upstream_status: int | None = None
    attempts: int = 1
    reader_duration_ms: int | None = None


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
        attempts=_bounded_int(data.get("attempts"), default=1, minimum=1, maximum=10),
    )


_STRUCTURED_FAILURE_KINDS = {
    "timeout",
    "upstream_auth",
    "rate_limited",
    "upstream_error",
    "request_error",
}

_FAILURE_MESSAGES = {
    "timeout": "网页读取超时，已跳过该来源",
    "upstream_auth": "网页读取服务暂时不可用，已跳过该来源",
    "rate_limited": "网页读取请求暂时受限，已跳过该来源",
    "upstream_error": "网页暂时无法读取，已跳过该来源",
    "request_error": "网页读取服务暂时不可用，已跳过该来源",
    "http_status": "网页暂时无法读取，已跳过该来源",
    "parse_error": "网页读取响应异常，已跳过该来源",
    "unknown": "网页读取发生异常，已跳过该来源",
}


def _failure(
    kind: str,
    *,
    retryable: bool = False,
    upstream_status: int | None = None,
    attempts: int = 1,
    reader_duration_ms: int | None = None,
) -> UrlReadFailure:
    return UrlReadFailure(
        kind=kind,
        message=_FAILURE_MESSAGES[kind],
        retryable=retryable,
        upstream_status=upstream_status,
        attempts=attempts,
        reader_duration_ms=reader_duration_ms,
    )


def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(minimum, min(maximum, value))


def _optional_bounded_int(value, *, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(minimum, min(maximum, value))


def _structured_failure_from_response(response: httpx.Response | None) -> UrlReadFailure | None:
    if response is None:
        return None
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        return None
    kind = detail.get("kind")
    if kind not in _STRUCTURED_FAILURE_KINDS:
        return None
    return _failure(
        kind,
        retryable=detail.get("retryable") if isinstance(detail.get("retryable"), bool) else False,
        upstream_status=_optional_bounded_int(
            detail.get("upstream_status"),
            minimum=100,
            maximum=599,
        ),
        attempts=_bounded_int(detail.get("attempts"), default=1, minimum=1, maximum=10),
        reader_duration_ms=_optional_bounded_int(
            detail.get("duration_ms"),
            minimum=0,
            maximum=300_000,
        ),
    )


def _log_failure(
    failure: UrlReadFailure,
    *,
    domain: str,
    service_status: int | str | None = None,
    error: bool = False,
) -> None:
    log = logger.error if error else logger.warning
    log(
        "网页读取已降级: "
        f"domain={domain}, kind={failure.kind}, "
        f"service_status={service_status}, upstream_status={failure.upstream_status}, "
        f"attempts={failure.attempts}, reader_duration_ms={failure.reader_duration_ms}"
    )


def _safe_log_domain(url: str) -> str:
    try:
        policy = evaluate_url_policy(url)
        safe_log_url = policy.safe_log_url or ""
        return (urlsplit(safe_log_url).hostname or "").lower()
    except Exception:
        return ""


async def read_url_with_diagnostics(url: str, timeout: float | None = None) -> UrlReadResponse:
    """
    调用 reader-service 读取网页内容。
    返回读取结果或结构化失败原因；失败时不阻断对话。
    """
    effective_timeout = settings.READER_SERVICE_TIMEOUT if timeout is None else timeout
    domain = _safe_log_domain(url)
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
            except (ValueError, KeyError, TypeError):
                failure = _failure("parse_error")
                _log_failure(failure, domain=domain, error=True)
                return UrlReadResponse(result=None, failure=failure)
    except httpx.TimeoutException:
        failure = _failure("timeout", retryable=True)
        _log_failure(failure, domain=domain)
        return UrlReadResponse(result=None, failure=failure)
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        failure = _structured_failure_from_response(e.response) or _failure("http_status")
        _log_failure(
            failure,
            domain=domain,
            service_status=status_code,
        )
        return UrlReadResponse(result=None, failure=failure)
    except httpx.RequestError:
        failure = _failure("request_error")
        _log_failure(failure, domain=domain)
        return UrlReadResponse(result=None, failure=failure)
    except Exception:
        failure = _failure("unknown")
        _log_failure(failure, domain=domain, error=True)
        return UrlReadResponse(result=None, failure=failure)


async def read_url(url: str, timeout: float | None = None) -> Optional[UrlReadResult]:
    """
    调用 reader-service 读取网页内容。
    返回 UrlReadResult；失败时返回 None（不阻断对话）。
    """
    response = await read_url_with_diagnostics(url, timeout=timeout)
    return response.result
