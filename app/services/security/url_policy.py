"""URL 读取安全策略。"""

from __future__ import annotations

import ipaddress
import posixpath
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

MAX_URL_LENGTH = 4096
SENSITIVE_QUERY_KEYS = ("token", "key", "signature", "auth", "code", "session")
PRIVATE_HOST_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home")
METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
}


class UrlPolicyResult(BaseModel):
    allowed: bool
    normalized_url: str | None = None
    reason: Literal[
        "ok",
        "unsupported_scheme",
        "private_host",
        "credentials_in_url",
        "sensitive_query",
        "url_too_long",
        "invalid_host",
        "blocked_domain",
    ]
    user_visible_message: str | None = None
    safe_log_url: str | None = None


def evaluate_url_policy(url: str) -> UrlPolicyResult:
    """检查 URL 是否允许自动或工具读取。

    Phase 1 只做确定性校验，不做 DNS 解析，避免测试和请求路径引入网络副作用。
    """
    raw_url = (url or "").strip()
    safe_log_url = _build_safe_log_url(raw_url)

    if len(raw_url) > MAX_URL_LENGTH:
        return UrlPolicyResult(
            allowed=False,
            reason="url_too_long",
            safe_log_url=safe_log_url,
            user_visible_message="链接过长，已跳过自动读取。",
        )

    parts = urlsplit(raw_url)
    if _get_port(parts) == "invalid":
        return UrlPolicyResult(
            allowed=False,
            reason="invalid_host",
            safe_log_url=None,
            user_visible_message="链接主机名无效，已跳过读取。",
        )
    if parts.scheme.lower() not in {"http", "https"}:
        return UrlPolicyResult(
            allowed=False,
            reason="unsupported_scheme",
            safe_log_url=safe_log_url,
            user_visible_message="仅支持读取 http 或 https 链接。",
        )

    hostname = (parts.hostname or "").strip().lower()
    if not hostname:
        return UrlPolicyResult(
            allowed=False,
            reason="invalid_host",
            safe_log_url=safe_log_url,
            user_visible_message="链接主机名无效，已跳过读取。",
        )

    if parts.username or parts.password:
        return UrlPolicyResult(
            allowed=False,
            reason="credentials_in_url",
            safe_log_url=safe_log_url,
            user_visible_message="链接包含用户名或密码，已跳过自动读取。",
        )

    if _is_private_host(hostname):
        return UrlPolicyResult(
            allowed=False,
            reason="private_host",
            safe_log_url=safe_log_url,
            user_visible_message="该链接指向私有或本地地址，已跳过读取。",
        )

    if _has_sensitive_query(parts.query):
        return UrlPolicyResult(
            allowed=False,
            reason="sensitive_query",
            safe_log_url=safe_log_url,
            user_visible_message="该链接可能包含敏感参数，需要确认后再读取。",
        )

    normalized_url = _normalize_public_url(parts)
    return UrlPolicyResult(
        allowed=True,
        normalized_url=normalized_url,
        reason="ok",
        safe_log_url=_build_safe_log_url(normalized_url),
    )


def _is_private_host(hostname: str) -> bool:
    if hostname == "localhost" or hostname.endswith(PRIVATE_HOST_SUFFIXES):
        return True

    try:
        ip = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return False

    return (
        ip in METADATA_IPS
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _has_sensitive_query(query: str) -> bool:
    if not query:
        return False
    for key, _value in parse_qsl(query, keep_blank_values=True):
        lower_key = key.lower()
        if any(marker in lower_key for marker in SENSITIVE_QUERY_KEYS):
            return True
    return False


def _normalize_public_url(parts) -> str:
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = _get_port(parts)
    netloc = host
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{host}:{port}"

    path = ""
    if parts.path:
        path = posixpath.normpath(parts.path)
        if parts.path.endswith("/") and not path.endswith("/"):
            path = f"{path}/"
        if not path.startswith("/"):
            path = f"/{path}"

    query = urlencode(parse_qsl(parts.query, keep_blank_values=True), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _build_safe_log_url(url: str) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    netloc = host
    port = _get_port(parts)
    if isinstance(port, int) and not (
        (parts.scheme == "https" and port == 443) or (parts.scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    path = ""
    if parts.path:
        path = posixpath.normpath(parts.path)
        if not path.startswith("/"):
            path = f"/{path}"
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def _get_port(parts) -> int | Literal["invalid"] | None:
    try:
        return parts.port
    except ValueError:
        return "invalid"
