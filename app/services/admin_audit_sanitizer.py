"""管理员审计响应的递归脱敏与有界投影。"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
TRUNCATED = "[TRUNCATED]"

_SECRET_KEYS = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "password",
    "passwd",
    "secret",
    "cookie",
    "setcookie",
    "signature",
    "credential",
    "privatekey",
    "session",
    "sessiontoken",
    "systemprompt",
    "developerprompt",
    "resolvedprompt",
    "prompt",
    "messages",
    "messagecontent",
    "content",
    "path",
    "storagekey",
    "thumbnailkey",
    "parsedcontent",
    "providerrequest",
    "providerresponse",
    "rawrequest",
    "rawresponse",
}
_SECRET_KEY_PREFIXES = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "password",
    "passwd",
    "secret",
    "cookie",
    "setcookie",
    "signature",
    "credential",
    "privatekey",
    "sessiontoken",
}
_SENSITIVE_QUERY_KEYS = _SECRET_KEYS | {
    "token",
    "xamzsignature",
    "xamzcredential",
    "xamzsecuritytoken",
    "sig",
    "key",
}
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_GITHUB_KEY_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_secret_key(value: str) -> bool:
    normalized = _normalize_key(value)
    return normalized in _SECRET_KEYS or any(normalized.startswith(prefix) for prefix in _SECRET_KEY_PREFIXES)


def _is_sensitive_query_key(value: str) -> bool:
    normalized = _normalize_key(value)
    return normalized in _SENSITIVE_QUERY_KEYS or any(normalized.startswith(prefix) for prefix in _SECRET_KEY_PREFIXES)


def mask_email(email: str | None) -> str | None:
    if not email:
        return None
    local, separator, domain = email.partition("@")
    if not separator:
        return "***"
    visible = local[:2] if len(local) > 1 else local[:1]
    return f"{visible}***@{domain}"


def _sanitize_url(value: str, path: str, redacted_fields: set[str]) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return value
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        if value.lower().startswith(("http://", "https://")):
            redacted_fields.add(path)
            return REDACTED
        return value
    netloc = f"{hostname}{port}"
    if parsed.username or parsed.password:
        redacted_fields.add(f"{path}.userinfo")

    query_items = []
    for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
        if _is_sensitive_query_key(key):
            query_items.append((key, REDACTED))
            redacted_fields.add(f"{path}.query.{key}")
        else:
            query_items.append((key, item_value))
    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query_items, doseq=True), ""))


def _sanitize_string(value: str, path: str, redacted_fields: set[str], max_string_chars: int) -> str:
    sanitized = _sanitize_url(value, path, redacted_fields)
    replacement_count = 0
    for pattern in (
        _BEARER_RE,
        _JWT_RE,
        _OPENAI_KEY_RE,
        _GITHUB_KEY_RE,
        _AWS_ACCESS_KEY_RE,
        _PEM_PRIVATE_KEY_RE,
    ):
        sanitized, count = pattern.subn(REDACTED, sanitized)
        replacement_count += count
    if replacement_count:
        redacted_fields.add(path)
    if len(sanitized) > max_string_chars:
        sanitized = sanitized[:max_string_chars] + "…"
        redacted_fields.add(path)
    return sanitized


def sanitize_admin_value(
    value: Any,
    *,
    max_string_chars: int = 4000,
    max_list_items: int = 100,
    max_dict_items: int = 100,
    max_depth: int = 12,
    max_nodes: int = 2000,
) -> tuple[Any, list[str]]:
    """递归清理管理员可见值，返回安全值和发生脱敏/截断的字段路径。"""

    redacted_fields: set[str] = set()
    remaining_nodes = max_nodes

    def visit(node: Any, path: str, depth: int) -> Any:
        nonlocal remaining_nodes
        remaining_nodes -= 1
        if remaining_nodes < 0:
            redacted_fields.add(path)
            return TRUNCATED
        if depth > max_depth:
            redacted_fields.add(path)
            return TRUNCATED
        if isinstance(node, dict):
            result: dict[str, Any] = {}
            items = list(node.items())
            for raw_key, child in items[:max_dict_items]:
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if _is_secret_key(key):
                    result[key] = REDACTED
                    redacted_fields.add(child_path)
                else:
                    result[key] = visit(child, child_path, depth + 1)
            if len(items) > max_dict_items:
                result[TRUNCATED] = len(items) - max_dict_items
                redacted_fields.add(path or "$")
            return result
        if isinstance(node, (list, tuple)):
            bounded = list(node[:max_list_items])
            result = [
                visit(child, f"{path}.{index}" if path else str(index), depth + 1)
                for index, child in enumerate(bounded)
            ]
            if len(node) > max_list_items:
                result.append(TRUNCATED)
                redacted_fields.add(path)
            return result
        if isinstance(node, str):
            return _sanitize_string(node, path or "$", redacted_fields, max_string_chars)
        if node is None or isinstance(node, (bool, int, float)):
            return node
        redacted_fields.add(path or "$")
        return _sanitize_string(str(node), path or "$", redacted_fields, max_string_chars)

    sanitized = visit(value, "", 0)
    return sanitized, sorted(redacted_fields)
