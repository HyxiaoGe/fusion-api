"""arguments 脱敏 hook + result_summary 截断."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from app.services.security.url_policy import evaluate_url_policy

URL_READ_REASON_MAX_CHARS = 160
EXTERNAL_TOOL_ARGUMENT_MAX_BYTES = 4096
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "apikey",
        "authorization",
        "bearer",
        "credential",
        "cookie",
        "password",
        "privatekey",
        "secret",
        "sessionid",
        "token",
    }
)


def sanitize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """对 tool arguments 做脱敏。

    agent_event 会进入 Redis Stream，工具执行前就会发出。
    url_read 的 URL 需要在这里先清理，不能等到 handler 执行后再处理。
    """
    if tool_name == "url_read":
        return sanitize_url_read_arguments(arguments)
    if tool_name.startswith("mcp_"):
        return sanitize_external_tool_arguments(arguments)
    return arguments


def sanitize_external_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """MCP 参数进入事件或日志前递归脱敏，并限制序列化体积。"""
    source = arguments if isinstance(arguments, dict) else {}
    sanitized = _sanitize_external_value(source, depth=0)
    if not isinstance(sanitized, dict):
        return {}
    return cap_and_truncate(sanitized, max_bytes=EXTERNAL_TOOL_ARGUMENT_MAX_BYTES)


def _sanitize_external_value(value: Any, *, depth: int) -> Any:
    if depth >= 8:
        return "[内容已省略]"
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:64]:
            key = str(raw_key)[:128]
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
                sanitized[key] = "[已脱敏]"
            else:
                sanitized[key] = _sanitize_external_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_external_value(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, tuple):
        return [_sanitize_external_value(item, depth=depth + 1) for item in value[:64]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1_024]


def sanitize_url_read_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """严格白名单化 url_read 参数，供 SSE 与 ToolCallLog 共用。"""
    source = arguments if isinstance(arguments, dict) else {}
    sanitized: dict[str, Any] = {}
    url = source.get("url")
    if isinstance(url, str):
        try:
            policy = evaluate_url_policy(url)
        except Exception:
            sanitized["url"] = ""
            sanitized["url_policy_reason"] = "invalid_url"
        else:
            sanitized["url"] = policy.safe_log_url or ""
            if not policy.allowed:
                sanitized["url_policy_reason"] = policy.reason
    elif url is not None:
        sanitized["url"] = ""
        sanitized["url_policy_reason"] = "invalid_url"

    reason = source.get("reason")
    if isinstance(reason, str):
        normalized_reason = reason.strip()
        if normalized_reason:
            sanitized["reason"] = normalized_reason[:URL_READ_REASON_MAX_CHARS]
    return sanitized


def _utf8_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _shrink_strings(node: Any) -> bool:
    """递归把 node 中所有 len>9 的 str 字段减半 + "…"。

    支持 dict / list 任意嵌套；in-place 修改。
    返回 True 表示本次至少有一个字段被缩短。
    """
    shrunk = False
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str) and len(v) > 9:
                node[k] = v[: max(8, len(v) // 2)] + "…"
                shrunk = True
            elif isinstance(v, (dict, list)):
                if _shrink_strings(v):
                    shrunk = True
    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, str) and len(item) > 9:
                node[i] = item[: max(8, len(item) // 2)] + "…"
                shrunk = True
            elif isinstance(item, (dict, list)):
                if _shrink_strings(item):
                    shrunk = True
    return shrunk


def cap_and_truncate(payload: dict[str, Any], max_bytes: int = 1024) -> dict[str, Any]:
    """把 result_summary 控制在 max_bytes 之内（UTF-8 JSON 字节，硬上限）。

    分三阶段：
      Phase 1 — 递归把任意层级的 str 字段反复减半 + "…"，置 truncated=True
      Phase 2 — 仍超限时，按序列化体积从大到小删除嵌套 dict/list 字段
      Phase 3 — 仍超限时回退到最小 {kind?, truncated:True}；极端情况回退 {}

    强保证：返回的 dict 序列化为 UTF-8 JSON 后字节数 ≤ max_bytes。
    """
    if _utf8_size(payload) <= max_bytes:
        return payload

    out = copy.deepcopy(payload)
    out["truncated"] = True

    # Phase 1: 递归裁字符串
    while _utf8_size(out) > max_bytes:
        if not _shrink_strings(out):
            break
    if _utf8_size(out) <= max_bytes:
        return out

    # Phase 2: 按体积从大到小删嵌套 container
    container_keys = [k for k, v in out.items() if isinstance(v, (dict, list)) and k != "truncated"]
    container_keys.sort(key=lambda k: _utf8_size({k: out[k]}), reverse=True)
    for k in container_keys:
        del out[k]
        if _utf8_size(out) <= max_bytes:
            return out

    # Phase 3: 最小回退 {kind?, truncated:True}
    kind = payload.get("kind")
    if isinstance(kind, str):
        kind_short = kind if len(kind) <= 32 else kind[:32] + "…"
        candidate = {"kind": kind_short, "truncated": True}
        if _utf8_size(candidate) <= max_bytes:
            return candidate

    minimal = {"truncated": True}
    if _utf8_size(minimal) <= max_bytes:
        return minimal

    # 极端：max_bytes 极小到连 {"truncated": true} 都装不下
    return {}
