"""Context7 单次 Agent run 的参数最小化与 Library ID 来源约束。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CONTEXT7_RESOLVE_TOOL = "resolve-library-id"
CONTEXT7_QUERY_TOOL = "query-docs"

_LIBRARY_ID_TOKEN = (
    r"/[A-Za-z0-9][A-Za-z0-9._-]{0,79}"
    r"/[A-Za-z0-9][A-Za-z0-9._-]{0,119}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._:+-]{0,119})?"
)
_LIBRARY_ID_PATTERN = re.compile(rf"^{_LIBRARY_ID_TOKEN}$")
_RESOLVED_LIBRARY_LINE_PATTERN = re.compile(
    rf"(?m)^- Context7-compatible library ID:\s*(?P<library_id>{_LIBRARY_ID_TOKEN})\s*$"
)
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*[^\s,;]{4,}",
        re.IGNORECASE,
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:sk|ctx7sk|gh[pousr])[-_][A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dX](?!\d)", re.IGNORECASE),
    re.compile(
        r"https?://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
        r"[^/\s:]+\.(?:local|internal|intranet|corp))(?:[/:?#][^\s]*)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb|redis|jdbc)://\S+", re.IGNORECASE),
    re.compile(
        r"(?:内部|私有)\s*(?:函数|方法|接口|类|模块|表|字段)|"
        r"\b(?:internal|private)\s+(?:function|method|api|class|module|table|field)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:SELECT\b.+\bFROM|INSERT\s+INTO|UPDATE\b.+\bSET|DELETE\s+FROM|"
        r"CREATE\s+(?:TABLE|VIEW)|ALTER\s+TABLE|DROP\s+(?:TABLE|VIEW))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:def|class|function)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*(?:\(|:)", re.IGNORECASE),
    re.compile(r"(?:=>|::|[A-Za-z_$][A-Za-z0-9_$]*\s*\([^)]*\)\s*\{)"),
    re.compile(r"(?:/Users/|/home/|[A-Z]:\\Users\\)\S+", re.IGNORECASE),
    re.compile(
        r"\b(?=[A-Za-z0-9_-]{32,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])"
        r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{32,}\b"
    ),
)
_MAX_RESOLVED_LIBRARY_IDS = 32


def validate_context7_arguments(tool_name: str, arguments: Any) -> list[dict[str, str]]:
    """按已知敏感模式拒绝不适合发送给第三方文档检索服务的参数。"""

    if not isinstance(arguments, dict):
        return [{"field": "$", "code": "type"}]

    errors: list[dict[str, str]] = []
    if tool_name == CONTEXT7_RESOLVE_TOOL:
        _validate_minimized_text(arguments.get("libraryName"), field="libraryName", max_length=120, errors=errors)
        _validate_minimized_text(arguments.get("query"), field="query", max_length=500, errors=errors)
    elif tool_name == CONTEXT7_QUERY_TOOL:
        library_id = arguments.get("libraryId")
        if (
            isinstance(library_id, str)
            and 3 <= len(library_id) <= 200
            and not _LIBRARY_ID_PATTERN.fullmatch(library_id)
        ):
            errors.append({"field": "libraryId", "code": "format"})
        _validate_minimized_text(library_id, field="libraryId", max_length=200, errors=errors)
        _validate_minimized_text(arguments.get("query"), field="query", max_length=500, errors=errors)
    return list({(item["field"], item["code"]): item for item in errors}.values())[:8]


def _validate_minimized_text(
    value: Any,
    *,
    field: str,
    max_length: int,
    errors: list[dict[str, str]],
) -> None:
    if not isinstance(value, str):
        return
    if len(value) > max_length:
        errors.append({"field": field, "code": "max_length"})
        return
    if "\n" in value or "\r" in value:
        errors.append({"field": field, "code": "multiline"})
    if "```" in value:
        errors.append({"field": field, "code": "code_block"})
    if any(pattern.search(value) for pattern in _SENSITIVE_TEXT_PATTERNS):
        errors.append({"field": field, "code": "sensitive_content"})


@dataclass
class Context7RunLibraryRegistry:
    """只在当前 Agent run 内记录官方解析结果，不写日志或持久化。"""

    resolved_library_ids: set[str] = field(default_factory=set)

    def register_from_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        content = payload.get("content")
        if not isinstance(content, list):
            return
        for item in content:
            if len(self.resolved_library_ids) >= _MAX_RESOLVED_LIBRARY_IDS:
                return
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            for match in _RESOLVED_LIBRARY_LINE_PATTERN.finditer(text):
                self.resolved_library_ids.add(match.group("library_id"))
                if len(self.resolved_library_ids) >= _MAX_RESOLVED_LIBRARY_IDS:
                    return

    def contains(self, library_id: Any) -> bool:
        return isinstance(library_id, str) and library_id in self.resolved_library_ids
