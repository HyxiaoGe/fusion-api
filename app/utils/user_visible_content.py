"""用户可见文本净化。

模型协议继续保留真实工具标识；这里只处理 SSE、消息内容和历史恢复等用户可见副本。
"""

from __future__ import annotations

import re

_MCP_ALIAS_PREFIX = "mcp_"
_MCP_ALIAS_TOKEN_LENGTH = 43
_MCP_ALIAS_RE = re.compile(rf"{_MCP_ALIAS_PREFIX}[A-Za-z0-9_-]{{{_MCP_ALIAS_TOKEN_LENGTH}}}")
_MCP_ALIAS_PARTIAL_RE = re.compile(rf"{_MCP_ALIAS_PREFIX}[A-Za-z0-9_-]+$")

_INTERNAL_TOOL_LABELS = {
    "local_place_search": "地点搜索",
    "route_compare": "路线比较",
    "search_flights": "航班查询",
    "search_trains": "高铁查询",
    "url_read": "网页读取",
    "web_search": "联网搜索",
}
_INTERNAL_TOOL_NAMES = tuple(sorted(_INTERNAL_TOOL_LABELS, key=len, reverse=True))
_INTERNAL_TOOL_NAME_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(re.escape(name) for name in _INTERNAL_TOOL_NAMES)})(?![A-Za-z0-9_])"
)


def _pending_mcp_alias_start(text: str) -> int | None:
    max_suffix_length = min(len(text), len(_MCP_ALIAS_PREFIX) + _MCP_ALIAS_TOKEN_LENGTH - 1)
    for suffix_length in range(max_suffix_length, 0, -1):
        suffix = text[-suffix_length:]
        if _MCP_ALIAS_PREFIX.startswith(suffix):
            return len(text) - suffix_length
        if (
            suffix.startswith(_MCP_ALIAS_PREFIX)
            and len(suffix) < len(_MCP_ALIAS_PREFIX) + _MCP_ALIAS_TOKEN_LENGTH
            and all(char.isalnum() or char in {"_", "-"} for char in suffix[len(_MCP_ALIAS_PREFIX) :])
        ):
            return len(text) - suffix_length
    return None


def _pending_internal_tool_name(text: str) -> tuple[int, tuple[str, ...]] | None:
    earliest_start = max(0, len(text) - max(len(name) for name in _INTERNAL_TOOL_NAMES))
    for start in range(earliest_start, len(text)):
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            continue
        suffix = text[start:]
        # 完整名称位于 chunk 末尾时仍需等待右边界。否则先输出中文标签，
        # 下一 chunk 若继续补成更长标识，累计可见文本就会发生回退。
        matches = tuple(name for name in _INTERNAL_TOOL_NAMES if name.startswith(suffix))
        if matches:
            return start, matches
    return None


def _pending_trailing_space_start(text: str) -> int | None:
    """暂存 chunk 尾部空白，为下一 chunk 的工具名前后间距决策保留余地。"""

    trailing = re.search(r"[ \t]+$", text)
    return trailing.start() if trailing is not None else None


def _pending_tool_visible_start(text: str, tool_start: int) -> int:
    """半截英文工具名暂不输出；连同前导空格一起等待，保证后续中文替换仍是单调追加。"""

    visible_start = tool_start
    while visible_start > 0 and text[visible_start - 1] in {" ", "\t"}:
        visible_start -= 1
    return visible_start


def _replace_internal_tool_name(match: re.Match[str]) -> str:
    return _INTERNAL_TOOL_LABELS[match.group(0)]


def _normalize_tool_label_spacing(text: str) -> str:
    """内部英文标识两侧常带空格；替换成中文标签后去掉中英文混排残留空格。"""

    normalized = text
    for label in set(_INTERNAL_TOOL_LABELS.values()):
        normalized = re.sub(rf"(?<=[\u4e00-\u9fff])\s+{re.escape(label)}", label, normalized)
        normalized = re.sub(rf"{re.escape(label)}\s+(?=[\u4e00-\u9fff])", label, normalized)
    return normalized


def sanitize_internal_tool_names(
    text: str,
    *,
    final: bool = False,
    include_named_tools: bool = True,
) -> str:
    """把内部函数标识改写为产品化名称，并正确处理跨 chunk 的半截标识。"""

    visible_source = text
    pending_starts: list[int] = []
    pending_mcp_start = _pending_mcp_alias_start(text)
    if pending_mcp_start is not None:
        pending_starts.append(pending_mcp_start)
    pending_tool = _pending_internal_tool_name(text) if include_named_tools else None
    if pending_tool is not None:
        pending_starts.append(_pending_tool_visible_start(text, pending_tool[0]))
    pending_space_start = _pending_trailing_space_start(text) if include_named_tools else None
    if pending_space_start is not None:
        pending_starts.append(pending_space_start)

    if not final and pending_starts:
        visible_source = text[: min(pending_starts)]
    elif final and pending_tool is not None:
        start, matches = pending_tool
        suffix = text[start:]
        exact_label = _INTERNAL_TOOL_LABELS.get(suffix)
        if exact_label is not None:
            visible_source = f"{text[:start]}{exact_label}"
        elif len(matches) == 1 and len(suffix) >= 4:
            visible_source = f"{text[:start]}{_INTERNAL_TOOL_LABELS[matches[0]]}"

    sanitized = _MCP_ALIAS_RE.sub("外部工具", visible_source)
    if include_named_tools:
        sanitized = _INTERNAL_TOOL_NAME_RE.sub(_replace_internal_tool_name, sanitized)
        sanitized = _normalize_tool_label_spacing(sanitized)
    if final:
        sanitized = _MCP_ALIAS_PARTIAL_RE.sub("外部工具", sanitized)
    return sanitized
