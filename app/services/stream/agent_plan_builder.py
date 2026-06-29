"""Agent 长任务计划构建器。"""

from __future__ import annotations

import re

from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.network_budget import (
    MAX_SEARCH_CALLS,
    MAX_SEARCH_COUNT,
    MAX_URL_READ_CALLS,
    MIN_SEARCH_COUNT,
)

_WHITESPACE_RE = re.compile(r"\s+")
_SEARCH_PREFIXES = (
    "请帮我查一下",
    "帮我查一下",
    "请查一下",
    "查一下",
    "帮我搜索一下",
    "请搜索一下",
    "搜索一下",
)
_MAX_FOCUS_LENGTH = 36


def build_long_task_plan_items(
    *,
    original_message: str,
    tools: list[str],
    limits: AgentLoopLimits,
) -> list[dict]:
    """构建用户可见的 deterministic 执行计划。"""
    _ = limits
    focus = _build_focus(original_message, strip_search_prefix=bool(tools))
    if tools:
        return _network_plan_items(focus=focus, tools=tools)
    return _direct_answer_plan_items(focus=focus)


def _network_plan_items(*, focus: str, tools: list[str]) -> list[dict]:
    return [
        {
            "id": "understand",
            "title": "制定执行计划",
            "status": "running",
            "kind": "reasoning",
            "summary": f"围绕「{focus}」判断资料需求和回答路径",
            "tool_names": [],
            "evidence_item_ids": [],
        },
        {
            "id": "search",
            "title": f"搜索：{focus}",
            "status": "pending",
            "kind": "search",
            "summary": _search_budget_summary(),
            "tool_names": tools,
            "evidence_item_ids": [],
        },
        {
            "id": "read",
            "title": "筛选关键来源",
            "status": "pending",
            "kind": "read",
            "summary": _read_budget_summary(),
            "tool_names": tools,
            "evidence_item_ids": [],
        },
        {
            "id": "answer",
            "title": "整理回答",
            "status": "pending",
            "kind": "answer",
            "summary": "基于可用依据给出结论、推荐和不确定性",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    ]


def _direct_answer_plan_items(*, focus: str) -> list[dict]:
    return [
        {
            "id": "understand",
            "title": "制定执行计划",
            "status": "running",
            "kind": "reasoning",
            "summary": f"确认「{focus}」的目标和回答结构",
            "tool_names": [],
            "evidence_item_ids": [],
        },
        {
            "id": "answer",
            "title": "整理回答",
            "status": "pending",
            "kind": "answer",
            "summary": "基于已有上下文直接回答，不使用联网工具",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    ]


def _build_focus(original_message: str, *, strip_search_prefix: bool) -> str:
    normalized = _WHITESPACE_RE.sub(" ", original_message).strip(" ，。！？!?")
    if strip_search_prefix:
        normalized = _strip_prefix(normalized)
    if not normalized:
        return "当前问题"
    if len(normalized) <= _MAX_FOCUS_LENGTH:
        return normalized
    return f"{normalized[:_MAX_FOCUS_LENGTH - 1]}…"


def _strip_prefix(value: str) -> str:
    for prefix in _SEARCH_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix):].strip(" ，。！？!?")
    return value


def _search_budget_summary() -> str:
    return f"工具：联网搜索；预算：最多 {MAX_SEARCH_CALLS} 次搜索，每次 {MIN_SEARCH_COUNT}-{MAX_SEARCH_COUNT} 条结果"


def _read_budget_summary() -> str:
    return f"必要时读取网页核验；预算：最多 {MAX_URL_READ_CALLS} 个网页"
