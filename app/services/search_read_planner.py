"""Search / Read Planner v1.1。

本模块负责把搜索结果排序计划转成可控的深读建议，不直接执行 url_read。
"""

from __future__ import annotations

import re

from app.services.source_candidate_ranker import (
    SearchResultForRanking,
    SourceSelectionPlan,
    format_source_selection_guidance,
    rank_search_sources,
)

QUICK_FACT_READ_LIMIT = 1
STANDARD_READ_LIMIT = 2
DEEP_READ_LIMIT = 3

_QUICK_BUDGET_NAMES = {"quick_fact", "quick_fact_followup"}
_FRESHNESS_BUDGET_NAMES = {"freshness", "freshness_followup"}
_DEEP_BUDGET_NAMES = {
    "comparison",
    "comparison_followup",
    "deep_research",
    "deep_research_followup",
    "official_source",
    "official_source_followup",
}
_DEEP_INTENTS = {"comparison", "deep_research", "official_source"}
_READ_REQUIRED_BUDGET_REASON = {
    "quick_fact": "quick_fact_requires_verification",
    "quick_fact_followup": "quick_fact_requires_verification",
    "freshness": "freshness_requires_verification",
    "freshness_followup": "freshness_requires_verification",
    "official_source": "official_source_requires_verification",
    "official_source_followup": "official_source_requires_verification",
    "comparison": "comparison_requires_verification",
    "comparison_followup": "comparison_requires_verification",
    "deep_research": "deep_research_requires_verification",
    "deep_research_followup": "deep_research_requires_verification",
}
_READ_REQUIRED_INTENT_REASON = {
    "quick_fact": "quick_fact_requires_verification",
    "freshness": "freshness_requires_verification",
    "official_source": "official_source_requires_verification",
    "comparison": "comparison_requires_verification",
    "deep_research": "deep_research_requires_verification",
}
_READ_REQUIRED_REASON_PRIORITY = (
    "deep_research_requires_verification",
    "official_source_requires_verification",
    "comparison_requires_verification",
    "freshness_requires_verification",
    "quick_fact_requires_verification",
    "current_fact_query_requires_verification",
)
_CURRENT_FACT_QUERY_RE = re.compile(
    r"("
    r"最新|最近|今天|今日|目前|当前|实时|现在|价格|估值|上市|发布|公告|上线|下线|政策|法规|财报|"
    r"latest|recent|today|current|now|price|valuation|ipo|release|launch|announcement|policy|earnings"
    r")",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)20\d{2}(?!\d)")


def build_search_read_plan(search_results: list[SearchResultForRanking]) -> SourceSelectionPlan:
    """构造本轮搜索后的读源建议计划。"""

    recommended_limit = _recommended_read_limit(search_results)
    read_required, reason = _read_required(search_results)
    return rank_search_sources(
        search_results,
        max_recommended=recommended_limit,
        read_required=read_required,
        minimum_required_reads=1 if read_required else 0,
        read_required_reason=reason,
    )


def format_search_read_plan_guidance(plan: SourceSelectionPlan) -> str:
    """生成给 LLM 的读源选择建议。"""

    return format_source_selection_guidance(plan)


def _recommended_read_limit(search_results: list[SearchResultForRanking]) -> int:
    if not search_results:
        return 0

    intents = {result.intent for result in search_results if result.intent}
    budgets = {result.search_budget for result in search_results if result.search_budget}

    if intents & _DEEP_INTENTS or budgets & _DEEP_BUDGET_NAMES:
        return DEEP_READ_LIMIT
    if budgets & _FRESHNESS_BUDGET_NAMES:
        return STANDARD_READ_LIMIT
    if budgets and budgets <= _QUICK_BUDGET_NAMES:
        return QUICK_FACT_READ_LIMIT
    if intents == {"quick_fact"}:
        return QUICK_FACT_READ_LIMIT
    return STANDARD_READ_LIMIT


def _read_required(search_results: list[SearchResultForRanking]) -> tuple[bool, str]:
    """判断本轮搜索是否需要至少读取一个关键来源后再下事实结论。"""

    reasons: set[str] = set()

    for result in search_results:
        budget_reason = _READ_REQUIRED_BUDGET_REASON.get(result.search_budget or "")
        if budget_reason:
            reasons.add(budget_reason)

    for result in search_results:
        intent_reason = _READ_REQUIRED_INTENT_REASON.get(result.intent or "")
        if intent_reason:
            reasons.add(intent_reason)

    for result in search_results:
        query = result.query or ""
        if _CURRENT_FACT_QUERY_RE.search(query) or _YEAR_RE.search(query):
            reasons.add("current_fact_query_requires_verification")

    for reason in _READ_REQUIRED_REASON_PRIORITY:
        if reason in reasons:
            return True, reason

    return False, ""
