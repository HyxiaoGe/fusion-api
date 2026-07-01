"""Search / Read Planner v1.1。

本模块负责把搜索结果排序计划转成可控的深读建议，不直接执行 url_read。
"""

from __future__ import annotations

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
_DEEP_BUDGET_NAMES = {
    "comparison",
    "comparison_followup",
    "deep_research",
    "deep_research_followup",
    "official_source",
    "official_source_followup",
}
_DEEP_INTENTS = {"comparison", "deep_research", "official_source"}


def build_search_read_plan(search_results: list[SearchResultForRanking]) -> SourceSelectionPlan:
    """构造本轮搜索后的读源建议计划。"""

    recommended_limit = _recommended_read_limit(search_results)
    return rank_search_sources(search_results, max_recommended=recommended_limit)


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
    if budgets and budgets <= _QUICK_BUDGET_NAMES:
        return QUICK_FACT_READ_LIMIT
    if intents == {"quick_fact"}:
        return QUICK_FACT_READ_LIMIT
    return STANDARD_READ_LIMIT
