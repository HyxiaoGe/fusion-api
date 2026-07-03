"""Search / Read Planner v1.1。

本模块负责把搜索结果排序计划转成可控的深读建议，不直接执行 url_read。
"""

from __future__ import annotations

import re

from app.services.agent_strategy_config import get_agent_strategy_config
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

    strategy_config, _meta = get_agent_strategy_config()
    read_planner_config = _read_planner_config(strategy_config)
    recommended_limit = _recommended_read_limit(search_results, read_planner_config=read_planner_config)
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


def _recommended_read_limit(search_results: list[SearchResultForRanking], *, read_planner_config: dict) -> int:
    if not search_results:
        return 0

    intents = {result.intent for result in search_results if result.intent}
    budgets = {result.search_budget for result in search_results if result.search_budget}
    read_limits = read_planner_config.get("read_limits") or {}
    quick_limit = _read_limit(read_limits, "quick_fact", QUICK_FACT_READ_LIMIT)
    standard_limit = _read_limit(read_limits, "standard", STANDARD_READ_LIMIT)
    deep_limit = _read_limit(read_limits, "deep", DEEP_READ_LIMIT)
    quick_budget_names = set(read_planner_config.get("quick_budget_names") or _QUICK_BUDGET_NAMES)
    freshness_budget_names = set(read_planner_config.get("freshness_budget_names") or _FRESHNESS_BUDGET_NAMES)
    deep_budget_names = set(read_planner_config.get("deep_budget_names") or _DEEP_BUDGET_NAMES)
    deep_intents = set(read_planner_config.get("deep_intents") or _DEEP_INTENTS)

    if intents & deep_intents or budgets & deep_budget_names:
        return deep_limit
    if budgets & freshness_budget_names:
        return standard_limit
    if budgets and budgets <= quick_budget_names:
        return quick_limit
    if intents == {"quick_fact"}:
        return quick_limit
    return standard_limit


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


def _read_planner_config(strategy_config: dict | None) -> dict:
    return (strategy_config or {}).get("read_planner") or {}


def _read_limit(read_limits: dict, key: str, fallback: int) -> int:
    try:
        return max(0, int(read_limits.get(key, fallback)))
    except (TypeError, ValueError):
        return fallback
