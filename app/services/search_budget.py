"""联网搜索预算策略。

本模块只负责把搜索意图映射为内部预算，避免 LLM 直接决定 provider count。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    name: str
    requested_count: int
    context_source_limit: int


STANDARD_SEARCH_BUDGET = SearchBudget(name="standard", requested_count=5, context_source_limit=5)

SEARCH_BUDGETS_BY_INTENT = {
    "quick_fact": SearchBudget(name="quick_fact", requested_count=3, context_source_limit=3),
    "freshness": SearchBudget(name="freshness", requested_count=5, context_source_limit=5),
    "comparison": SearchBudget(name="comparison", requested_count=8, context_source_limit=6),
    "deep_research": SearchBudget(name="deep_research", requested_count=10, context_source_limit=8),
    "official_source": SearchBudget(name="official_source", requested_count=5, context_source_limit=4),
}


def derive_search_budget(intent: str | None) -> SearchBudget:
    if not intent:
        return STANDARD_SEARCH_BUDGET
    return SEARCH_BUDGETS_BY_INTENT.get(intent, STANDARD_SEARCH_BUDGET)
