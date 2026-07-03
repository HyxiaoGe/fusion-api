"""联网搜索预算策略。

本模块只负责把搜索意图映射为内部预算，避免 LLM 直接决定 provider count。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.services.agent_strategy_config import get_agent_strategy_config


@dataclass(frozen=True)
class SearchBudget:
    name: str
    requested_count: int
    context_source_limit: int


@dataclass(frozen=True)
class SearchBudgetDecision:
    query: str
    intent: str | None
    action: str
    budget_name: str
    requested_count: int
    context_source_limit: int
    reason_code: str
    previous_query_count: int
    planned_search_limit: int


STANDARD_SEARCH_BUDGET = SearchBudget(name="standard", requested_count=5, context_source_limit=5)

SEARCH_BUDGETS_BY_INTENT = {
    "quick_fact": SearchBudget(name="quick_fact", requested_count=3, context_source_limit=3),
    "freshness": SearchBudget(name="freshness", requested_count=5, context_source_limit=5),
    "comparison": SearchBudget(name="comparison", requested_count=8, context_source_limit=6),
    "deep_research": SearchBudget(name="deep_research", requested_count=10, context_source_limit=8),
    "official_source": SearchBudget(name="official_source", requested_count=5, context_source_limit=4),
}

FOLLOWUP_SEARCH_BUDGETS_BY_NAME = {
    "standard": SearchBudget(name="standard_followup", requested_count=3, context_source_limit=3),
    "quick_fact": SearchBudget(name="quick_fact_followup", requested_count=3, context_source_limit=3),
    "freshness": SearchBudget(name="freshness_followup", requested_count=3, context_source_limit=3),
    "official_source": SearchBudget(name="official_source_followup", requested_count=3, context_source_limit=3),
    "comparison": SearchBudget(name="comparison_followup", requested_count=5, context_source_limit=4),
    "deep_research": SearchBudget(name="deep_research_followup", requested_count=5, context_source_limit=5),
}

SUPPORTED_SEARCH_INTENTS = set(SEARCH_BUDGETS_BY_INTENT)

_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]+")
_CURRENT_YEAR_RE = re.compile(r"(?<!\d)20\d{2}(?!\d)")

_COMPARISON_KEYWORDS = (
    "权威媒体",
    "媒体",
    "报道",
    "对照",
    "对比",
    "比较",
    "compare",
    "comparison",
    "versus",
    "media",
    "reuters",
    "bloomberg",
    "techcrunch",
    "axios",
    "bbc",
    "nytimes",
    "new york times",
    "wall street journal",
    "wsj",
    "the verge",
)
_OFFICIAL_SOURCE_KEYWORDS = (
    "官方",
    "官网",
    "公告",
    "发布",
    "official",
    "announcement",
    "announces",
    "announced",
    "press release",
    "release notes",
    "official blog",
    "openai.com",
)
_DEEP_RESEARCH_KEYWORDS = (
    "深入",
    "调研",
    "研究",
    "论文",
    "白皮书",
    "技术报告",
    "technical report",
    "system card",
    "research",
    "paper",
    "whitepaper",
)
_FRESHNESS_KEYWORDS = (
    "最新",
    "今天",
    "今日",
    "目前",
    "实时",
    "current",
    "latest",
    "today",
    "recent",
    "new",
)
_QUICK_FACT_KEYWORDS = (
    "是谁",
    "是什么",
    "多少",
    "价格",
    "上市日期",
    "who is",
    "what is",
    "when did",
    "how much",
)

_SIMILAR_FOLLOWUP_THRESHOLD = 0.55
_DUPLICATE_SEARCH_THRESHOLD = 0.82


def normalize_search_intent(value, *, strategy_config: dict | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    intent = value.strip().lower()
    if intent in _supported_search_intents(strategy_config):
        return intent
    return None


def resolve_search_intent(value, query: str | None = None, *, strategy_config: dict | None = None) -> str | None:
    """优先使用模型显式 intent，否则从 query 里做保守推断。"""

    explicit_intent = normalize_search_intent(value, strategy_config=strategy_config)
    if explicit_intent:
        return explicit_intent
    return infer_search_intent(query or "", strategy_config=strategy_config)


def infer_search_intent(query: str, *, strategy_config: dict | None = None) -> str | None:
    normalized = _normalize_query_text(query)
    if not normalized:
        return None

    keywords = _search_config(strategy_config).get("intent_keywords", {})
    comparison_keywords = tuple(keywords.get("comparison") or _COMPARISON_KEYWORDS)
    official_source_keywords = tuple(keywords.get("official_source") or _OFFICIAL_SOURCE_KEYWORDS)
    deep_research_keywords = tuple(keywords.get("deep_research") or _DEEP_RESEARCH_KEYWORDS)
    freshness_keywords = tuple(keywords.get("freshness") or _FRESHNESS_KEYWORDS)
    quick_fact_keywords = tuple(keywords.get("quick_fact") or _QUICK_FACT_KEYWORDS)

    if _contains_any(normalized, comparison_keywords):
        return "comparison"
    if _contains_any(normalized, official_source_keywords):
        return "official_source"
    if _contains_any(normalized, deep_research_keywords):
        return "deep_research"
    if _contains_any(normalized, freshness_keywords) or _CURRENT_YEAR_RE.search(normalized):
        return "freshness"
    if _contains_any(normalized, quick_fact_keywords) and len(_query_tokens(normalized)) <= 8:
        return "quick_fact"
    return None


def derive_search_budget(
    intent: str | None,
    *,
    query: str | None = None,
    previous_queries: Sequence[str] = (),
    previous_intents: Sequence[str | None] = (),
    strategy_config: dict | None = None,
) -> SearchBudget:
    search_config = _search_config(strategy_config)
    standard_budget = _budget_from_config(search_config.get("standard_budget"), STANDARD_SEARCH_BUDGET)
    budgets_by_intent = {
        name: _budget_from_config(payload, SEARCH_BUDGETS_BY_INTENT.get(name, standard_budget))
        for name, payload in (search_config.get("budgets_by_intent") or {}).items()
    }
    followup_budgets_by_name = {
        name: _budget_from_config(payload, FOLLOWUP_SEARCH_BUDGETS_BY_NAME.get(name, standard_budget))
        for name, payload in (search_config.get("followup_budgets_by_name") or {}).items()
    }
    base_budget = budgets_by_intent.get(intent, standard_budget) if intent else standard_budget
    if _is_similar_followup_query(
        query or "",
        intent,
        previous_queries=previous_queries,
        previous_intents=previous_intents,
        strategy_config=strategy_config,
    ):
        return followup_budgets_by_name.get(base_budget.name, base_budget)
    return base_budget


def is_duplicate_search_query(
    query: str,
    intent: str | None,
    *,
    previous_queries: Sequence[str],
    previous_intents: Sequence[str | None],
    strategy_config: dict | None = None,
) -> bool:
    """判断本次搜索是否与已执行搜索重复到应跳过真实 provider 调用。"""

    normalized_query = _normalize_query_text(query)
    if not normalized_query or not previous_queries:
        return False

    padded_intents: list[str | None] = list(previous_intents)
    if len(padded_intents) < len(previous_queries):
        padded_intents.extend([None] * (len(previous_queries) - len(padded_intents)))

    for previous_query, previous_intent in zip(previous_queries, padded_intents):
        normalized_previous = _normalize_query_text(previous_query)
        if not normalized_previous:
            continue
        if normalized_query == normalized_previous:
            return True
        if previous_intent != intent:
            continue
        threshold = _threshold("duplicate_search", _DUPLICATE_SEARCH_THRESHOLD, strategy_config)
        if _query_similarity(query, previous_query) >= threshold:
            return True
    return False


def _is_similar_followup_query(
    query: str,
    intent: str | None,
    *,
    previous_queries: Sequence[str],
    previous_intents: Sequence[str | None],
    strategy_config: dict | None = None,
) -> bool:
    if not query or not previous_queries:
        return False

    padded_intents: list[str | None] = list(previous_intents)
    if len(padded_intents) < len(previous_queries):
        padded_intents.extend([None] * (len(previous_queries) - len(padded_intents)))

    for previous_query, previous_intent in zip(previous_queries, padded_intents):
        if previous_intent != intent:
            continue
        threshold = _threshold("similar_followup", _SIMILAR_FOLLOWUP_THRESHOLD, strategy_config)
        if _query_similarity(query, previous_query) >= threshold:
            return True
    return False


def _query_similarity(left: str, right: str) -> float:
    left_tokens = _query_tokens(left)
    right_tokens = _query_tokens(right)
    if not left_tokens or not right_tokens:
        return 0
    shared_count = len(left_tokens & right_tokens)
    return shared_count / min(len(left_tokens), len(right_tokens))


def _query_tokens(query: str) -> set[str]:
    normalized = _normalize_query_text(query)
    tokens = set(_LATIN_TOKEN_RE.findall(normalized))
    for sequence in _CJK_SEQUENCE_RE.findall(normalized):
        if len(sequence) == 1:
            tokens.add(sequence)
            continue
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _search_config(strategy_config: dict | None = None) -> dict:
    if strategy_config is None:
        strategy_config, _meta = get_agent_strategy_config()
    return strategy_config.get("search") or {}


def _supported_search_intents(strategy_config: dict | None = None) -> set[str]:
    search_config = _search_config(strategy_config)
    configured = search_config.get("budgets_by_intent")
    if isinstance(configured, dict) and configured:
        return set(configured)
    return SUPPORTED_SEARCH_INTENTS


def _budget_from_config(value, fallback: SearchBudget) -> SearchBudget:
    if not isinstance(value, dict):
        return fallback
    try:
        return SearchBudget(
            name=str(value.get("name") or fallback.name),
            requested_count=max(0, int(value.get("requested_count", fallback.requested_count))),
            context_source_limit=max(0, int(value.get("context_source_limit", fallback.context_source_limit))),
        )
    except (TypeError, ValueError):
        return fallback


def _threshold(name: str, fallback: float, strategy_config: dict | None = None) -> float:
    try:
        return float((_search_config(strategy_config).get("thresholds") or {}).get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _normalize_query_text(query: str) -> str:
    return str(query or "").strip().lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)
