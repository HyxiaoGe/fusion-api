"""联网搜索预算策略。

本模块只负责把搜索意图映射为内部预算，避免 LLM 直接决定 provider count。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


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


def normalize_search_intent(value) -> str | None:
    if not isinstance(value, str):
        return None
    intent = value.strip().lower()
    if intent in SUPPORTED_SEARCH_INTENTS:
        return intent
    return None


def resolve_search_intent(value, query: str | None = None) -> str | None:
    """优先使用模型显式 intent，否则从 query 里做保守推断。"""

    explicit_intent = normalize_search_intent(value)
    if explicit_intent:
        return explicit_intent
    return infer_search_intent(query or "")


def infer_search_intent(query: str) -> str | None:
    normalized = _normalize_query_text(query)
    if not normalized:
        return None

    if _contains_any(normalized, _COMPARISON_KEYWORDS):
        return "comparison"
    if _contains_any(normalized, _OFFICIAL_SOURCE_KEYWORDS):
        return "official_source"
    if _contains_any(normalized, _DEEP_RESEARCH_KEYWORDS):
        return "deep_research"
    if _contains_any(normalized, _FRESHNESS_KEYWORDS) or _CURRENT_YEAR_RE.search(normalized):
        return "freshness"
    if _contains_any(normalized, _QUICK_FACT_KEYWORDS) and len(_query_tokens(normalized)) <= 8:
        return "quick_fact"
    return None


def derive_search_budget(
    intent: str | None,
    *,
    query: str | None = None,
    previous_queries: Sequence[str] = (),
    previous_intents: Sequence[str | None] = (),
) -> SearchBudget:
    base_budget = SEARCH_BUDGETS_BY_INTENT.get(intent, STANDARD_SEARCH_BUDGET) if intent else STANDARD_SEARCH_BUDGET
    if _is_similar_followup_query(
        query or "",
        intent,
        previous_queries=previous_queries,
        previous_intents=previous_intents,
    ):
        return FOLLOWUP_SEARCH_BUDGETS_BY_NAME.get(base_budget.name, base_budget)
    return base_budget


def is_duplicate_search_query(
    query: str,
    intent: str | None,
    *,
    previous_queries: Sequence[str],
    previous_intents: Sequence[str | None],
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
        if _query_similarity(query, previous_query) >= _DUPLICATE_SEARCH_THRESHOLD:
            return True
    return False


def _is_similar_followup_query(
    query: str,
    intent: str | None,
    *,
    previous_queries: Sequence[str],
    previous_intents: Sequence[str | None],
) -> bool:
    if not query or not previous_queries:
        return False

    padded_intents: list[str | None] = list(previous_intents)
    if len(padded_intents) < len(previous_queries):
        padded_intents.extend([None] * (len(previous_queries) - len(padded_intents)))

    for previous_query, previous_intent in zip(previous_queries, padded_intents):
        if previous_intent != intent:
            continue
        if _query_similarity(query, previous_query) >= _SIMILAR_FOLLOWUP_THRESHOLD:
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


def _normalize_query_text(query: str) -> str:
    return str(query or "").strip().lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)
