"""单轮联网工具预算与参数归一化。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.services.search_budget import (
    SearchBudgetDecision,
    derive_search_budget,
    is_duplicate_search_query,
    resolve_search_intent,
)
from app.services.tool_handlers.base import ToolResult

MAX_SEARCH_CALLS = 4
DEFAULT_PLANNED_SEARCH_CALLS = 2
DEEP_RESEARCH_PLANNED_SEARCH_CALLS = 3
MAX_URL_READ_CALLS = 5
MAX_DOMAINS = 5
MIN_RECENCY_DAYS = 1
MAX_RECENCY_DAYS = 365
DEEP_RESEARCH_INTENT = "deep_research"

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


@dataclass
class NetworkToolBudget:
    """限制一次 assistant run 内的联网工具调用次数。"""

    web_search_calls: int = 0
    url_read_calls: int = 0
    web_search_queries: list[str] = field(default_factory=list)
    web_search_intents: list[str | None] = field(default_factory=list)

    def prepare_web_search_args(self, args: dict) -> tuple[dict, ToolResult | None]:
        normalized = dict(args or {})

        query = str(normalized.get("query") or "")
        intent = resolve_search_intent(normalized.get("intent"), query)
        if intent:
            normalized["intent"] = intent
        else:
            normalized.pop("intent", None)

        domains = _normalize_domains(normalized.get("domains"))
        if domains:
            normalized["domains"] = domains
        else:
            normalized.pop("domains", None)

        planned_search_limit = _planned_search_call_limit(intent, self.web_search_intents)
        previous_query_count = len(self.web_search_queries)
        if not domains and is_duplicate_search_query(
            query,
            intent,
            previous_queries=self.web_search_queries,
            previous_intents=self.web_search_intents,
        ):
            normalized["count"] = 0
            normalized["context_source_limit"] = 0
            normalized["search_budget"] = "duplicate_skipped"
            decision = _search_budget_decision(
                query=query,
                intent=intent,
                action="skip_duplicate",
                budget_name="duplicate_skipped",
                requested_count=0,
                context_source_limit=0,
                reason_code="duplicate_query",
                previous_query_count=previous_query_count,
                planned_search_limit=planned_search_limit,
            )
            normalized["budget_decision"] = decision
            return normalized, ToolResult(
                status="degraded",
                error_message="重复搜索已跳过",
                data={
                    "query": query,
                    "sources": [],
                    "result_count": 0,
                    "requested_count": 0,
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": 0,
                    "search_budget": "duplicate_skipped",
                    "intent": intent,
                    "domains": domains,
                    "recency_days": normalized.get("recency_days"),
                    "budget_limited": False,
                    "duplicate_search_skipped": True,
                    "budget_decision": decision,
                },
            )

        search_budget = derive_search_budget(
            intent,
            query=query,
            previous_queries=self.web_search_queries,
            previous_intents=self.web_search_intents,
        )
        normalized["count"] = search_budget.requested_count
        normalized["context_source_limit"] = search_budget.context_source_limit
        normalized["search_budget"] = search_budget.name
        decision = _search_budget_decision(
            query=query,
            intent=intent,
            action=_allowed_search_action(search_budget.name, previous_query_count),
            budget_name=search_budget.name,
            requested_count=search_budget.requested_count,
            context_source_limit=search_budget.context_source_limit,
            reason_code=_allowed_search_reason_code(search_budget.name, previous_query_count),
            previous_query_count=previous_query_count,
            planned_search_limit=planned_search_limit,
        )
        normalized["budget_decision"] = decision

        if normalized.get("recency_days") is not None:
            normalized["recency_days"] = _clamp_int(
                normalized.get("recency_days"),
                MIN_RECENCY_DAYS,
                MIN_RECENCY_DAYS,
                MAX_RECENCY_DAYS,
            )

        if self.web_search_calls >= MAX_SEARCH_CALLS:
            decision = _search_budget_decision(
                query=str(normalized.get("query") or ""),
                intent=normalized.get("intent"),
                action="limit_budget",
                budget_name=str(normalized.get("search_budget") or search_budget.name),
                requested_count=normalized.get("count", search_budget.requested_count),
                context_source_limit=normalized.get("context_source_limit", search_budget.context_source_limit),
                reason_code="hard_search_limit_reached",
                previous_query_count=previous_query_count,
                planned_search_limit=planned_search_limit,
            )
            normalized["budget_decision"] = decision
            return normalized, ToolResult(
                status="degraded",
                error_message="web_search 已达到本轮联网预算",
                data={
                    "query": normalized.get("query", ""),
                    "sources": [],
                    "result_count": 0,
                    "requested_count": normalized.get("count", search_budget.requested_count),
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": normalized.get(
                        "context_source_limit",
                        search_budget.context_source_limit,
                    ),
                    "search_budget": normalized.get("search_budget", search_budget.name),
                    "intent": normalized.get("intent"),
                    "domains": normalized.get("domains", []),
                    "recency_days": normalized.get("recency_days"),
                    "budget_limited": True,
                    "budget_decision": decision,
                },
            )

        if self.web_search_calls >= planned_search_limit:
            normalized["count"] = 0
            normalized["context_source_limit"] = 0
            normalized["search_budget"] = "planner_limited"
            decision = _search_budget_decision(
                query=str(normalized.get("query") or ""),
                intent=normalized.get("intent"),
                action="limit_planner",
                budget_name="planner_limited",
                requested_count=0,
                context_source_limit=0,
                reason_code="planned_search_limit_reached",
                previous_query_count=previous_query_count,
                planned_search_limit=planned_search_limit,
            )
            normalized["budget_decision"] = decision
            return normalized, ToolResult(
                status="degraded",
                error_message="搜索计划已收敛",
                data={
                    "query": normalized.get("query", ""),
                    "sources": [],
                    "result_count": 0,
                    "requested_count": 0,
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": 0,
                    "search_budget": "planner_limited",
                    "intent": normalized.get("intent"),
                    "domains": normalized.get("domains", []),
                    "recency_days": normalized.get("recency_days"),
                    "budget_limited": False,
                    "search_plan_limited": True,
                    "planned_search_limit": planned_search_limit,
                    "executed_search_count": self.web_search_calls,
                    "budget_decision": decision,
                },
            )

        self.web_search_calls += 1
        self.web_search_queries.append(query)
        self.web_search_intents.append(intent)
        return normalized, None

    def prepare_url_read_args(self, args: dict) -> tuple[dict, ToolResult | None]:
        normalized = dict(args or {})
        if self.url_read_calls >= MAX_URL_READ_CALLS:
            return normalized, ToolResult(
                status="degraded",
                error_message="url_read 已达到本轮联网预算",
                data={
                    "url": normalized.get("url", ""),
                    "reason": normalized.get("reason"),
                    "budget_limited": True,
                },
            )

        self.url_read_calls += 1
        return normalized, None


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _normalize_domains(value) -> list[str]:
    if not isinstance(value, list):
        return []

    domains: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        domain = _extract_domain(item)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        domains.append(domain)
        if len(domains) >= MAX_DOMAINS:
            break
    return domains


def _extract_domain(value: str) -> str | None:
    raw = value.strip().lower()
    if not raw:
        return None
    if any(char in raw for char in ("://", "/", "?", "#", ":", "*")):
        return None
    if raw.startswith("www."):
        raw = raw[4:]
    if _DOMAIN_RE.match(raw):
        return raw
    return None


def _planned_search_call_limit(intent: str | None, previous_intents: list[str | None]) -> int:
    """控制真实 provider 搜索轮次，避免 LLM 机械扩写 query。"""

    if intent == DEEP_RESEARCH_INTENT or DEEP_RESEARCH_INTENT in previous_intents:
        return DEEP_RESEARCH_PLANNED_SEARCH_CALLS
    return DEFAULT_PLANNED_SEARCH_CALLS


def _allowed_search_action(budget_name: str, previous_query_count: int) -> str:
    if budget_name.endswith("_followup"):
        return "narrow_followup"
    return "execute"


def _allowed_search_reason_code(budget_name: str, previous_query_count: int) -> str:
    if budget_name.endswith("_followup"):
        return "similar_followup"
    if previous_query_count > 0:
        return "complementary_search"
    return "initial_search"


def _search_budget_decision(
    *,
    query: str,
    intent: str | None,
    action: str,
    budget_name: str,
    requested_count: int,
    context_source_limit: int,
    reason_code: str,
    previous_query_count: int,
    planned_search_limit: int,
) -> dict:
    return asdict(
        SearchBudgetDecision(
            query=query,
            intent=intent,
            action=action,
            budget_name=budget_name,
            requested_count=requested_count,
            context_source_limit=context_source_limit,
            reason_code=reason_code,
            previous_query_count=previous_query_count,
            planned_search_limit=planned_search_limit,
        )
    )
