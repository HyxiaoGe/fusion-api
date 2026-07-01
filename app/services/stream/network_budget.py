"""单轮联网工具预算与参数归一化。"""

from __future__ import annotations

import json
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
REPAIR_SEARCH_COUNT = 3
REPAIR_CONTEXT_SOURCE_LIMIT = 3
WEAK_SEARCH_RESULT_THRESHOLD = 2
MIN_RECENCY_DAYS = 1
MAX_RECENCY_DAYS = 365
DEEP_RESEARCH_INTENT = "deep_research"
READ_ALTERNATIVE_ACTIONS = {"recommend_read"}
PROVIDER_SEARCH_ACTIONS = {"execute", "narrow_followup", "repair_search"}

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


@dataclass
class NetworkToolBudget:
    """限制一次 assistant run 内的联网工具调用次数。"""

    web_search_calls: int = 0
    url_read_calls: int = 0
    web_search_queries: list[str] = field(default_factory=list)
    web_search_intents: list[str | None] = field(default_factory=list)
    repair_search_used: bool = False
    pending_search_repair_reason_code: str | None = None
    read_failure_pending: bool = False
    candidate_read_urls: set[str] = field(default_factory=set)
    attempted_read_urls: set[str] = field(default_factory=set)

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

        repair_reason_code = self._consume_search_repair_reason_code()

        if not repair_reason_code and not domains and self._has_unread_read_alternatives():
            normalized["count"] = 0
            normalized["context_source_limit"] = 0
            normalized["search_budget"] = "read_alternative_redirect"
            decision = _search_budget_decision(
                query=query,
                intent=intent,
                action="redirect_to_read_alternative",
                budget_name="read_alternative_redirect",
                requested_count=0,
                context_source_limit=0,
                reason_code="read_alternatives_available",
                previous_query_count=previous_query_count,
                planned_search_limit=planned_search_limit,
            )
            normalized["budget_decision"] = decision
            return normalized, ToolResult(
                status="degraded",
                error_message="已有未读取候选来源，已暂停继续搜索",
                data={
                    "query": query,
                    "sources": [],
                    "result_count": 0,
                    "requested_count": 0,
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": 0,
                    "search_budget": "read_alternative_redirect",
                    "intent": intent,
                    "domains": domains,
                    "recency_days": normalized.get("recency_days"),
                    "budget_limited": False,
                    "read_alternatives_available": True,
                    "unread_candidate_count": len(self._unread_candidate_urls()),
                    "budget_decision": decision,
                },
            )

        if (
            not repair_reason_code
            and not domains
            and is_duplicate_search_query(
                query,
                intent,
                previous_queries=self.web_search_queries,
                previous_intents=self.web_search_intents,
            )
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
        if repair_reason_code:
            normalized["count"] = REPAIR_SEARCH_COUNT
            normalized["context_source_limit"] = REPAIR_CONTEXT_SOURCE_LIMIT
            normalized["search_budget"] = "repair"
        else:
            normalized["count"] = search_budget.requested_count
            normalized["context_source_limit"] = search_budget.context_source_limit
            normalized["search_budget"] = search_budget.name

        effective_planned_search_limit = planned_search_limit
        if repair_reason_code and self.web_search_calls >= planned_search_limit:
            effective_planned_search_limit = min(MAX_SEARCH_CALLS, self.web_search_calls + 1)
        decision = _search_budget_decision(
            query=query,
            intent=intent,
            action="repair_search" if repair_reason_code else _allowed_search_action(search_budget.name),
            budget_name=normalized["search_budget"],
            requested_count=normalized["count"],
            context_source_limit=normalized["context_source_limit"],
            reason_code=repair_reason_code or _allowed_search_reason_code(search_budget.name, previous_query_count),
            previous_query_count=previous_query_count,
            planned_search_limit=effective_planned_search_limit,
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

        if self.web_search_calls >= effective_planned_search_limit:
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
                planned_search_limit=effective_planned_search_limit,
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
                    "planned_search_limit": effective_planned_search_limit,
                    "executed_search_count": self.web_search_calls,
                    "budget_decision": decision,
                },
            )

        self.web_search_calls += 1
        self.web_search_queries.append(query)
        self.web_search_intents.append(intent)
        if repair_reason_code:
            self.repair_search_used = True
            self.pending_search_repair_reason_code = None
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

    def record_tool_results(self, results: list, *, source_plan=None) -> None:
        """回填本轮工具执行结果，供下一次预算决策使用。"""

        self._record_source_plan_candidates(source_plan)
        for record in results or []:
            tool_name = getattr(record, "tool_name", "")
            result = getattr(record, "result", None)
            if result is None:
                continue
            if tool_name == "web_search":
                self._record_search_result(result)
            elif tool_name == "url_read":
                self._record_url_read_result(record, result)
        self._drop_attempted_read_candidates()

    def _record_search_result(self, result) -> None:
        data = getattr(result, "data", None) or {}
        decision = data.get("budget_decision") if isinstance(data, dict) else {}
        action = decision.get("action") if isinstance(decision, dict) else ""
        if action and action not in PROVIDER_SEARCH_ACTIONS:
            return

        result_count = _result_count(data)
        if result.status != "success" or result_count == 0:
            self._set_pending_search_repair("previous_search_no_results")
            return
        if result_count < WEAK_SEARCH_RESULT_THRESHOLD:
            self._set_pending_search_repair("previous_search_weak_results")
            return
        self.pending_search_repair_reason_code = None

    def _record_url_read_result(self, record, result) -> None:
        url = _record_url(record, result)
        if url:
            self.attempted_read_urls.add(url)
        if result.status != "success":
            self.read_failure_pending = True
        elif url:
            self.read_failure_pending = False

    def _record_source_plan_candidates(self, source_plan) -> None:
        decisions = getattr(source_plan, "read_decisions", ()) if source_plan is not None else ()
        for decision in decisions or ():
            if getattr(decision, "action", "") not in READ_ALTERNATIVE_ACTIONS:
                continue
            url = getattr(getattr(decision, "candidate", None), "url", "")
            if url:
                self.candidate_read_urls.add(str(url))
        self._drop_attempted_read_candidates()

    def _drop_attempted_read_candidates(self) -> None:
        if self.attempted_read_urls:
            self.candidate_read_urls.difference_update(self.attempted_read_urls)

    def _set_pending_search_repair(self, reason_code: str) -> None:
        if not self.repair_search_used:
            self.pending_search_repair_reason_code = reason_code

    def _consume_search_repair_reason_code(self) -> str | None:
        if self.repair_search_used:
            self.pending_search_repair_reason_code = None
            return None
        return self.pending_search_repair_reason_code

    def _unread_candidate_urls(self) -> set[str]:
        return self.candidate_read_urls - self.attempted_read_urls

    def _has_unread_read_alternatives(self) -> bool:
        has_alternatives = bool(self._unread_candidate_urls())
        if not has_alternatives:
            self.read_failure_pending = False
        return self.read_failure_pending and has_alternatives


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


def _allowed_search_action(budget_name: str) -> str:
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


def _result_count(data: dict) -> int:
    value = data.get("result_count")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        sources = data.get("sources")
        return len(sources) if isinstance(sources, list) else 0


def _record_url(record, result) -> str:
    data = getattr(result, "data", None) or {}
    url = data.get("url") if isinstance(data, dict) else ""
    if url:
        return str(url)
    raw_arguments = getattr(record, "tool_call", {}).get("arguments", {})
    if isinstance(raw_arguments, dict):
        return str(raw_arguments.get("url") or "")
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return ""
        if isinstance(parsed, dict):
            return str(parsed.get("url") or "")
    return ""
