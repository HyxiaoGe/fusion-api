"""单轮联网工具预算与参数归一化。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.search_budget import SEARCH_BUDGETS_BY_INTENT, derive_search_budget
from app.services.tool_handlers.base import ToolResult

MAX_SEARCH_CALLS = 4
MAX_URL_READ_CALLS = 5
MAX_DOMAINS = 5
MIN_RECENCY_DAYS = 1
MAX_RECENCY_DAYS = 365

SUPPORTED_SEARCH_INTENTS = set(SEARCH_BUDGETS_BY_INTENT)

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


@dataclass
class NetworkToolBudget:
    """限制一次 assistant run 内的联网工具调用次数。"""

    web_search_calls: int = 0
    url_read_calls: int = 0

    def prepare_web_search_args(self, args: dict) -> tuple[dict, ToolResult | None]:
        normalized = dict(args or {})

        intent = _normalize_intent(normalized.get("intent"))
        if intent:
            normalized["intent"] = intent
        else:
            normalized.pop("intent", None)

        search_budget = derive_search_budget(intent)
        normalized["count"] = search_budget.requested_count
        normalized["context_source_limit"] = search_budget.context_source_limit
        normalized["search_budget"] = search_budget.name

        domains = _normalize_domains(normalized.get("domains"))
        if domains:
            normalized["domains"] = domains
        else:
            normalized.pop("domains", None)

        if normalized.get("recency_days") is not None:
            normalized["recency_days"] = _clamp_int(
                normalized.get("recency_days"),
                MIN_RECENCY_DAYS,
                MIN_RECENCY_DAYS,
                MAX_RECENCY_DAYS,
            )

        if self.web_search_calls >= MAX_SEARCH_CALLS:
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
                },
            )

        self.web_search_calls += 1
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


def _normalize_intent(value) -> str | None:
    if not isinstance(value, str):
        return None
    intent = value.strip().lower()
    if intent in SUPPORTED_SEARCH_INTENTS:
        return intent
    return None


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
