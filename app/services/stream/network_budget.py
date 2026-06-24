"""单轮联网工具预算与参数归一化。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.services.tool_handlers.base import ToolResult

MIN_SEARCH_COUNT = 3
DEFAULT_SEARCH_COUNT = 5
MAX_SEARCH_COUNT = 10
MAX_SEARCH_CALLS = 3
MAX_URL_READ_CALLS = 5
MAX_DOMAINS = 5
MIN_RECENCY_DAYS = 1
MAX_RECENCY_DAYS = 365

SUPPORTED_SEARCH_INTENTS = {
    "lookup",
    "news",
    "comparison",
    "official",
    "research",
    "verification",
}

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


@dataclass
class NetworkToolBudget:
    """限制一次 assistant run 内的联网工具调用次数。"""

    web_search_calls: int = 0
    url_read_calls: int = 0

    def prepare_web_search_args(self, args: dict) -> tuple[dict, ToolResult | None]:
        normalized = dict(args or {})
        normalized["count"] = _clamp_int(
            normalized.get("count"), DEFAULT_SEARCH_COUNT, MIN_SEARCH_COUNT, MAX_SEARCH_COUNT
        )

        intent = _normalize_intent(normalized.get("intent"))
        if intent:
            normalized["intent"] = intent
        else:
            normalized.pop("intent", None)

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
                    "requested_count": normalized.get("count", DEFAULT_SEARCH_COUNT),
                    "actual_count": 0,
                    "context_source_count": 0,
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
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    if _DOMAIN_RE.match(host):
        return host
    return None
