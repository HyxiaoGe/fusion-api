"""
WebSearchHandler — 网络搜索工具处理器
从 stream_handler.py 提取，行为保持不变
"""

import re
import time
import unicodedata
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.ai.prompts.agent_loop import (
    SEARCH_CONTEXT_CITATION_RULE,
    SEARCH_CONTEXT_FOLLOW_UP_RULES,
    SEARCH_CONTEXT_OPENING,
    SEARCH_CONTEXT_TRUST_BOUNDARY,
)
from app.schemas.chat import SearchBlock, SearchSource, SearchSourceSummary, SourceReference
from app.services.external.search_client import search_web
from app.services.source_context import UntrustedSourceContext, format_untrusted_source_context
from app.services.tool_handlers.base import BaseToolHandler, ToolResult

MAX_CONTEXT_SOURCES = 8
DEFAULT_MAX_SOURCES_PER_DOMAIN = 2
TRACKING_QUERY_PARAMS = {
    "_hsenc",
    "_hsmi",
    "dclid",
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "msclkid",
    "spm",
    "ttclid",
    "twclid",
    "vero_conv",
    "vero_id",
    "yclid",
}


class WebSearchHandler(BaseToolHandler):
    @property
    def tool_name(self) -> str:
        return "web_search"

    @property
    def sse_event_prefix(self) -> str:
        return "search"

    async def execute(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        context_source_limit = _normalize_context_source_limit(args.get("context_source_limit"))
        search_budget = args.get("search_budget")
        if not query:
            return ToolResult(
                status="degraded",
                error_message="query 为空",
                data={
                    "query": query,
                    "sources": [],
                    "result_count": 0,
                    "requested_count": args.get("count", 5),
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": context_source_limit,
                    "search_budget": search_budget,
                    "intent": args.get("intent"),
                    "domains": args.get("domains", []),
                    "recency_days": args.get("recency_days"),
                    "budget_limited": bool(args.get("budget_limited", False)),
                },
            )

        requested_count = args.get("count", 5)
        domains = args.get("domains") or []
        recency_days = args.get("recency_days")
        intent = args.get("intent")
        start = time.monotonic()
        try:
            raw_sources = await search_web(query, count=requested_count, domains=domains, recency_days=recency_days)
            duration_ms = int((time.monotonic() - start) * 1000)

            if not raw_sources:
                return ToolResult(
                    status="degraded",
                    duration_ms=duration_ms,
                    error_message="搜索返回空结果",
                    data={
                        "query": query,
                        "sources": [],
                        "result_count": 0,
                        "requested_count": requested_count,
                        "actual_count": 0,
                        "context_source_count": 0,
                        "context_source_limit": context_source_limit,
                        "search_budget": search_budget,
                        "intent": intent,
                        "domains": domains,
                        "recency_days": recency_days,
                        "budget_limited": False,
                    },
                )

            provider_metadata = _extract_provider_metadata(raw_sources)
            sources = _post_process_sources(raw_sources, intent=intent, domains=domains)
            context_source_count = min(len(sources), context_source_limit)
            return ToolResult(
                status="success",
                duration_ms=duration_ms,
                data={
                    "query": query,
                    "sources": sources,
                    "result_count": len(sources),
                    "requested_count": requested_count,
                    "actual_count": len(raw_sources),
                    "context_source_count": context_source_count,
                    "context_source_limit": context_source_limit,
                    "search_budget": search_budget,
                    "intent": intent,
                    "domains": domains,
                    "recency_days": recency_days,
                    "budget_limited": False,
                    **provider_metadata,
                },
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                status="failed",
                duration_ms=duration_ms,
                error_message=str(e),
                data={
                    "query": query,
                    "sources": [],
                    "result_count": 0,
                    "requested_count": requested_count,
                    "actual_count": 0,
                    "context_source_count": 0,
                    "context_source_limit": context_source_limit,
                    "search_budget": search_budget,
                    "intent": intent,
                    "domains": domains,
                    "recency_days": recency_days,
                    "budget_limited": False,
                },
            )

    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> SearchBlock:
        sources: List[SearchSource] = result.data.get("sources", [])
        source_refs = [
            SourceReference(
                kind="search",
                title=s.title,
                url=s.url,
                favicon=s.favicon,
                status=result.status,
                tool_call_log_id=log_id,
                error_message=result.error_message,
            )
            for s in sources
        ]
        return SearchBlock(
            type="search",
            id=block_id,
            query=result.data.get("query", ""),
            tool_call_log_id=log_id,
            sources=[
                SearchSourceSummary(
                    title=s.title,
                    url=s.url,
                    favicon=s.favicon,
                )
                for s in sources
            ],
            status=result.status,
            error_message=result.error_message,
            source_count=len(source_refs),
            source_refs=source_refs,
            requested_provider=result.data.get("requested_provider"),
            result_provider=result.data.get("result_provider"),
            fallback_used=bool(result.data.get("fallback_used", False)),
            provider_chain=result.data.get("provider_chain", []),
            requested_count=result.data.get("requested_count"),
            actual_count=result.data.get("actual_count"),
            context_source_count=result.data.get("context_source_count"),
            context_source_limit=result.data.get("context_source_limit"),
            search_budget=result.data.get("search_budget"),
            intent=result.data.get("intent"),
            domains=result.data.get("domains", []),
            recency_days=result.data.get("recency_days"),
            budget_limited=bool(result.data.get("budget_limited", False)),
        )

    def format_llm_context(self, result: ToolResult) -> str:
        sources: List[SearchSource] = result.data.get("sources", [])
        if not sources:
            return (
                "搜索未取得可用结果，不能把这次搜索作为依据；如需回答，请说明搜索来源不可用，或仅基于其他可用信息回答。"
            )

        parts = [SEARCH_CONTEXT_OPENING]
        parts.append(SEARCH_CONTEXT_TRUST_BOUNDARY)
        parts.append(SEARCH_CONTEXT_CITATION_RULE)

        context_source_limit = _normalize_context_source_limit(result.data.get("context_source_limit"))
        context_sources = sources[:context_source_limit]
        if len(sources) > len(context_sources):
            parts.append(f"搜索返回 {len(sources)} 条结果，仅前 {len(context_sources)} 条注入上下文。\n")

        for i, source in enumerate(context_sources, 1):
            parts.append(f"[{i}] {source.title}")
            parts.append(f"    来源: {source.url}")
            content = source.content or source.description
            parts.append(
                format_untrusted_source_context(
                    UntrustedSourceContext(
                        source_id=f"S{i}",
                        source_type="search",
                        title=source.title,
                        url=source.url,
                        content=content,
                        provider="search-service",
                    ),
                    max_chars=1000,
                )
            )
            parts.append("")

        parts.append("注意：")
        parts.extend(f"- {rule}" for rule in SEARCH_CONTEXT_FOLLOW_UP_RULES)

        return "\n".join(parts)

    def _build_result_summary(self, result: ToolResult) -> dict:
        """搜索结果轻量摘要：命中数 + 首条标题/favicon。

        emitter.tool_call_completed 内部还会经 cap_and_truncate(1024) 兜底。
        """
        if result.status != "success":
            return {"kind": "search", "truncated": False}
        sources = (result.data or {}).get("sources") or []
        first = sources[0] if sources else None
        return {
            "kind": "search",
            "title": getattr(first, "title", "") if first else "",
            "count": len(sources),
            "favicon": getattr(first, "favicon", None) if first else None,
            "result_provider": result.data.get("result_provider"),
            "truncated": False,
        }


def _extract_provider_metadata(sources: List[SearchSource]) -> dict:
    first = next((source for source in sources if source.result_provider or source.requested_provider), None)
    if not first:
        return {}

    return {
        "requested_provider": first.requested_provider,
        "result_provider": first.result_provider,
        "fallback_used": first.fallback_used,
        "provider_chain": first.provider_chain,
    }


def _normalize_context_source_limit(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = MAX_CONTEXT_SOURCES
    return max(1, min(MAX_CONTEXT_SOURCES, parsed))


def _post_process_sources(sources: List[SearchSource], intent: Optional[str], domains: list[str]) -> List[SearchSource]:
    relax_domain_limit = intent == "official_source" or _has_single_domain_filter(domains)
    seen_urls: set[str] = set()
    seen_domain_titles: set[tuple[str, str]] = set()
    domain_counts: dict[str, int] = {}
    processed: List[SearchSource] = []

    for source in sources:
        canonical_url, normalized_domain = _canonicalize_search_url(source.url)
        url_key = canonical_url or source.url.strip()
        if url_key in seen_urls:
            continue

        title_key = _normalize_title(source.title)
        domain_title_key = (normalized_domain, title_key) if normalized_domain and title_key else None
        if domain_title_key and domain_title_key in seen_domain_titles:
            continue

        if (
            not relax_domain_limit
            and normalized_domain
            and domain_counts.get(normalized_domain, 0) >= DEFAULT_MAX_SOURCES_PER_DOMAIN
        ):
            continue

        seen_urls.add(url_key)
        if domain_title_key:
            seen_domain_titles.add(domain_title_key)
        if normalized_domain:
            domain_counts[normalized_domain] = domain_counts.get(normalized_domain, 0) + 1
        processed.append(_copy_source_with_url(source, canonical_url))

    return processed


def _canonicalize_search_url(url: str) -> tuple[str, str]:
    stripped_url = (url or "").strip()
    if not stripped_url:
        return "", ""

    try:
        parsed = urlsplit(stripped_url)
    except ValueError:
        return stripped_url, ""

    if not parsed.netloc:
        return stripped_url, ""

    scheme = parsed.scheme.lower() or "https"
    normalized_domain = _normalize_domain(parsed.hostname or "")
    if not normalized_domain:
        return stripped_url, ""

    try:
        port = parsed.port
    except ValueError:
        return stripped_url, normalized_domain

    include_port = port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443))
    netloc = f"{normalized_domain}:{port}" if include_port else normalized_domain
    query = _canonicalize_query(parsed.query)
    path = "" if parsed.path == "/" else parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, query, "")), normalized_domain


def _canonicalize_query(query: str) -> str:
    params = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_PARAMS:
            continue
        params.append((key, value))

    params.sort(key=lambda item: (item[0].lower(), item[1]))
    return urlencode(params, doseq=True)


def _normalize_domain(domain: str) -> str:
    normalized = domain.strip().rstrip(".").lower()
    while normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _normalize_domain_filter(domain: str) -> str:
    stripped_domain = (domain or "").strip()
    if not stripped_domain:
        return ""

    try:
        parsed = urlsplit(stripped_domain)
    except ValueError:
        parsed = None

    if parsed and parsed.hostname:
        host = parsed.hostname
    else:
        host = stripped_domain.split("/", 1)[0]
        host = host.split(":", 1)[0]

    return _normalize_domain(host.removeprefix("*."))


def _has_single_domain_filter(domains: list[str]) -> bool:
    normalized_domains = {_normalize_domain_filter(domain) for domain in domains if _normalize_domain_filter(domain)}
    return len(normalized_domains) == 1


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title or "").casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _copy_source_with_url(source: SearchSource, url: str) -> SearchSource:
    if not url or source.url == url:
        return source
    return source.model_copy(update={"url": url})
