"""最终回答来源使用判定。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from app.services.source_evidence_ledger import canonicalize_evidence_url, stable_web_evidence_id

_CITATION_PATTERN = re.compile(r"(?:\[(\d{1,3})\]|⟦(\d{1,3})⟧)")
_URL_PATTERN = re.compile(r"https?://[^\s\])}>\"'，。；、]+", re.IGNORECASE)


@dataclass(frozen=True)
class _AnswerSource:
    kind: str
    title: str
    url: str
    canonical_url: str
    domain: str | None
    favicon: str | None = None

    @property
    def evidence_id(self) -> str:
        raw_key = self.url or self.canonical_url or self.title
        digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
        return stable_web_evidence_id(self.url, fallback=f"ev-final-{digest}")


def build_used_final_answer_evidence(
    *, content_blocks: list[Any], answer_text: str
) -> list[dict[str, Any]]:
    """从最终回答文本里保守识别真正使用过的网页来源。"""
    normalized_answer = (answer_text or "").strip()
    if not normalized_answer:
        return []

    search_sources, read_sources = _collect_sources(content_blocks)
    all_sources = _dedupe_sources([*search_sources, *read_sources])
    if not all_sources:
        return []

    used: list[_AnswerSource] = []
    _extend_unique(used, _sources_from_citations(normalized_answer, search_sources))
    _extend_unique(used, _sources_from_url_mentions(normalized_answer, all_sources))
    _extend_unique(used, _sources_from_unique_domain_mentions(normalized_answer, all_sources))

    if not used and len(read_sources) == 1:
        used.append(read_sources[0])

    return [_to_evidence_item(source) for source in used]


def _collect_sources(content_blocks: list[Any]) -> tuple[list[_AnswerSource], list[_AnswerSource]]:
    search_sources: list[_AnswerSource] = []
    read_sources: list[_AnswerSource] = []

    for block in content_blocks:
        block_type = _value(block, "type")
        refs = _source_refs(block)
        if refs:
            for ref in refs:
                if _value(ref, "status") not in {"", None, "success"}:
                    continue
                source = _source_from_ref(ref)
                if source is None:
                    continue
                if source.kind == "url_read" or block_type == "url_read":
                    read_sources.append(source)
                else:
                    search_sources.append(source)
            continue

        if block_type == "search":
            for source_summary in _value(block, "sources") or []:
                source = _source_from_values(
                    kind="search",
                    title=_value(source_summary, "title") or "",
                    url=_value(source_summary, "url") or "",
                    favicon=_value(source_summary, "favicon"),
                )
                if source is not None:
                    search_sources.append(source)
        elif block_type == "url_read" and _value(block, "status") in {"", None, "success"}:
            source = _source_from_values(
                kind="url_read",
                title=_value(block, "title") or "",
                url=_value(block, "url") or "",
                favicon=_value(block, "favicon"),
            )
            if source is not None:
                read_sources.append(source)

    return _dedupe_sources(search_sources), _dedupe_sources(read_sources)


def _source_refs(block: Any) -> list[Any]:
    refs = _value(block, "source_refs")
    return refs if isinstance(refs, list) else []


def _source_from_ref(ref: Any) -> _AnswerSource | None:
    return _source_from_values(
        kind=_value(ref, "kind") or "search",
        title=_value(ref, "title") or "",
        url=_value(ref, "url") or "",
        favicon=_value(ref, "favicon"),
    )


def _source_from_values(*, kind: str, title: str, url: str, favicon: str | None = None) -> _AnswerSource | None:
    raw_url = str(url or "").strip()
    canonical_url = canonicalize_evidence_url(raw_url)
    evidence_url = canonical_url or raw_url
    if not evidence_url:
        return None

    domain = _domain(evidence_url)
    return _AnswerSource(
        kind=kind,
        title=str(title or "").strip() or domain or "网页来源",
        url=raw_url,
        canonical_url=evidence_url,
        domain=domain,
        favicon=favicon,
    )


def _sources_from_citations(answer_text: str, search_sources: list[_AnswerSource]) -> list[_AnswerSource]:
    sources: list[_AnswerSource] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        index_text = match.group(1) or match.group(2)
        if not index_text:
            continue
        index = int(index_text) - 1
        if 0 <= index < len(search_sources):
            sources.append(search_sources[index])
    return sources


def _sources_from_url_mentions(answer_text: str, sources: list[_AnswerSource]) -> list[_AnswerSource]:
    lowered = answer_text.lower()
    mentioned_urls = {
        canonical
        for url in _URL_PATTERN.findall(answer_text)
        if (canonical := canonicalize_evidence_url(url))
    }
    matched: list[_AnswerSource] = []

    for source in sources:
        canonical = source.canonical_url.lower()
        raw = source.url.lower()
        if canonical in mentioned_urls or canonical in lowered or (raw and raw in lowered):
            matched.append(source)
    return matched


def _sources_from_unique_domain_mentions(answer_text: str, sources: list[_AnswerSource]) -> list[_AnswerSource]:
    lowered = answer_text.lower()
    by_domain: dict[str, list[_AnswerSource]] = {}
    for source in sources:
        if source.domain:
            by_domain.setdefault(source.domain.lower(), []).append(source)

    matched: list[_AnswerSource] = []
    for domain, domain_sources in by_domain.items():
        if domain in lowered and len(_dedupe_sources(domain_sources)) == 1:
            matched.append(domain_sources[0])
    return matched


def _extend_unique(target: list[_AnswerSource], candidates: list[_AnswerSource]) -> None:
    seen = {source.evidence_id for source in target}
    for source in candidates:
        if source.evidence_id in seen:
            continue
        target.append(source)
        seen.add(source.evidence_id)


def _dedupe_sources(sources: list[_AnswerSource]) -> list[_AnswerSource]:
    deduped: list[_AnswerSource] = []
    seen: set[str] = set()
    for source in sources:
        key = source.canonical_url or source.url
        if not key or key in seen:
            continue
        deduped.append(source)
        seen.add(key)
    return deduped


def _to_evidence_item(source: _AnswerSource) -> dict[str, Any]:
    evidence_url = source.canonical_url or source.url
    return {
        "id": source.evidence_id,
        "kind": "web",
        "status": "used",
        "title": source.title,
        "url": evidence_url,
        "domain": source.domain or _domain(evidence_url),
        "claim": "最终回答引用了该来源。",
        "snippet": None,
        "used_by_final_answer": True,
    }


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = parsed.hostname or parsed.netloc
    if not host:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
