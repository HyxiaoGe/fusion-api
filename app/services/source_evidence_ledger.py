"""联网来源 evidence ledger 的轻量构造工具。"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.services.source_candidate_ranker import RankedSourceCandidate

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
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
    "yclid",
}

URL_READ_STATUS_TO_EVIDENCE_STATUS = {
    "success": "read_success",
    "degraded": "read_degraded",
    "failed": "read_failed",
    "interrupted": "read_failed",
}


def canonicalize_evidence_url(url: str) -> str:
    """生成用于 evidence 去重的稳定 URL。"""
    stripped_url = (url or "").strip()
    if not stripped_url:
        return ""
    try:
        parsed = urlsplit(stripped_url)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parsed.port and not ((scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, netloc, parsed.path or "", query, ""))


def stable_web_evidence_id(url: str, *, fallback: str) -> str:
    canonical_url = canonicalize_evidence_url(url)
    if not canonical_url:
        return fallback
    digest = hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"ev-web-{digest}"


def build_search_source_evidence_item(source: Any, *, tool_call_id: str, source_index: int) -> dict[str, Any]:
    url = _source_value(source, "url") or ""
    canonical_url = canonicalize_evidence_url(url)
    evidence_url = canonical_url or url
    title = _safe_text(_source_value(source, "title"), 80) or "搜索结果"
    claim = _safe_text(
        _source_value(source, "description") or _source_value(source, "content") or title,
        120,
    )
    return {
        "id": stable_web_evidence_id(url, fallback=f"ev-{tool_call_id}-{source_index}"),
        "kind": "web",
        "status": "candidate",
        "title": title,
        "url": evidence_url,
        "domain": _domain(evidence_url),
        "claim": claim,
        "snippet": _safe_text(_source_value(source, "content") or _source_value(source, "description"), 180),
        "used_by_final_answer": False,
    }


def build_url_read_evidence_item(
    result_data: dict[str, Any], *, status: str, tool_call_id: str
) -> dict[str, Any] | None:
    url = str(result_data.get("url") or result_data.get("safe_log_url") or "").strip()
    if not url:
        return None
    canonical_url = canonicalize_evidence_url(url)
    evidence_url = canonical_url or url
    evidence_status = URL_READ_STATUS_TO_EVIDENCE_STATUS.get(status, "read_failed")
    title = _safe_text(result_data.get("title"), 80) or _domain(evidence_url) or "网页来源"
    content = result_data.get("content") or result_data.get("reason") or result_data.get("failure_detail")
    if evidence_status == "read_success":
        claim = _safe_text(content or "已读取网页内容，供后续回答核验。", 120)
    elif evidence_status == "read_degraded":
        claim = _safe_text(content or "网页暂时无法完整读取，已降级处理。", 120)
    else:
        claim = _safe_text(content or "网页暂时无法读取，已跳过该来源。", 120)
    return {
        "id": stable_web_evidence_id(url, fallback=f"ev-{tool_call_id}-url"),
        "kind": "web",
        "status": evidence_status,
        "title": title,
        "url": evidence_url,
        "domain": _domain(evidence_url),
        "claim": claim,
        "snippet": _safe_text(content, 180) if content else None,
        "used_by_final_answer": False,
    }


def build_selected_source_evidence_item(candidate: RankedSourceCandidate) -> dict[str, Any]:
    reasons = " / ".join(candidate.reasons[:3]) or "高优先级来源"
    claim = _safe_text(f"建议深读：{reasons}", 120)
    snippet = _safe_text(f"来自搜索关键词：{candidate.query}", 180) if candidate.query else None
    return {
        "id": stable_web_evidence_id(candidate.url, fallback=f"ev-{candidate.tool_call_id}-{candidate.source_index}"),
        "kind": "web",
        "status": "selected",
        "title": _safe_text(candidate.title, 80) or "建议深读来源",
        "url": canonicalize_evidence_url(candidate.url) or candidate.url,
        "domain": candidate.domain or _domain(candidate.url),
        "claim": claim,
        "snippet": snippet,
        "used_by_final_answer": False,
    }


def _source_value(source: Any, key: str) -> str | None:
    if isinstance(source, dict):
        value = source.get(key)
    else:
        value = getattr(source, key, None)
    return str(value) if value else None


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    return parsed.netloc or None


def _safe_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_chars]
