"""搜索候选来源排序器。

本模块只做确定性的候选来源评分、去重和解释，不直接触发 url_read。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.schemas.chat import SearchSource

MAX_LOW_PRIORITY_EXAMPLES = 3
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
    "yclid",
}
STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "news",
    "the",
    "with",
    "发布",
    "新闻",
    "最新",
    "官方",
    "公告",
}
AUTHORITY_MEDIA_DOMAINS = {
    "apnews.com",
    "axios.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "nytimes.com",
    "reuters.com",
    "techcrunch.com",
    "theverge.com",
    "venturebeat.com",
    "wired.com",
    "wsj.com",
}
LOW_PRIORITY_DOMAINS = {
    "bilibili.com",
    "douyin.com",
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "threads.com",
    "tiktok.com",
    "twitter.com",
    "weibo.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "zhihu.com",
}
VIDEO_DOMAINS = {"bilibili.com", "douyin.com", "tiktok.com", "youtube.com", "youtu.be"}
FORUM_DOMAINS = {"reddit.com", "threads.com", "twitter.com", "weibo.com", "x.com", "zhihu.com"}


@dataclass(frozen=True)
class SearchResultForRanking:
    tool_call_id: str
    query: str
    sources: list[SearchSource | dict]
    intent: str | None = None
    search_budget: str | None = None


@dataclass(frozen=True)
class RankedSourceCandidate:
    rank: int
    title: str
    url: str
    domain: str
    query: str
    tool_call_id: str
    source_index: int
    score: int
    priority: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SourceReadDecision:
    candidate: RankedSourceCandidate
    action: str
    reason_code: str


@dataclass(frozen=True)
class SourceSelectionPlan:
    total_source_count: int
    unique_source_count: int
    search_queries: tuple[str, ...]
    candidates: tuple[RankedSourceCandidate, ...]
    recommended: tuple[RankedSourceCandidate, ...]
    low_priority: tuple[RankedSourceCandidate, ...]
    read_decisions: tuple[SourceReadDecision, ...]
    decision_summary: dict[str, int]
    recommended_read_limit: int = 3
    not_recommended_count: int = 0
    read_required: bool = False
    minimum_required_reads: int = 0
    read_required_reason: str = ""


@dataclass(frozen=True)
class _CandidateDraft:
    title: str
    url: str
    canonical_url: str
    domain: str
    query: str
    tool_call_id: str
    source_index: int
    source_order: int
    score: int
    priority: str
    reasons: tuple[str, ...]


def rank_search_sources(
    search_results: list[SearchResultForRanking],
    *,
    max_recommended: int = 3,
    read_required: bool = False,
    minimum_required_reads: int = 0,
    read_required_reason: str = "",
) -> SourceSelectionPlan:
    """对同一轮多个搜索结果做跨搜索去重和深读候选排序。"""
    total_source_count = sum(len(result.sources) for result in search_results)
    drafts = _build_candidate_drafts(search_results)
    deduped = _dedupe_candidates(drafts)
    ranked = tuple(
        RankedSourceCandidate(
            rank=index,
            title=draft.title,
            url=draft.canonical_url or draft.url,
            domain=draft.domain,
            query=draft.query,
            tool_call_id=draft.tool_call_id,
            source_index=draft.source_index,
            score=draft.score,
            priority=draft.priority,
            reasons=draft.reasons,
        )
        for index, draft in enumerate(sorted(deduped, key=_sort_candidate), 1)
    )
    recommended_limit = max(0, max_recommended)
    recommended = tuple(candidate for candidate in ranked if candidate.priority != "low")[:recommended_limit]
    low_priority = tuple(candidate for candidate in ranked if candidate.priority == "low")
    read_decisions = _build_read_decisions(ranked, recommended)
    normalized_minimum_reads = 0
    if read_required:
        normalized_minimum_reads = min(len(recommended), max(0, minimum_required_reads))
    return SourceSelectionPlan(
        total_source_count=total_source_count,
        unique_source_count=len(ranked),
        search_queries=tuple(result.query for result in search_results if result.query),
        candidates=ranked,
        recommended=recommended,
        low_priority=low_priority,
        read_decisions=read_decisions,
        decision_summary=_summarize_read_decisions(read_decisions),
        recommended_read_limit=recommended_limit,
        not_recommended_count=max(0, len(ranked) - len(recommended)),
        read_required=read_required and normalized_minimum_reads > 0,
        minimum_required_reads=normalized_minimum_reads,
        read_required_reason=read_required_reason if read_required and normalized_minimum_reads > 0 else "",
    )


def format_source_selection_guidance(plan: SourceSelectionPlan) -> str:
    """生成给 LLM 的本轮搜索候选选择建议。"""
    if not plan.candidates:
        return ""

    parts = [
        "【结构化来源选择建议】",
        (
            f"本轮搜索合并候选 {plan.total_source_count} 条，去重后 {plan.unique_source_count} 条；"
            "以下排序由 SourceCandidateRanker 基于官方性、原文性、相关性和来源类型生成。"
        ),
    ]
    if plan.search_queries:
        parts.append("搜索关键词：")
        parts.extend(f"{index}. {query}" for index, query in enumerate(plan.search_queries, 1))

    parts.append(
        f"建议深读最多 {plan.recommended_read_limit} 个来源；"
        "优先覆盖官方原文、技术报告、权威媒体或与问题高度相关的来源。"
    )
    if plan.read_required:
        parts.append(
            f"本轮属于需核验的当前/关键事实场景；回答事实结论前，"
            f"必须先读取至少 {plan.minimum_required_reads} 个建议优先深读来源。"
        )

    if plan.recommended:
        parts.append("建议优先深读：")
        for candidate in plan.recommended:
            parts.append(_format_candidate_line(candidate))

    if plan.low_priority:
        parts.append("低优先级候选：")
        for candidate in plan.low_priority[:MAX_LOW_PRIORITY_EXAMPLES]:
            parts.append(_format_candidate_line(candidate))

    if plan.not_recommended_count:
        parts.append(
            f"未建议深读：剩余 {plan.not_recommended_count} 条候选优先级低于已推荐来源，"
            "或仅作为搜索摘要候选保留。"
        )
        reason_summary = _format_not_recommended_reason_summary(plan.read_decisions)
        if reason_summary:
            parts.append("未建议深读原因：")
            parts.extend(reason_summary)

    if plan.read_required:
        parts.append(
            "执行规则：回答当前事实结论前，必须优先对“建议优先深读”的来源调用 url_read；"
            "不要为了形式读满所有搜索结果；如果推荐来源不可用，再读取下一个高价值候选，"
            "仍无法核验时必须降低结论确定性。"
        )
    else:
        parts.append(
            "执行规则：如果搜索摘要不足以回答，应优先对“建议优先深读”的少量来源调用 url_read；"
            "不要为了形式读满所有搜索结果；只有当推荐来源无法回答关键事实，才读取未推荐来源。"
        )
    return "\n".join(parts)


def _build_candidate_drafts(search_results: list[SearchResultForRanking]) -> list[_CandidateDraft]:
    drafts: list[_CandidateDraft] = []
    source_order = 0
    for result in search_results:
        for source_index, source in enumerate(result.sources, 1):
            source_order += 1
            drafts.append(_score_source(source, result.query, result.tool_call_id, source_index, source_order))
    return drafts


def _score_source(
    source: SearchSource | dict,
    query: str,
    tool_call_id: str,
    source_index: int,
    source_order: int,
) -> _CandidateDraft:
    source_url = _source_field(source, "url")
    source_description = _source_field(source, "description")
    source_content = _source_field(source, "content")
    canonical_url, domain = _canonicalize_url(source_url)
    title = _source_field(source, "title") or source_url
    text = " ".join([title, source_description, source_content, canonical_url or source_url])
    text_lower = text.lower()
    query_terms = _tokenize(query)
    domain_tokens = set(_tokenize(domain.replace(".", " ")))
    score = max(0, 22 - source_index * 2)
    reasons: list[str] = []
    is_low_priority = False

    is_official = _is_official_source(domain_tokens, query_terms)
    is_authority_media = _is_authority_media(domain)
    if is_official:
        score += 38
        reasons.append("官方来源")
    if _has_original_signal(text_lower, canonical_url, is_official, is_authority_media):
        score += 22
        reasons.append("原文公告")
    has_specific_original = _has_specific_original_signal(text_lower, canonical_url)
    is_pdf = _is_pdf(canonical_url, title)
    if has_specific_original:
        score += 18
        reasons.append("具体原文页面")
    if is_official and has_specific_original and not is_pdf and not _is_news_listing(text_lower, canonical_url):
        score += 35
        reasons.append("官方原文优先")
    if is_pdf:
        score += 35
        reasons.append("官方 PDF/技术报告" if "官方来源" in reasons else "PDF/技术报告")
    if is_authority_media:
        score += 36
        reasons.append("权威媒体")
    if _is_news_listing(text_lower, canonical_url):
        score -= 28
        reasons.append("聚合页降权")

    relevance_score = _relevance_score(query_terms, text_lower)
    if relevance_score:
        score += relevance_score
        reasons.append("高相关")

    if _is_video_source(domain, title):
        score -= 28
        reasons.append("视频来源默认降权")
        is_low_priority = True
    elif _is_forum_source(domain):
        score -= 24
        reasons.append("社交/论坛来源默认降权")
        is_low_priority = True
    elif domain in LOW_PRIORITY_DOMAINS:
        score -= 18
        reasons.append("低相关来源默认降权")
        is_low_priority = True

    priority = _priority(score, is_low_priority)
    return _CandidateDraft(
        title=title,
        url=source_url,
        canonical_url=canonical_url,
        domain=domain,
        query=query,
        tool_call_id=tool_call_id,
        source_index=source_index,
        source_order=source_order,
        score=score,
        priority=priority,
        reasons=tuple(dict.fromkeys(reasons or ["普通候选"])),
    )


def _dedupe_candidates(drafts: list[_CandidateDraft]) -> list[_CandidateDraft]:
    by_url: dict[str, _CandidateDraft] = {}
    for draft in drafts:
        key = draft.canonical_url or draft.url.strip()
        previous = by_url.get(key)
        if previous is None or (draft.score, -draft.source_order) > (previous.score, -previous.source_order):
            by_url[key] = draft
    return list(by_url.values())


def _build_read_decisions(
    candidates: tuple[RankedSourceCandidate, ...],
    recommended: tuple[RankedSourceCandidate, ...],
) -> tuple[SourceReadDecision, ...]:
    recommended_urls = {candidate.url for candidate in recommended}
    decisions: list[SourceReadDecision] = []
    for candidate in candidates:
        if candidate.url in recommended_urls:
            decisions.append(
                SourceReadDecision(
                    candidate=candidate,
                    action="recommend_read",
                    reason_code=_recommended_reason_code(candidate),
                )
            )
            continue
        if candidate.priority == "low":
            decisions.append(
                SourceReadDecision(
                    candidate=candidate,
                    action="deprioritize",
                    reason_code="low_priority_source_type",
                )
            )
            continue
        decisions.append(
            SourceReadDecision(
                candidate=candidate,
                action="keep_candidate",
                reason_code="outside_read_limit",
            )
        )
    return tuple(decisions)


def _recommended_reason_code(candidate: RankedSourceCandidate) -> str:
    reasons = set(candidate.reasons)
    if any("PDF" in reason or "技术报告" in reason for reason in reasons):
        return "official_document"
    if "官方原文优先" in reasons or ("官方来源" in reasons and ("原文公告" in reasons or "具体原文页面" in reasons)):
        return "official_original"
    if "权威媒体" in reasons:
        return "authority_media"
    return "high_relevance"


def _summarize_read_decisions(decisions: tuple[SourceReadDecision, ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for decision in decisions:
        counter[decision.action] += 1
        counter[decision.reason_code] += 1
    return dict(counter)


def _format_not_recommended_reason_summary(decisions: tuple[SourceReadDecision, ...]) -> list[str]:
    labels = {
        "low_priority_source_type": "低优先级来源",
        "outside_read_limit": "超过本轮推荐深读上限",
        "covered_by_recommended_source": "被更高质量来源覆盖",
    }
    counter: Counter[str] = Counter(
        decision.reason_code for decision in decisions if decision.action != "recommend_read"
    )
    return [f"- {label}：{counter[reason_code]} 条" for reason_code, label in labels.items() if counter[reason_code]]


def _source_field(source: SearchSource | dict, field_name: str) -> str:
    if isinstance(source, dict):
        value = source.get(field_name)
    else:
        value = getattr(source, field_name, None)
    return str(value or "")


def _sort_candidate(candidate: _CandidateDraft) -> tuple[int, int]:
    return (-candidate.score, candidate.source_order)


def _format_candidate_line(candidate: RankedSourceCandidate) -> str:
    priority_label = {"high": "高优先级", "medium": "中优先级", "low": "低优先级"}.get(
        candidate.priority, candidate.priority
    )
    reasons = "、".join(candidate.reasons)
    return f"- R{candidate.rank} {priority_label} | {candidate.domain or 'unknown'} | {candidate.title}\n  URL: {candidate.url}\n  原因: {reasons}"


def _canonicalize_url(url: str) -> tuple[str, str]:
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
    domain = _normalize_domain(parsed.hostname or "")
    if not domain:
        return stripped_url, ""
    try:
        port = parsed.port
    except ValueError:
        return stripped_url, domain

    include_port = port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443))
    netloc = f"{domain}:{port}" if include_port else domain
    path = "" if parsed.path == "/" else parsed.path.rstrip("/")
    query = _canonicalize_query(parsed.query)
    return urlunsplit((scheme, netloc, path, query, "")), domain


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


def _tokenize(text: str) -> set[str]:
    tokens = {token.casefold() for token in re.findall(r"[\w.-]+", text or "", flags=re.UNICODE)}
    return {token for token in tokens if len(token) >= 3 and not token.isdigit() and token not in STOP_WORDS}


def _is_official_source(domain_tokens: set[str], query_terms: set[str]) -> bool:
    return bool(domain_tokens & {term for term in query_terms if len(term) >= 4})


def _has_original_signal(text_lower: str, canonical_url: str, is_official: bool, is_authority_media: bool) -> bool:
    original_keywords = (
        "announcement",
        "announcing",
        "changelog",
        "docs",
        "documentation",
        "newsroom",
        "previewing",
        "press",
        "release",
        "released",
        "system card",
        "公告",
        "官方",
        "新闻中心",
    )
    if is_official:
        return any(keyword in text_lower or keyword in canonical_url.lower() for keyword in original_keywords)
    if is_authority_media:
        return "release" in text_lower or "released" in text_lower or "reports" in text_lower
    return False


def _has_specific_original_signal(text_lower: str, canonical_url: str) -> bool:
    specific_keywords = ("previewing", "/index/", "/blog/", "/docs/", "system-card", "system card")
    return any(keyword in text_lower or keyword in canonical_url.lower() for keyword in specific_keywords)


def _is_news_listing(text_lower: str, canonical_url: str) -> bool:
    listing_keywords = ("company-announcements", "news/company", "新闻中心", "最新动态", "newsroom")
    return any(keyword in text_lower or keyword in canonical_url.lower() for keyword in listing_keywords)


def _is_pdf(url: str, title: str) -> bool:
    lowered_url = (url or "").lower()
    lowered_title = (title or "").lower()
    return lowered_url.endswith(".pdf") or "[pdf]" in lowered_title or "system card" in lowered_title


def _is_authority_media(domain: str) -> bool:
    return domain in AUTHORITY_MEDIA_DOMAINS


def _is_video_source(domain: str, title: str) -> bool:
    title_lower = (title or "").lower()
    return domain in VIDEO_DOMAINS or "youtube" in title_lower or "视频" in title_lower or "video" in title_lower


def _is_forum_source(domain: str) -> bool:
    return domain in FORUM_DOMAINS


def _relevance_score(query_terms: set[str], text_lower: str) -> int:
    if not query_terms:
        return 0
    matched = sum(1 for term in query_terms if term in text_lower)
    return min(18, matched * 4)


def _priority(score: int, is_low_priority: bool) -> str:
    if is_low_priority:
        return "low"
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
