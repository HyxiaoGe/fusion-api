"""从工具执行结果派生用户可读 digest/evidence。"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.services.stream.tool_execution_result import ToolExecutionRecord


def build_tool_result_digest(record: ToolExecutionRecord) -> dict[str, Any]:
    evidence_items = build_evidence_items(record)
    summary = _result_summary(record)
    status = _digest_status(record.result.status)
    title = _safe_text(summary.get("title"), 80) or _default_title(record.tool_name, status, summary)

    return {
        "tool_call_id": str(record.tool_call.get("id", "")),
        "tool_name": record.tool_name,
        "status": status,
        "title": title,
        "summary": _digest_summary(record, status, summary),
        "key_findings": _key_findings(evidence_items),
        "source_refs": [item["id"] for item in evidence_items],
        "truncated": bool(summary.get("truncated", False)),
    }


def build_evidence_items(record: ToolExecutionRecord) -> list[dict[str, Any]]:
    sources = _extract_sources(record)
    tool_call_id = str(record.tool_call.get("id", "tool"))
    evidence = []
    for index, source in enumerate(sources[:12]):
        url = _source_value(source, "url")
        if url and not _is_public_url(url):
            continue
        title = _safe_text(_source_value(source, "title"), 80) or "工具结果"
        claim = _safe_text(
            _source_value(source, "description") or _source_value(source, "content") or title,
            120,
        )
        evidence.append(
            {
                "id": f"ev-{tool_call_id}-{index}",
                "kind": "web",
                "status": "candidate",
                "title": title,
                "url": url,
                "domain": _domain(url),
                "claim": claim,
                "snippet": _safe_text(_source_value(source, "content") or _source_value(source, "description"), 180),
                "used_by_final_answer": False,
            }
        )
    return evidence


def _result_summary(record: ToolExecutionRecord) -> dict[str, Any]:
    if record.handler is None:
        return {"kind": record.tool_name, "truncated": False}
    builder = getattr(record.handler, "_build_result_summary", None)
    if builder is None:
        return {"kind": record.tool_name, "truncated": False}
    try:
        summary = builder(record.result)
    except Exception:
        return {"kind": record.tool_name, "truncated": True}
    return summary if isinstance(summary, dict) else {"kind": record.tool_name, "truncated": False}


def _digest_status(status: str) -> str:
    if status in {"success", "failed", "degraded", "interrupted"}:
        return status
    return "failed"


def _default_title(tool_name: str, status: str, summary: dict[str, Any]) -> str:
    count = summary.get("count")
    if status == "success" and isinstance(count, int):
        return f"找到 {count} 条结果"
    if status == "success":
        return f"{tool_name} 已完成"
    if status == "degraded":
        return f"{tool_name} 降级完成"
    return f"{tool_name} 未取得可用结果"


def _digest_summary(record: ToolExecutionRecord, status: str, summary: dict[str, Any]) -> str:
    if status == "success":
        count = summary.get("count")
        if isinstance(count, int):
            return _safe_text(f"保留 {count} 条候选结果，供后续回答筛选。", 120)
        return "工具返回了可用结果。"
    if record.result.error_message:
        return _safe_text(record.result.error_message, 120)
    if status == "degraded":
        return "工具降级返回，结果可能不完整。"
    return "工具未取得可用结果。"


def _key_findings(evidence_items: list[dict[str, Any]]) -> list[str]:
    findings = []
    for item in evidence_items[:5]:
        claim = item.get("claim")
        if claim:
            findings.append(_safe_text(claim, 80))
    return findings


def _extract_sources(record: ToolExecutionRecord) -> list[Any]:
    data = record.result.data or {}
    sources = data.get("sources") or data.get("source_refs") or []
    return sources if isinstance(sources, list) else []


def _source_value(source: Any, key: str) -> str | None:
    if isinstance(source, dict):
        value = source.get(key)
    else:
        value = getattr(source, key, None)
    return str(value) if value else None


def _is_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname or ""
    return host not in {"localhost", "127.0.0.1", "::1"}


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or None


def _safe_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_chars]
