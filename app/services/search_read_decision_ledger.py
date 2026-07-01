"""Search / Read Planner 决策账本。

本模块只聚合已有工具结果和来源选择计划，供测试和后续 UI 消费。
"""

from __future__ import annotations

from typing import Any

from app.services.source_candidate_ranker import SourceSelectionPlan
from app.services.stream.tool_execution_result import ToolExecutionRecord

PROVIDER_SEARCH_ACTIONS = {"execute", "narrow_followup", "repair_search"}


def build_search_read_decision_ledger(
    results: list[ToolExecutionRecord],
    *,
    source_plan: SourceSelectionPlan | None = None,
) -> dict[str, Any]:
    """把一次工具回合里的搜索预算和读源选择决策聚合成可评估结构。"""

    search_decisions = [_search_decision(record) for record in results if record.tool_name == "web_search"]
    read_decisions = _read_decisions(source_plan)
    reason_codes = _unique_reason_codes(search_decisions, read_decisions)
    return {
        "search_decisions": search_decisions,
        "read_decisions": read_decisions,
        "summary": {
            "executed_search_count": sum(
                1 for record in results if record.tool_name == "web_search" and record.result.status == "success"
            ),
            "provider_search_count": sum(
                1 for decision in search_decisions if decision.get("action") in PROVIDER_SEARCH_ACTIONS
            ),
            "query_repair_count": sum(1 for decision in search_decisions if decision.get("action") == "repair_search"),
            "recommended_read_count": sum(
                1 for decision in read_decisions if decision.get("action") == "recommend_read"
            ),
            "deprioritized_count": sum(1 for decision in read_decisions if decision.get("action") == "deprioritize"),
            "kept_candidate_count": sum(1 for decision in read_decisions if decision.get("action") == "keep_candidate"),
            "decision_reason_codes": reason_codes,
        },
    }


def _search_decision(record: ToolExecutionRecord) -> dict[str, Any]:
    data = getattr(record.result, "data", None) or {}
    raw_decision = data.get("budget_decision") if isinstance(data, dict) else None
    decision = raw_decision if isinstance(raw_decision, dict) else {}
    return {
        "tool_call_id": str(record.tool_call.get("id", "")),
        "query": str(decision.get("query") or data.get("query") or ""),
        "intent": decision.get("intent") or data.get("intent"),
        "action": str(decision.get("action") or _fallback_search_action(record)),
        "budget_name": str(decision.get("budget_name") or data.get("search_budget") or ""),
        "requested_count": _int_value(decision.get("requested_count"), data.get("requested_count")),
        "context_source_limit": _int_value(decision.get("context_source_limit"), data.get("context_source_limit")),
        "reason_code": str(decision.get("reason_code") or _fallback_search_reason_code(record)),
        "previous_query_count": _int_value(decision.get("previous_query_count"), 0),
        "planned_search_limit": _int_value(decision.get("planned_search_limit"), 0),
        "status": record.result.status,
    }


def _read_decisions(source_plan: SourceSelectionPlan | None) -> list[dict[str, Any]]:
    if source_plan is None:
        return []
    decisions = getattr(source_plan, "read_decisions", ()) or ()
    return [
        {
            "rank": decision.candidate.rank,
            "title": decision.candidate.title,
            "url": decision.candidate.url,
            "domain": decision.candidate.domain,
            "query": decision.candidate.query,
            "tool_call_id": decision.candidate.tool_call_id,
            "source_index": decision.candidate.source_index,
            "priority": decision.candidate.priority,
            "score": decision.candidate.score,
            "action": decision.action,
            "reason_code": decision.reason_code,
        }
        for decision in decisions
    ]


def _unique_reason_codes(search_decisions: list[dict[str, Any]], read_decisions: list[dict[str, Any]]) -> list[str]:
    reason_codes: list[str] = []
    seen: set[str] = set()
    for decision in [*search_decisions, *read_decisions]:
        reason_code = str(decision.get("reason_code") or "")
        if not reason_code or reason_code in seen:
            continue
        seen.add(reason_code)
        reason_codes.append(reason_code)
    return reason_codes


def _fallback_search_action(record: ToolExecutionRecord) -> str:
    if record.result.status == "success":
        return "execute"
    return "unknown"


def _fallback_search_reason_code(record: ToolExecutionRecord) -> str:
    if record.result.status == "success":
        return "provider_result"
    return "provider_unavailable"


def _int_value(value: Any, fallback: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(fallback)
        except (TypeError, ValueError):
            return 0
