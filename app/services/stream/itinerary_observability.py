"""智能行程运行的低基数、安全可聚合观测。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from app.core.logger import app_logger as logger

ItineraryResult = Literal["complete", "partial", "failed"]
ItineraryToolOutcome = Literal["success", "degraded", "failed", "timeout"]

LOG_PREFIX = "ITINERARY_RUN"
ITINERARY_PRIMARY_TOOL_NAMES = frozenset({"search_flights", "search_trains"})
ITINERARY_PRODUCT_TOOL_NAMES = frozenset(
    {
        "local_place_search",
        "route_compare",
        "search_flights",
        "search_trains",
        "weather_forecast",
    }
)
_OUTCOME_NAMES = ("success", "degraded", "failed", "timeout")
_TIMEOUT_ERROR_CODES = frozenset({"call_timeout", "connect_timeout", "tool_timeout", "upstream_timeout"})
_UPSTREAM_ERROR_CODES = frozenset(
    {
        "network_error",
        "rate_limit",
        "search_unavailable",
        "upstream_error",
        "upstream_unavailable",
    }
)
_ARGUMENT_ERROR_CODES = frozenset(
    {
        "argument_repair_coalesced",
        "argument_repair_constraint_violation",
        "argument_repair_deferred_to_next_round",
        "argument_repair_exhausted",
        "arguments_json_invalid",
        "arguments_schema_invalid",
        "invalid_arguments",
        "remote_invalid_arguments",
        "runtime_argument_guard_failed",
    }
)
_INTERNAL_ERROR_CODES = frozenset({"internal_error", "tool_execution_failed"})
_SAFE_RUN_STATUSES = frozenset(
    {
        "completed",
        "error",
        "incomplete",
        "interrupted",
        "limit_reached",
    }
)
_SAFE_FINISH_REASONS = frozenset(
    {
        "completed",
        "max_steps",
        "max_tool_calls",
        "stop",
        "timeout",
        "tool_calls",
        "unknown",
    }
)
_SAFE_LIMIT_REASONS = frozenset({"max_steps", "max_tool_calls", "timeout"})


@dataclass(frozen=True)
class ItineraryToolObservation:
    """仅保留固定分类，不持有工具参数、第三方返回或原始错误文本。"""

    tool_name: str
    outcome: ItineraryToolOutcome
    error_category: str | None
    duration_ms: int | None
    controlled_repair: bool
    travel_budget_exhausted: bool
    server_budget_exhausted: bool
    upstream_error: bool
    repair_required: bool = False
    repair_retryable: bool = False
    repair_requires_user_input: bool = False
    repair_retry_exhausted: bool = False


def build_itinerary_tool_observation(record: Any) -> ItineraryToolObservation | None:
    tool_name = getattr(record, "tool_name", None)
    if tool_name not in ITINERARY_PRODUCT_TOOL_NAMES:
        return None
    result = getattr(record, "result", None)
    return build_itinerary_tool_observation_from_values(
        tool_name=tool_name,
        status=getattr(result, "status", None),
        duration_ms=getattr(result, "duration_ms", None),
        output_data=getattr(result, "data", None),
    )


def build_itinerary_tool_observation_from_values(
    *,
    tool_name: str,
    status: str | None,
    duration_ms: Any,
    output_data: Any,
) -> ItineraryToolObservation | None:
    if tool_name not in ITINERARY_PRODUCT_TOOL_NAMES:
        return None
    data = output_data if isinstance(output_data, dict) else {}
    error_code = data.get("error_code") if isinstance(data.get("error_code"), str) else None
    repair = data.get("repair")
    controlled_repair = isinstance(repair, dict) or isinstance(data.get("resolves_repair_id"), str)
    error_category = normalize_error_category(error_code)
    if error_category == "timeout":
        outcome: ItineraryToolOutcome = "timeout"
    elif isinstance(repair, dict) and repair.get("retryable") is True:
        outcome = "degraded"
    elif status in {"success", "degraded"}:
        outcome = status
    else:
        outcome = "failed"
    return ItineraryToolObservation(
        tool_name=tool_name,
        outcome=outcome,
        error_category=error_category,
        duration_ms=_safe_duration(duration_ms),
        controlled_repair=controlled_repair,
        travel_budget_exhausted=error_code == "travel_run_budget_exhausted",
        server_budget_exhausted=error_code == "server_run_budget_exhausted",
        upstream_error=error_category == "upstream_error",
        repair_required=isinstance(repair, dict),
        repair_retryable=isinstance(repair, dict) and repair.get("retryable") is True,
        repair_requires_user_input=isinstance(repair, dict) and repair.get("requires_user_input") is True,
        repair_retry_exhausted=isinstance(repair, dict) and repair.get("retry_exhausted") is True,
    )


def build_itinerary_run_payload(
    *,
    run_id: str,
    model_id: str,
    provider: str,
    content_blocks: list[Any],
    tool_observations: list[ItineraryToolObservation],
    run_status: str,
    finish_reason: str | None,
    limit_reason: str | None,
    total_duration_ms: int,
    total_steps: int,
    total_tool_calls: int,
) -> dict[str, Any] | None:
    """构造固定字段的行程终态；普通天气、地点和通勤不进入行程指标。"""

    has_itinerary_block = _itinerary_block_status(content_blocks) is not None
    has_primary_tool = any(item.tool_name in ITINERARY_PRIMARY_TOOL_NAMES for item in tool_observations)
    if not has_itinerary_block and not has_primary_tool:
        return None

    tool_outcomes: dict[str, dict[str, int]] = {}
    for tool_name in sorted({item.tool_name for item in tool_observations}):
        counts = Counter(item.outcome for item in tool_observations if item.tool_name == tool_name)
        tool_outcomes[tool_name] = {outcome: counts.get(outcome, 0) for outcome in _OUTCOME_NAMES}

    return {
        "event": "itinerary_run",
        "schema_version": 1,
        "run_id": _safe_identifier(run_id, max_chars=100),
        "model_id": _safe_identifier(model_id, max_chars=200),
        "provider": _safe_identifier(provider, max_chars=100).lower(),
        "result": classify_itinerary_result(
            content_blocks,
            run_status=run_status,
        ),
        "run_status": run_status if run_status in _SAFE_RUN_STATUSES else "error",
        "finish_reason": finish_reason if finish_reason in _SAFE_FINISH_REASONS else "unknown",
        "limit_reason": limit_reason if limit_reason in _SAFE_LIMIT_REASONS else None,
        "total_duration_ms": _safe_duration(total_duration_ms) or 0,
        "total_steps": _safe_count(total_steps),
        "total_tool_calls": _safe_count(total_tool_calls),
        "product_tool_call_count": len(tool_observations),
        "tool_outcomes": tool_outcomes,
        "upstream_error_count": sum(item.upstream_error for item in tool_observations),
        "controlled_argument_repair_count": sum(item.controlled_repair for item in tool_observations),
        "repair_required_count": sum(item.repair_required for item in tool_observations),
        "repair_retryable_count": sum(item.repair_retryable for item in tool_observations),
        "repair_requires_user_input_count": sum(item.repair_requires_user_input for item in tool_observations),
        "repair_retry_exhausted_count": sum(item.repair_retry_exhausted for item in tool_observations),
        "travel_budget_exhausted_count": sum(item.travel_budget_exhausted for item in tool_observations),
        "server_budget_exhausted_count": sum(item.server_budget_exhausted for item in tool_observations),
    }


def classify_itinerary_result(
    content_blocks: list[Any],
    *,
    run_status: str,
) -> ItineraryResult:
    block_status = _itinerary_block_status(content_blocks)
    if run_status == "error":
        return "failed"
    if run_status in {"interrupted", "limit_reached", "incomplete"}:
        return "partial" if block_status is not None else "failed"
    if block_status == "success":
        return "complete"
    if block_status is not None:
        return "partial"
    return "failed"


def normalize_error_category(error_code: str | None) -> str | None:
    if error_code in _TIMEOUT_ERROR_CODES:
        return "timeout"
    if error_code in _UPSTREAM_ERROR_CODES:
        return "upstream_error"
    if error_code == "travel_run_budget_exhausted":
        return "travel_budget_exhausted"
    if error_code == "server_run_budget_exhausted":
        return "server_budget_exhausted"
    if error_code in _ARGUMENT_ERROR_CODES:
        return "argument_repair"
    if error_code in _INTERNAL_ERROR_CODES:
        return "internal_error"
    return "other" if error_code else None


def emit_itinerary_run_log(payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    try:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        logger.info(f"{LOG_PREFIX} {serialized}")
    except Exception:
        logger.warning("智能行程可观测性日志写入失败，已忽略")


def aggregate_itinerary_stability(
    rows: dict[str, Any],
    *,
    created_from: Any,
    created_to: Any,
    model_id: str | None,
) -> dict[str, Any]:
    """把有界 ORM 行聚合为管理员稳定性视图，只输出计数、分位数和固定分类。"""

    sessions = list(rows.get("sessions") or [])
    messages = {row.id: row for row in rows.get("messages") or []}
    sessions_by_id = {row.id: row for row in sessions}
    sessions_by_message = {row.message_id: row for row in sessions if row.message_id}

    def session_for(log: Any) -> Any:
        return sessions_by_id.get(log.trace_id) or sessions_by_message.get(log.message_id)

    def key_for(log: Any) -> str:
        session = session_for(log)
        if session is not None:
            return f"run:{session.id}"
        if log.trace_id:
            return f"trace:{log.trace_id}"
        if log.message_id:
            return f"message:{log.message_id}"
        return f"log:{log.id}"

    anchor_keys = {key_for(log) for log in rows.get("anchors") or []}
    logs_by_run: dict[str, list[Any]] = {key: [] for key in anchor_keys}
    for log in rows.get("tool_logs") or []:
        key = key_for(log)
        if key in logs_by_run:
            logs_by_run[key].append(log)

    completed_runs: list[dict[str, Any]] = []
    excluded_running_count = 0
    excluded_interrupted_count = 0
    excluded_unlinked_count = 0
    for key in sorted(anchor_keys):
        logs = logs_by_run.get(key, [])
        anchor = next((item for item in rows.get("anchors") or [] if key_for(item) == key), None)
        if anchor is None:
            continue
        session = session_for(anchor)
        if session is None:
            excluded_unlinked_count += 1
            continue
        if session is not None and session.status == "running":
            excluded_running_count += 1
            continue
        if session.status == "interrupted":
            excluded_interrupted_count += 1
            continue
        observations = [
            observation
            for log in logs
            if (
                observation := build_itinerary_tool_observation_from_values(
                    tool_name=log.tool_name,
                    status=log.status,
                    duration_ms=log.duration_ms,
                    output_data=log.output_data,
                )
            )
            is not None
        ]
        message = messages.get(session.message_id if session is not None else anchor.message_id)
        content_blocks = message.content if message is not None and isinstance(message.content, list) else []
        run_model_id = session.model_id if session is not None else anchor.model_id
        run_provider = session.provider if session is not None else anchor.provider
        result = classify_itinerary_result(
            content_blocks,
            run_status=session.status,
        )
        completed_runs.append(
            {
                "model_id": _safe_identifier(run_model_id, max_chars=200),
                "provider": _safe_identifier(run_provider, max_chars=100).lower(),
                "result": result,
                "duration_ms": _safe_duration(session.total_duration_ms) if session is not None else None,
                "agent_limit_reached": session is not None and session.status == "limit_reached",
                "observations": observations,
            }
        )

    return {
        "scope": {
            "created_from": created_from.isoformat(),
            "created_to": created_to.isoformat(),
            "timezone": "Asia/Shanghai",
            "sample_definition": "terminal_run_with_travel_tool",
            "excluded_running_count": excluded_running_count,
            "excluded_interrupted_count": excluded_interrupted_count,
            "excluded_unlinked_count": excluded_unlinked_count,
        },
        "summary": _aggregate_run_group(completed_runs),
        "by_model": [
            {
                "model_id": current_model,
                **_aggregate_run_group([row for row in completed_runs if row["model_id"] == current_model]),
            }
            for current_model in sorted({row["model_id"] for row in completed_runs})
        ],
        "by_tool": _aggregate_tools(completed_runs),
    }


def _aggregate_run_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [item for run in runs for item in run["observations"]]
    results = Counter(run["result"] for run in runs)
    outcomes = Counter(item.outcome for item in observations)
    return {
        "itinerary": {
            "total": len(runs),
            "complete": results.get("complete", 0),
            "partial": results.get("partial", 0),
            "failed": results.get("failed", 0),
        },
        "run_latency_ms": _percentiles([run["duration_ms"] for run in runs]),
        "product_tools": {
            "total": len(observations),
            "success": outcomes.get("success", 0),
            "degraded": outcomes.get("degraded", 0),
            "failed": outcomes.get("failed", 0) + outcomes.get("timeout", 0),
            "timeout": outcomes.get("timeout", 0),
        },
        "tool_latency_ms": _percentiles([item.duration_ms for item in observations]),
        "signals": {
            "upstream_error": sum(item.upstream_error for item in observations),
            "repair_required": sum(item.repair_required for item in observations),
            "repair_retryable": sum(item.repair_retryable for item in observations),
            "repair_requires_user_input": sum(item.repair_requires_user_input for item in observations),
            "repair_retry_exhausted": sum(item.repair_retry_exhausted for item in observations),
            "travel_budget_exhausted": sum(item.travel_budget_exhausted for item in observations),
            "server_budget_exhausted": sum(item.server_budget_exhausted for item in observations),
            "agent_limit_reached": sum(run["agent_limit_reached"] for run in runs),
            "other_safe_error": sum(item.error_category == "other" for item in observations),
        },
    }


def _aggregate_tools(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = [item for run in runs for item in run["observations"]]
    result = []
    for tool_name in sorted({item.tool_name for item in observations}):
        items = [item for item in observations if item.tool_name == tool_name]
        outcomes = Counter(item.outcome for item in items)
        result.append(
            {
                "tool_name": tool_name,
                "calls": {
                    "total": len(items),
                    "success": outcomes.get("success", 0),
                    "degraded": outcomes.get("degraded", 0),
                    "failed": outcomes.get("failed", 0) + outcomes.get("timeout", 0),
                    "timeout": outcomes.get("timeout", 0),
                },
                "latency_ms": _percentiles([item.duration_ms for item in items]),
                "upstream_error": sum(item.upstream_error for item in items),
                "budget_exhausted": sum(item.travel_budget_exhausted or item.server_budget_exhausted for item in items),
            }
        )
    return result


def _percentiles(values: list[int | None]) -> dict[str, int | None]:
    samples = sorted(value for value in values if value is not None)
    if not samples:
        return {"sample_count": 0, "p50_ms": None, "p95_ms": None}

    def nearest_rank(percentile: float) -> int:
        return samples[max(0, math.ceil(percentile * len(samples)) - 1)]

    return {
        "sample_count": len(samples),
        "p50_ms": nearest_rank(0.5),
        "p95_ms": nearest_rank(0.95),
    }


def _itinerary_block_status(content_blocks: list[Any]) -> str | None:
    for block in reversed(content_blocks):
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type != "itinerary_results":
            continue
        status = block.get("status") if isinstance(block, dict) else getattr(block, "status", None)
        return status if status in {"success", "degraded"} else "degraded"
    return None


def _safe_identifier(value: Any, *, max_chars: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[A-Za-z0-9._:/-]{{1,{max_chars}}}", value):
        return "unknown"
    return value


def _safe_duration(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(max(int(value), 0), 86_400_000)


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return min(max(value, 0), 1_000)
