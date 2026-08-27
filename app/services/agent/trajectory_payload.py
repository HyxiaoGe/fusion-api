"""轨迹账本 payload 的事件类型白名单与有界脱敏。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from app.schemas.trajectory import TrajectoryCapabilityResolution

MAX_LEDGER_TEXT_LENGTH = 512
MAX_LEDGER_LIST_ITEMS = 50

_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "type",
        "run_id",
        "parent_run_id",
        "step_id",
        "parent_step_id",
        "tool_call_id",
        "sequence",
        "trace_id",
        "ts",
    }
)

_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "run_started": frozenset({"conversation_id", "message_id", "task_id", "model", "tools", "capability_resolution"}),
    "step_started": frozenset({"step_number"}),
    "tool_call_started": frozenset({"tool_name", "plan_item_id"}),
    "tool_call_delta": frozenset({"tool_name"}),
    "tool_call_completed": frozenset({"tool_name", "status", "duration_ms", "plan_item_id"}),
    "step_completed": frozenset({"step_number", "tool_call_count", "duration_ms"}),
    "run_limit_reached": frozenset({"reason"}),
    "run_interrupted": frozenset({"reason"}),
    "run_failed": frozenset({"error_code", "message"}),
    "run_completed": frozenset({"total_steps", "total_tool_calls", "finish_reason"}),
    "llm_round_started": frozenset({"llm_round_id", "round_index", "model", "provider", "system_prompt_fingerprint"}),
    "llm_round_first_output_delta": frozenset({"llm_round_id", "delta_kind", "ttft_ms"}),
    "llm_round_completed": frozenset(
        {
            "llm_round_id",
            "status",
            "finish_reason",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "ttft_ms",
            "duration_ms",
        }
    ),
    "llm_round_failed": frozenset({"llm_round_id", "status", "error_code", "message"}),
    "llm_round_cancelled": frozenset({"llm_round_id", "status", "reason"}),
    "retrieval_started": frozenset({"retrieval_id", "query_summary"}),
    "retrieval_completed": frozenset({"retrieval_id", "status", "document_count", "duration_ms"}),
    "retrieval_failed": frozenset({"retrieval_id", "status", "error_code", "message"}),
    "retrieval_cancelled": frozenset({"retrieval_id", "status", "reason"}),
    "tool_attempt_started": frozenset({"tool_attempt_id", "tool_name", "attempt_index"}),
    "tool_attempt_completed": frozenset({"tool_attempt_id", "status", "error_code", "duration_ms"}),
    "suggested_questions_pending": frozenset({"protocol_version", "message_id", "revision", "status"}),
    "run_progress_updated": frozenset(
        {
            "protocol_version",
            "phase",
            "label",
            "completed_steps",
            "total_steps",
            "completed_tool_calls",
            "max_tool_calls",
        }
    ),
    "plan_snapshot": frozenset({"protocol_version", "plan_id", "mode", "source", "revision", "reason", "items"}),
    "plan_step_updated": frozenset({"protocol_version", "plan_id", "mode", "source", "revision", "reason", "item"}),
    "tool_result_digest": frozenset(
        {
            "protocol_version",
            "tool_name",
            "status",
            "title",
            "summary",
            "key_findings",
            "source_refs",
            "truncated",
            "repair_state",
            "repair_id",
            "plan_item_id",
        }
    ),
    "evidence_item_upserted": frozenset({"protocol_version", "evidence"}),
    # 完整 content block 不属于 P0 脱敏账本，只保留事件存在性与协议版本。
    "content_block_upserted": frozenset({"protocol_version"}),
    "content_block_discarded": frozenset({"protocol_version", "block_id"}),
    "system_prompt_prepared": frozenset(
        {
            "protocol_version",
            "status",
            "source",
            "template_version",
            "section_ids",
            "detail_status",
            "fingerprint",
            "char_count",
            "duration_ms",
            "error_code",
            "message",
        }
    ),
    "context_status_updated": frozenset(
        {
            "protocol_version",
            "message_id",
            "phase",
            "status",
            "round_index",
            "window_tokens",
            "estimated_tokens_before",
            "estimated_tokens_after",
            "actual_prompt_tokens",
            "removed_turns",
            "removed_messages",
            "removed_tool_transactions",
        }
    ),
    "context_required": frozenset(
        {"protocol_version", "context_type", "request_id", "purpose", "reason", "expires_at"}
    ),
    "context_result": frozenset({"protocol_version", "context_type", "request_id", "status"}),
}

_PLAN_ITEM_FIELDS = frozenset(
    {
        "id",
        "title",
        "phase_id",
        "phase_title",
        "status",
        "kind",
        "summary",
        "tool_names",
        "evidence_item_ids",
        "depends_on",
        "planned_tools",
    }
)
_PLAN_ITEM_LIST_FIELDS = frozenset({"tool_names", "evidence_item_ids", "depends_on", "planned_tools"})
_EVIDENCE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "status",
        "title",
        "url",
        "domain",
        "claim",
        "snippet",
        "used_by_final_answer",
        "citation_index",
    }
)
_LIST_FIELDS = frozenset({"tools", "key_findings", "source_refs", "section_ids"})
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|access[_-]?token|token|password|secret)\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;]+"
)


class UnsupportedTrajectoryEventError(ValueError):
    """事件类型尚未定义账本白名单。"""


def _bounded_text(value: Any) -> str:
    redacted = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(value))
    return redacted[:MAX_LEDGER_TEXT_LENGTH]


def _bounded_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_bounded_text(item) for item in value[:MAX_LEDGER_LIST_ITEMS]]


def _safe_url(value: Any) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(_bounded_text(value))
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return None
    hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    try:
        netloc = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    except ValueError:
        return None
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))[:MAX_LEDGER_TEXT_LENGTH]


def _sanitize_plan_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    item: dict[str, Any] = {}
    for field in _PLAN_ITEM_FIELDS:
        if field not in value:
            continue
        if field in _PLAN_ITEM_LIST_FIELDS:
            item[field] = _bounded_list(value[field])
        else:
            item[field] = _sanitize_scalar(value[field])
    return item


def _sanitize_plan_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_sanitize_plan_item(item) for item in value[:MAX_LEDGER_LIST_ITEMS]]


def _sanitize_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    evidence: dict[str, Any] = {}
    for field in _EVIDENCE_FIELDS:
        if field not in value:
            continue
        evidence[field] = _safe_url(value[field]) if field == "url" else _sanitize_scalar(value[field])
    return evidence


_CAPABILITY_RESOLUTION_FIELDS = (
    "schema_version",
    "router_version",
    "package_id",
    "confidence",
    "resolution_mode",
    "reason_codes",
    "external_tool_names",
    "effective_plan_mode",
    "include_current_date",
    "network_boundary_required",
    "bundle_fingerprint",
)


def _sanitize_capability_resolution(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    candidate = {field: value[field] for field in _CAPABILITY_RESOLUTION_FIELDS if field in value}
    try:
        return TrajectoryCapabilityResolution.model_validate(candidate).model_dump()
    except ValidationError:
        return None


def _sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value)


_SPECIAL_SANITIZERS: dict[str, Callable[[Any], Any]] = {
    "items": _sanitize_plan_items,
    "item": _sanitize_plan_item,
    "evidence": _sanitize_evidence,
    "capability_resolution": _sanitize_capability_resolution,
}


def build_trajectory_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """仅从显式事件白名单构造可落库 DTO；未知事件拒绝原样入库。"""
    event_type = payload.get("type")
    if not isinstance(event_type, str) or event_type not in _EVENT_FIELDS:
        raise UnsupportedTrajectoryEventError(f"不支持的轨迹事件类型: {event_type!r}")

    stored: dict[str, Any] = {}
    allowed_fields = _COMMON_FIELDS | _EVENT_FIELDS[event_type]
    for field in allowed_fields:
        if field not in payload:
            continue
        sanitizer = _SPECIAL_SANITIZERS.get(field)
        if sanitizer is not None:
            stored[field] = sanitizer(payload[field])
        elif field in _LIST_FIELDS:
            stored[field] = _bounded_list(payload[field])
        else:
            stored[field] = _sanitize_scalar(payload[field])
    return stored
