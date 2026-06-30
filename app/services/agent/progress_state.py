"""Agent progress compact snapshot 的纯函数 reducer。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

MAX_EVIDENCE_ITEMS = 12
MAX_TOOL_DIGESTS = 20


def empty_progress_state(*, run_id: str, message_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "message_id": message_id,
        "status": "running",
        "progress": None,
        "plan": None,
        "tool_digests": [],
        "evidence": [],
        "updated_at": None,
    }


def apply_progress_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type")
    next_state = deepcopy(state)

    if event_type == "run_progress_updated" and event.get("protocol_version") == 2:
        next_state["progress"] = {
            "phase": event.get("phase"),
            "label": _truncate(event.get("label"), 40),
            "completed_steps": event.get("completed_steps"),
            "total_steps": event.get("total_steps"),
            "completed_tool_calls": event.get("completed_tool_calls"),
            "max_tool_calls": event.get("max_tool_calls"),
        }
    elif event_type == "plan_snapshot" and event.get("protocol_version") == 2:
        next_state["plan"] = {
            "plan_id": event.get("plan_id"),
            "revision": event.get("revision", 0),
            "items": [_normalize_plan_item(item) for item in event.get("items", [])],
        }
    elif event_type == "plan_step_updated" and event.get("protocol_version") == 2:
        _apply_plan_step_update(next_state, event)
    elif event_type == "tool_result_digest" and event.get("protocol_version") == 2:
        _upsert_by_key(
            next_state["tool_digests"],
            _normalize_tool_digest(event),
            key="tool_call_id",
            limit=MAX_TOOL_DIGESTS,
        )
    elif event_type == "evidence_item_upserted" and event.get("protocol_version") == 2:
        evidence = event.get("evidence")
        if isinstance(evidence, dict):
            _upsert_by_key(
                next_state["evidence"],
                _normalize_evidence(evidence),
                key="id",
                limit=None,
            )
            next_state["evidence"] = _cap_evidence(next_state["evidence"])
    elif event_type in {"run_completed", "run_failed", "run_interrupted"}:
        next_state["status"] = _terminal_status(event)
    else:
        return state

    next_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return next_state


def _apply_plan_step_update(state: dict[str, Any], event: dict[str, Any]) -> None:
    plan = state.get("plan")
    if not plan or plan.get("plan_id") != event.get("plan_id"):
        return

    revision = event.get("revision", 0)
    if revision <= plan.get("revision", 0):
        return

    item = _normalize_plan_item(event.get("item", {}))
    items = plan.setdefault("items", [])
    for index, existing in enumerate(items):
        if existing.get("id") == item.get("id"):
            items[index] = item
            break
    else:
        items.append(item)
    plan["revision"] = revision


def _normalize_plan_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "title": _truncate(item.get("title"), 80),
        "status": item.get("status", "pending"),
        "kind": item.get("kind", "other"),
        "summary": _truncate(item.get("summary"), 120) if item.get("summary") else None,
        "tool_names": _string_list(item.get("tool_names"), limit=8, max_chars=40),
        "evidence_item_ids": _string_list(item.get("evidence_item_ids"), limit=12, max_chars=80),
    }


def _normalize_tool_digest(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_call_id": str(event.get("tool_call_id", "")),
        "tool_name": str(event.get("tool_name", "")),
        "status": event.get("status", "success"),
        "title": _truncate(event.get("title"), 80),
        "summary": _truncate(event.get("summary"), 120),
        "key_findings": _string_list(event.get("key_findings"), limit=5, max_chars=80),
        "source_refs": _string_list(event.get("source_refs"), limit=12, max_chars=80),
        "truncated": bool(event.get("truncated", False)),
    }


def _normalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(evidence.get("id", "")),
        "kind": evidence.get("kind", "tool"),
        "status": evidence.get("status", "candidate"),
        "title": _truncate(evidence.get("title"), 80),
        "url": _truncate(evidence.get("url"), 500) if evidence.get("url") else None,
        "domain": _truncate(evidence.get("domain"), 120) if evidence.get("domain") else None,
        "claim": _truncate(evidence.get("claim"), 120),
        "snippet": _truncate(evidence.get("snippet"), 180) if evidence.get("snippet") else None,
        "used_by_final_answer": bool(evidence.get("used_by_final_answer", False)),
    }


def _upsert_by_key(items: list[dict[str, Any]], item: dict[str, Any], *, key: str, limit: int | None) -> None:
    item_key = item.get(key)
    for index, existing in enumerate(items):
        if existing.get(key) == item_key:
            items[index] = item
            break
    else:
        items.append(item)

    if limit is not None and len(items) > limit:
        del items[: len(items) - limit]


def _cap_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(items) <= MAX_EVIDENCE_ITEMS:
        return items

    indexed_items = list(enumerate(items))
    kept = sorted(
        indexed_items,
        key=lambda entry: (_evidence_priority(entry[1]), entry[0]),
        reverse=True,
    )[:MAX_EVIDENCE_ITEMS]
    return [item for _, item in sorted(kept, key=lambda entry: entry[0])]


def _evidence_priority(item: dict[str, Any]) -> int:
    status = item.get("status")
    if status == "used" or item.get("used_by_final_answer"):
        return 5
    if status == "read_success":
        return 4
    if status == "selected":
        return 3
    if status in {"read_degraded", "read_failed"}:
        return 2
    return 1


def _terminal_status(event: dict[str, Any]) -> str:
    if event.get("type") == "run_failed":
        return "failed"
    if event.get("type") == "run_interrupted":
        return "interrupted"
    finish_reason = event.get("finish_reason")
    if finish_reason == "limit_reached":
        return "limit_reached"
    if finish_reason == "incomplete":
        return "incomplete"
    return "completed"


def _truncate(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text[:max_chars]


def _string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_truncate(item, max_chars) for item in value[:limit]]
