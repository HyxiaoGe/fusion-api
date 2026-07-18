"""模型选择工具后、真实执行前的运行上下文握手策略。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.services.agent.context_broker import (
    Geolocation,
    PendingContextRequest,
    ResolvedContext,
    create_context_request,
    wait_for_context_result,
)
from app.services.stream.agent_loop_state import AgentLoopState

_CONTEXT_TYPE = "geolocation"
_CONTEXT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ToolRuntimeContext:
    geolocation: Geolocation | None = None


@dataclass(frozen=True)
class BlockedToolContext:
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class ToolContextResolution:
    executable_calls: list[dict]
    blocked_calls: dict[str, BlockedToolContext] = field(default_factory=dict)
    runtime_context: ToolRuntimeContext = field(default_factory=ToolRuntimeContext)


def _arguments(tool_call: dict) -> dict[str, Any]:
    raw = tool_call.get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _geolocation_purpose(tool_call: dict) -> str | None:
    args = _arguments(tool_call)
    if tool_call.get("name") == "local_place_search" and args.get("anchor_source") == "current_location":
        return "nearby_search"
    if tool_call.get("name") == "route_compare":
        if args.get("origin_source") == "current_location":
            return "route_origin"
        if args.get("destination_source") == "current_location":
            return "route_destination"
    return None


def _dependent_calls(tool_calls: list[dict]) -> tuple[list[dict], list[dict], str | None]:
    dependent: list[dict] = []
    independent: list[dict] = []
    purpose = None
    for tool_call in tool_calls:
        call_purpose = _geolocation_purpose(tool_call)
        if call_purpose is None:
            independent.append(tool_call)
            continue
        dependent.append(tool_call)
        purpose = purpose or call_purpose
    return dependent, independent, purpose


def _blocked(dependent: list[dict], *, status: str, reason: str | None) -> dict[str, BlockedToolContext]:
    return {str(tool_call.get("id", "")): BlockedToolContext(status=status, reason=reason) for tool_call in dependent}


async def resolve_tool_context(
    *,
    tool_calls: list[dict],
    state: AgentLoopState,
    emitter: Any,
    user_id: str,
    conversation_id: str,
    message_id: str,
    run_id: str,
    task_id: str,
    clock: Callable[[], float] = time.time,
    request_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    create_request_fn: Callable[..., Awaitable[PendingContextRequest]] = create_context_request,
    wait_result_fn: Callable[[PendingContextRequest], Awaitable[ResolvedContext]] = wait_for_context_result,
) -> ToolContextResolution:
    dependent, _independent, purpose = _dependent_calls(tool_calls)
    if not dependent:
        return ToolContextResolution(executable_calls=tool_calls)

    cached = state.runtime_contexts.get(_CONTEXT_TYPE)
    if isinstance(cached, Geolocation):
        return ToolContextResolution(
            executable_calls=tool_calls,
            runtime_context=ToolRuntimeContext(geolocation=cached),
        )

    unavailable_status = state.unavailable_contexts.get(_CONTEXT_TYPE)
    if unavailable_status:
        return ToolContextResolution(
            executable_calls=[call for call in tool_calls if call not in dependent],
            blocked_calls=_blocked(dependent, status=unavailable_status, reason="context_unavailable_for_run"),
        )

    started_at = clock()
    request_id = request_id_factory()
    expires_at = started_at + _CONTEXT_TIMEOUT_SECONDS
    reason = {
        "nearby_search": "搜索当前位置附近的地点",
        "route_origin": "使用当前位置作为路线起点",
        "route_destination": "使用当前位置作为路线终点",
    }.get(purpose, "完成当前工具调用需要位置信息")
    try:
        pending = await create_request_fn(
            request_id=request_id,
            context_type=_CONTEXT_TYPE,
            purpose=purpose,
            reason=reason,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            run_id=run_id,
            task_id=task_id,
            expires_at=expires_at,
        )
    except Exception:  # noqa: BLE001 — broker 故障只降级，不暴露内部异常
        state.unavailable_contexts[_CONTEXT_TYPE] = "unavailable"
        return ToolContextResolution(
            executable_calls=[call for call in tool_calls if call not in dependent],
            blocked_calls=_blocked(dependent, status="unavailable", reason="context_broker_unavailable"),
        )
    await emitter.context_required(
        request_id=request_id,
        context_type=_CONTEXT_TYPE,
        purpose=purpose,
        reason=reason,
        expires_at=expires_at,
    )
    try:
        resolved = await wait_result_fn(pending)
    except Exception:  # noqa: BLE001 — 等待链路故障必须安全续接同一 run
        resolved = ResolvedContext(
            request_id=request_id,
            status="unavailable",
            reason="context_broker_unavailable",
        )
    state.record_context_wait(clock() - started_at)
    await emitter.context_result(
        request_id=request_id,
        context_type=_CONTEXT_TYPE,
        status=resolved.status,
    )

    if resolved.status == "provided" and resolved.location is not None:
        state.runtime_contexts[_CONTEXT_TYPE] = resolved.location
        return ToolContextResolution(
            executable_calls=tool_calls,
            runtime_context=ToolRuntimeContext(geolocation=resolved.location),
        )

    state.unavailable_contexts[_CONTEXT_TYPE] = resolved.status
    return ToolContextResolution(
        executable_calls=[call for call in tool_calls if call not in dependent],
        blocked_calls=_blocked(dependent, status=resolved.status, reason=resolved.reason),
    )
