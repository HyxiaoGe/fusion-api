"""模型选择工具后、真实执行前的运行上下文握手策略。"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.schemas.chat import FlightResultsBlock, TrainResultsBlock
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


@dataclass
class ToolRuntimeContext:
    geolocation: Geolocation | None = None
    step_number: int = 0
    argument_repair_state: dict[str, dict[str, Any]] | None = None
    route_city_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    remaining_agent_steps: int | None = None
    remaining_agent_tool_calls: int | None = None
    remaining_agent_tool_calls_before_round: int | None = None
    remaining_agent_tool_calls_after_batch: int | None = None
    remaining_argument_repair_slots: int | None = None


@dataclass(frozen=True)
class BlockedToolContext:
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class ToolContextResolution:
    executable_calls: list[dict]
    blocked_calls: dict[str, BlockedToolContext] = field(default_factory=dict)
    runtime_context: ToolRuntimeContext = field(default_factory=ToolRuntimeContext)


def enrich_tool_runtime_context(
    context: ToolRuntimeContext,
    *,
    messages: list[dict],
    state: AgentLoopState,
    step_number: int,
) -> ToolRuntimeContext:
    """补入本轮步骤号与 run 级通用修参状态。"""

    del messages
    context.step_number = step_number
    context.argument_repair_state = state.argument_repair_state
    context.route_city_hints = _trusted_route_city_hints(state.content_blocks)
    return context


def _trusted_route_city_hints(content_blocks: list[Any]) -> dict[str, tuple[str, ...]]:
    """只从当前 run 已投影的班次结果中提取精确站点城市，冲突值不进入修参通道。"""

    collected: dict[str, dict[str, set[str]]] = {}
    for block in content_blocks:
        if isinstance(block, FlightResultsBlock):
            options = block.flights
        elif isinstance(block, TrainResultsBlock):
            options = block.trains
        else:
            continue
        expected_cities = {
            "departure": _normalize_route_city(block.origin),
            "arrival": _normalize_route_city(block.destination),
        }
        for option in options[:5]:
            for endpoint_field in ("departure", "arrival"):
                endpoint = _value(option, endpoint_field)
                station_name = _value(endpoint, "station_name")
                city = _value(endpoint, "city")
                if not isinstance(station_name, str) or not isinstance(city, str):
                    continue
                station_name = station_name.strip()
                city = city.strip()
                if not station_name or not city or len(station_name) > 120 or len(city) > 80:
                    continue
                normalized_city = _normalize_route_city(city)
                if not normalized_city or normalized_city != expected_cities[endpoint_field]:
                    continue
                variants = collected.setdefault(station_name, {}).setdefault(normalized_city, set())
                variants.add(city)
    hints: dict[str, tuple[str, ...]] = {}
    for station_name, normalized_cities in list(collected.items())[:40]:
        if len(normalized_cities) != 1:
            continue
        variants = next(iter(normalized_cities.values()))
        hints[station_name] = (min(variants, key=lambda value: (len(value), value)),)
    return hints


def _normalize_route_city(value: str) -> str:
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())
    for suffix in ("自治州", "地区", "市", "盟"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


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
    if tool_call.get("name") == "weather_forecast" and args.get("location_source") == "current_location":
        return "local_weather"
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
        "local_weather": "查询当前位置所在行政区的天气预报",
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
