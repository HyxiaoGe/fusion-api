"""Agent 工具回合编排。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.schemas.chat import (
    FlightResultsBlock,
    ItineraryResultsBlock,
    RouteResultsBlock,
    ThinkingBlock,
    TrainResultsBlock,
    WeatherResultsBlock,
)
from app.schemas.content_block_registry import is_registered_rich_content_block
from app.services.search_read_planner import build_search_read_plan, format_search_read_plan_guidance
from app.services.source_candidate_ranker import (
    SearchResultForRanking,
    SourceSelectionPlan,
)
from app.services.source_evidence_ledger import build_selected_source_evidence_item, canonicalize_evidence_url
from app.services.stream.agent_loop_state import AgentLoopState, ProductToolOutcome
from app.services.stream.itinerary_observability import build_itinerary_tool_observation
from app.services.stream.itinerary_result_composer import compose_itinerary_result
from app.services.stream.plan_control import process_plan_control_calls
from app.services.stream.step_lifecycle import AgentStepContext, mark_tool_round_started
from app.services.stream.tool_context import (
    BlockedToolContext,
    ToolContextResolution,
    ToolRuntimeContext,
    enrich_tool_runtime_context,
    resolve_tool_context,
)
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_executor import (
    build_successful_call_signature,
    is_local_argument_preflight_failure,
)
from app.services.stream_state_service import StreamWriteTerminalError

logger = logging.getLogger(__name__)

# 本地预校验不消耗真实工具额度，但仍限制单轮事件、日志与上下文放大。
MAX_LOCAL_PREFLIGHT_ATTEMPTS_PER_ROUND = 4


@dataclass(frozen=True)
class ToolRoundOutcome:
    tool_call_count: int
    tool_names: list[str]
    no_progress_search_results: tuple[bool, ...] = ()
    product_result_count: int = 0
    itinerary_result_count: int = 0
    product_outcomes: tuple[ProductToolOutcome, ...] = ()
    control_repair_exhausted: bool = False


@dataclass(frozen=True)
class ToolRoundRequest:
    db: Any
    assistant_message_id: str
    conversation_id: str
    user_id: str
    model_id: str
    provider: str
    content_blocks: list
    messages: list[dict]
    tool_calls: list[dict]
    reasoning_buf: str
    should_use_reasoning: bool
    step_context: AgentStepContext
    step_number: int
    run_id: str
    emitter: Any
    session_cache: Any
    network_budget: Any
    call_kwargs: dict
    persist_message_fn: Callable[..., Any]
    execute_tools_fn: Callable[..., Awaitable[list[ToolExecutionRecord]]]
    complete_step_fn: Callable[..., Awaitable[Any]]
    protocol_reasoning_buf: str | None = None
    assistant_message_sequence: int | None = None
    on_tools_executed: Callable[[int], None] | None = None
    completed_tool_calls: int | None = None
    max_tool_calls: int | None = None
    max_steps: int | None = None
    clock: Callable[[], float] = time.time
    tool_handlers: dict[str, Any] | None = None
    announced_tool_names: frozenset[str] | None = None
    task_id: str = ""
    agent_state: AgentLoopState | None = None
    output_deferred: bool = False
    resolve_tool_context_fn: Callable[..., Awaitable[ToolContextResolution]] = resolve_tool_context


def build_assistant_tool_message(
    *,
    tool_calls: list[dict],
    reasoning_buf: str,
    should_use_reasoning: bool,
) -> dict:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"],
                },
            }
            for tool_call in tool_calls
        ],
    }
    if should_use_reasoning and reasoning_buf:
        message["reasoning_content"] = reasoning_buf
    return message


def restore_reasoning_after_tool_decision(call_kwargs: dict) -> None:
    extra_body = call_kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        return

    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        call_kwargs.pop("extra_body", None)


def append_tool_round_reasoning(request: ToolRoundRequest) -> None:
    if request.reasoning_buf:
        request.content_blocks.append(
            ThinkingBlock(
                type="thinking",
                id=request.step_context.thinking_block_id,
                thinking=request.reasoning_buf,
            )
        )


def persist_tool_round_checkpoint(request: ToolRoundRequest) -> None:
    persistence_kwargs = (
        {"sequence": request.assistant_message_sequence} if request.assistant_message_sequence is not None else {}
    )
    request.persist_message_fn(
        request.db,
        request.assistant_message_id,
        request.conversation_id,
        request.model_id,
        request.content_blocks,
        partial=True,
        **persistence_kwargs,
    )


async def execute_tool_round_tools(
    request: ToolRoundRequest,
    *,
    selected_tool_calls: list[dict] | None = None,
    runtime_context: ToolRuntimeContext | None = None,
) -> list[ToolExecutionRecord]:
    if selected_tool_calls is None:
        announced_tool_calls, _ = _partition_tool_calls_by_announcement(request)
        executed_tool_calls = _select_tool_calls_within_limit(request, announced_tool_calls)[0]
    else:
        executed_tool_calls = selected_tool_calls
    if not executed_tool_calls:
        if request.on_tools_executed is not None:
            request.on_tools_executed(0)
        return []

    execute_kwargs = {
        "trace_id": request.run_id,
        "step_number": request.step_number,
        "message_id": request.assistant_message_id,
        "emitter": request.emitter,
        "network_budget": request.network_budget,
    }
    if request.tool_handlers:
        execute_kwargs["tool_handlers"] = request.tool_handlers
    if runtime_context is not None:
        execute_kwargs["runtime_context"] = runtime_context
    if request.agent_state is not None:
        execute_kwargs["successful_tool_call_signatures"] = request.agent_state.successful_tool_call_signatures
    try:
        results = await request.execute_tools_fn(
            executed_tool_calls,
            request.conversation_id,
            request.user_id,
            request.model_id,
            request.provider,
            **execute_kwargs,
        )
    except BaseException:
        if request.on_tools_executed is not None:
            request.on_tools_executed(len(executed_tool_calls))
        raise
    else:
        if request.on_tools_executed is not None:
            request.on_tools_executed(_actual_tool_execution_count(executed_tool_calls, results))
        return results


async def update_tool_round_plan_started(request: ToolRoundRequest, *, tool_calls: list[dict] | None = None) -> None:
    planned_tool_calls = request.tool_calls if tool_calls is None else tool_calls
    await mark_tool_round_started(
        context=request.step_context,
        emitter=request.emitter,
        tool_call_count=len(planned_tool_calls),
        tool_names=[tool_call["name"] for tool_call in planned_tool_calls],
        tool_arguments=_tool_arguments(planned_tool_calls),
        completed_tool_calls=_completed_tool_calls_before_round(request),
        max_tool_calls=_max_tool_calls(request),
    )


def append_tool_round_messages(request: ToolRoundRequest, results: list[ToolExecutionRecord]) -> None:
    append_tool_round_messages_with_plan(request, results, source_plan=None)


def append_tool_round_messages_with_plan(
    request: ToolRoundRequest,
    results: list[ToolExecutionRecord],
    *,
    source_plan: SourceSelectionPlan | None,
    missing_result_tool_calls: list[dict] | None = None,
    not_executed_tool_calls: list[dict] | None = None,
    unavailable_tool_calls: list[dict] | None = None,
    reused_tool_calls: list[dict] | None = None,
    context_blocked_calls: dict[str, BlockedToolContext] | None = None,
    built_content_blocks: dict[str, Any] | None = None,
    control_tool_responses: dict[str, str] | None = None,
) -> None:
    request.messages.append(
        build_assistant_tool_message(
            tool_calls=request.tool_calls,
            reasoning_buf=getattr(request, "protocol_reasoning_buf", None) or request.reasoning_buf,
            should_use_reasoning=request.should_use_reasoning,
        )
    )

    source_selection_guidance_by_tool_call_id = _build_source_selection_guidance_by_tool_call_id(
        results,
        source_plan=source_plan,
    )
    citation_registry = _build_search_citation_registry(request.content_blocks)
    records_by_id = {str(record.tool_call.get("id", "")): record for record in results}
    missing_result_ids = {str(tool_call.get("id", "")) for tool_call in missing_result_tool_calls or []}
    not_executed_ids = {str(tool_call.get("id", "")) for tool_call in not_executed_tool_calls or []}
    unavailable_ids = {str(tool_call.get("id", "")) for tool_call in unavailable_tool_calls or []}
    reused_ids = {str(tool_call.get("id", "")) for tool_call in reused_tool_calls or []}
    context_blocked_by_id = context_blocked_calls or {}
    control_responses_by_id = control_tool_responses or {}

    for tool_call in request.tool_calls:
        tool_call_id = str(tool_call.get("id", ""))
        if tool_call_id in control_responses_by_id:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": control_responses_by_id[tool_call_id],
                }
            )
            continue
        record = records_by_id.get(tool_call_id)
        if record is not None:
            citation_numbers = _assign_search_citation_numbers(citation_registry, record)
            tool_context = record.format_llm_context(citation_numbers=citation_numbers)
            source_selection_guidance = source_selection_guidance_by_tool_call_id.get(tool_call_id)
            if source_selection_guidance:
                tool_context = f"{tool_context}\n\n{source_selection_guidance}"
            content_block = (
                built_content_blocks[tool_call_id]
                if built_content_blocks is not None and tool_call_id in built_content_blocks
                else record.build_content_block()
            )
            if content_block is not None:
                request.content_blocks.append(content_block)
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_context,
                }
            )
            continue

        if tool_call_id in missing_result_ids:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _format_missing_tool_result_context(),
                }
            )
            continue

        blocked_context = context_blocked_by_id.get(tool_call_id)
        if blocked_context is not None:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _format_context_unavailable_tool_context(blocked_context),
                }
            )
            continue

        if tool_call_id in reused_ids:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _format_reused_tool_context(),
                }
            )
            continue

        if tool_call_id in not_executed_ids:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _format_not_executed_tool_context(),
                }
            )
            continue

        if tool_call_id in unavailable_ids:
            request.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _format_unavailable_tool_context(),
                }
            )


def _build_search_citation_registry(content_blocks: list[Any]) -> dict[str, int]:
    search_blocks = [block for block in content_blocks if _value(block, "type") == "search"]
    use_source_refs = any(_value(block, "source_refs") for block in search_blocks)
    registry: dict[str, int] = {}

    for block in search_blocks:
        sources = (_value(block, "source_refs") or []) if use_source_refs else (_value(block, "sources") or [])
        for source in sources:
            if use_source_refs:
                if _value(source, "kind") not in {None, "", "search"}:
                    continue
                if _value(source, "status") not in {None, "", "success"}:
                    continue
            key = _citation_source_key(source)
            if key and key not in registry:
                registry[key] = len(registry) + 1
    return registry


def _assign_search_citation_numbers(
    registry: dict[str, int],
    record: ToolExecutionRecord,
) -> list[int] | None:
    if record.tool_name != "web_search":
        return None
    result_data = _value(record.result, "data") or {}
    sources = _value(result_data, "sources") or []
    if not sources:
        return None

    numbers: list[int] = []
    next_number = max(registry.values(), default=0) + 1
    for source_index, source in enumerate(sources, 1):
        key = _citation_source_key(source) or f"{record.tool_call.get('id', '')}:{source_index}"
        citation_number = registry.get(key)
        if citation_number is None:
            citation_number = next_number
            next_number += 1
            registry[key] = citation_number
        numbers.append(citation_number)
    return numbers


def _citation_source_key(source: Any) -> str:
    raw_url = str(_value(source, "url") or "").strip()
    canonical_url = canonicalize_evidence_url(raw_url)
    if canonical_url:
        return canonical_url
    if raw_url:
        return raw_url
    return str(_value(source, "title") or "").strip()


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


async def complete_tool_round_step(
    request: ToolRoundRequest,
    results: list[ToolExecutionRecord],
    *,
    executed_count: int | None = None,
) -> list[str]:
    tool_names = [record.tool_name for record in results]
    actual_count = len(results) if executed_count is None else executed_count
    await request.complete_step_fn(
        context=request.step_context,
        emitter=request.emitter,
        session_cache=request.session_cache,
        tool_names=tool_names,
        tool_call_count=actual_count,
        completed_tool_calls=_completed_tool_calls_after_round(request, actual_count),
        max_tool_calls=_max_tool_calls(request),
        clock=request.clock,
    )
    return tool_names


async def handle_tool_calls_round(*, request: ToolRoundRequest) -> ToolRoundOutcome:
    if not request.output_deferred:
        append_tool_round_reasoning(request)
        persist_tool_round_checkpoint(request)

    control_result = await process_plan_control_calls(
        tool_calls=request.tool_calls,
        coordinator=request.agent_state.plan_coordinator
        if request.agent_state is not None
        else AgentLoopState().plan_coordinator,
        emitter=request.emitter,
    )
    if request.agent_state is not None and request.agent_state.plan_coordinator.has_valid_model_plan:
        object.__setattr__(request.step_context, "model_plan_managed", True)
    announced_tool_calls, unavailable_tool_calls = _partition_tool_calls_by_announcement(
        request,
        tool_calls=control_result.external_tool_calls,
    )
    selected_tool_calls = _select_tool_calls_within_limit(request, announced_tool_calls)
    context_resolution = ToolContextResolution(executable_calls=selected_tool_calls[0])
    if request.agent_state is not None and selected_tool_calls[0]:
        context_resolution = await request.resolve_tool_context_fn(
            tool_calls=selected_tool_calls[0],
            state=request.agent_state,
            emitter=request.emitter,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            message_id=request.assistant_message_id,
            run_id=request.run_id,
            task_id=request.task_id,
            clock=request.clock,
        )
        runtime_context = enrich_tool_runtime_context(
            context_resolution.runtime_context,
            messages=request.messages,
            state=request.agent_state,
            step_number=request.step_number,
        )
        if request.max_steps is not None:
            runtime_context.remaining_agent_steps = max(0, request.max_steps - request.step_number)
        if request.max_tool_calls is not None:
            completed = request.completed_tool_calls or 0
            runtime_context.remaining_agent_tool_calls_before_round = max(
                0,
                request.max_tool_calls - completed,
            )
            planned_current_calls = sum(
                not _is_successfully_reusable_call(request, tool_call)
                for tool_call in context_resolution.executable_calls
            )
            runtime_context.remaining_agent_tool_calls = max(
                0,
                request.max_tool_calls - completed - planned_current_calls,
            )
        context_resolution = ToolContextResolution(
            executable_calls=context_resolution.executable_calls,
            blocked_calls=context_resolution.blocked_calls,
            runtime_context=runtime_context,
        )
    executable_tool_calls = context_resolution.executable_calls
    if request.agent_state is not None:
        started_snapshot = request.agent_state.plan_coordinator.mark_tools_started(
            [
                str(tool_call["plan_item_id"])
                for tool_call in executable_tool_calls
                if isinstance(tool_call.get("plan_item_id"), str)
            ]
        )
        if started_snapshot is not None:
            await request.emitter.plan_snapshot(**started_snapshot)
    planned_execution_tool_calls = [
        tool_call for tool_call in executable_tool_calls if not _is_successfully_reusable_call(request, tool_call)
    ]
    if planned_execution_tool_calls:
        await update_tool_round_plan_started(request, tool_calls=planned_execution_tool_calls)
    raw_results = await execute_tool_round_tools(
        request,
        selected_tool_calls=executable_tool_calls,
        runtime_context=context_resolution.runtime_context,
    )
    results, missing_result_tool_calls = _reconcile_tool_execution_results(
        selected_tool_calls=executable_tool_calls,
        results=raw_results,
        run_id=request.run_id,
    )
    executed_results = [record for record in results if not record.reused]
    if request.agent_state is not None:
        tool_statuses = _plan_item_statuses_from_results(results)
        result_snapshot = request.agent_state.plan_coordinator.mark_tool_results(tool_statuses)
        if result_snapshot is not None:
            await request.emitter.plan_snapshot(**result_snapshot)
    _record_tool_repairs(request.agent_state, executed_results)
    _record_itinerary_tool_observations(request.agent_state, executed_results)
    reused_limited_tool_calls, not_executed_tool_calls = _partition_successfully_reusable_calls(
        request,
        selected_tool_calls[1],
    )
    source_plan = _build_source_selection_plan(executed_results)
    record_network_budget_feedback(request, executed_results, source_plan=source_plan)
    await emit_selected_source_evidence(request, executed_results, source_plan=source_plan)
    built_content_blocks = build_tool_round_content_blocks(results)
    append_tool_round_messages_with_plan(
        request,
        results,
        source_plan=source_plan,
        missing_result_tool_calls=missing_result_tool_calls,
        not_executed_tool_calls=not_executed_tool_calls,
        unavailable_tool_calls=unavailable_tool_calls,
        reused_tool_calls=reused_limited_tool_calls,
        context_blocked_calls=context_resolution.blocked_calls,
        built_content_blocks=built_content_blocks,
        control_tool_responses=control_result.tool_responses,
    )
    product_outcomes = build_product_tool_outcomes(results, built_content_blocks=built_content_blocks)
    if request.agent_state is not None:
        request.agent_state.record_product_tool_outcomes(product_outcomes)
        all_product_outcomes = request.agent_state.product_tool_outcomes
    else:
        all_product_outcomes = list(product_outcomes)
    previous_itinerary_id = next(
        (block.id for block in request.content_blocks if isinstance(block, ItineraryResultsBlock)),
        None,
    )
    itinerary_result = upsert_itinerary_result(
        request.content_blocks,
        product_outcomes=all_product_outcomes,
    )
    current_itinerary_id = next(
        (block.id for block in request.content_blocks if isinstance(block, ItineraryResultsBlock)),
        None,
    )
    persist_tool_round_checkpoint(request)
    await emit_product_result_blocks(request, results, built_content_blocks=built_content_blocks)
    if previous_itinerary_id is not None and previous_itinerary_id != current_itinerary_id:
        await emit_replaced_itinerary_discard(request, previous_itinerary_id)
    if itinerary_result is not None:
        await emit_itinerary_result_block(request, itinerary_result)

    executed_count = _actual_tool_execution_count(executable_tool_calls, results)
    tool_names = await complete_tool_round_step(request, executed_results, executed_count=executed_count)
    restore_reasoning_after_tool_decision(request.call_kwargs)
    return ToolRoundOutcome(
        tool_call_count=executed_count,
        tool_names=tool_names,
        no_progress_search_results=_classify_no_progress_search_results(
            selected_tool_calls=executable_tool_calls,
            results=executed_results,
        ),
        product_result_count=sum(is_registered_rich_content_block(block) for block in built_content_blocks.values()),
        itinerary_result_count=1 if itinerary_result is not None else 0,
        product_outcomes=product_outcomes,
        control_repair_exhausted=control_result.repair_exhausted,
    )


def _plan_item_statuses_from_results(
    results: list[ToolExecutionRecord],
) -> dict[str, str]:
    """同一计划项可能并行调用多个工具，以最严重结果决定本轮状态。"""

    candidates: dict[str, list[str]] = {}
    for record in results:
        plan_item_id = record.tool_call.get("plan_item_id")
        if not isinstance(plan_item_id, str):
            continue
        data = record.result.data if isinstance(record.result.data, dict) else {}
        repair = data.get("repair") if isinstance(data, dict) else None
        if isinstance(repair, dict) and repair.get("retryable") is True:
            status = "running"
        elif record.result.status in {"success", "degraded"}:
            status = "completed"
        else:
            status = "failed"
        candidates.setdefault(plan_item_id, []).append(status)

    priority = {"completed": 1, "running": 2, "failed": 3}
    return {item_id: max(statuses, key=priority.__getitem__) for item_id, statuses in candidates.items()}


def _record_tool_repairs(
    state: AgentLoopState | None,
    results: list[ToolExecutionRecord],
) -> None:
    if state is None:
        return
    for record in results:
        data = getattr(record.result, "data", None)
        resolves_repair_id = data.get("resolves_repair_id") if isinstance(data, dict) else None
        if isinstance(resolves_repair_id, str):
            state.pending_tool_repairs.pop(resolves_repair_id, None)
        repair = data.get("repair") if isinstance(data, dict) else None
        if not isinstance(repair, dict):
            continue
        repair_id = repair.get("repair_id")
        if not isinstance(repair_id, str) or not repair_id:
            repair_id = str(record.tool_call.get("id", "")) or record.tool_name
        state.pending_tool_repairs[repair_id] = {
            "required_fields": [field for field in repair.get("required_fields", []) if isinstance(field, str)][:4],
            "retryable": repair.get("retryable") is True,
            "requires_user_input": repair.get("requires_user_input") is True,
            "retry_exhausted": repair.get("retry_exhausted") is True,
        }


def _record_itinerary_tool_observations(
    state: AgentLoopState | None,
    results: list[ToolExecutionRecord],
) -> None:
    if state is None:
        return
    observations = [
        observation for record in results if (observation := build_itinerary_tool_observation(record)) is not None
    ]
    state.record_itinerary_tool_observations(observations)


def build_tool_round_content_blocks(results: list[ToolExecutionRecord]) -> dict[str, Any]:
    """每条工具结果只构造一次 content block，供 checkpoint 与事件共同复用。"""
    return {str(record.tool_call.get("id", "")): record.build_content_block() for record in results}


def build_product_tool_outcomes(
    results: list[ToolExecutionRecord],
    *,
    built_content_blocks: dict[str, Any],
) -> tuple[ProductToolOutcome, ...]:
    outcomes: list[ProductToolOutcome] = []
    for record in results:
        if record.reused:
            continue
        tool_call_id = str(record.tool_call.get("id", ""))
        content_block = built_content_blocks.get(tool_call_id)
        outcome = _build_product_tool_outcome(record, content_block)
        if outcome is not None:
            outcomes.append(outcome)
    return tuple(outcomes)


def _build_product_tool_outcome(
    record: ToolExecutionRecord,
    content_block: Any,
) -> ProductToolOutcome | None:
    tool_name = record.tool_name
    mode = {
        "search_flights": "flight",
        "search_trains": "train",
        "weather_forecast": "weather",
        "route_compare": "route",
    }.get(tool_name)
    if mode is None or _has_retryable_product_repair(record):
        return None
    if isinstance(content_block, (FlightResultsBlock, TrainResultsBlock)):
        return ProductToolOutcome(
            tool_name=tool_name,
            mode=mode,
            status="available",
            block_id=content_block.id,
            origin=content_block.origin,
            destination=content_block.destination,
            departure_date=content_block.departure_date,
        )
    if isinstance(content_block, WeatherResultsBlock):
        return ProductToolOutcome(
            tool_name=tool_name,
            mode="weather",
            status="available",
            block_id=content_block.id,
            location=content_block.resolved_location,
        )
    if isinstance(content_block, RouteResultsBlock):
        return ProductToolOutcome(
            tool_name=tool_name,
            mode="route",
            status="available",
            block_id=content_block.id,
            origin=content_block.origin.label,
            destination=content_block.destination.label,
        )
    return _unavailable_product_tool_outcome(record, mode)


def _unavailable_product_tool_outcome(
    record: ToolExecutionRecord,
    mode: str,
) -> ProductToolOutcome | None:
    arguments = _safe_tool_arguments(record.tool_call)
    if mode in {"flight", "train"}:
        origin = _safe_argument_text(arguments.get("origin"), 80)
        destination = _safe_argument_text(arguments.get("destination"), 80)
        departure_date = _safe_departure_date(arguments.get("departure_date"))
        if not origin or not destination or departure_date is None:
            return None
        return ProductToolOutcome(
            tool_name=record.tool_name,
            mode=mode,
            status="unavailable",
            origin=origin,
            destination=destination,
            departure_date=departure_date,
        )
    if mode == "weather":
        location = _safe_argument_text(arguments.get("location"), 120)
        return (
            ProductToolOutcome(
                tool_name=record.tool_name,
                mode="weather",
                status="unavailable",
                location=location,
            )
            if location
            else None
        )
    origin = _safe_argument_text(arguments.get("origin"), 120)
    destination = _safe_argument_text(arguments.get("destination"), 120)
    if not origin or not destination:
        return None
    return ProductToolOutcome(
        tool_name=record.tool_name,
        mode="route",
        status="unavailable",
        origin=origin,
        destination=destination,
    )


def _has_retryable_product_repair(record: ToolExecutionRecord) -> bool:
    data = record.result.data if isinstance(record.result.data, dict) else {}
    repair = data.get("repair")
    return isinstance(repair, dict) and repair.get("retryable") is True


def _safe_tool_arguments(tool_call: dict) -> dict[str, Any]:
    raw_arguments = tool_call.get("arguments")
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_argument_text(value: Any, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized and len(normalized) <= max_chars else None


def _safe_departure_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def upsert_itinerary_result(
    content_blocks: list[Any],
    *,
    product_outcomes: list[ProductToolOutcome],
) -> ItineraryResultsBlock | None:
    itinerary = compose_itinerary_result(content_blocks, product_outcomes=product_outcomes)
    if itinerary is None:
        content_blocks[:] = [block for block in content_blocks if not isinstance(block, ItineraryResultsBlock)]
        return None
    existing_index = next(
        (index for index, block in enumerate(content_blocks) if isinstance(block, ItineraryResultsBlock)),
        None,
    )
    if existing_index is None:
        content_blocks.append(itinerary)
        return itinerary
    if content_blocks[existing_index] == itinerary:
        return None
    content_blocks[existing_index] = itinerary
    return itinerary


async def emit_product_result_blocks(
    request: ToolRoundRequest,
    results: list[ToolExecutionRecord],
    *,
    built_content_blocks: dict[str, Any],
) -> None:
    """checkpoint 成功后发送同一产品结果块；非终止写入失败不影响后续回答。"""
    emit = getattr(request.emitter, "content_block_upserted", None)
    if emit is None:
        return
    for record in results:
        tool_call_id = str(record.tool_call.get("id", ""))
        content_block = built_content_blocks.get(tool_call_id)
        if is_registered_rich_content_block(content_block):
            try:
                await emit(tool_call_id=tool_call_id, content_block=content_block)
            except StreamWriteTerminalError:
                raise
            except Exception:
                logger.warning("发送产品结果 content block 事件失败", exc_info=True)


async def emit_itinerary_result_block(
    request: ToolRoundRequest,
    itinerary: ItineraryResultsBlock,
) -> None:
    emit = getattr(request.emitter, "content_block_upserted", None)
    if emit is None:
        return
    try:
        await emit(
            tool_call_id=f"itinerary:{itinerary.id}",
            content_block=itinerary,
        )
    except StreamWriteTerminalError:
        raise
    except Exception:
        logger.warning("发送行程结果 content block 事件失败", exc_info=True)


async def emit_replaced_itinerary_discard(
    request: ToolRoundRequest,
    block_id: str,
) -> None:
    """行程身份变化时先撤回直播页旧块，确保流式与刷新恢复一致。"""

    discard = getattr(request.emitter, "content_block_discarded", None)
    if discard is None:
        return
    try:
        await discard(block_id=block_id)
    except StreamWriteTerminalError:
        raise
    except Exception:
        logger.warning("撤回旧行程 content block 事件失败", exc_info=True)


async def emit_selected_source_evidence(
    request: ToolRoundRequest,
    results: list[ToolExecutionRecord],
    *,
    source_plan: SourceSelectionPlan | None = None,
) -> None:
    emit = getattr(request.emitter, "evidence_item_upserted", None)
    if emit is None:
        return
    plan = source_plan if source_plan is not None else _build_source_selection_plan(results)
    if plan is None:
        return

    for candidate in plan.recommended:
        try:
            await emit(
                tool_call_id=candidate.tool_call_id,
                evidence=build_selected_source_evidence_item(candidate),
            )
        except StreamWriteTerminalError:
            raise
        except Exception:
            logger.warning("发送推荐深读 evidence 失败", exc_info=True)


def record_network_budget_feedback(
    request: ToolRoundRequest,
    results: list[ToolExecutionRecord],
    *,
    source_plan: SourceSelectionPlan | None,
) -> None:
    record_tool_results = getattr(request.network_budget, "record_tool_results", None)
    if not callable(record_tool_results):
        return
    record_tool_results(results=results, source_plan=source_plan)


def _completed_tool_calls_before_round(request: ToolRoundRequest) -> int | None:
    if request.completed_tool_calls is not None:
        return request.completed_tool_calls
    value = getattr(request.network_budget, "completed_tool_calls", None)
    return value if isinstance(value, int) else None


def _completed_tool_calls_after_round(request: ToolRoundRequest, executed_count: int) -> int | None:
    before = _completed_tool_calls_before_round(request)
    return before + executed_count if before is not None else None


def _max_tool_calls(request: ToolRoundRequest) -> int | None:
    if request.max_tool_calls is not None:
        return request.max_tool_calls
    value = getattr(request.network_budget, "max_tool_calls", None)
    return value if isinstance(value, int) else None


def _partition_tool_calls_by_announcement(
    request: ToolRoundRequest,
    *,
    tool_calls: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    candidate_tool_calls = request.tool_calls if tool_calls is None else tool_calls
    if request.announced_tool_names is None:
        return candidate_tool_calls, []

    announced_tool_calls: list[dict] = []
    unavailable_tool_calls: list[dict] = []
    for tool_call in candidate_tool_calls:
        if tool_call.get("name") in request.announced_tool_names:
            announced_tool_calls.append(tool_call)
        else:
            unavailable_tool_calls.append(tool_call)
    if unavailable_tool_calls:
        logger.warning(
            "模型返回本轮未公告工具调用: run_id=%s requested=%s announced=%s unavailable=%s",
            request.run_id,
            len(candidate_tool_calls),
            len(announced_tool_calls),
            len(unavailable_tool_calls),
        )
    return announced_tool_calls, unavailable_tool_calls


def _select_tool_calls_within_limit(
    request: ToolRoundRequest,
    tool_calls: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    candidate_tool_calls = request.tool_calls if tool_calls is None else tool_calls
    completed_tool_calls = _completed_tool_calls_before_round(request)
    max_tool_calls = _max_tool_calls(request)
    if completed_tool_calls is None or max_tool_calls is None:
        return candidate_tool_calls, []

    remaining_capacity = max(0, max_tool_calls - completed_tool_calls)
    remaining_preflight_attempts = MAX_LOCAL_PREFLIGHT_ATTEMPTS_PER_ROUND
    executed_tool_calls: list[dict] = []
    not_executed_tool_calls: list[dict] = []
    for tool_call in candidate_tool_calls:
        if _is_successfully_reusable_call(request, tool_call):
            executed_tool_calls.append(tool_call)
            continue
        if is_local_argument_preflight_failure(
            tool_call,
            request.tool_handlers,
        ):
            if remaining_preflight_attempts > 0:
                executed_tool_calls.append(tool_call)
                remaining_preflight_attempts -= 1
            else:
                not_executed_tool_calls.append(tool_call)
            continue
        if remaining_capacity > 0:
            executed_tool_calls.append(tool_call)
            remaining_capacity -= 1
            continue
        not_executed_tool_calls.append(tool_call)
    if not_executed_tool_calls:
        logger.info(
            "工具调用批次受全局上限截断: "
            f"run_id={request.run_id}, completed={completed_tool_calls}, max={max_tool_calls}, "
            f"requested={len(candidate_tool_calls)}, executed={len(executed_tool_calls)}, "
            f"not_executed={len(not_executed_tool_calls)}"
        )
    return executed_tool_calls, not_executed_tool_calls


def _actual_tool_execution_count(
    submitted_tool_calls: list[dict],
    results: list[ToolExecutionRecord],
) -> int:
    submitted_ids = {str(tool_call.get("id", "")) for tool_call in submitted_tool_calls}
    reused_ids = {
        str(record.tool_call.get("id", ""))
        for record in results
        if record.reused and str(record.tool_call.get("id", "")) in submitted_ids
    }
    local_preflight_ids = {
        str(record.tool_call.get("id", ""))
        for record in results
        if isinstance(getattr(record.result, "data", None), dict)
        and record.result.data.get("local_preflight") is True
        and str(record.tool_call.get("id", "")) in submitted_ids
    }
    return max(0, len(submitted_tool_calls) - len(reused_ids) - len(local_preflight_ids))


def _is_successfully_reusable_call(request: ToolRoundRequest, tool_call: dict) -> bool:
    state = request.agent_state
    if not isinstance(state, AgentLoopState):
        return False
    tool_handlers = request.tool_handlers if isinstance(request.tool_handlers, dict) else None
    signature = build_successful_call_signature(tool_call, tool_handlers)
    return signature is not None and signature in state.successful_tool_call_signatures


def _partition_successfully_reusable_calls(
    request: ToolRoundRequest,
    tool_calls: list[dict],
) -> tuple[list[dict], list[dict]]:
    reused: list[dict] = []
    not_executed: list[dict] = []
    for tool_call in tool_calls:
        target = reused if _is_successfully_reusable_call(request, tool_call) else not_executed
        target.append(tool_call)
    return reused, not_executed


def _reconcile_tool_execution_results(
    *,
    selected_tool_calls: list[dict],
    results: list[ToolExecutionRecord],
    run_id: str,
) -> tuple[list[ToolExecutionRecord], list[dict]]:
    selected_ids = {str(tool_call.get("id", "")) for tool_call in selected_tool_calls}
    records_by_id: dict[str, ToolExecutionRecord] = {}
    duplicate_count = 0
    extra_count = 0

    for record in results:
        tool_call_id = str(record.tool_call.get("id", ""))
        if tool_call_id not in selected_ids:
            extra_count += 1
            continue
        if tool_call_id in records_by_id:
            duplicate_count += 1
            continue
        records_by_id[tool_call_id] = record

    reconciled_results: list[ToolExecutionRecord] = []
    missing_result_tool_calls: list[dict] = []
    for tool_call in selected_tool_calls:
        record = records_by_id.get(str(tool_call.get("id", "")))
        if record is None:
            missing_result_tool_calls.append(tool_call)
            continue
        reconciled_results.append(record)

    if missing_result_tool_calls or duplicate_count or extra_count:
        logger.warning(
            "工具执行结果与提交批次不一致: "
            f"run_id={run_id}, selected={len(selected_tool_calls)}, returned={len(results)}, "
            f"accepted={len(reconciled_results)}, missing={len(missing_result_tool_calls)}, "
            f"duplicate={duplicate_count}, extra={extra_count}"
        )

    return reconciled_results, missing_result_tool_calls


def _classify_no_progress_search_results(
    *,
    selected_tool_calls: list[dict],
    results: list[ToolExecutionRecord],
) -> tuple[bool, ...]:
    records_by_id = {str(record.tool_call.get("id", "")): record for record in results}
    return tuple(
        _is_no_progress_search_result(records_by_id.get(str(tool_call.get("id", ""))))
        for tool_call in selected_tool_calls
    )


def _is_no_progress_search_result(record: ToolExecutionRecord | None) -> bool:
    if record is None or record.tool_name != "web_search" or record.result.status == "success":
        return False
    data = record.result.data if isinstance(record.result.data, dict) else {}
    return data.get("search_budget") in {"planner_limited", "duplicate_skipped"}


def _format_missing_tool_result_context() -> str:
    return json.dumps(
        {
            "status": "failed",
            "reason": "execution_result_missing",
            "message": "工具执行未返回可用记录，本次结果不能作为事实依据。",
        },
        ensure_ascii=False,
    )


def _format_not_executed_tool_context() -> str:
    return json.dumps(
        {
            "status": "not_executed",
            "reason": "limit_reached",
            "limit_reason": "max_tool_calls",
            "message": "Agent 工具调用额度已耗尽，本次调用未执行，不能将其视为事实依据。",
        },
        ensure_ascii=False,
    )


def _format_reused_tool_context() -> str:
    return json.dumps(
        {
            "status": "reused",
            "reason": "identical_successful_result",
            "message": "同名同参查询已成功执行；请复用上一条成功结果并直接回答，不要再次调用或重复展示。",
        },
        ensure_ascii=False,
    )


def _format_unavailable_tool_context() -> str:
    return json.dumps(
        {
            "status": "not_executed",
            "reason": "tool_not_announced_this_round",
            "message": "该工具本轮不可用，本次调用未执行，不能作为事实依据；请停止重复调用。",
        },
        ensure_ascii=False,
    )


def _format_context_unavailable_tool_context(blocked: BlockedToolContext) -> str:
    return json.dumps(
        {
            "status": "unavailable",
            "error_code": "context_required_not_provided",
            "context_type": "geolocation",
            "context_status": blocked.status,
            "reason": blocked.reason,
            "instruction": "未执行工具；不得声称已获取用户位置，本 run 不要再次请求当前位置。",
        },
        ensure_ascii=False,
    )


def _build_source_selection_guidance_by_tool_call_id(
    results: list[ToolExecutionRecord],
    *,
    source_plan: SourceSelectionPlan | None = None,
) -> dict[str, str]:
    target_search_tool_call_id = _last_successful_search_tool_call_id(results)
    plan = source_plan if source_plan is not None else _build_source_selection_plan(results)
    if plan is None or not target_search_tool_call_id:
        return {}

    guidance = format_search_read_plan_guidance(plan)
    if not guidance:
        return {}
    return {target_search_tool_call_id: guidance}


def _last_successful_search_tool_call_id(results: list[ToolExecutionRecord]) -> str:
    target = ""
    for record in results:
        if record.tool_name == "web_search" and record.result.status == "success":
            target = str(record.tool_call.get("id", ""))
    return target


def _build_source_selection_plan(results: list[ToolExecutionRecord]) -> SourceSelectionPlan | None:
    search_results: list[SearchResultForRanking] = []
    for record in results:
        if record.tool_name != "web_search" or record.result.status != "success":
            continue
        result_data = getattr(record.result, "data", None) or {}
        sources = result_data.get("sources") or []
        if not sources:
            continue
        tool_call_id = str(record.tool_call.get("id", ""))
        search_results.append(
            SearchResultForRanking(
                tool_call_id=tool_call_id,
                query=str(result_data.get("query") or ""),
                intent=result_data.get("intent"),
                search_budget=result_data.get("search_budget"),
                sources=sources,
            )
        )

    if not search_results:
        return None

    return build_search_read_plan(search_results)


def _tool_arguments(tool_calls: list[dict]) -> list[dict]:
    arguments: list[dict] = []
    for tool_call in tool_calls:
        raw_arguments = tool_call.get("arguments", {})
        if isinstance(raw_arguments, dict):
            arguments.append(raw_arguments)
            continue
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed = {}
            arguments.append(parsed if isinstance(parsed, dict) else {})
            continue
        arguments.append({})
    return arguments
