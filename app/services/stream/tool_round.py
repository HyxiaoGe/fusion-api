"""Agent 工具回合编排。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas.chat import ThinkingBlock
from app.services.search_read_planner import build_search_read_plan, format_search_read_plan_guidance
from app.services.source_candidate_ranker import (
    SearchResultForRanking,
    SourceSelectionPlan,
)
from app.services.source_evidence_ledger import build_selected_source_evidence_item
from app.services.stream.step_lifecycle import AgentStepContext, mark_tool_round_started
from app.services.stream.tool_execution_result import ToolExecutionRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolRoundOutcome:
    tool_call_count: int
    tool_names: list[str]


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
    on_tools_executed: Callable[[int], None] | None = None
    completed_tool_calls: int | None = None
    max_tool_calls: int | None = None
    clock: Callable[[], float] = time.time


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
    request.persist_message_fn(
        request.db,
        request.assistant_message_id,
        request.conversation_id,
        request.model_id,
        request.content_blocks,
        partial=True,
    )


async def execute_tool_round_tools(request: ToolRoundRequest) -> list[ToolExecutionRecord]:
    results = await request.execute_tools_fn(
        request.tool_calls,
        request.conversation_id,
        request.user_id,
        request.model_id,
        request.provider,
        trace_id=request.run_id,
        step_number=request.step_number,
        message_id=request.assistant_message_id,
        emitter=request.emitter,
        network_budget=request.network_budget,
    )
    if request.on_tools_executed is not None:
        request.on_tools_executed(len(request.tool_calls))
    return results


async def update_tool_round_plan_started(request: ToolRoundRequest) -> None:
    await mark_tool_round_started(
        context=request.step_context,
        emitter=request.emitter,
        tool_call_count=len(request.tool_calls),
        tool_names=[tool_call["name"] for tool_call in request.tool_calls],
        tool_arguments=_tool_arguments(request.tool_calls),
        completed_tool_calls=_completed_tool_calls_before_round(request),
        max_tool_calls=_max_tool_calls(request),
    )


def append_tool_round_messages(request: ToolRoundRequest, results: list[ToolExecutionRecord]) -> None:
    request.messages.append(
        build_assistant_tool_message(
            tool_calls=request.tool_calls,
            reasoning_buf=request.reasoning_buf,
            should_use_reasoning=request.should_use_reasoning,
        )
    )

    source_selection_guidance_by_tool_call_id = _build_source_selection_guidance_by_tool_call_id(results)
    for record in results:
        tool_context = record.format_llm_context()
        source_selection_guidance = source_selection_guidance_by_tool_call_id.get(str(record.tool_call.get("id", "")))
        if source_selection_guidance:
            tool_context = f"{tool_context}\n\n{source_selection_guidance}"
        content_block = record.build_content_block()
        if content_block is not None:
            request.content_blocks.append(content_block)

        request.messages.append(
            {
                "role": "tool",
                "tool_call_id": record.tool_call["id"],
                "content": tool_context,
            }
        )


async def complete_tool_round_step(request: ToolRoundRequest, results: list[ToolExecutionRecord]) -> list[str]:
    tool_names = [record.tool_name for record in results]
    await request.complete_step_fn(
        context=request.step_context,
        emitter=request.emitter,
        session_cache=request.session_cache,
        tool_names=tool_names,
        tool_call_count=len(results),
        completed_tool_calls=_completed_tool_calls_after_round(request, len(results)),
        max_tool_calls=_max_tool_calls(request),
        clock=request.clock,
    )
    return tool_names


async def handle_tool_calls_round(*, request: ToolRoundRequest) -> ToolRoundOutcome:
    append_tool_round_reasoning(request)
    persist_tool_round_checkpoint(request)

    await update_tool_round_plan_started(request)
    results = await execute_tool_round_tools(request)
    await emit_selected_source_evidence(request, results)
    append_tool_round_messages(request, results)
    persist_tool_round_checkpoint(request)

    tool_names = await complete_tool_round_step(request, results)
    restore_reasoning_after_tool_decision(request.call_kwargs)
    return ToolRoundOutcome(tool_call_count=len(request.tool_calls), tool_names=tool_names)


async def emit_selected_source_evidence(request: ToolRoundRequest, results: list[ToolExecutionRecord]) -> None:
    emit = getattr(request.emitter, "evidence_item_upserted", None)
    if emit is None:
        return
    plan = _build_source_selection_plan(results)
    if plan is None:
        return

    for candidate in plan.recommended:
        try:
            await emit(
                tool_call_id=candidate.tool_call_id,
                evidence=build_selected_source_evidence_item(candidate),
            )
        except Exception:
            logger.warning("发送推荐深读 evidence 失败", exc_info=True)


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


def _build_source_selection_guidance_by_tool_call_id(results: list[ToolExecutionRecord]) -> dict[str, str]:
    target_search_tool_call_id = _last_successful_search_tool_call_id(results)
    plan = _build_source_selection_plan(results)
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
