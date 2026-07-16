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
from app.services.source_evidence_ledger import build_selected_source_evidence_item, canonicalize_evidence_url
from app.services.stream.step_lifecycle import AgentStepContext, mark_tool_round_started
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream_state_service import StreamWriteTerminalError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolRoundOutcome:
    tool_call_count: int
    tool_names: list[str]
    no_progress_search_results: tuple[bool, ...] = ()


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
    assistant_message_sequence: int | None = None
    on_tools_executed: Callable[[int], None] | None = None
    completed_tool_calls: int | None = None
    max_tool_calls: int | None = None
    clock: Callable[[], float] = time.time
    tool_handlers: dict[str, Any] | None = None
    announced_tool_names: frozenset[str] | None = None


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
    try:
        return await request.execute_tools_fn(
            executed_tool_calls,
            request.conversation_id,
            request.user_id,
            request.model_id,
            request.provider,
            **execute_kwargs,
        )
    finally:
        if request.on_tools_executed is not None:
            request.on_tools_executed(len(executed_tool_calls))


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
) -> None:
    request.messages.append(
        build_assistant_tool_message(
            tool_calls=request.tool_calls,
            reasoning_buf=request.reasoning_buf,
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

    for tool_call in request.tool_calls:
        tool_call_id = str(tool_call.get("id", ""))
        record = records_by_id.get(tool_call_id)
        if record is not None:
            citation_numbers = _assign_search_citation_numbers(citation_registry, record)
            tool_context = record.format_llm_context(citation_numbers=citation_numbers)
            source_selection_guidance = source_selection_guidance_by_tool_call_id.get(tool_call_id)
            if source_selection_guidance:
                tool_context = f"{tool_context}\n\n{source_selection_guidance}"
            content_block = record.build_content_block()
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
    append_tool_round_reasoning(request)
    persist_tool_round_checkpoint(request)

    announced_tool_calls, unavailable_tool_calls = _partition_tool_calls_by_announcement(request)
    selected_tool_calls = _select_tool_calls_within_limit(request, announced_tool_calls)
    if selected_tool_calls[0]:
        await update_tool_round_plan_started(request, tool_calls=selected_tool_calls[0])
    raw_results = await execute_tool_round_tools(request, selected_tool_calls=selected_tool_calls[0])
    results, missing_result_tool_calls = _reconcile_tool_execution_results(
        selected_tool_calls=selected_tool_calls[0],
        results=raw_results,
        run_id=request.run_id,
    )
    source_plan = _build_source_selection_plan(results)
    record_network_budget_feedback(request, results, source_plan=source_plan)
    await emit_selected_source_evidence(request, results, source_plan=source_plan)
    append_tool_round_messages_with_plan(
        request,
        results,
        source_plan=source_plan,
        missing_result_tool_calls=missing_result_tool_calls,
        not_executed_tool_calls=selected_tool_calls[1],
        unavailable_tool_calls=unavailable_tool_calls,
    )
    persist_tool_round_checkpoint(request)

    executed_count = len(selected_tool_calls[0])
    tool_names = await complete_tool_round_step(request, results, executed_count=executed_count)
    restore_reasoning_after_tool_decision(request.call_kwargs)
    return ToolRoundOutcome(
        tool_call_count=executed_count,
        tool_names=tool_names,
        no_progress_search_results=_classify_no_progress_search_results(
            selected_tool_calls=selected_tool_calls[0],
            results=results,
        ),
    )


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


def _partition_tool_calls_by_announcement(request: ToolRoundRequest) -> tuple[list[dict], list[dict]]:
    if request.announced_tool_names is None:
        return request.tool_calls, []

    announced_tool_calls: list[dict] = []
    unavailable_tool_calls: list[dict] = []
    for tool_call in request.tool_calls:
        if tool_call.get("name") in request.announced_tool_names:
            announced_tool_calls.append(tool_call)
        else:
            unavailable_tool_calls.append(tool_call)
    if unavailable_tool_calls:
        logger.warning(
            "模型返回本轮未公告工具调用: run_id=%s requested=%s announced=%s unavailable=%s",
            request.run_id,
            len(request.tool_calls),
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
    executed_tool_calls = candidate_tool_calls[:remaining_capacity]
    not_executed_tool_calls = candidate_tool_calls[remaining_capacity:]
    if not_executed_tool_calls:
        logger.info(
            "工具调用批次受全局上限截断: "
            f"run_id={request.run_id}, completed={completed_tool_calls}, max={max_tool_calls}, "
            f"requested={len(candidate_tool_calls)}, executed={len(executed_tool_calls)}, "
            f"not_executed={len(not_executed_tool_calls)}"
        )
    return executed_tool_calls, not_executed_tool_calls


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


def _format_unavailable_tool_context() -> str:
    return json.dumps(
        {
            "status": "not_executed",
            "reason": "tool_not_announced_this_round",
            "message": "该工具本轮不可用，本次调用未执行，不能作为事实依据；请停止重复调用。",
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
