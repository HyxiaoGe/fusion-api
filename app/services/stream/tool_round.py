"""Agent 工具回合编排。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas.chat import ThinkingBlock
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_execution_result import ToolExecutionRecord


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


def append_tool_round_messages(request: ToolRoundRequest, results: list[ToolExecutionRecord]) -> None:
    request.messages.append(
        build_assistant_tool_message(
            tool_calls=request.tool_calls,
            reasoning_buf=request.reasoning_buf,
            should_use_reasoning=request.should_use_reasoning,
        )
    )

    for record in results:
        tool_context = record.format_llm_context()
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
        clock=request.clock,
    )
    return tool_names


async def handle_tool_calls_round(*, request: ToolRoundRequest) -> ToolRoundOutcome:
    append_tool_round_reasoning(request)
    persist_tool_round_checkpoint(request)

    results = await execute_tool_round_tools(request)
    append_tool_round_messages(request, results)
    persist_tool_round_checkpoint(request)

    tool_names = await complete_tool_round_step(request, results)
    restore_reasoning_after_tool_decision(request.call_kwargs)
    return ToolRoundOutcome(tool_call_count=len(request.tool_calls), tool_names=tool_names)
