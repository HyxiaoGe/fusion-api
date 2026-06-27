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


async def handle_tool_calls_round(
    *,
    db: Any,
    assistant_message_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    content_blocks: list,
    messages: list[dict],
    tool_calls: list[dict],
    reasoning_buf: str,
    should_use_reasoning: bool,
    step_context: AgentStepContext,
    step_number: int,
    run_id: str,
    emitter: Any,
    session_cache: Any,
    network_budget: Any,
    call_kwargs: dict,
    persist_message_fn: Callable[..., Any],
    execute_tools_fn: Callable[..., Awaitable[list[ToolExecutionRecord]]],
    complete_step_fn: Callable[..., Awaitable[Any]],
    on_tools_executed: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.time,
) -> ToolRoundOutcome:
    if reasoning_buf:
        content_blocks.append(
            ThinkingBlock(
                type="thinking",
                id=step_context.thinking_block_id,
                thinking=reasoning_buf,
            )
        )

    persist_message_fn(db, assistant_message_id, conversation_id, model_id, content_blocks, partial=True)

    results = await execute_tools_fn(
        tool_calls,
        conversation_id,
        user_id,
        model_id,
        provider,
        trace_id=run_id,
        step_number=step_number,
        message_id=assistant_message_id,
        emitter=emitter,
        network_budget=network_budget,
    )
    if on_tools_executed is not None:
        on_tools_executed(len(tool_calls))

    messages.append(
        build_assistant_tool_message(
            tool_calls=tool_calls,
            reasoning_buf=reasoning_buf,
            should_use_reasoning=should_use_reasoning,
        )
    )

    for record in results:
        tool_context = record.format_llm_context()
        content_block = record.build_content_block()
        if content_block is not None:
            content_blocks.append(content_block)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": record.tool_call["id"],
                "content": tool_context,
            }
        )

    persist_message_fn(db, assistant_message_id, conversation_id, model_id, content_blocks, partial=True)

    tool_names = [record.tool_name for record in results]
    await complete_step_fn(
        context=step_context,
        emitter=emitter,
        session_cache=session_cache,
        tool_names=tool_names,
        tool_call_count=len(results),
        clock=clock,
    )

    restore_reasoning_after_tool_decision(call_kwargs)
    return ToolRoundOutcome(tool_call_count=len(tool_calls), tool_names=tool_names)
