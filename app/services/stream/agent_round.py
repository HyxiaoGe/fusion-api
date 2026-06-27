"""Agent loop 单轮普通 LLM 调用编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas.chat import Usage


@dataclass(frozen=True)
class AgentRoundResult:
    reasoning_buf: str
    content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    accumulated_usage: Usage


def accumulate_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage:
    if not usage_data:
        return accumulated_usage
    return Usage(
        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
    )


async def run_agent_round(
    *,
    conversation_id: str,
    task_id: str,
    run_id: str,
    step_number: int,
    model_id: str,
    provider: str,
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    should_use_reasoning: bool,
    call_kwargs: dict,
    accumulated_usage: Usage,
    step_context: Any,
    llm_call_fn: Callable[..., Awaitable[Any]],
    stream_round_fn: Callable[..., Awaitable[tuple[str, str, list[dict], str, Usage | None]]],
    log_round_summary_fn: Callable[..., None],
) -> AgentRoundResult:
    response = await llm_call_fn(
        litellm_model,
        litellm_kwargs,
        messages,
        **call_kwargs,
    )
    reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = await stream_round_fn(
        response,
        conversation_id,
        task_id,
        should_use_reasoning,
        step_context.thinking_block_id,
        step_context.text_block_id,
        run_id=run_id,
        step_id=step_context.step_id,
    )
    log_round_summary_fn(
        conversation_id=conversation_id,
        run_id=run_id,
        step_number=step_number,
        model_id=model_id,
        provider=provider,
        finish_reason=finish_reason,
        tool_calls_count=len(tool_calls),
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
    )

    return AgentRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        accumulated_usage=accumulate_usage(accumulated_usage, usage_data),
    )
