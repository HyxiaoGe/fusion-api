"""Agent 触顶后的强制总结 step 编排。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.ai.prompts.agent_loop import LIMIT_SUMMARY_PROMPT as _LIMIT_SUMMARY_PROMPT
from app.ai.prompts.agent_loop import get_limit_summary_prompt
from app.core.logger import app_logger as logger
from app.schemas.chat import TextBlock, ThinkingBlock, Usage

LIMIT_SUMMARY_PROMPT = _LIMIT_SUMMARY_PROMPT


@dataclass(frozen=True)
class LimitSummaryOutcome:
    accumulated_usage: Usage


@dataclass(frozen=True)
class LimitSummaryRoundResult:
    reasoning_buf: str
    content_buf: str
    usage_data: Usage | None


@dataclass(frozen=True)
class LimitSummaryStepRequest:
    conversation_id: str
    task_id: str
    run_id: str
    step_number: int
    model_id: str
    provider: str
    litellm_model: str
    litellm_kwargs: dict
    messages: list[dict]
    should_use_reasoning: bool
    content_blocks: list
    call_kwargs: dict
    accumulated_usage: Usage
    emitter: Any
    session_cache: Any
    total_timeout_s: int
    run_start: float
    start_step_fn: Callable[..., Awaitable[Any]]
    complete_step_fn: Callable[..., Awaitable[Any]]
    llm_call_fn: Callable[..., Awaitable[Any]]
    stream_round_fn: Callable[..., Awaitable[tuple[str, str, list[dict], str, Usage | None]]]
    log_round_summary_fn: Callable[..., None]
    warning_fn: Callable[[str], None] | None = None
    clock: Callable[[], float] = time.time
    on_step_started: Callable[[str], None] | None = None


def build_limit_summary_call_kwargs(call_kwargs: dict) -> dict:
    return {key: value for key, value in call_kwargs.items() if key not in ("tools", "tool_choice")}


def compute_summary_timeout(*, total_timeout_s: int, run_start: float, clock: Callable[[], float]) -> float:
    return max(10, total_timeout_s - (clock() - run_start))


def append_limit_summary_prompt(messages: list[dict]) -> None:
    messages.append({"role": "system", "content": get_limit_summary_prompt()})


async def call_limit_summary_round(
    *,
    request: LimitSummaryStepRequest,
    thinking_block_id: str,
    text_block_id: str,
    step_id: str,
) -> LimitSummaryRoundResult:
    final_call_kwargs = build_limit_summary_call_kwargs(request.call_kwargs)
    response = await request.llm_call_fn(
        request.litellm_model,
        request.litellm_kwargs,
        request.messages,
        **final_call_kwargs,
    )
    reasoning_buf, content_buf, _, _, usage_data = await request.stream_round_fn(
        response,
        request.conversation_id,
        request.task_id,
        request.should_use_reasoning,
        thinking_block_id,
        text_block_id,
        run_id=request.run_id,
        step_id=step_id,
    )
    request.log_round_summary_fn(
        conversation_id=request.conversation_id,
        run_id=request.run_id,
        step_number=request.step_number,
        model_id=request.model_id,
        provider=request.provider,
        finish_reason="limit_summary",
        tool_calls_count=0,
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
    )
    return LimitSummaryRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        usage_data=usage_data,
    )


def accumulate_summary_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage:
    if not usage_data:
        return accumulated_usage
    return Usage(
        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
    )


def append_summary_content_blocks(
    *,
    content_blocks: list,
    reasoning_buf: str,
    content_buf: str,
    thinking_block_id: str,
    text_block_id: str,
) -> None:
    if content_buf:
        if reasoning_buf:
            content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
        return

    if reasoning_buf:
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=reasoning_buf))


async def complete_limit_summary_step(
    *,
    summary_context: Any,
    emitter: Any,
    session_cache: Any,
    complete_step_fn: Callable[..., Awaitable[Any]],
    clock: Callable[[], float],
) -> None:
    await complete_step_fn(
        context=summary_context,
        emitter=emitter,
        session_cache=session_cache,
        tool_names=[],
        tool_call_count=0,
        clock=clock,
    )


async def start_limit_summary_step(*, request: LimitSummaryStepRequest) -> Any:
    return await request.start_step_fn(
        emitter=request.emitter,
        session_cache=request.session_cache,
        run_id=request.run_id,
        step_number=request.step_number,
        clock=request.clock,
        on_step_started=request.on_step_started,
    )


async def run_summary_round_with_timeout(
    *,
    request: LimitSummaryStepRequest,
    summary_context: Any,
    thinking_block_id: str,
    text_block_id: str,
    remaining: float,
) -> LimitSummaryRoundResult:
    try:
        return await asyncio.wait_for(
            call_limit_summary_round(
                request=request,
                thinking_block_id=thinking_block_id,
                text_block_id=text_block_id,
                step_id=summary_context.step_id,
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(f"触顶总结超出剩余预算: conv_id={request.conversation_id}, budget={remaining}s")
        return LimitSummaryRoundResult(reasoning_buf="", content_buf="", usage_data=None)


async def run_limit_summary_step(
    *,
    request: LimitSummaryStepRequest,
) -> LimitSummaryOutcome:
    summary_context = await start_limit_summary_step(request=request)

    append_limit_summary_prompt(request.messages)
    thinking_block_id = summary_context.thinking_block_id
    text_block_id = summary_context.text_block_id
    remaining = compute_summary_timeout(
        total_timeout_s=request.total_timeout_s,
        run_start=request.run_start,
        clock=request.clock,
    )

    round_result = await run_summary_round_with_timeout(
        request=request,
        summary_context=summary_context,
        thinking_block_id=thinking_block_id,
        text_block_id=text_block_id,
        remaining=remaining,
    )

    next_usage = accumulate_summary_usage(request.accumulated_usage, round_result.usage_data)
    append_summary_content_blocks(
        content_blocks=request.content_blocks,
        reasoning_buf=round_result.reasoning_buf,
        content_buf=round_result.content_buf,
        thinking_block_id=thinking_block_id,
        text_block_id=text_block_id,
    )
    await complete_limit_summary_step(
        summary_context=summary_context,
        emitter=request.emitter,
        session_cache=request.session_cache,
        complete_step_fn=request.complete_step_fn,
        clock=request.clock,
    )
    return LimitSummaryOutcome(accumulated_usage=next_usage)
