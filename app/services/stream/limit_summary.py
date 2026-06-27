"""Agent 触顶后的强制总结 step 编排。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.logger import app_logger as logger
from app.schemas.chat import TextBlock, ThinkingBlock, Usage

LIMIT_SUMMARY_PROMPT = "你已达到工具调用上限，请基于已收集的信息给出最终回答。不要再调用任何工具。"


@dataclass(frozen=True)
class LimitSummaryOutcome:
    accumulated_usage: Usage


def build_limit_summary_call_kwargs(call_kwargs: dict) -> dict:
    return {key: value for key, value in call_kwargs.items() if key not in ("tools", "tool_choice")}


def compute_summary_timeout(*, total_timeout_s: int, run_start: float, clock: Callable[[], float]) -> float:
    return max(10, total_timeout_s - (clock() - run_start))


async def run_limit_summary_step(
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
    content_blocks: list,
    call_kwargs: dict,
    accumulated_usage: Usage,
    emitter: Any,
    session_cache: Any,
    total_timeout_s: int,
    run_start: float,
    start_step_fn: Callable[..., Awaitable[Any]],
    complete_step_fn: Callable[..., Awaitable[Any]],
    llm_call_fn: Callable[..., Awaitable[Any]],
    stream_round_fn: Callable[..., Awaitable[tuple[str, str, list[dict], str, Usage | None]]],
    log_round_summary_fn: Callable[..., None],
    warning_fn: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.time,
    on_step_started: Callable[[str], None] | None = None,
) -> LimitSummaryOutcome:
    summary_context = await start_step_fn(
        emitter=emitter,
        session_cache=session_cache,
        run_id=run_id,
        step_number=step_number,
        clock=clock,
        on_step_started=on_step_started,
    )

    messages.append({"role": "system", "content": LIMIT_SUMMARY_PROMPT})
    final_call_kwargs = build_limit_summary_call_kwargs(call_kwargs)
    thinking_block_id = summary_context.thinking_block_id
    text_block_id = summary_context.text_block_id
    remaining = compute_summary_timeout(total_timeout_s=total_timeout_s, run_start=run_start, clock=clock)

    try:

        async def _do_summary():
            response = await llm_call_fn(
                litellm_model,
                litellm_kwargs,
                messages,
                **final_call_kwargs,
            )
            return await stream_round_fn(
                response,
                conversation_id,
                task_id,
                should_use_reasoning,
                thinking_block_id,
                text_block_id,
                run_id=run_id,
                step_id=summary_context.step_id,
            )

        reasoning_buf, content_buf, _, _, usage_data = await asyncio.wait_for(_do_summary(), timeout=remaining)
        log_round_summary_fn(
            conversation_id=conversation_id,
            run_id=run_id,
            step_number=step_number,
            model_id=model_id,
            provider=provider,
            finish_reason="limit_summary",
            tool_calls_count=0,
            reasoning_buf=reasoning_buf,
            content_buf=content_buf,
        )
    except asyncio.TimeoutError:
        warning = warning_fn if warning_fn is not None else logger.warning
        warning(f"触顶总结超出剩余预算: conv_id={conversation_id}, budget={remaining}s")
        reasoning_buf, content_buf, usage_data = "", "", None

    next_usage = accumulated_usage
    if usage_data:
        next_usage = Usage(
            input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
            output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
        )
    if reasoning_buf:
        content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
    if content_buf:
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))

    await complete_step_fn(
        context=summary_context,
        emitter=emitter,
        session_cache=session_cache,
        tool_names=[],
        tool_call_count=0,
        clock=clock,
    )
    return LimitSummaryOutcome(accumulated_usage=next_usage)
