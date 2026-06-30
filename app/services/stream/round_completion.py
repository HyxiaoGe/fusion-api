"""普通文本回合收尾边界。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from app.schemas.chat import ContentBlock, TextBlock, ThinkingBlock
from app.services.stream.step_lifecycle import AgentStepContext, AgentStepEmitter, AgentStepSessionCache

CompleteStepFn = Callable[..., Awaitable[int]]


def append_round_content_blocks(
    content_blocks: list[ContentBlock],
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


async def complete_text_response_step(
    *,
    context: AgentStepContext,
    emitter: AgentStepEmitter,
    session_cache: AgentStepSessionCache,
    complete_step_fn: CompleteStepFn,
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
    clock: Callable[[], float] = time.time,
) -> int:
    return await complete_step_fn(
        context=context,
        emitter=emitter,
        session_cache=session_cache,
        tool_names=[],
        tool_call_count=0,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
        clock=clock,
    )
