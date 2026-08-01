"""Agent step 生命周期边界。

计划状态完全由 PlanCoordinator 管理；本模块只负责 step、session 与粗粒度运行进度。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Protocol


class AgentStepEmitter(Protocol):
    async def step_started(self, *, step_number: int) -> str: ...

    async def step_completed(self, *, step_number: int, tool_call_count: int, duration_ms: int) -> None: ...


class AgentStepSessionCache(Protocol):
    async def write_step_started(self, *, run_id: str, step_id: str, step_number: int) -> None: ...

    async def write_step_completed(
        self,
        *,
        step_id: str,
        tool_names: list[str],
        tool_calls_count: int,
        duration_ms: int,
    ) -> None: ...


@dataclass(frozen=True)
class AgentStepContext:
    step_id: str
    step_number: int
    started_at: float
    thinking_block_id: str
    text_block_id: str
    run_id: str = ""


def _make_block_id() -> str:
    return f"blk_{uuid.uuid4().hex[:12]}"


async def _maybe_emit(emitter: Any, method_name: str, **kwargs: Any) -> None:
    method = getattr(emitter, method_name, None)
    if not callable(method):
        return
    result = method(**kwargs)
    if isawaitable(result):
        await result


async def start_agent_step(
    *,
    emitter: AgentStepEmitter,
    session_cache: AgentStepSessionCache,
    run_id: str,
    step_number: int,
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
    clock: Callable[[], float] = time.time,
    block_id_factory: Callable[[], str] = _make_block_id,
    on_step_started: Callable[[str], None] | None = None,
) -> AgentStepContext:
    del completed_tool_calls, max_tool_calls
    started_at = clock()
    step_id = await emitter.step_started(step_number=step_number)
    if on_step_started is not None:
        on_step_started(step_id)
    await session_cache.write_step_started(
        run_id=run_id,
        step_id=step_id,
        step_number=step_number,
    )
    return AgentStepContext(
        step_id=step_id,
        run_id=run_id,
        step_number=step_number,
        started_at=started_at,
        thinking_block_id=block_id_factory(),
        text_block_id=block_id_factory(),
    )


async def complete_agent_step(
    *,
    context: AgentStepContext,
    emitter: AgentStepEmitter,
    session_cache: AgentStepSessionCache,
    tool_names: Sequence[str],
    tool_call_count: int,
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
    clock: Callable[[], float] = time.time,
) -> int:
    duration_ms = int((clock() - context.started_at) * 1000)
    await emitter.step_completed(
        step_number=context.step_number,
        tool_call_count=tool_call_count,
        duration_ms=duration_ms,
    )
    await session_cache.write_step_completed(
        step_id=context.step_id,
        tool_names=list(tool_names),
        tool_calls_count=tool_call_count,
        duration_ms=duration_ms,
    )
    if tool_call_count > 0:
        await _maybe_emit(
            emitter,
            "run_progress_updated",
            phase="researching",
            label="已完成外部工具调用",
            completed_steps=None,
            total_steps=None,
            completed_tool_calls=completed_tool_calls,
            max_tool_calls=max_tool_calls,
        )
    return duration_ms


async def mark_tool_round_started(
    *,
    context: AgentStepContext,
    emitter: AgentStepEmitter,
    tool_call_count: int,
    tool_names: Sequence[str] = (),
    tool_arguments: Sequence[dict] = (),
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
) -> None:
    del context, tool_call_count, tool_arguments
    stage = "reading" if tool_names and all(name == "url_read" for name in tool_names) else "researching"
    await _maybe_emit(
        emitter,
        "run_progress_updated",
        phase=stage,
        label="正在读取关键来源" if stage == "reading" else "正在调用外部工具",
        completed_steps=None,
        total_steps=None,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
    )
