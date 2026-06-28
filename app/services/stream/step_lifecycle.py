"""Agent step 生命周期薄边界。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import Protocol


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


async def start_agent_step(
    *,
    emitter: AgentStepEmitter,
    session_cache: AgentStepSessionCache,
    run_id: str,
    step_number: int,
    clock: Callable[[], float] = time.time,
    block_id_factory: Callable[[], str] = _make_block_id,
    on_step_started: Callable[[str], None] | None = None,
) -> AgentStepContext:
    started_at = clock()
    step_id = await emitter.step_started(step_number=step_number)
    if on_step_started is not None:
        on_step_started(step_id)
    await _maybe_emit_plan_step_started(
        emitter=emitter,
        run_id=run_id,
        step_number=step_number,
    )
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
    clock: Callable[[], float] = time.time,
) -> int:
    duration_ms = int((clock() - context.started_at) * 1000)
    await emitter.step_completed(
        step_number=context.step_number,
        tool_call_count=tool_call_count,
        duration_ms=duration_ms,
    )
    await _maybe_emit_plan_step_completed(
        emitter=emitter,
        context=context,
        tool_names=list(tool_names),
        tool_call_count=tool_call_count,
    )
    await session_cache.write_step_completed(
        step_id=context.step_id,
        tool_names=list(tool_names),
        tool_calls_count=tool_call_count,
        duration_ms=duration_ms,
    )
    return duration_ms


async def _maybe_emit_plan_step_started(*, emitter: AgentStepEmitter, run_id: str, step_number: int) -> None:
    await _maybe_call_async(
        emitter,
        "plan_step_updated",
        plan_id=f"plan-{run_id}",
        revision=step_number * 2,
        item={
            "id": "search",
            "title": "查找资料",
            "status": "running",
            "kind": "search",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    )


async def _maybe_emit_plan_step_completed(
    *,
    emitter: AgentStepEmitter,
    context: AgentStepContext,
    tool_names: list[str],
    tool_call_count: int,
) -> None:
    await _maybe_call_async(
        emitter,
        "plan_step_updated",
        plan_id=f"plan-{context.run_id}",
        revision=context.step_number * 2 + 1,
        item={
            "id": "search",
            "title": "查找资料",
            "status": "completed",
            "kind": "search",
            "summary": f"完成 {tool_call_count} 个工具调用",
            "tool_names": tool_names,
            "evidence_item_ids": [],
        },
    )
    await _maybe_call_async(
        emitter,
        "run_progress_updated",
        phase="reading" if tool_call_count > 0 else "synthesizing",
        label="正在读取关键来源" if tool_call_count > 0 else "正在整理回答",
        completed_steps=2,
        total_steps=None,
        completed_tool_calls=tool_call_count,
        max_tool_calls=None,
    )


async def _maybe_call_async(emitter: AgentStepEmitter, method_name: str, **kwargs) -> None:
    method = getattr(emitter, method_name, None)
    if method is None or not iscoroutinefunction(method):
        return
    await method(**kwargs)
