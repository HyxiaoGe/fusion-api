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


PLAN_REVISION_WINDOW = 10
PLAN_REV_UNDERSTAND_RUNNING = 2
PLAN_REV_UNDERSTAND_COMPLETED = 3
PLAN_REV_FOLLOWUP_READ_COMPLETED = 2
PLAN_REV_FOLLOWUP_ANSWER_RUNNING = 3
PLAN_REV_ANSWER_PENDING_FOR_TOOLS = 4
PLAN_REV_FIRST_TOOL_RUNNING = 4
PLAN_REV_FOLLOWUP_TOOL_RUNNING = 5
PLAN_REV_FIRST_TOOL_COMPLETED = 5
PLAN_REV_FOLLOWUP_TOOL_COMPLETED = 6
PLAN_REV_FIRST_READ_RUNNING_AFTER_SEARCH = 6
PLAN_REV_FOLLOWUP_READ_RUNNING_AFTER_SEARCH = 7
PLAN_REV_ANSWER_COMPLETED = 9


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
    started_at = clock()
    step_id = await emitter.step_started(step_number=step_number)
    if on_step_started is not None:
        on_step_started(step_id)
    await _maybe_emit_plan_step_started(
        emitter=emitter,
        run_id=run_id,
        step_number=step_number,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
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
    await _maybe_emit_plan_step_completed(
        emitter=emitter,
        context=context,
        tool_names=list(tool_names),
        tool_call_count=tool_call_count,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
    )
    await session_cache.write_step_completed(
        step_id=context.step_id,
        tool_names=list(tool_names),
        tool_calls_count=tool_call_count,
        duration_ms=duration_ms,
    )
    return duration_ms


async def mark_tool_round_started(
    *,
    context: AgentStepContext,
    emitter: AgentStepEmitter,
    tool_call_count: int,
    tool_names: Sequence[str] = (),
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
) -> None:
    stage = _tool_stage(tool_names)
    if context.step_number == 1:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_plan_revision(context.step_number, PLAN_REV_UNDERSTAND_COMPLETED),
            item={
                "id": "understand",
                "title": "理解问题",
                "status": "completed",
                "kind": "reasoning",
                "summary": "已完成问题理解",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )
    else:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_plan_revision(context.step_number, PLAN_REV_ANSWER_PENDING_FOR_TOOLS),
            item={
                "id": "answer",
                "title": "整理回答",
                "status": "pending",
                "kind": "answer",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )

    if stage == "read":
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_tool_running_revision(context.step_number),
            item={
                "id": "read",
                "title": "读取关键来源",
                "status": "running",
                "kind": "read",
                "summary": f"正在读取 {tool_call_count} 个关键来源",
                "tool_names": list(tool_names),
                "evidence_item_ids": [],
            },
        )
        await _maybe_call_async(
            emitter,
            "run_progress_updated",
            phase="reading",
            label="正在读取关键来源",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=completed_tool_calls,
            max_tool_calls=max_tool_calls,
        )
        return

    await _emit_plan_step_update(
        emitter=emitter,
        run_id=context.run_id,
        revision=_tool_running_revision(context.step_number),
        item={
            "id": "search",
            "title": "查找资料",
            "status": "running",
            "kind": "search",
            "summary": f"正在执行 {tool_call_count} 个工具调用",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    )
    await _maybe_call_async(
        emitter,
        "run_progress_updated",
        phase="researching",
        label="正在查找资料",
        completed_steps=1,
        total_steps=None,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
    )


async def _maybe_emit_plan_step_started(
    *,
    emitter: AgentStepEmitter,
    run_id: str,
    step_number: int,
    completed_tool_calls: int | None,
    max_tool_calls: int | None,
) -> None:
    if step_number == 1:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=run_id,
            revision=_plan_revision(step_number, PLAN_REV_UNDERSTAND_RUNNING),
            item={
                "id": "understand",
                "title": "理解问题",
                "status": "running",
                "kind": "reasoning",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )
        return

    await _emit_plan_step_update(
        emitter=emitter,
        run_id=run_id,
        revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_READ_COMPLETED),
        item={
            "id": "read",
            "title": "读取关键来源",
            "status": "completed",
            "kind": "read",
            "summary": "已完成关键来源读取",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    )
    await _emit_plan_step_update(
        emitter=emitter,
        run_id=run_id,
        revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_ANSWER_RUNNING),
        item={
            "id": "answer",
            "title": "整理回答",
            "status": "running",
            "kind": "answer",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    )
    await _maybe_call_async(
        emitter,
        "run_progress_updated",
        phase="synthesizing",
        label="正在整理回答",
        completed_steps=3,
        total_steps=None,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
    )


async def _maybe_emit_plan_step_completed(
    *,
    emitter: AgentStepEmitter,
    context: AgentStepContext,
    tool_names: list[str],
    tool_call_count: int,
    completed_tool_calls: int | None,
    max_tool_calls: int | None,
) -> None:
    if tool_call_count > 0:
        stage = _tool_stage(tool_names)
        if stage == "read":
            await _emit_plan_step_update(
                emitter=emitter,
                run_id=context.run_id,
                revision=_tool_completed_revision(context.step_number),
                item={
                    "id": "read",
                    "title": "读取关键来源",
                    "status": "completed",
                    "kind": "read",
                    "summary": "已完成关键来源读取",
                    "tool_names": tool_names,
                    "evidence_item_ids": [],
                },
            )
            await _maybe_call_async(
                emitter,
                "run_progress_updated",
                phase="reading",
                label="已完成关键来源读取",
                completed_steps=2,
                total_steps=None,
                completed_tool_calls=completed_tool_calls if completed_tool_calls is not None else tool_call_count,
                max_tool_calls=max_tool_calls,
            )
            return

        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_tool_completed_revision(context.step_number),
            item={
                "id": "search",
                "title": "查找资料",
                "status": "completed",
                "kind": "search",
                "summary": f"完成 {_completed_tool_summary_count(tool_call_count, completed_tool_calls)} 个工具调用",
                "tool_names": tool_names,
                "evidence_item_ids": [],
            },
        )
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_read_running_after_search_revision(context.step_number),
            item={
                "id": "read",
                "title": "读取关键来源",
                "status": "running",
                "kind": "read",
                "summary": "正在整理关键来源",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )
        await _maybe_call_async(
            emitter,
            "run_progress_updated",
            phase="reading",
            label="正在读取关键来源",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=completed_tool_calls if completed_tool_calls is not None else tool_call_count,
            max_tool_calls=max_tool_calls,
        )
        return

    if context.step_number == 1:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=context.run_id,
            revision=_plan_revision(context.step_number, PLAN_REV_UNDERSTAND_COMPLETED),
            item={
                "id": "understand",
                "title": "理解问题",
                "status": "completed",
                "kind": "reasoning",
                "summary": "已完成问题理解",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )

    await _emit_plan_step_update(
        emitter=emitter,
        run_id=context.run_id,
        revision=_plan_revision(context.step_number, PLAN_REV_ANSWER_COMPLETED),
        item={
            "id": "answer",
            "title": "整理回答",
            "status": "completed",
            "kind": "answer",
            "summary": "已完成回答整理",
            "tool_names": [],
            "evidence_item_ids": [],
        },
    )
    await _maybe_call_async(
        emitter,
        "run_progress_updated",
        phase="answering",
        label="已完成回答整理",
        completed_steps=4,
        total_steps=None,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
    )


def _plan_revision(step_number: int, offset: int) -> int:
    normalized_step = max(1, step_number)
    return (normalized_step - 1) * PLAN_REVISION_WINDOW + offset


def _tool_running_revision(step_number: int) -> int:
    offset = PLAN_REV_FIRST_TOOL_RUNNING if step_number == 1 else PLAN_REV_FOLLOWUP_TOOL_RUNNING
    return _plan_revision(step_number, offset)


def _tool_completed_revision(step_number: int) -> int:
    offset = PLAN_REV_FIRST_TOOL_COMPLETED if step_number == 1 else PLAN_REV_FOLLOWUP_TOOL_COMPLETED
    return _plan_revision(step_number, offset)


def _read_running_after_search_revision(step_number: int) -> int:
    offset = PLAN_REV_FIRST_READ_RUNNING_AFTER_SEARCH if step_number == 1 else PLAN_REV_FOLLOWUP_READ_RUNNING_AFTER_SEARCH
    return _plan_revision(step_number, offset)


def _tool_stage(tool_names: Sequence[str]) -> str:
    names = [tool_name for tool_name in tool_names if tool_name]
    if names and all(tool_name == "url_read" for tool_name in names):
        return "read"
    return "search"


def _completed_tool_summary_count(tool_call_count: int, completed_tool_calls: int | None) -> int:
    if completed_tool_calls is not None:
        return max(tool_call_count, completed_tool_calls)
    return tool_call_count


async def _emit_plan_step_update(
    *,
    emitter: AgentStepEmitter,
    run_id: str,
    revision: int,
    item: dict,
) -> None:
    await _maybe_call_async(
        emitter,
        "plan_step_updated",
        plan_id=f"plan-{run_id}",
        revision=revision,
        item=item,
    )


async def _maybe_call_async(emitter: AgentStepEmitter, method_name: str, **kwargs) -> None:
    method = getattr(emitter, method_name, None)
    if method is None or not iscoroutinefunction(method):
        return
    await method(**kwargs)
