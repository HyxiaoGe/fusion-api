"""Agent step 生命周期薄边界。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    plan_items: dict[str, dict] = field(default_factory=dict)


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
    plan_items: dict[str, dict] | None = None,
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
        plan_items=plan_items,
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
        plan_items=_copy_plan_items(plan_items),
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
    tool_arguments: Sequence[dict] = (),
    completed_tool_calls: int | None = None,
    max_tool_calls: int | None = None,
) -> None:
    stage = _tool_stage(tool_names)
    await _ensure_tool_plan_initialized(
        emitter=emitter,
        context=context,
        stage=stage,
        tool_call_count=tool_call_count,
        tool_names=tool_names,
        tool_arguments=tool_arguments,
    )
    if context.step_number == 1:
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_plan_revision(context.step_number, PLAN_REV_UNDERSTAND_COMPLETED),
            item=_plan_item_update(
                context.plan_items,
                item_id="understand",
                title="理解问题",
                status="completed",
                kind="reasoning",
                summary="已完成问题理解",
                tool_names=[],
                evidence_item_ids=[],
            ),
        )
    else:
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_plan_revision(context.step_number, PLAN_REV_ANSWER_PENDING_FOR_TOOLS),
            item=_plan_item_update(
                context.plan_items,
                item_id="answer",
                title="整理回答",
                status="pending",
                kind="answer",
                summary=_answer_running_summary(max(completed_tool_calls or 0, tool_call_count)),
                tool_names=[],
                evidence_item_ids=[],
            ),
        )

    if stage == "read":
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_tool_running_revision(context.step_number),
            item=_plan_item_update(
                context.plan_items,
                item_id="read",
                title="读取关键来源",
                status="running",
                kind="read",
                summary=f"正在读取 {tool_call_count} 个关键来源",
                tool_names=list(tool_names),
                evidence_item_ids=[],
            ),
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

    if stage == "tool":
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_tool_running_revision(context.step_number),
            item=_plan_item_update(
                context.plan_items,
                item_id="tool",
                title="调用外部工具",
                status="running",
                kind="other",
                summary=f"正在调用 {tool_call_count} 个外部工具",
                tool_names=list(tool_names),
                evidence_item_ids=[],
            ),
        )
        await _maybe_call_async(
            emitter,
            "run_progress_updated",
            phase="researching",
            label="正在调用外部工具",
            completed_steps=1,
            total_steps=None,
            completed_tool_calls=completed_tool_calls,
            max_tool_calls=max_tool_calls,
        )
        return

    await _emit_context_plan_step_update(
        emitter=emitter,
        context=context,
        revision=_tool_running_revision(context.step_number),
        item=_plan_item_update(
            context.plan_items,
            item_id="search",
            title=_search_running_title(tool_arguments, tool_call_count),
            status="running",
            kind="search",
            summary=_search_running_summary(tool_arguments, tool_call_count),
            tool_names=list(tool_names),
            evidence_item_ids=[],
        ),
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
    plan_items: dict[str, dict] | None = None,
) -> None:
    if not plan_items:
        return

    if step_number == 1:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=run_id,
            revision=_plan_revision(step_number, PLAN_REV_UNDERSTAND_RUNNING),
            item=_plan_item_update(
                plan_items,
                item_id="understand",
                title="理解问题",
                status="running",
                kind="reasoning",
                tool_names=[],
                evidence_item_ids=[],
            ),
        )
        return

    if "tool" in plan_items:
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=run_id,
            revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_READ_COMPLETED),
            item=_plan_item_update(
                plan_items,
                item_id="tool",
                title="调用外部工具",
                status="completed",
                kind="other",
                summary="已完成外部工具调用",
                tool_names=[],
                evidence_item_ids=[],
            ),
        )
        await _emit_plan_step_update(
            emitter=emitter,
            run_id=run_id,
            revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_ANSWER_RUNNING),
            item=_plan_item_update(
                plan_items,
                item_id="answer",
                title="整理回答",
                status="running",
                kind="answer",
                summary=_answer_running_summary(completed_tool_calls),
                tool_names=[],
                evidence_item_ids=[],
            ),
        )
        await _maybe_call_async(
            emitter,
            "run_progress_updated",
            phase="synthesizing",
            label="正在整理工具结果",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=completed_tool_calls,
            max_tool_calls=max_tool_calls,
        )
        return

    await _emit_plan_step_update(
        emitter=emitter,
        run_id=run_id,
        revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_READ_COMPLETED),
        item=_plan_item_update(
            plan_items,
            item_id="read",
            title="读取关键来源",
            status="completed",
            kind="read",
            summary="已完成关键来源读取",
            tool_names=[],
            evidence_item_ids=[],
        ),
    )
    await _emit_plan_step_update(
        emitter=emitter,
        run_id=run_id,
        revision=_plan_revision(step_number, PLAN_REV_FOLLOWUP_ANSWER_RUNNING),
        item=_plan_item_update(
            plan_items,
            item_id="answer",
            title="整理回答",
            status="running",
            kind="answer",
            summary=_answer_running_summary(completed_tool_calls),
            tool_names=[],
            evidence_item_ids=[],
        ),
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
    if not context.plan_items and tool_call_count <= 0:
        return

    if tool_call_count > 0:
        stage = _tool_stage(tool_names)
        if stage == "read":
            await _emit_context_plan_step_update(
                emitter=emitter,
                context=context,
                revision=_tool_completed_revision(context.step_number),
                item=_plan_item_update(
                    context.plan_items,
                    item_id="read",
                    title="读取关键来源",
                    status="completed",
                    kind="read",
                    summary="已完成关键来源读取",
                    tool_names=tool_names,
                    evidence_item_ids=[],
                ),
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

        if stage == "tool":
            await _emit_context_plan_step_update(
                emitter=emitter,
                context=context,
                revision=_tool_completed_revision(context.step_number),
                item=_plan_item_update(
                    context.plan_items,
                    item_id="tool",
                    title="调用外部工具",
                    status="completed",
                    kind="other",
                    summary=f"完成 {_completed_tool_summary_count(tool_call_count, completed_tool_calls)} 个工具调用",
                    tool_names=tool_names,
                    evidence_item_ids=[],
                ),
            )
            await _maybe_call_async(
                emitter,
                "run_progress_updated",
                phase="synthesizing",
                label="正在整理工具结果",
                completed_steps=2,
                total_steps=None,
                completed_tool_calls=(completed_tool_calls if completed_tool_calls is not None else tool_call_count),
                max_tool_calls=max_tool_calls,
            )
            return

        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_tool_completed_revision(context.step_number),
            item=_plan_item_update(
                context.plan_items,
                item_id="search",
                title="查找资料",
                status="completed",
                kind="search",
                summary=f"完成 {_completed_tool_summary_count(tool_call_count, completed_tool_calls)} 个工具调用",
                tool_names=tool_names,
                evidence_item_ids=[],
            ),
        )
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_read_running_after_search_revision(context.step_number),
            item=_plan_item_update(
                context.plan_items,
                item_id="read",
                title="读取关键来源",
                status="running",
                kind="read",
                summary="正在整理关键来源",
                tool_names=[],
                evidence_item_ids=[],
            ),
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
        await _emit_context_plan_step_update(
            emitter=emitter,
            context=context,
            revision=_plan_revision(context.step_number, PLAN_REV_UNDERSTAND_COMPLETED),
            item=_plan_item_update(
                context.plan_items,
                item_id="understand",
                title="理解问题",
                status="completed",
                kind="reasoning",
                summary="已完成问题理解",
                tool_names=[],
                evidence_item_ids=[],
            ),
        )

    await _emit_context_plan_step_update(
        emitter=emitter,
        context=context,
        revision=_plan_revision(context.step_number, PLAN_REV_ANSWER_COMPLETED),
        item=_plan_item_update(
            context.plan_items,
            item_id="answer",
            title="整理回答",
            status="completed",
            kind="answer",
            summary="已完成回答整理",
            tool_names=[],
            evidence_item_ids=[],
        ),
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


def _copy_plan_items(plan_items: dict[str, dict] | None) -> dict[str, dict]:
    return {str(item_id): dict(item) for item_id, item in (plan_items or {}).items()}


async def _ensure_tool_plan_initialized(
    *,
    emitter: AgentStepEmitter,
    context: AgentStepContext,
    stage: str,
    tool_call_count: int,
    tool_names: Sequence[str],
    tool_arguments: Sequence[dict],
) -> None:
    if context.plan_items:
        return

    items = _initial_tool_plan_items(
        stage=stage,
        tool_call_count=tool_call_count,
        tool_names=tool_names,
        tool_arguments=tool_arguments,
    )
    context.plan_items.update({str(item["id"]): dict(item) for item in items})
    await _maybe_call_async(
        emitter,
        "plan_snapshot",
        plan_id=f"plan-{context.run_id}",
        revision=1,
        items=items,
    )


def _initial_tool_plan_items(
    *,
    stage: str,
    tool_call_count: int,
    tool_names: Sequence[str],
    tool_arguments: Sequence[dict],
) -> list[dict]:
    items = [
        {
            "id": "understand",
            "title": "理解问题",
            "status": "running",
            "kind": "reasoning",
            "summary": "判断资料需求和回答路径",
            "tool_names": [],
            "evidence_item_ids": [],
        }
    ]
    if stage == "tool":
        items.extend(
            [
                {
                    "id": "tool",
                    "title": "调用外部工具",
                    "status": "pending",
                    "kind": "other",
                    "summary": f"准备调用 {tool_call_count} 个外部工具",
                    "tool_names": list(tool_names),
                    "evidence_item_ids": [],
                },
                {
                    "id": "answer",
                    "title": "整理回答",
                    "status": "pending",
                    "kind": "answer",
                    "summary": "基于工具结果给出结论和必要说明",
                    "tool_names": [],
                    "evidence_item_ids": [],
                },
            ]
        )
        return items
    if stage != "read":
        items.append(
            {
                "id": "search",
                "title": _search_running_title(tool_arguments, tool_call_count),
                "status": "pending",
                "kind": "search",
                "summary": _search_running_summary(tool_arguments, tool_call_count),
                "tool_names": list(tool_names),
                "evidence_item_ids": [],
            }
        )
    items.extend(
        [
            {
                "id": "read",
                "title": "读取关键来源",
                "status": "pending",
                "kind": "read",
                "summary": "必要时读取关键来源核验",
                "tool_names": list(tool_names),
                "evidence_item_ids": [],
            },
            {
                "id": "answer",
                "title": "整理回答",
                "status": "pending",
                "kind": "answer",
                "summary": "基于可用依据给出结论、推荐和不确定性",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        ]
    )
    return items


def _plan_item_update(
    plan_items: dict[str, dict] | None,
    *,
    item_id: str,
    title: str,
    status: str,
    kind: str,
    summary: str | None = None,
    tool_names: list[str] | None = None,
    evidence_item_ids: list[str] | None = None,
) -> dict:
    template = (plan_items or {}).get(item_id) or {}
    item = {
        "id": item_id,
        "title": str(template.get("title") or title),
        "status": status,
        "kind": kind,
        "tool_names": tool_names if tool_names is not None else list(template.get("tool_names") or []),
        "evidence_item_ids": (
            evidence_item_ids if evidence_item_ids is not None else list(template.get("evidence_item_ids") or [])
        ),
    }
    merged_summary = _merge_plan_summary(template.get("summary"), summary)
    if merged_summary:
        item["summary"] = merged_summary
    return item


def _merge_plan_summary(template_summary: object, summary: str | None) -> str | None:
    if not summary:
        return template_summary if isinstance(template_summary, str) and template_summary else None
    budget = _extract_budget(template_summary)
    if budget:
        return f"{summary} · 预算：{budget}"
    return summary


def _extract_budget(template_summary: object) -> str | None:
    if not isinstance(template_summary, str):
        return None
    marker = "预算："
    if marker not in template_summary:
        return None
    return template_summary.split(marker, 1)[1].strip() or None


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
    offset = (
        PLAN_REV_FIRST_READ_RUNNING_AFTER_SEARCH if step_number == 1 else PLAN_REV_FOLLOWUP_READ_RUNNING_AFTER_SEARCH
    )
    return _plan_revision(step_number, offset)


def _tool_stage(tool_names: Sequence[str]) -> str:
    names = [tool_name for tool_name in tool_names if tool_name]
    if not names:
        return "search"
    if names and all(tool_name == "url_read" for tool_name in names):
        return "read"
    if names and all(tool_name in {"web_search", "url_read"} for tool_name in names):
        return "search"
    return "tool"


def _completed_tool_summary_count(tool_call_count: int, completed_tool_calls: int | None) -> int:
    if completed_tool_calls is not None:
        return max(tool_call_count, completed_tool_calls)
    return tool_call_count


def _search_running_title(tool_arguments: Sequence[dict], tool_call_count: int) -> str:
    queries = _search_queries(tool_arguments)
    if len(queries) == 1:
        return f"搜索：{_truncate_plan_text(queries[0], 36)}"
    if len(queries) > 1:
        return f"搜索：{len(queries)} 个查询"
    return "查找资料"


def _search_running_summary(tool_arguments: Sequence[dict], tool_call_count: int) -> str:
    queries = _search_queries(tool_arguments)
    if queries:
        visible_queries = [_truncate_plan_text(query, 36) for query in queries[:3]]
        suffix = " 等" if len(queries) > 3 else ""
        return f"正在搜索：{'、'.join(visible_queries)}{suffix}"
    return f"正在执行 {tool_call_count} 个工具调用"


def _search_queries(tool_arguments: Sequence[dict]) -> list[str]:
    queries: list[str] = []
    for arguments in tool_arguments:
        if not isinstance(arguments, dict):
            continue
        query = arguments.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return queries


def _truncate_plan_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def _answer_running_summary(completed_tool_calls: int | None) -> str:
    if completed_tool_calls and completed_tool_calls > 0:
        return "基于可用依据给出结论、推荐和不确定性"
    return "基于已有上下文直接回答，不使用联网工具"


async def _emit_context_plan_step_update(
    *,
    emitter: AgentStepEmitter,
    context: AgentStepContext,
    revision: int,
    item: dict,
) -> None:
    context.plan_items[str(item["id"])] = dict(item)
    await _emit_plan_step_update(
        emitter=emitter,
        run_id=context.run_id,
        revision=revision,
        item=item,
    )


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
