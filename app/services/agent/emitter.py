"""AgentEventEmitter — 控制面事件唯一发送方."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from app.services.agent import events as ev
from app.services.agent.sanitizer import cap_and_truncate, sanitize_arguments

# Sentinel 用于 _envelope 区分"未传 step_id（用 current）"vs"显式传 None"
_USE_CURRENT_STEP = object()


class _RedisWriter(Protocol):
    """emitter 的 Redis 写入抽象。

    本仓库目前没有具体实现：Task 9 (stream_handler) 会提供一个 adapter，
    把 (conv_id, chunk_type, payload: dict) 桥接到现有
    stream_state_service.append_chunk(conv_id, chunk_type, content, block_id, task_id=task_id)
    （payload JSON 序列化进 content，block_id 留空）。
    单元测试用 unittest.mock.AsyncMock 满足此 Protocol。
    """

    async def append_chunk(
        self,
        conversation_id: str,
        task_id: str,
        chunk_type: str,
        payload: dict[str, Any],
    ) -> None: ...


class AgentEventEmitter:
    """单 run 内发 agent_event；并发安全；维护 step 上下文。"""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        conversation_id: str,
        task_id: str,
        redis_writer: _RedisWriter,
    ) -> None:
        self._run_id = run_id
        self._trace_id = trace_id
        self._conv_id = conversation_id
        self._task_id = task_id
        self._writer = redis_writer
        self._sequence = 0
        self._current_step_id: str | None = None
        self._lock = asyncio.Lock()

    async def _emit(self, event: ev.AgentEventBase) -> None:
        """在 lock 内原子分配 sequence + ts，再 dump + write，最后递增。

        依赖 Pydantic v2 默认行为：模型字段可赋值且不重新校验
        （未启用 frozen / validate_assignment）。extra="forbid" 只拒绝额外字段，
        不阻塞已声明字段的 mutation。若未来在 AgentEventBase 启用
        validate_assignment，本方法的 mutation 会触发额外校验开销。
        """
        async with self._lock:
            event.sequence = self._sequence
            event.ts = time.time()
            payload = event.model_dump(mode="json")
            await self._writer.append_chunk(self._conv_id, self._task_id, "agent_event", payload)
            self._sequence += 1

    def _envelope(self, *, tool_call_id: str | None = None, step_id: Any = _USE_CURRENT_STEP) -> dict[str, Any]:
        """构造 envelope 字段；sequence 与 ts 用占位值，由 _emit 在 lock 内回填。

        返回的 dict 不可直接发出 — sequence/ts 必须由 _emit 在 lock 内回填，
        否则会和真实顺序错位。

        step_id 默认从 _current_step_id 派生；run-level 事件需显式传 None。
        """
        return dict(
            run_id=self._run_id,
            trace_id=self._trace_id,
            sequence=0,  # 占位，_emit 回填
            ts=0.0,  # 占位，_emit 回填
            step_id=self._current_step_id if step_id is _USE_CURRENT_STEP else step_id,
            tool_call_id=tool_call_id,
            parent_run_id=None,
            parent_step_id=None,
        )

    async def run_started(self, *, message_id: str, model: str, tools: list[str], config: dict[str, Any]) -> None:
        await self._emit(
            ev.RunStarted(
                type="run_started",
                conversation_id=self._conv_id,
                message_id=message_id,
                model=model,
                tools=tools,
                config=config,
                **self._envelope(step_id=None),
            )
        )

    async def step_started(self, *, step_number: int) -> str:
        step_id = str(uuid.uuid4())
        # 在发事件前先设 current_step_id，让 envelope 带上自己
        self._current_step_id = step_id
        await self._emit(
            ev.StepStarted(
                type="step_started",
                step_number=step_number,
                **self._envelope(),
            )
        )
        return step_id

    async def tool_call_started(self, *, tool_call_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
        sanitized = sanitize_arguments(tool_name, arguments)
        await self._emit(
            ev.ToolCallStarted(
                type="tool_call_started",
                tool_name=tool_name,
                arguments=sanitized,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def tool_call_delta(self, *, tool_call_id: str, tool_name: str, delta: dict[str, Any]) -> None:
        await self._emit(
            ev.ToolCallDelta(
                type="tool_call_delta",
                tool_name=tool_name,
                delta=delta,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def tool_call_completed(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        status: str,
        duration_ms: int,
        result_summary: dict[str, Any],
        error: str | None = None,
    ) -> None:
        capped = cap_and_truncate(result_summary, max_bytes=1024)
        await self._emit(
            ev.ToolCallCompleted(
                type="tool_call_completed",
                tool_name=tool_name,
                status=status,
                duration_ms=duration_ms,
                result_summary=capped,
                error=error,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def step_completed(self, *, step_number: int, tool_call_count: int, duration_ms: int) -> None:
        await self._emit(
            ev.StepCompleted(
                type="step_completed",
                step_number=step_number,
                tool_call_count=tool_call_count,
                duration_ms=duration_ms,
                **self._envelope(),
            )
        )
        self._current_step_id = None

    async def run_limit_reached(self, *, reason: str) -> None:
        await self._emit(
            ev.RunLimitReached(
                type="run_limit_reached",
                reason=reason,
                **self._envelope(step_id=None),
            )
        )

    async def run_interrupted(self, *, reason: str) -> None:
        await self._emit(
            ev.RunInterrupted(
                type="run_interrupted",
                reason=reason,
                **self._envelope(step_id=None),
            )
        )

    async def run_failed(self, *, error_code: str, message: str) -> None:
        await self._emit(
            ev.RunFailed(
                type="run_failed",
                error_code=error_code,
                message=message,
                **self._envelope(step_id=None),
            )
        )

    async def run_completed(self, *, total_steps: int, total_tool_calls: int, finish_reason: str) -> None:
        await self._emit(
            ev.RunCompleted(
                type="run_completed",
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                finish_reason=finish_reason,
                **self._envelope(step_id=None),
            )
        )

    async def run_progress_updated(
        self,
        *,
        phase: str,
        label: str,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        completed_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        await self._emit(
            ev.RunProgressUpdated(
                type="run_progress_updated",
                protocol_version=2,
                phase=phase,
                label=label,
                completed_steps=completed_steps,
                total_steps=total_steps,
                completed_tool_calls=completed_tool_calls,
                max_tool_calls=max_tool_calls,
                **self._envelope(step_id=None),
            )
        )

    async def plan_snapshot(self, *, plan_id: str, revision: int, items: list[dict[str, Any]]) -> None:
        await self._emit(
            ev.PlanSnapshot(
                type="plan_snapshot",
                protocol_version=2,
                plan_id=plan_id,
                revision=revision,
                items=items,
                **self._envelope(step_id=None),
            )
        )

    async def plan_step_updated(self, *, plan_id: str, revision: int, item: dict[str, Any]) -> None:
        await self._emit(
            ev.PlanStepUpdated(
                type="plan_step_updated",
                protocol_version=2,
                plan_id=plan_id,
                revision=revision,
                item=item,
                **self._envelope(),
            )
        )

    async def tool_result_digest(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        status: str,
        title: str,
        summary: str,
        key_findings: list[str] | None = None,
        source_refs: list[str] | None = None,
        truncated: bool = False,
    ) -> None:
        await self._emit(
            ev.ToolResultDigest(
                type="tool_result_digest",
                protocol_version=2,
                tool_name=tool_name,
                status=status,
                title=title,
                summary=summary,
                key_findings=key_findings or [],
                source_refs=source_refs or [],
                truncated=truncated,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def evidence_item_upserted(
        self,
        *,
        tool_call_id: str | None = None,
        evidence: dict[str, Any],
    ) -> None:
        await self._emit(
            ev.EvidenceItemUpserted(
                type="evidence_item_upserted",
                protocol_version=2,
                evidence=evidence,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )
