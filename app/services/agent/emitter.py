"""AgentEventEmitter — 控制面事件唯一发送方."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from app.services.agent import events as ev
from app.services.agent.sanitizer import cap_and_truncate, sanitize_arguments


class _RedisWriter(Protocol):
    async def append_chunk(self, conversation_id: str, chunk_type: str,
                           payload: dict[str, Any]) -> None: ...


class AgentEventEmitter:
    """单 run 内发 agent_event；并发安全；维护 step 上下文。"""

    def __init__(self, *, run_id: str, trace_id: str,
                 conversation_id: str, redis_writer: _RedisWriter) -> None:
        self._run_id = run_id
        self._trace_id = trace_id
        self._conv_id = conversation_id
        self._writer = redis_writer
        self._sequence = 0
        self._current_step_id: str | None = None
        self._lock = asyncio.Lock()

    async def _emit(self, event: ev.AgentEventBase) -> None:
        """在 lock 内原子分配 sequence + ts，再 dump + write，最后递增。"""
        async with self._lock:
            event.sequence = self._sequence
            event.ts = time.time()
            payload = event.model_dump(mode="json")
            await self._writer.append_chunk(self._conv_id, "agent_event", payload)
            self._sequence += 1

    def _envelope(self, *, tool_call_id: str | None = None,
                  step_id: str | None = ...) -> dict[str, Any]:
        """构造 envelope；sequence 与 ts 用占位值，由 _emit 在 lock 内回填。"""
        return dict(
            run_id=self._run_id,
            trace_id=self._trace_id,
            sequence=0,           # 占位，_emit 回填
            ts=0.0,               # 占位，_emit 回填
            step_id=self._current_step_id if step_id is ... else step_id,
            tool_call_id=tool_call_id,
            parent_run_id=None,
            parent_step_id=None,
        )

    async def run_started(self, *, model: str, tools: list[str],
                          config: dict[str, Any]) -> None:
        await self._emit(ev.RunStarted(
            type="run_started",
            conversation_id=self._conv_id, model=model, tools=tools, config=config,
            **self._envelope(step_id=None),
        ))

    async def step_started(self, *, step_number: int) -> str:
        step_id = str(uuid.uuid4())
        # 在发事件前先设 current_step_id，让 envelope 带上自己
        self._current_step_id = step_id
        await self._emit(ev.StepStarted(
            type="step_started", step_number=step_number,
            **self._envelope(),
        ))
        return step_id

    async def tool_call_started(self, *, tool_call_id: str, tool_name: str,
                                arguments: dict[str, Any]) -> None:
        sanitized = sanitize_arguments(tool_name, arguments)
        await self._emit(ev.ToolCallStarted(
            type="tool_call_started", tool_name=tool_name, arguments=sanitized,
            **self._envelope(tool_call_id=tool_call_id),
        ))

    async def tool_call_delta(self, *, tool_call_id: str, tool_name: str,
                              delta: dict[str, Any]) -> None:
        await self._emit(ev.ToolCallDelta(
            type="tool_call_delta", tool_name=tool_name, delta=delta,
            **self._envelope(tool_call_id=tool_call_id),
        ))

    async def tool_call_completed(self, *, tool_call_id: str, tool_name: str,
                                  status: str, duration_ms: int,
                                  result_summary: dict[str, Any],
                                  error: str | None = None) -> None:
        capped = cap_and_truncate(result_summary, max_bytes=1024)
        await self._emit(ev.ToolCallCompleted(
            type="tool_call_completed", tool_name=tool_name, status=status,
            duration_ms=duration_ms, result_summary=capped, error=error,
            **self._envelope(tool_call_id=tool_call_id),
        ))

    async def step_completed(self, *, step_number: int, tool_call_count: int,
                             duration_ms: int) -> None:
        await self._emit(ev.StepCompleted(
            type="step_completed", step_number=step_number,
            tool_call_count=tool_call_count, duration_ms=duration_ms,
            **self._envelope(),
        ))
        self._current_step_id = None

    async def run_limit_reached(self, *, reason: str) -> None:
        await self._emit(ev.RunLimitReached(
            type="run_limit_reached", reason=reason,
            **self._envelope(step_id=None),
        ))

    async def run_interrupted(self, *, reason: str) -> None:
        await self._emit(ev.RunInterrupted(
            type="run_interrupted", reason=reason,
            **self._envelope(step_id=None),
        ))

    async def run_failed(self, *, error_code: str, message: str) -> None:
        await self._emit(ev.RunFailed(
            type="run_failed", error_code=error_code, message=message,
            **self._envelope(step_id=None),
        ))

    async def run_completed(self, *, total_steps: int, total_tool_calls: int,
                            finish_reason: str) -> None:
        await self._emit(ev.RunCompleted(
            type="run_completed", total_steps=total_steps,
            total_tool_calls=total_tool_calls, finish_reason=finish_reason,
            **self._envelope(step_id=None),
        ))
