"""Agent run 终态事件与 session_cache 写入边界。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.services.stream.agent_loop_policy import AgentSessionStatus, RunCompletedFinishReason


@dataclass(frozen=True)
class AgentRunStats:
    run_id: str
    total_steps: int
    total_tool_calls: int


InterruptionReason = Literal["user_cancelled", "superseded"]
StepTerminalStatus = Literal["failed", "interrupted"]
ErrorSessionStatus = Literal["error"]
InterruptedSessionStatus = Literal["interrupted"]
TerminalSessionStatus = AgentSessionStatus | ErrorSessionStatus | InterruptedSessionStatus


class AgentRunEmitter(Protocol):
    async def run_started(
        self,
        *,
        message_id: str,
        model: str,
        tools: list[str],
        config: dict[str, Any],
    ) -> None: ...

    async def run_completed(
        self,
        *,
        total_steps: int,
        total_tool_calls: int,
        finish_reason: RunCompletedFinishReason,
    ) -> None: ...

    async def run_interrupted(self, *, reason: InterruptionReason) -> None: ...

    async def run_failed(self, *, error_code: str, message: str) -> None: ...


class AgentRunSessionCache(Protocol):
    async def write_session_started(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
        message_id: str,
    ) -> None: ...

    async def write_step_terminal(self, *, step_id: str, status: StepTerminalStatus) -> None: ...

    async def write_session_status(
        self,
        *,
        run_id: str,
        status: TerminalSessionStatus,
        total_steps: int,
        total_tool_calls: int,
        total_duration_ms: int | None = None,
    ) -> None: ...


DurationMsFactory = Callable[[], int]


async def start_agent_run(
    *,
    emitter: AgentRunEmitter,
    session_cache: AgentRunSessionCache,
    run_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    message_id: str,
    tools: list[str],
    config: dict[str, Any],
) -> None:
    await session_cache.write_session_started(
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=user_id,
        model_id=model_id,
        provider=provider,
        message_id=message_id,
    )
    await emitter.run_started(
        message_id=message_id,
        model=model_id,
        tools=tools,
        config=config,
    )


async def complete_agent_run(
    *,
    emitter: AgentRunEmitter,
    session_cache: AgentRunSessionCache,
    stats: AgentRunStats,
    duration_ms_factory: DurationMsFactory,
    session_status: AgentSessionStatus,
    finish_reason: RunCompletedFinishReason,
) -> None:
    await emitter.run_completed(
        total_steps=stats.total_steps,
        total_tool_calls=stats.total_tool_calls,
        finish_reason=finish_reason,
    )
    await _write_session_status(
        session_cache=session_cache,
        stats=stats,
        duration_ms_factory=duration_ms_factory,
        status=session_status,
    )


async def interrupt_agent_run(
    *,
    emitter: AgentRunEmitter,
    session_cache: AgentRunSessionCache,
    stats: AgentRunStats,
    duration_ms_factory: DurationMsFactory,
    current_step_id: str | None,
    reason: InterruptionReason,
) -> None:
    if current_step_id is not None:
        await session_cache.write_step_terminal(step_id=current_step_id, status="interrupted")
    await emitter.run_interrupted(reason=reason)
    await _write_session_status(
        session_cache=session_cache,
        stats=stats,
        duration_ms_factory=duration_ms_factory,
        status="interrupted",
    )


async def fail_agent_run(
    *,
    emitter: AgentRunEmitter,
    session_cache: AgentRunSessionCache,
    stats: AgentRunStats,
    duration_ms_factory: DurationMsFactory,
    current_step_id: str | None,
    error_code: str,
    message: str,
) -> None:
    if current_step_id is not None:
        await session_cache.write_step_terminal(step_id=current_step_id, status="failed")
    await emitter.run_failed(error_code=error_code, message=message)
    await _write_session_status(
        session_cache=session_cache,
        stats=stats,
        duration_ms_factory=duration_ms_factory,
        status="error",
    )


async def write_fallback_error_status(
    *,
    session_cache: AgentRunSessionCache,
    stats: AgentRunStats,
    duration_ms_factory: DurationMsFactory,
) -> None:
    await _write_session_status(
        session_cache=session_cache,
        stats=stats,
        duration_ms_factory=duration_ms_factory,
        status="error",
    )


async def _write_session_status(
    *,
    session_cache: AgentRunSessionCache,
    stats: AgentRunStats,
    duration_ms_factory: DurationMsFactory,
    status: TerminalSessionStatus,
) -> None:
    await session_cache.write_session_status(
        run_id=stats.run_id,
        status=status,
        total_steps=stats.total_steps,
        total_tool_calls=stats.total_tool_calls,
        total_duration_ms=duration_ms_factory(),
    )
