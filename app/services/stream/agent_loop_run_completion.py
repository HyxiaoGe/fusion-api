"""Agent loop run 终态收尾编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.stream.agent_loop_policy import AgentRunTerminalState
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.run_finalizer import InterruptedStatusWriteError
from app.services.stream_state_service import StreamOwnershipLostError, StreamWriteTerminalError

PersistMessageFn = Callable[..., Any]
FinalizeStreamFn = Callable[..., Awaitable[Any]]
TerminalRunFn = Callable[..., Awaitable[Any]]
WarningFn = Callable[[str], None]
DurationMsFactory = Callable[[], int]


@dataclass(frozen=True)
class AgentLoopRunCompletionContext:
    db: Any
    conversation_id: str
    task_id: str
    run_id: str
    model_id: str
    assistant_message_id: str
    emitter: Any
    session_cache: Any
    state: AgentLoopState
    duration_ms_factory: DurationMsFactory


def persist_run_message(
    *,
    context: AgentLoopRunCompletionContext,
    persist_message_fn: PersistMessageFn,
    only_if_content: bool = False,
    partial: bool = False,
) -> None:
    if only_if_content and not context.state.content_blocks and context.state.final_usage() is None:
        return

    persist_message_fn(
        context.db,
        context.assistant_message_id,
        context.conversation_id,
        context.model_id,
        context.state.content_blocks,
        context.state.final_usage(),
        partial,
    )


async def finalize_completed_run(
    *,
    context: AgentLoopRunCompletionContext,
    terminal_state: AgentRunTerminalState,
    persist_message_fn: PersistMessageFn,
    complete_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
) -> None:
    persist_run_message(context=context, persist_message_fn=persist_message_fn, partial=False)
    await complete_agent_run_fn(
        emitter=context.emitter,
        session_cache=context.session_cache,
        stats=context.state.run_stats(context.run_id),
        duration_ms_factory=context.duration_ms_factory,
        session_status=terminal_state.session_status,
        finish_reason=terminal_state.run_finish_reason,
        limit_reason=context.state.limit_reason,
    )
    context.state.mark_terminal_emitted()
    await finalize_stream_fn(context.conversation_id, success=True, task_id=context.task_id)


async def finalize_superseded_run(
    *,
    context: AgentLoopRunCompletionContext,
    error_msg: str | None,
    persist_message_fn: PersistMessageFn,
    interrupt_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
) -> None:
    persist_run_message(context=context, persist_message_fn=persist_message_fn, partial=True)
    await interrupt_agent_run_fn(
        emitter=context.emitter,
        session_cache=context.session_cache,
        stats=context.state.run_stats(context.run_id),
        duration_ms_factory=context.duration_ms_factory,
        current_step_id=context.state.current_step_id,
        reason="superseded",
    )
    context.state.mark_terminal_emitted()
    await finalize_stream_fn(
        context.conversation_id,
        success=False,
        error_msg=error_msg or "被新请求取代",
        task_id=context.task_id,
    )


async def finalize_cancelled_run(
    *,
    context: AgentLoopRunCompletionContext,
    persist_message_fn: PersistMessageFn,
    interrupt_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
    warning_fn: WarningFn,
) -> None:
    persist_run_message(
        context=context,
        persist_message_fn=persist_message_fn,
        only_if_content=True,
        partial=True,
    )
    try:
        await interrupt_agent_run_fn(
            emitter=context.emitter,
            session_cache=context.session_cache,
            stats=context.state.run_stats(context.run_id),
            duration_ms_factory=context.duration_ms_factory,
            current_step_id=context.state.current_step_id,
            reason="user_cancelled",
        )
        context.state.mark_terminal_emitted()
    except InterruptedStatusWriteError:
        raise
    except StreamOwnershipLostError as emit_exc:
        warning_fn(f"emit run_interrupted ownership lost，外部 stop 已接管流终态: {emit_exc}")
        context.state.mark_terminal_emitted()
    except StreamWriteTerminalError:
        raise
    except Exception as emit_exc:  # noqa: BLE001 — 非 Stream 写终止错误不能阻塞 cancel 传播
        warning_fn(f"emit run_interrupted 失败: {emit_exc}")
    await finalize_stream_fn(context.conversation_id, success=False, error_msg="用户中止", task_id=context.task_id)


async def finalize_failed_run(
    *,
    context: AgentLoopRunCompletionContext,
    error: Exception,
    persist_message_fn: PersistMessageFn,
    fail_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
    warning_fn: WarningFn,
) -> None:
    persist_run_message(
        context=context,
        persist_message_fn=persist_message_fn,
        only_if_content=True,
        partial=True,
    )
    structured_error_code = _safe_structured_error_code(error)
    try:
        await fail_agent_run_fn(
            emitter=context.emitter,
            session_cache=context.session_cache,
            stats=context.state.run_stats(context.run_id),
            duration_ms_factory=context.duration_ms_factory,
            current_step_id=context.state.current_step_id,
            error_code=structured_error_code or type(error).__name__,
            message=str(error),
        )
        context.state.mark_terminal_emitted()
    except StreamWriteTerminalError:
        raise
    except Exception as emit_exc:  # noqa: BLE001
        warning_fn(f"emit run_failed 失败: {emit_exc}")
    finalize_kwargs = {
        "success": False,
        "error_msg": str(error),
        "task_id": context.task_id,
    }
    if structured_error_code:
        finalize_kwargs["error_code"] = structured_error_code
    await finalize_stream_fn(context.conversation_id, **finalize_kwargs)


def _safe_structured_error_code(error: Exception) -> str:
    candidate = getattr(error, "error_code", None)
    if not isinstance(candidate, str) or not 1 <= len(candidate) <= 64:
        return ""
    if not all(char.isascii() and (char.isalnum() or char == "_") for char in candidate):
        return ""
    return candidate


async def write_fallback_run_error(
    *,
    context: AgentLoopRunCompletionContext,
    write_fallback_error_status_fn: TerminalRunFn,
    warning_fn: WarningFn,
) -> None:
    if context.state.terminal_emitted:
        return

    try:
        await write_fallback_error_status_fn(
            session_cache=context.session_cache,
            stats=context.state.run_stats(context.run_id),
            duration_ms_factory=context.duration_ms_factory,
        )
    except Exception as exc:  # noqa: BLE001
        warning_fn(f"finally 兜底 write_session_status 失败: {exc}")
