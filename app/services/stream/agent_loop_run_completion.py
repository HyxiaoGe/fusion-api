"""Agent loop run 终态收尾编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.stream.agent_loop_policy import AgentRunTerminalState
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.itinerary_observability import build_itinerary_run_payload, emit_itinerary_run_log
from app.services.stream.run_finalizer import InterruptedStatusWriteError
from app.services.stream_state_service import StreamOwnershipLostError, StreamWriteTerminalError

PersistMessageFn = Callable[..., Any]
FinalizeStreamFn = Callable[..., Awaitable[Any]]
TerminalRunFn = Callable[..., Awaitable[Any]]
WarningFn = Callable[[str], None]
DurationMsFactory = Callable[[], int]

AGENT_RUN_FAILED_MESSAGE = "生成服务暂时不可用，请稍后重试"
AGENT_RUN_FAILED_ERROR_CODE = "agent_run_failed"
_PUBLIC_ERROR_MESSAGES = {
    "context_budget_exceeded": "当前消息与必要上下文过长，请缩短本次输入或移除较大的文件后重试",
    "context_estimation_unavailable": "上下文预算暂时无法校验，请稍后重试",
}


@dataclass(frozen=True)
class AgentLoopRunCompletionContext:
    db: Any
    conversation_id: str
    task_id: str
    run_id: str
    model_id: str
    provider: str
    assistant_message_id: str
    emitter: Any
    session_cache: Any
    state: AgentLoopState
    duration_ms_factory: DurationMsFactory
    assistant_message_sequence: int | None = None


def persist_run_message(
    *,
    context: AgentLoopRunCompletionContext,
    persist_message_fn: PersistMessageFn,
    only_if_content: bool = False,
    partial: bool = False,
) -> None:
    if only_if_content and not context.state.content_blocks and context.state.final_usage() is None:
        return

    persistence_kwargs = (
        {"sequence": context.assistant_message_sequence} if context.assistant_message_sequence is not None else {}
    )
    persist_message_fn(
        context.db,
        context.assistant_message_id,
        context.conversation_id,
        context.model_id,
        context.state.content_blocks,
        context.state.final_usage(),
        partial,
        **persistence_kwargs,
    )


async def finalize_completed_run(
    *,
    context: AgentLoopRunCompletionContext,
    terminal_state: AgentRunTerminalState,
    persist_message_fn: PersistMessageFn,
    complete_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
) -> None:
    try:
        await _emit_terminal_plan(context, terminal_state.run_finish_reason)
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
    finally:
        _emit_itinerary_observation(
            context,
            run_status=terminal_state.session_status,
            finish_reason=terminal_state.run_finish_reason,
            limit_reason=context.state.limit_reason,
        )


async def finalize_superseded_run(
    *,
    context: AgentLoopRunCompletionContext,
    error_msg: str | None,
    persist_message_fn: PersistMessageFn,
    interrupt_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
) -> None:
    try:
        await _emit_terminal_plan(context, "superseded")
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
    finally:
        _emit_itinerary_observation(
            context,
            run_status="interrupted",
            finish_reason="unknown",
            limit_reason=None,
        )


async def finalize_cancelled_run(
    *,
    context: AgentLoopRunCompletionContext,
    persist_message_fn: PersistMessageFn,
    interrupt_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
    warning_fn: WarningFn,
) -> None:
    try:
        await _emit_terminal_plan(context, "interrupted")
    except StreamOwnershipLostError as emit_exc:
        warning_fn(f"terminal plan ownership lost，外部 stop 已接管流终态: {emit_exc}")
    persist_run_message(
        context=context,
        persist_message_fn=persist_message_fn,
        only_if_content=True,
        partial=True,
    )
    try:
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
        await finalize_stream_fn(
            context.conversation_id,
            success=False,
            error_msg="用户中止",
            task_id=context.task_id,
            error_code="stream_interrupted",
            error_data={"reason": "user_cancelled"},
        )
    finally:
        _emit_itinerary_observation(
            context,
            run_status="interrupted",
            finish_reason="unknown",
            limit_reason=None,
        )


async def finalize_failed_run(
    *,
    context: AgentLoopRunCompletionContext,
    error: Exception,
    persist_message_fn: PersistMessageFn,
    fail_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
    warning_fn: WarningFn,
) -> None:
    await _emit_terminal_plan(context, "failed")
    persist_run_message(
        context=context,
        persist_message_fn=persist_message_fn,
        only_if_content=True,
        partial=True,
    )
    structured_error_code = _safe_structured_error_code(error)
    public_error_code = structured_error_code or AGENT_RUN_FAILED_ERROR_CODE
    public_error_message = _PUBLIC_ERROR_MESSAGES.get(
        public_error_code,
        AGENT_RUN_FAILED_MESSAGE,
    )
    try:
        try:
            await fail_agent_run_fn(
                emitter=context.emitter,
                session_cache=context.session_cache,
                stats=context.state.run_stats(context.run_id),
                duration_ms_factory=context.duration_ms_factory,
                current_step_id=context.state.current_step_id,
                error_code=public_error_code,
                message=public_error_message,
            )
            context.state.mark_terminal_emitted()
        except StreamWriteTerminalError:
            raise
        except Exception as emit_exc:  # noqa: BLE001
            warning_fn(f"emit run_failed 失败: error_type={type(emit_exc).__name__}")
        finalize_kwargs = {
            "success": False,
            "error_msg": public_error_message,
            "error_code": public_error_code,
            "task_id": context.task_id,
        }
        await finalize_stream_fn(context.conversation_id, **finalize_kwargs)
    finally:
        _emit_itinerary_observation(
            context,
            run_status="error",
            finish_reason="unknown",
            limit_reason=None,
        )


def _safe_structured_error_code(error: Exception) -> str:
    candidate = getattr(error, "error_code", None)
    if not isinstance(candidate, str) or candidate not in _PUBLIC_ERROR_MESSAGES:
        return ""
    return candidate


async def _emit_terminal_plan(context: AgentLoopRunCompletionContext, outcome: str) -> None:
    has_final_answer = any(
        (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "text"
        and bool(str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")).strip())
        for block in context.state.content_blocks
    )
    snapshot = context.state.plan_coordinator.terminalize(
        outcome,
        has_final_answer=has_final_answer,
    )
    if snapshot is not None:
        await context.emitter.plan_snapshot(**snapshot)


def _emit_itinerary_observation(
    context: AgentLoopRunCompletionContext,
    *,
    run_status: str,
    finish_reason: str,
    limit_reason: str | None,
) -> None:
    try:
        stats = context.state.run_stats(context.run_id)
        emit_itinerary_run_log(
            build_itinerary_run_payload(
                run_id=context.run_id,
                model_id=context.model_id,
                provider=context.provider,
                content_blocks=context.state.content_blocks,
                tool_observations=context.state.itinerary_tool_observations,
                run_status=run_status,
                finish_reason=finish_reason,
                limit_reason=limit_reason,
                total_duration_ms=context.duration_ms_factory(),
                total_steps=stats.total_steps,
                total_tool_calls=stats.total_tool_calls,
            )
        )
    except Exception:  # noqa: BLE001 — 观测失败绝不能改变 Agent 主链终态
        return


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
        warning_fn(f"finally 兜底 write_session_status 失败: error_type={type(exc).__name__}")
