"""Agent loop run 终态收尾编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.stream.agent_loop_policy import AgentRunTerminalState
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.itinerary_observability import build_itinerary_run_payload, emit_itinerary_run_log
from app.services.stream.run_finalizer import InterruptedStatusWriteError, interrupt_agent_run
from app.services.stream_state_service import StreamOwnershipLostError, StreamWriteTerminalError

PersistMessageFn = Callable[..., Any]
FinalizeStreamFn = Callable[..., Awaitable[Any]]
TerminalRunFn = Callable[..., Awaitable[Any]]
WarningFn = Callable[[str], None]
DurationMsFactory = Callable[[], int]
ClaimSuggestedQuestionsFn = Callable[..., Any]
GenerateSuggestedQuestionsFn = Callable[..., Any]
FailSuggestedQuestionsFn = Callable[..., Any]

AGENT_RUN_FAILED_MESSAGE = "生成服务暂时不可用，请稍后重试"
AGENT_RUN_FAILED_ERROR_CODE = "agent_run_failed"
ASSISTANT_PERSISTENCE_FAILED_MESSAGE = "回答保存失败，请稍后重试"
ASSISTANT_PERSISTENCE_FAILED_ERROR_CODE = "assistant_persistence_failed"
_PUBLIC_ERROR_MESSAGES = {
    "context_budget_exceeded": "当前消息与必要上下文过长，请缩短本次输入或移除较大的文件后重试",
    "context_estimation_unavailable": "上下文预算暂时无法校验，请稍后重试",
    "knowledge_selection_unavailable": "所选知识库当前不可用，请刷新后重新选择",
    "knowledge_retrieval_unavailable": "知识库检索暂时不可用，请稍后重试",
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
    trajectory_recorder: Any | None = None


def persist_run_message(
    *,
    context: AgentLoopRunCompletionContext,
    persist_message_fn: PersistMessageFn,
    only_if_content: bool = False,
    partial: bool = False,
) -> bool | None:
    if only_if_content and not context.state.content_blocks and context.state.final_usage() is None:
        return None

    persistence_kwargs = (
        {"sequence": context.assistant_message_sequence} if context.assistant_message_sequence is not None else {}
    )
    return persist_message_fn(
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
    interrupt_agent_run_fn: TerminalRunFn = interrupt_agent_run,
    claim_suggested_questions_fn: ClaimSuggestedQuestionsFn | None = None,
    generate_suggested_questions_fn: GenerateSuggestedQuestionsFn | None = None,
    fail_suggested_questions_fn: FailSuggestedQuestionsFn | None = None,
    warning_fn: WarningFn | None = None,
) -> None:
    suggestion_claim = None
    try:
        await _emit_terminal_plan(context, terminal_state.run_finish_reason)
        persisted = persist_run_message(context=context, persist_message_fn=persist_message_fn, partial=False)
        if persisted is False:
            if warning_fn is not None:
                warning_fn("assistant 终态写入已被更新任务接管，终结旧任务")
            await _finalize_superseded_terminal(
                context=context,
                error_msg="被新请求取代",
                interrupt_agent_run_fn=interrupt_agent_run_fn,
                finalize_stream_fn=finalize_stream_fn,
                error_code="generation_superseded",
            )
            return
        if persisted is not True:
            if warning_fn is not None:
                warning_fn("assistant 终态写入失败，不发布成功终态")
            try:
                await context.emitter.run_failed(
                    error_code=ASSISTANT_PERSISTENCE_FAILED_ERROR_CODE,
                    message=ASSISTANT_PERSISTENCE_FAILED_MESSAGE,
                )
            except StreamOwnershipLostError:
                pass
            failed_stats = context.state.run_stats(context.run_id)
            await context.session_cache.write_session_status(
                run_id=failed_stats.run_id,
                status="error",
                total_steps=failed_stats.total_steps,
                total_tool_calls=failed_stats.total_tool_calls,
                total_duration_ms=context.duration_ms_factory(),
            )
            await finalize_stream_fn(
                context.conversation_id,
                success=False,
                error_msg=ASSISTANT_PERSISTENCE_FAILED_MESSAGE,
                task_id=context.task_id,
                error_code=ASSISTANT_PERSISTENCE_FAILED_ERROR_CODE,
            )
            return
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
        if claim_suggested_questions_fn is not None and _has_formal_text(context.state.content_blocks):
            try:
                suggestion_claim = claim_suggested_questions_fn(
                    db=context.db,
                    assistant_message_id=context.assistant_message_id,
                    run_id=context.run_id,
                )
            except Exception as error:  # noqa: BLE001 — 推荐问题绝不能阻塞 SSE 终态
                if warning_fn is not None:
                    warning_fn(f"领取推荐问题版本失败: error_type={type(error).__name__}")
        if suggestion_claim is not None:
            emit_pending = getattr(context.emitter, "suggested_questions_pending", None)
            if emit_pending is not None:
                try:
                    await emit_pending(
                        message_id=getattr(suggestion_claim, "message_id", context.assistant_message_id),
                        revision=suggestion_claim.revision,
                    )
                except Exception as error:  # noqa: BLE001 — 辅助事件不能阻塞 SSE 终态
                    if warning_fn is not None:
                        warning_fn(f"发送推荐问题 pending 事件失败: error_type={type(error).__name__}")
        try:
            await finalize_stream_fn(context.conversation_id, success=True, task_id=context.task_id)
        except BaseException:
            if suggestion_claim is not None and fail_suggested_questions_fn is not None:
                fail_suggested_questions_fn(claim=suggestion_claim)
            raise
        if suggestion_claim is not None and generate_suggested_questions_fn is not None:
            try:
                generate_suggested_questions_fn(claim=suggestion_claim)
            except Exception as error:  # noqa: BLE001 — SSE 已终态，辅助任务异常仅记录
                if warning_fn is not None:
                    warning_fn(f"终态后调度推荐问题失败: error_type={type(error).__name__}")
    finally:
        _emit_itinerary_observation(
            context,
            run_status=terminal_state.session_status,
            finish_reason=terminal_state.run_finish_reason,
            limit_reason=context.state.limit_reason,
        )


def _has_formal_text(content_blocks: list[Any]) -> bool:
    return any(
        (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "text"
        and bool(str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")).strip())
        for block in content_blocks
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
        await _finalize_superseded_terminal(
            context=context,
            error_msg=error_msg or "被新请求取代",
            interrupt_agent_run_fn=interrupt_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
        )
    finally:
        _emit_itinerary_observation(
            context,
            run_status="interrupted",
            finish_reason="unknown",
            limit_reason=None,
        )


async def _finalize_superseded_terminal(
    *,
    context: AgentLoopRunCompletionContext,
    error_msg: str,
    interrupt_agent_run_fn: TerminalRunFn,
    finalize_stream_fn: FinalizeStreamFn,
    error_code: str | None = None,
) -> None:
    """统一 superseded 业务终态；仅 required writer 接受事件后确认已发送。"""
    if context.state.superseded_terminal_decided:
        return
    context.state.mark_superseded_terminal_decided()
    await interrupt_agent_run_fn(
        emitter=context.emitter,
        session_cache=context.session_cache,
        stats=context.state.run_stats(context.run_id),
        duration_ms_factory=context.duration_ms_factory,
        current_step_id=context.state.current_step_id,
        reason="superseded",
    )
    context.state.mark_terminal_emitted()
    finalize_kwargs = {
        "success": False,
        "error_msg": error_msg,
        "task_id": context.task_id,
    }
    if error_code is not None:
        finalize_kwargs["error_code"] = error_code
    await finalize_stream_fn(context.conversation_id, **finalize_kwargs)


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
    has_final_answer = not context.state.unknown_terminated and any(
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
    if context.state.terminal_emitted or context.state.superseded_terminal_decided:
        return

    try:
        await write_fallback_error_status_fn(
            session_cache=context.session_cache,
            stats=context.state.run_stats(context.run_id),
            duration_ms_factory=context.duration_ms_factory,
        )
    except Exception as exc:  # noqa: BLE001
        warning_fn(f"finally 兜底 write_session_status 失败: error_type={type(exc).__name__}")
