"""Agent loop 单次运行生命周期 facade。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.services.stream.agent_loop_execution import AgentLoopExecutionContext
from app.services.stream.agent_loop_outcome import AgentLoopExit
from app.services.stream.agent_loop_policy import AgentLoopLimits, map_run_terminal_state
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig
from app.services.stream.agent_plan_builder import build_long_task_plan_items

AsyncFn = Callable[..., Awaitable[Any]]
PersistMessageFn = Callable[..., Any]
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class AgentLoopLifecycleRequest:
    raw_messages: list
    has_vision: bool
    file_ids: list | None
    original_message: str
    call_config: AgentLoopCallConfig
    limits: AgentLoopLimits
    initial_content_blocks: list[Any] = field(default_factory=list)
    extra_system_prompts: list[str] = field(default_factory=list)
    preprocess_user_input: bool = True


@dataclass(frozen=True)
class AgentLoopLifecycleDependencies:
    append_chunk_fn: AsyncFn
    start_agent_run_fn: AsyncFn
    prepare_messages_fn: AsyncFn
    run_agent_loop_fn: AsyncFn
    finalize_completed_run_fn: AsyncFn
    finalize_superseded_run_fn: AsyncFn
    finalize_cancelled_run_fn: AsyncFn
    finalize_failed_run_fn: AsyncFn
    write_fallback_run_error_fn: AsyncFn
    persist_message_fn: PersistMessageFn
    complete_agent_run_fn: AsyncFn
    interrupt_agent_run_fn: AsyncFn
    fail_agent_run_fn: AsyncFn
    finalize_stream_fn: AsyncFn
    write_fallback_error_status_fn: AsyncFn
    info_fn: LogFn
    error_fn: LogFn
    warning_fn: LogFn


async def run_agent_loop_lifecycle(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    try:
        await _run_success_path(request=request, execution=execution, dependencies=dependencies)
    except asyncio.CancelledError:
        await _finalize_cancelled(execution=execution, dependencies=dependencies)
        raise
    except Exception as error:
        await _finalize_failed(error=error, execution=execution, dependencies=dependencies)
        raise
    finally:
        await _write_fallback(execution=execution, dependencies=dependencies)


async def _run_success_path(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await _start_run(request=request, execution=execution, dependencies=dependencies)
    prepared_messages = await _prepare_messages(request=request, execution=execution, dependencies=dependencies)
    execution.state.content_blocks.extend(request.initial_content_blocks)
    execution.state.content_blocks.extend(prepared_messages.initial_content_blocks)

    loop_outcome = await dependencies.run_agent_loop_fn(
        db=execution.completion_context.db,
        messages=prepared_messages.messages,
        state=execution.state,
        runtime=execution.runtime,
    )
    if loop_outcome.exit == AgentLoopExit.SUPERSEDED:
        await _finalize_superseded(
            error_msg=loop_outcome.error_msg,
            execution=execution,
            dependencies=dependencies,
        )
        return

    await _finalize_completed(execution=execution, dependencies=dependencies)


async def _start_run(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    context = execution.completion_context
    plan_items = build_long_task_plan_items(
        original_message=request.original_message,
        tools=[],
        limits=request.limits,
        tool_decision_pending=bool(request.call_config.announced_tools),
    )
    execution.state.set_plan_items(plan_items)
    await dependencies.append_chunk_fn(context.conversation_id, "preparing", "", "")
    await dependencies.start_agent_run_fn(
        emitter=execution.emitter,
        session_cache=context.session_cache,
        run_id=execution.run_id,
        conversation_id=context.conversation_id,
        user_id=execution.runtime.user_id,
        model_id=context.model_id,
        provider=execution.runtime.provider,
        message_id=context.assistant_message_id,
        tools=request.call_config.announced_tools,
        config=_run_config(request.limits),
    )
    await execution.emitter.run_progress_updated(
        phase="planning",
        label="正在制定执行计划",
        completed_steps=0,
        total_steps=len(plan_items),
        completed_tool_calls=0,
        max_tool_calls=request.limits.max_tool_calls,
    )
    await execution.emitter.plan_snapshot(
        plan_id=f"plan-{execution.run_id}",
        revision=1,
        items=plan_items,
    )


async def _prepare_messages(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> Any:
    return await dependencies.prepare_messages_fn(
        db=execution.completion_context.db,
        user_id=execution.runtime.user_id,
        raw_messages=request.raw_messages,
        has_vision=request.has_vision,
        file_ids=request.file_ids,
        original_message=request.original_message,
        call_config=request.call_config,
        extra_system_prompts=request.extra_system_prompts,
        preprocess_user_input=request.preprocess_user_input,
    )


async def _finalize_completed(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    terminal_state = map_run_terminal_state(
        unknown_terminated=execution.state.unknown_terminated,
        limit_reason=execution.state.limit_reason,
    )
    await dependencies.finalize_completed_run_fn(
        context=execution.completion_context,
        terminal_state=terminal_state,
        persist_message_fn=dependencies.persist_message_fn,
        complete_agent_run_fn=dependencies.complete_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
    )


async def _finalize_superseded(
    *,
    error_msg: str | None,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await dependencies.finalize_superseded_run_fn(
        context=execution.completion_context,
        error_msg=error_msg,
        persist_message_fn=dependencies.persist_message_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
    )


async def _finalize_cancelled(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    dependencies.info_fn(f"Agent 任务被取消: conv_id={execution.completion_context.conversation_id}")
    await dependencies.finalize_cancelled_run_fn(
        context=execution.completion_context,
        persist_message_fn=dependencies.persist_message_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        warning_fn=dependencies.warning_fn,
    )


async def _finalize_failed(
    *,
    error: Exception,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    dependencies.error_fn(f"Agent 生成异常: conv_id={execution.completion_context.conversation_id}, error={error}")
    await dependencies.finalize_failed_run_fn(
        context=execution.completion_context,
        error=error,
        persist_message_fn=dependencies.persist_message_fn,
        fail_agent_run_fn=dependencies.fail_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        warning_fn=dependencies.warning_fn,
    )


async def _write_fallback(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await dependencies.write_fallback_run_error_fn(
        context=execution.completion_context,
        write_fallback_error_status_fn=dependencies.write_fallback_error_status_fn,
        warning_fn=dependencies.warning_fn,
    )


def _run_config(limits: AgentLoopLimits) -> dict:
    return {
        "max_steps": limits.max_steps,
        "max_tool_calls": limits.max_tool_calls,
        "timeout_s": limits.total_timeout_s,
    }
