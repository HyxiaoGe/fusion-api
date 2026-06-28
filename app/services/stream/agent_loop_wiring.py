"""Agent loop runner 依赖装配。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.stream.agent_loop_execution import (
    AgentLoopDependencies,
    AgentLoopExecutionRequest,
)
from app.services.stream.agent_loop_lifecycle import (
    AgentLoopLifecycleDependencies,
    AgentLoopLifecycleRequest,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig

AsyncFn = Callable[..., Awaitable[Any]]
PersistMessageFn = Callable[..., Any]
LogFn = Callable[[str], None]


@dataclass(frozen=True)
class AgentLoopRunInput:
    conversation_id: str
    user_id: str
    model_id: str
    litellm_model: str
    litellm_kwargs: dict
    provider: str
    raw_messages: list
    has_vision: bool
    file_ids: list | None
    original_message: str
    assistant_message_id: str
    task_id: str
    options: dict | None
    capabilities: dict | None
    trace_id: str | None


@dataclass(frozen=True)
class AgentLoopWiringDependencies:
    build_call_config_fn: Callable[..., AgentLoopCallConfig]
    build_execution_fn: Callable[..., Any]
    session_cache: Any
    redis_writer_factory: Callable[[], Any]
    start_step_fn: AsyncFn
    complete_step_fn: AsyncFn
    run_round_fn: AsyncFn
    handle_tool_calls_round_fn: AsyncFn
    run_limit_summary_step_fn: AsyncFn
    llm_call_fn: AsyncFn
    stream_round_fn: AsyncFn
    execute_tools_fn: AsyncFn
    persist_message_fn: PersistMessageFn
    log_round_summary_fn: Callable[..., None]
    clock: Callable[[], float]
    append_chunk_fn: AsyncFn
    start_agent_run_fn: AsyncFn
    prepare_messages_fn: AsyncFn
    run_agent_loop_fn: AsyncFn
    finalize_completed_run_fn: AsyncFn
    finalize_superseded_run_fn: AsyncFn
    finalize_cancelled_run_fn: AsyncFn
    finalize_failed_run_fn: AsyncFn
    write_fallback_run_error_fn: AsyncFn
    complete_agent_run_fn: AsyncFn
    interrupt_agent_run_fn: AsyncFn
    fail_agent_run_fn: AsyncFn
    finalize_stream_fn: AsyncFn
    write_fallback_error_status_fn: AsyncFn
    info_fn: LogFn
    error_fn: LogFn
    warning_fn: LogFn


@dataclass(frozen=True)
class AgentLoopLifecycleCall:
    request: AgentLoopLifecycleRequest
    execution: Any
    dependencies: AgentLoopLifecycleDependencies


def build_agent_loop_lifecycle_call(
    *,
    run_input: AgentLoopRunInput,
    db: Any,
    limits: AgentLoopLimits,
    dependencies: AgentLoopWiringDependencies,
) -> AgentLoopLifecycleCall:
    options = {} if run_input.options is None else run_input.options
    capabilities = {} if run_input.capabilities is None else run_input.capabilities
    call_config = dependencies.build_call_config_fn(
        provider=run_input.provider,
        options=options,
        capabilities=capabilities,
    )
    execution = dependencies.build_execution_fn(
        request=_execution_request(run_input=run_input, db=db, call_config=call_config),
        limits=limits,
        dependencies=_execution_dependencies(dependencies),
    )
    return AgentLoopLifecycleCall(
        request=_lifecycle_request(run_input=run_input, call_config=call_config, limits=limits),
        execution=execution,
        dependencies=_lifecycle_dependencies(dependencies),
    )


def _execution_request(
    *,
    run_input: AgentLoopRunInput,
    db: Any,
    call_config: AgentLoopCallConfig,
) -> AgentLoopExecutionRequest:
    return AgentLoopExecutionRequest(
        db=db,
        conversation_id=run_input.conversation_id,
        user_id=run_input.user_id,
        model_id=run_input.model_id,
        litellm_model=run_input.litellm_model,
        litellm_kwargs=run_input.litellm_kwargs,
        provider=run_input.provider,
        assistant_message_id=run_input.assistant_message_id,
        task_id=run_input.task_id,
        call_config=call_config,
        trace_id=run_input.trace_id,
    )


def _execution_dependencies(dependencies: AgentLoopWiringDependencies) -> AgentLoopDependencies:
    return AgentLoopDependencies(
        session_cache=dependencies.session_cache,
        redis_writer=dependencies.redis_writer_factory(),
        start_step_fn=dependencies.start_step_fn,
        complete_step_fn=dependencies.complete_step_fn,
        run_round_fn=dependencies.run_round_fn,
        handle_tool_calls_round_fn=dependencies.handle_tool_calls_round_fn,
        run_limit_summary_step_fn=dependencies.run_limit_summary_step_fn,
        llm_call_fn=dependencies.llm_call_fn,
        stream_round_fn=dependencies.stream_round_fn,
        execute_tools_fn=dependencies.execute_tools_fn,
        persist_message_fn=dependencies.persist_message_fn,
        log_round_summary_fn=dependencies.log_round_summary_fn,
        warning_fn=dependencies.warning_fn,
        clock=dependencies.clock,
    )


def _lifecycle_request(
    *,
    run_input: AgentLoopRunInput,
    call_config: AgentLoopCallConfig,
    limits: AgentLoopLimits,
) -> AgentLoopLifecycleRequest:
    return AgentLoopLifecycleRequest(
        raw_messages=run_input.raw_messages,
        has_vision=run_input.has_vision,
        file_ids=run_input.file_ids,
        original_message=run_input.original_message,
        call_config=call_config,
        limits=limits,
    )


def _lifecycle_dependencies(dependencies: AgentLoopWiringDependencies) -> AgentLoopLifecycleDependencies:
    return AgentLoopLifecycleDependencies(
        append_chunk_fn=dependencies.append_chunk_fn,
        start_agent_run_fn=dependencies.start_agent_run_fn,
        prepare_messages_fn=dependencies.prepare_messages_fn,
        run_agent_loop_fn=dependencies.run_agent_loop_fn,
        finalize_completed_run_fn=dependencies.finalize_completed_run_fn,
        finalize_superseded_run_fn=dependencies.finalize_superseded_run_fn,
        finalize_cancelled_run_fn=dependencies.finalize_cancelled_run_fn,
        finalize_failed_run_fn=dependencies.finalize_failed_run_fn,
        write_fallback_run_error_fn=dependencies.write_fallback_run_error_fn,
        persist_message_fn=dependencies.persist_message_fn,
        complete_agent_run_fn=dependencies.complete_agent_run_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        fail_agent_run_fn=dependencies.fail_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        write_fallback_error_status_fn=dependencies.write_fallback_error_status_fn,
        info_fn=dependencies.info_fn,
        error_fn=dependencies.error_fn,
        warning_fn=dependencies.warning_fn,
    )
