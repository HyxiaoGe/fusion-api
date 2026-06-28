"""Agent loop runtime 与收尾上下文装配。"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.agent.emitter import AgentEventEmitter
from app.services.agent.progress_recorder import AgentProgressRecorder
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig
from app.services.stream.agent_loop_run_completion import AgentLoopRunCompletionContext
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.network_budget import NetworkToolBudget
from app.services.stream.tool_executor import AgentEventCompositeWriter


@dataclass(frozen=True)
class AgentLoopDependencies:
    session_cache: Any
    redis_writer: Any
    start_step_fn: Callable[..., Awaitable[Any]]
    complete_step_fn: Callable[..., Awaitable[Any]]
    run_round_fn: Callable[..., Awaitable[Any]]
    handle_tool_calls_round_fn: Callable[..., Awaitable[Any]]
    run_limit_summary_step_fn: Callable[..., Awaitable[Any]]
    llm_call_fn: Callable[..., Awaitable[Any]]
    stream_round_fn: Callable[..., Awaitable[Any]]
    execute_tools_fn: Callable[..., Awaitable[Any]]
    persist_message_fn: Callable[..., Any]
    log_round_summary_fn: Callable[..., None]
    warning_fn: Callable[[str], None]
    clock: Callable[[], float]


@dataclass(frozen=True)
class AgentLoopExecutionRequest:
    db: Any
    conversation_id: str
    user_id: str
    model_id: str
    litellm_model: str
    litellm_kwargs: dict
    provider: str
    assistant_message_id: str
    task_id: str
    call_config: AgentLoopCallConfig
    trace_id: str | None


@dataclass(frozen=True)
class AgentLoopExecutionContext:
    run_id: str
    run_start: float
    state: AgentLoopState
    network_budget: NetworkToolBudget
    emitter: AgentEventEmitter
    runtime: AgentLoopRuntime
    completion_context: AgentLoopRunCompletionContext


@dataclass(frozen=True)
class AgentLoopExecutionParts:
    run_id: str
    run_start: float
    state: AgentLoopState
    network_budget: NetworkToolBudget
    emitter: AgentEventEmitter


def _build_execution_parts(
    *,
    request: AgentLoopExecutionRequest,
    dependencies: AgentLoopDependencies,
) -> AgentLoopExecutionParts:
    run_id = request.trace_id or str(uuid.uuid4())
    progress_recorder = AgentProgressRecorder(
        db=request.db,
        run_id=run_id,
        conversation_id=request.conversation_id,
        message_id=request.assistant_message_id,
        user_id=request.user_id,
    )
    event_writer = AgentEventCompositeWriter(
        redis_writer=dependencies.redis_writer,
        recorder=progress_recorder,
    )
    emitter = AgentEventEmitter(
        run_id=run_id,
        trace_id=run_id,
        conversation_id=request.conversation_id,
        redis_writer=event_writer,
    )
    return AgentLoopExecutionParts(
        run_id=run_id,
        run_start=dependencies.clock(),
        state=AgentLoopState(),
        network_budget=NetworkToolBudget(),
        emitter=emitter,
    )


def _build_completion_context(
    *,
    request: AgentLoopExecutionRequest,
    parts: AgentLoopExecutionParts,
    dependencies: AgentLoopDependencies,
) -> AgentLoopRunCompletionContext:
    def _run_duration_ms() -> int:
        return int((dependencies.clock() - parts.run_start) * 1000)

    return AgentLoopRunCompletionContext(
        db=request.db,
        conversation_id=request.conversation_id,
        task_id=request.task_id,
        run_id=parts.run_id,
        model_id=request.model_id,
        assistant_message_id=request.assistant_message_id,
        emitter=parts.emitter,
        session_cache=dependencies.session_cache,
        state=parts.state,
        duration_ms_factory=_run_duration_ms,
    )


def build_agent_loop_runtime(
    *,
    request: AgentLoopExecutionRequest,
    limits: AgentLoopLimits,
    dependencies: AgentLoopDependencies,
    parts: AgentLoopExecutionParts,
) -> AgentLoopRuntime:
    return AgentLoopRuntime(
        conversation_id=request.conversation_id,
        task_id=request.task_id,
        run_id=parts.run_id,
        user_id=request.user_id,
        model_id=request.model_id,
        provider=request.provider,
        litellm_model=request.litellm_model,
        litellm_kwargs=request.litellm_kwargs,
        should_use_reasoning=request.call_config.should_use_reasoning,
        call_kwargs=request.call_config.call_kwargs,
        assistant_message_id=request.assistant_message_id,
        run_start=parts.run_start,
        limits=limits,
        emitter=parts.emitter,
        session_cache=dependencies.session_cache,
        network_budget=parts.network_budget,
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


def build_agent_loop_execution(
    *,
    request: AgentLoopExecutionRequest,
    limits: AgentLoopLimits,
    dependencies: AgentLoopDependencies,
) -> AgentLoopExecutionContext:
    """集中创建单次 agent loop 运行需要共享的 runtime 对象。"""
    parts = _build_execution_parts(request=request, dependencies=dependencies)
    completion_context = _build_completion_context(
        request=request,
        parts=parts,
        dependencies=dependencies,
    )
    runtime = build_agent_loop_runtime(
        request=request,
        limits=limits,
        dependencies=dependencies,
        parts=parts,
    )
    return AgentLoopExecutionContext(
        run_id=parts.run_id,
        run_start=parts.run_start,
        state=parts.state,
        network_budget=parts.network_budget,
        emitter=parts.emitter,
        runtime=runtime,
        completion_context=completion_context,
    )
