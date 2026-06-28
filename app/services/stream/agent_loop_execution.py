"""Agent loop runtime 与收尾上下文装配。"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.agent.emitter import AgentEventEmitter
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig
from app.services.stream.agent_loop_run_completion import AgentLoopRunCompletionContext
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.network_budget import NetworkToolBudget


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


def _build_completion_context(
    *,
    request: AgentLoopExecutionRequest,
    run_id: str,
    emitter: AgentEventEmitter,
    state: AgentLoopState,
    dependencies: AgentLoopDependencies,
    run_start: float,
) -> AgentLoopRunCompletionContext:
    def _run_duration_ms() -> int:
        return int((dependencies.clock() - run_start) * 1000)

    return AgentLoopRunCompletionContext(
        db=request.db,
        conversation_id=request.conversation_id,
        task_id=request.task_id,
        run_id=run_id,
        model_id=request.model_id,
        assistant_message_id=request.assistant_message_id,
        emitter=emitter,
        session_cache=dependencies.session_cache,
        state=state,
        duration_ms_factory=_run_duration_ms,
    )


def _build_runtime(
    *,
    request: AgentLoopExecutionRequest,
    limits: AgentLoopLimits,
    dependencies: AgentLoopDependencies,
    run_id: str,
    run_start: float,
    emitter: AgentEventEmitter,
    network_budget: NetworkToolBudget,
) -> AgentLoopRuntime:
    return AgentLoopRuntime(
        conversation_id=request.conversation_id,
        task_id=request.task_id,
        run_id=run_id,
        user_id=request.user_id,
        model_id=request.model_id,
        provider=request.provider,
        litellm_model=request.litellm_model,
        litellm_kwargs=request.litellm_kwargs,
        should_use_reasoning=request.call_config.should_use_reasoning,
        call_kwargs=request.call_config.call_kwargs,
        assistant_message_id=request.assistant_message_id,
        run_start=run_start,
        limits=limits,
        emitter=emitter,
        session_cache=dependencies.session_cache,
        network_budget=network_budget,
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
    state = AgentLoopState()
    network_budget = NetworkToolBudget()
    run_id = request.trace_id or str(uuid.uuid4())
    run_start = dependencies.clock()
    emitter = AgentEventEmitter(
        run_id=run_id,
        trace_id=run_id,
        conversation_id=request.conversation_id,
        redis_writer=dependencies.redis_writer,
    )
    completion_context = _build_completion_context(
        request=request,
        run_id=run_id,
        emitter=emitter,
        state=state,
        dependencies=dependencies,
        run_start=run_start,
    )
    runtime = _build_runtime(
        request=request,
        limits=limits,
        dependencies=dependencies,
        run_id=run_id,
        run_start=run_start,
        emitter=emitter,
        network_budget=network_budget,
    )
    return AgentLoopExecutionContext(
        run_id=run_id,
        run_start=run_start,
        state=state,
        network_budget=network_budget,
        emitter=emitter,
        runtime=runtime,
        completion_context=completion_context,
    )
