"""Agent loop runtime 与收尾上下文装配。"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.services.agent.emitter import AgentEventEmitter
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.agent.progress_recorder import AgentProgressRecorder
from app.services.agent.trajectory_recorder import TrajectoryRecorder
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
    turn_message_id: str | None = None
    previous_run_id: str | None = None
    run_attempt_kind: str = "initial"
    assistant_message_sequence: int | None = None


@dataclass
class TrajectoryBarrierState:
    """单次 run 的轨迹提交屏障幂等状态。"""

    started: bool = False


@dataclass(frozen=True)
class AgentLoopExecutionContext:
    run_id: str
    run_start: float
    state: AgentLoopState
    network_budget: NetworkToolBudget
    emitter: AgentEventEmitter
    runtime: AgentLoopRuntime
    completion_context: AgentLoopRunCompletionContext
    trajectory_recorder: TrajectoryRecorder
    turn_message_id: str | None
    previous_run_id: str | None
    run_attempt_kind: str
    trajectory_barrier_state: TrajectoryBarrierState = field(default_factory=TrajectoryBarrierState)


@dataclass(frozen=True)
class AgentLoopExecutionParts:
    run_id: str
    run_start: float
    state: AgentLoopState
    network_budget: NetworkToolBudget
    emitter: AgentEventEmitter
    trajectory_recorder: TrajectoryRecorder


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
    trajectory_recorder = TrajectoryRecorder(
        run_id=run_id,
        conversation_id=request.conversation_id,
        message_id=request.assistant_message_id,
    )
    event_writer = AgentEventCompositeWriter(
        redis_writer=dependencies.redis_writer,
        recorder=progress_recorder,
        trajectory_recorder=trajectory_recorder,
    )
    emitter = AgentEventEmitter(
        run_id=run_id,
        trace_id=run_id,
        conversation_id=request.conversation_id,
        task_id=request.task_id,
        redis_writer=event_writer,
    )
    return AgentLoopExecutionParts(
        run_id=run_id,
        run_start=dependencies.clock(),
        state=AgentLoopState(
            plan_coordinator=PlanCoordinator(
                run_id=run_id,
                mode=getattr(request.call_config, "plan_mode", "auto"),
                allowed_tool_names=frozenset(getattr(request.call_config, "announced_tools", [])),
                required_initial_tool_counts=dict(getattr(request.call_config, "required_initial_tool_counts", {})),
            )
        ),
        network_budget=NetworkToolBudget(
            profile=getattr(request.call_config, "network_profile", "standard"),
            require_distinct_read_urls=(
                "verified_research_request"
                in set((getattr(request.call_config, "plan_tool_policy_reason", "") or "").split("+"))
            ),
        ),
        emitter=emitter,
        trajectory_recorder=trajectory_recorder,
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
        provider=request.provider,
        assistant_message_id=request.assistant_message_id,
        assistant_message_sequence=request.assistant_message_sequence,
        emitter=parts.emitter,
        session_cache=dependencies.session_cache,
        state=parts.state,
        duration_ms_factory=_run_duration_ms,
        trajectory_recorder=parts.trajectory_recorder,
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
        assistant_message_sequence=request.assistant_message_sequence,
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
        dynamic_tool_handlers=getattr(request.call_config, "dynamic_tool_handlers", {}),
        plan_mode=getattr(request.call_config, "plan_mode", "auto"),
        control_tool_names=getattr(request.call_config, "control_tool_names", frozenset()),
        task_mode=getattr(request.call_config, "task_mode", "standard"),
        evidence_policy=getattr(request.call_config, "evidence_policy", "standard"),
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
        trajectory_recorder=parts.trajectory_recorder,
        turn_message_id=request.turn_message_id,
        previous_run_id=request.previous_run_id,
        run_attempt_kind=request.run_attempt_kind,
    )
