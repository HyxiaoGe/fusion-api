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
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig, supports_dynamic_agent_tools

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
    assistant_message_sequence: int | None = None
    initial_content_blocks: list | None = None
    extra_system_prompts: list[str] | None = None
    preprocess_user_input: bool = True

    def to_execution_request(
        self,
        *,
        db: Any,
        call_config: AgentLoopCallConfig,
    ) -> AgentLoopExecutionRequest:
        return AgentLoopExecutionRequest(
            db=db,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            model_id=self.model_id,
            litellm_model=self.litellm_model,
            litellm_kwargs=self.litellm_kwargs,
            provider=self.provider,
            assistant_message_id=self.assistant_message_id,
            assistant_message_sequence=self.assistant_message_sequence,
            task_id=self.task_id,
            call_config=call_config,
            trace_id=self.trace_id,
        )

    def to_lifecycle_request(
        self,
        *,
        call_config: AgentLoopCallConfig,
        limits: AgentLoopLimits,
    ) -> AgentLoopLifecycleRequest:
        return AgentLoopLifecycleRequest(
            raw_messages=self.raw_messages,
            has_vision=self.has_vision,
            file_ids=self.file_ids,
            original_message=self.original_message,
            call_config=call_config,
            limits=limits,
            initial_content_blocks=self.initial_content_blocks or [],
            extra_system_prompts=self.extra_system_prompts or [],
            preprocess_user_input=self.preprocess_user_input,
        )


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
    load_dynamic_tools_fn: Callable[[Any], Any] | None = None

    def to_execution_dependencies(self) -> AgentLoopDependencies:
        return AgentLoopDependencies(
            session_cache=self.session_cache,
            redis_writer=self.redis_writer_factory(),
            start_step_fn=self.start_step_fn,
            complete_step_fn=self.complete_step_fn,
            run_round_fn=self.run_round_fn,
            handle_tool_calls_round_fn=self.handle_tool_calls_round_fn,
            run_limit_summary_step_fn=self.run_limit_summary_step_fn,
            llm_call_fn=self.llm_call_fn,
            stream_round_fn=self.stream_round_fn,
            execute_tools_fn=self.execute_tools_fn,
            persist_message_fn=self.persist_message_fn,
            log_round_summary_fn=self.log_round_summary_fn,
            warning_fn=self.warning_fn,
            clock=self.clock,
        )

    def to_lifecycle_dependencies(self) -> AgentLoopLifecycleDependencies:
        return AgentLoopLifecycleDependencies(
            append_chunk_fn=self.append_chunk_fn,
            start_agent_run_fn=self.start_agent_run_fn,
            prepare_messages_fn=self.prepare_messages_fn,
            run_agent_loop_fn=self.run_agent_loop_fn,
            finalize_completed_run_fn=self.finalize_completed_run_fn,
            finalize_superseded_run_fn=self.finalize_superseded_run_fn,
            finalize_cancelled_run_fn=self.finalize_cancelled_run_fn,
            finalize_failed_run_fn=self.finalize_failed_run_fn,
            write_fallback_run_error_fn=self.write_fallback_run_error_fn,
            persist_message_fn=self.persist_message_fn,
            complete_agent_run_fn=self.complete_agent_run_fn,
            interrupt_agent_run_fn=self.interrupt_agent_run_fn,
            fail_agent_run_fn=self.fail_agent_run_fn,
            finalize_stream_fn=self.finalize_stream_fn,
            write_fallback_error_status_fn=self.write_fallback_error_status_fn,
            info_fn=self.info_fn,
            error_fn=self.error_fn,
            warning_fn=self.warning_fn,
        )


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
    should_load_dynamic_tools = (
        supports_dynamic_agent_tools(capabilities)
        and options.get("disable_tools") is not True
        and dependencies.load_dynamic_tools_fn is not None
    )
    dynamic_tool_set = dependencies.load_dynamic_tools_fn(db) if should_load_dynamic_tools else None
    call_config = dependencies.build_call_config_fn(
        provider=run_input.provider,
        options=options,
        capabilities=capabilities,
        additional_tools=list(getattr(dynamic_tool_set, "definitions", []) or []),
        dynamic_tool_handlers=dict(getattr(dynamic_tool_set, "handlers", {}) or {}),
        tool_bindings=list(getattr(dynamic_tool_set, "audit_bindings", []) or []),
    )
    execution = dependencies.build_execution_fn(
        request=run_input.to_execution_request(db=db, call_config=call_config),
        limits=limits,
        dependencies=dependencies.to_execution_dependencies(),
    )
    return AgentLoopLifecycleCall(
        request=run_input.to_lifecycle_request(call_config=call_config, limits=limits),
        execution=execution,
        dependencies=dependencies.to_lifecycle_dependencies(),
    )
