"""Agent loop step request 构造。"""

from __future__ import annotations

from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.limit_summary import LimitSummaryStepRequest
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundRequest


def build_tool_round_request(
    *,
    db,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_number: int,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> ToolRoundRequest:
    return ToolRoundRequest(
        db=db,
        assistant_message_id=runtime.assistant_message_id,
        conversation_id=runtime.conversation_id,
        user_id=runtime.user_id,
        model_id=runtime.model_id,
        provider=runtime.provider,
        content_blocks=state.content_blocks,
        messages=messages,
        tool_calls=round_result.tool_calls,
        reasoning_buf=round_result.reasoning_buf,
        should_use_reasoning=runtime.should_use_reasoning,
        step_context=step_context,
        step_number=step_number,
        run_id=runtime.run_id,
        emitter=runtime.emitter,
        session_cache=runtime.session_cache,
        network_budget=runtime.network_budget,
        call_kwargs=runtime.call_kwargs,
        persist_message_fn=runtime.persist_message_fn,
        execute_tools_fn=runtime.execute_tools_fn,
        complete_step_fn=runtime.complete_step_fn,
        on_tools_executed=state.record_executed_tool_calls,
        completed_tool_calls=state.total_tool_calls,
        max_tool_calls=runtime.limits.max_tool_calls,
        clock=runtime.clock,
    )


def build_limit_summary_step_request(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    messages: list[dict],
) -> LimitSummaryStepRequest:
    return LimitSummaryStepRequest(
        conversation_id=runtime.conversation_id,
        task_id=runtime.task_id,
        run_id=runtime.run_id,
        step_number=state.next_step_number(),
        model_id=runtime.model_id,
        provider=runtime.provider,
        litellm_model=runtime.litellm_model,
        litellm_kwargs=runtime.litellm_kwargs,
        messages=messages,
        should_use_reasoning=runtime.should_use_reasoning,
        content_blocks=state.content_blocks,
        call_kwargs=runtime.call_kwargs,
        accumulated_usage=state.accumulated_usage,
        emitter=runtime.emitter,
        session_cache=runtime.session_cache,
        total_timeout_s=runtime.limits.total_timeout_s,
        run_start=runtime.run_start,
        start_step_fn=runtime.start_step_fn,
        complete_step_fn=runtime.complete_step_fn,
        llm_call_fn=runtime.llm_call_fn,
        stream_round_fn=runtime.stream_round_fn,
        log_round_summary_fn=runtime.log_round_summary_fn,
        warning_fn=runtime.warning_fn,
        clock=runtime.clock,
        on_step_started=state.mark_current_step,
        on_context_updated=state.update_context,
        assistant_message_id=runtime.assistant_message_id,
    )
