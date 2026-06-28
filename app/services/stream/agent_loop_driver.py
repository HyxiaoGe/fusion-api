"""Agent loop 状态机 driver。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.stream.agent_loop_policy import check_agent_loop_limit
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.limit_summary import LimitSummaryStepRequest
from app.services.stream.round_completion import append_round_content_blocks, complete_text_response_step
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundRequest


class AgentLoopExit(Enum):
    COMPLETED = "completed"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class AgentLoopOutcome:
    exit: AgentLoopExit
    error_msg: str | None = None


async def run_agent_loop(
    *,
    db,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
) -> AgentLoopOutcome:
    while True:
        if await _stop_if_limit_reached(state=state, runtime=runtime):
            break

        step_number, step_context = await _start_next_step(state=state, runtime=runtime)
        round_result = await _run_round(
            messages=messages,
            state=state,
            runtime=runtime,
            step_number=step_number,
            step_context=step_context,
        )
        outcome = await _handle_round_result(
            db=db,
            messages=messages,
            state=state,
            runtime=runtime,
            step_number=step_number,
            step_context=step_context,
            round_result=round_result,
        )
        if outcome is None:
            continue
        if outcome.exit == AgentLoopExit.SUPERSEDED:
            return outcome
        break

    if state.limit_reason is not None:
        await _run_limit_summary(state=state, runtime=runtime, messages=messages)

    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


async def _stop_if_limit_reached(*, state: AgentLoopState, runtime: AgentLoopRuntime) -> bool:
    state.limit_reason = check_agent_loop_limit(
        elapsed_seconds=runtime.clock() - runtime.run_start,
        step=state.step,
        total_tool_calls=state.total_tool_calls,
        limits=runtime.limits,
    )
    if state.limit_reason is None:
        return False

    state.finish_reason = "timeout" if state.limit_reason == "timeout" else "tool_calls"
    await runtime.emitter.run_limit_reached(reason=state.limit_reason)
    return True


async def _start_next_step(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
) -> tuple[int, AgentStepContext]:
    step_number = state.next_step_number()
    step_context = await runtime.start_step_fn(
        emitter=runtime.emitter,
        session_cache=runtime.session_cache,
        run_id=runtime.run_id,
        step_number=step_number,
        clock=runtime.clock,
        on_step_started=state.mark_current_step,
    )
    state.mark_current_step(step_context.step_id)
    return step_number, step_context


async def _run_round(
    *,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_number: int,
    step_context: AgentStepContext,
) -> AgentRoundResult:
    round_result = await runtime.run_round_fn(
        conversation_id=runtime.conversation_id,
        task_id=runtime.task_id,
        run_id=runtime.run_id,
        step_number=step_number,
        model_id=runtime.model_id,
        provider=runtime.provider,
        litellm_model=runtime.litellm_model,
        litellm_kwargs=runtime.litellm_kwargs,
        messages=messages,
        should_use_reasoning=runtime.should_use_reasoning,
        call_kwargs=runtime.call_kwargs,
        accumulated_usage=state.accumulated_usage,
        step_context=step_context,
        llm_call_fn=runtime.llm_call_fn,
        stream_round_fn=runtime.stream_round_fn,
        log_round_summary_fn=runtime.log_round_summary_fn,
    )
    state.finish_reason = round_result.finish_reason
    state.update_usage(round_result.accumulated_usage)
    return round_result


async def _handle_round_result(
    *,
    db,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_number: int,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> AgentLoopOutcome | None:
    if state.finish_reason == "stop":
        await _complete_text_round(state=state, runtime=runtime, step_context=step_context, round_result=round_result)
        return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

    if state.finish_reason == "cancelled":
        _append_round_blocks(state=state, step_context=step_context, round_result=round_result)
        return AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED, error_msg="被新请求取代")

    if state.finish_reason == "tool_calls" and round_result.tool_calls:
        await _handle_tool_calls_round(
            db=db,
            messages=messages,
            state=state,
            runtime=runtime,
            step_number=step_number,
            step_context=step_context,
            round_result=round_result,
        )
        return None

    await _complete_unknown_round(state=state, runtime=runtime, step_context=step_context, round_result=round_result)
    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


def _append_round_blocks(
    *,
    state: AgentLoopState,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> None:
    append_round_content_blocks(
        state.content_blocks,
        round_result.reasoning_buf,
        round_result.content_buf,
        step_context.thinking_block_id,
        step_context.text_block_id,
    )


async def _complete_text_round(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> None:
    _append_round_blocks(state=state, step_context=step_context, round_result=round_result)
    await complete_text_response_step(
        context=step_context,
        emitter=runtime.emitter,
        session_cache=runtime.session_cache,
        complete_step_fn=runtime.complete_step_fn,
        clock=runtime.clock,
    )
    state.clear_current_step()


async def _handle_tool_calls_round(
    *,
    db,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_number: int,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> None:
    await runtime.handle_tool_calls_round_fn(
        request=ToolRoundRequest(
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
            clock=runtime.clock,
        ),
    )
    state.clear_current_step()


async def _complete_unknown_round(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_context: AgentStepContext,
    round_result: AgentRoundResult,
) -> None:
    state.mark_unknown_terminated()
    await _complete_text_round(state=state, runtime=runtime, step_context=step_context, round_result=round_result)


async def _run_limit_summary(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    messages: list[dict],
) -> None:
    step_number = state.next_step_number()
    summary_outcome = await runtime.run_limit_summary_step_fn(
        request=LimitSummaryStepRequest(
            conversation_id=runtime.conversation_id,
            task_id=runtime.task_id,
            run_id=runtime.run_id,
            step_number=step_number,
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
        ),
    )
    state.update_usage(summary_outcome.accumulated_usage)
    state.clear_current_step()
