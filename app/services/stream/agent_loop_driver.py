"""Agent loop 状态机 driver。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.stream.agent_loop_policy import check_agent_loop_limit
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.round_completion import append_round_content_blocks, complete_text_response_step


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
        state.limit_reason = check_agent_loop_limit(
            elapsed_seconds=runtime.clock() - runtime.run_start,
            step=state.step,
            total_tool_calls=state.total_tool_calls,
            limits=runtime.limits,
        )
        if state.limit_reason is not None:
            state.finish_reason = "timeout" if state.limit_reason == "timeout" else "tool_calls"
            await runtime.emitter.run_limit_reached(reason=state.limit_reason)
            break

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
        thinking_block_id = step_context.thinking_block_id
        text_block_id = step_context.text_block_id

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
        reasoning_buf = round_result.reasoning_buf
        content_buf = round_result.content_buf
        tool_calls_list = round_result.tool_calls
        state.finish_reason = round_result.finish_reason
        state.update_usage(round_result.accumulated_usage)

        if state.finish_reason == "stop":
            append_round_content_blocks(
                state.content_blocks,
                reasoning_buf,
                content_buf,
                thinking_block_id,
                text_block_id,
            )
            await complete_text_response_step(
                context=step_context,
                emitter=runtime.emitter,
                session_cache=runtime.session_cache,
                complete_step_fn=runtime.complete_step_fn,
                clock=runtime.clock,
            )
            state.clear_current_step()
            break

        if state.finish_reason == "cancelled":
            append_round_content_blocks(
                state.content_blocks,
                reasoning_buf,
                content_buf,
                thinking_block_id,
                text_block_id,
            )
            return AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED, error_msg="被新请求取代")

        if state.finish_reason == "tool_calls" and tool_calls_list:
            await runtime.handle_tool_calls_round_fn(
                db=db,
                assistant_message_id=runtime.assistant_message_id,
                conversation_id=runtime.conversation_id,
                user_id=runtime.user_id,
                model_id=runtime.model_id,
                provider=runtime.provider,
                content_blocks=state.content_blocks,
                messages=messages,
                tool_calls=tool_calls_list,
                reasoning_buf=reasoning_buf,
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
            )
            state.clear_current_step()
            continue

        append_round_content_blocks(
            state.content_blocks,
            reasoning_buf,
            content_buf,
            thinking_block_id,
            text_block_id,
        )
        state.mark_unknown_terminated()
        await complete_text_response_step(
            context=step_context,
            emitter=runtime.emitter,
            session_cache=runtime.session_cache,
            complete_step_fn=runtime.complete_step_fn,
            clock=runtime.clock,
        )
        state.clear_current_step()
        break

    if state.limit_reason is not None:
        step_number = state.next_step_number()
        summary_outcome = await runtime.run_limit_summary_step_fn(
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
        )
        state.update_usage(summary_outcome.accumulated_usage)
        state.clear_current_step()

    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)
