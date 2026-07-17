"""Agent loop 状态机 driver。"""

from __future__ import annotations

from inspect import Parameter, signature

from app.services.stream.agent_loop_outcome import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_policy import check_agent_loop_limit
from app.services.stream.agent_loop_round_outcome import AgentRoundOutcomeRequest, handle_agent_round_outcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_limit_summary_step_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.product_result_answer import has_product_result_blocks
from app.services.stream.step_lifecycle import AgentStepContext


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
        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db=db,
                messages=messages,
                state=state,
                runtime=runtime,
                step_number=step_number,
                step_context=step_context,
                round_result=round_result,
            ),
        )
        if outcome is None:
            continue
        if outcome.exit == AgentLoopExit.SUPERSEDED:
            return outcome
        if outcome.exit == AgentLoopExit.PRODUCT_RESULT_READY:
            await _complete_product_result_without_llm(
                db=db,
                messages=messages,
                state=state,
                runtime=runtime,
            )
            break
        if outcome.exit == AgentLoopExit.SUMMARY_REQUIRED:
            state.finish_reason = outcome.summary_finish_reason or "empty_answer_summary"
            await _run_limit_summary(
                state=state,
                runtime=runtime,
                messages=messages,
                summary_finish_reason=outcome.summary_finish_reason or "limit_summary",
            )
            break
        break

    if state.limit_reason is not None:
        if has_product_result_blocks(state.content_blocks):
            limit_finish_reason = state.finish_reason
            try:
                await _complete_product_result_without_llm(
                    db=db,
                    messages=messages,
                    state=state,
                    runtime=runtime,
                )
            finally:
                state.finish_reason = limit_finish_reason
        else:
            await _run_limit_summary(state=state, runtime=runtime, messages=messages)

    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


async def _complete_product_result_without_llm(
    *,
    db,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
) -> None:
    """产品结果块本身已足够收尾，不再消耗一轮模型调用。"""
    step_number, step_context = await _start_next_step(state=state, runtime=runtime)
    state.finish_reason = "stop"
    await handle_agent_round_outcome(
        request=AgentRoundOutcomeRequest(
            db=db,
            messages=messages,
            state=state,
            runtime=runtime,
            step_number=step_number,
            step_context=step_context,
            round_result=AgentRoundResult(
                reasoning_buf="",
                content_buf="",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=state.accumulated_usage,
                context=state.last_context,
                output_deferred=True,
            ),
        )
    )


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
    start_step_kwargs = {
        "emitter": runtime.emitter,
        "session_cache": runtime.session_cache,
        "run_id": runtime.run_id,
        "step_number": step_number,
        "completed_tool_calls": state.total_tool_calls,
        "max_tool_calls": runtime.limits.max_tool_calls,
        "clock": runtime.clock,
        "on_step_started": state.mark_current_step,
    }
    if _accepts_keyword(runtime.start_step_fn, "plan_items"):
        start_step_kwargs["plan_items"] = state.plan_items
    step_context = await runtime.start_step_fn(**start_step_kwargs)
    state.mark_current_step(step_context.step_id)
    return step_number, step_context


def _accepts_keyword(fn, keyword: str) -> bool:
    try:
        parameters = signature(fn).parameters
    except (TypeError, ValueError):
        return True

    return keyword in parameters or any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


async def _run_round(
    *,
    messages: list[dict],
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    step_number: int,
    step_context: AgentStepContext,
) -> AgentRoundResult:
    call_kwargs = await _filter_exhausted_dynamic_tools(
        call_kwargs=runtime.call_kwargs,
        dynamic_tool_handlers=runtime.dynamic_tool_handlers,
    )
    run_round_kwargs = dict(
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
        call_kwargs=call_kwargs,
        accumulated_usage=state.accumulated_usage,
        step_context=step_context,
        llm_call_fn=runtime.llm_call_fn,
        stream_round_fn=runtime.stream_round_fn,
        log_round_summary_fn=runtime.log_round_summary_fn,
        assistant_message_id=runtime.assistant_message_id,
        emitter=runtime.emitter,
        on_context_updated=state.update_context,
    )
    if has_product_result_blocks(state.content_blocks) and _accepts_keyword(
        runtime.run_round_fn,
        "defer_output",
    ):
        run_round_kwargs["defer_output"] = True
    round_result = await runtime.run_round_fn(**run_round_kwargs)
    state.finish_reason = round_result.finish_reason
    state.update_usage(round_result.accumulated_usage)
    state.update_context(round_result.context)
    return round_result


async def _filter_exhausted_dynamic_tools(
    *,
    call_kwargs: dict,
    dynamic_tool_handlers: dict[str, object],
) -> dict:
    """在下一轮模型调用前隐藏已耗尽单服务预算的动态工具。"""

    if not dynamic_tool_handlers or not call_kwargs.get("tools"):
        return call_kwargs

    exhausted_aliases: set[str] = set()
    for alias, handler in dynamic_tool_handlers.items():
        is_exhausted = getattr(handler, "is_run_budget_exhausted", None)
        if is_exhausted is not None and await is_exhausted():
            exhausted_aliases.add(alias)
    if not exhausted_aliases:
        return call_kwargs

    filtered_tools = []
    for tool in call_kwargs["tools"]:
        function = tool.get("function") if isinstance(tool, dict) else None
        tool_name = function.get("name") if isinstance(function, dict) else None
        if tool_name not in exhausted_aliases:
            filtered_tools.append(tool)

    filtered_call_kwargs = dict(call_kwargs)
    if filtered_tools:
        filtered_call_kwargs["tools"] = filtered_tools
    else:
        filtered_call_kwargs.pop("tools", None)
        filtered_call_kwargs.pop("tool_choice", None)
    return filtered_call_kwargs


async def _run_limit_summary(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    messages: list[dict],
    summary_finish_reason: str = "limit_summary",
) -> None:
    summary_outcome = await runtime.run_limit_summary_step_fn(
        request=build_limit_summary_step_request(
            state=state,
            runtime=runtime,
            messages=messages,
            summary_finish_reason=summary_finish_reason,
        ),
    )
    state.update_usage(summary_outcome.accumulated_usage)
    state.update_context(summary_outcome.context)
    state.clear_current_step()
