"""Agent loop 单轮普通 LLM 调用编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from app.ai.llm_round_observability import create_llm_round_observation
from app.schemas.chat import ContextUsage, Usage
from app.services.chat.context_manager import ContextManagementError, ContextPlan, prepare_context
from app.services.stream.context_status import build_context_usage, emit_context_status


@dataclass(frozen=True)
class AgentRoundResult:
    reasoning_buf: str
    content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    accumulated_usage: Usage
    context: ContextUsage | None = None
    announced_tool_names: frozenset[str] | None = None
    output_deferred: bool = False


StreamRoundResult = tuple[str, str, list[dict], str, Usage | None]


def _create_agent_round_observation(
    *,
    context_plan: ContextPlan,
    conversation_id: str,
    run_id: str,
    step_number: int,
    step_id: str,
    model_id: str,
    provider: str,
    litellm_model: str,
    call_kwargs: dict,
    assistant_message_id: str | None,
    estimator_status: str | None = None,
) -> Any:
    return create_llm_round_observation(
        conversation_id=conversation_id,
        run_id=run_id,
        round_index=step_number,
        step_id=step_id,
        round_kind="agent",
        model_id=model_id,
        provider=provider,
        litellm_model=litellm_model,
        messages=context_plan.messages,
        call_kwargs=call_kwargs,
        assistant_message_id=assistant_message_id,
        context_management=context_plan.telemetry(),
        estimated_prompt_tokens=context_plan.estimated_tokens_after,
        estimator_status=estimator_status,
    )


def accumulate_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage:
    if not usage_data:
        return accumulated_usage
    return Usage(
        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
    )


def _announced_tool_names(call_kwargs: dict) -> frozenset[str]:
    names: set[str] = set()
    for tool in call_kwargs.get("tools", []) or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.add(str(name))
    return frozenset(names)


async def collect_agent_round_stream(
    *,
    conversation_id: str,
    task_id: str,
    run_id: str,
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    should_use_reasoning: bool,
    call_kwargs: dict,
    step_context: Any,
    llm_call_fn: Callable[..., Awaitable[Any]],
    stream_round_fn: Callable[..., Awaitable[StreamRoundResult]],
    observation: Any | None = None,
    defer_output: bool = False,
) -> StreamRoundResult:
    response = await llm_call_fn(
        litellm_model,
        litellm_kwargs,
        messages,
        **call_kwargs,
    )
    if observation is not None:
        response = observation.wrap_response(response)
    stream_kwargs = {"run_id": run_id, "step_id": step_context.step_id}
    if defer_output and _accepts_keyword(stream_round_fn, "defer_output"):
        stream_kwargs["defer_output"] = True
    return await stream_round_fn(
        response,
        conversation_id,
        task_id,
        should_use_reasoning,
        step_context.thinking_block_id,
        step_context.text_block_id,
        **stream_kwargs,
    )


def _accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def log_agent_round_summary(
    *,
    conversation_id: str,
    run_id: str,
    step_number: int,
    model_id: str,
    provider: str,
    stream_result: StreamRoundResult,
    log_round_summary_fn: Callable[..., None],
) -> None:
    reasoning_buf, content_buf, tool_calls, finish_reason, _usage_data = stream_result
    log_round_summary_fn(
        conversation_id=conversation_id,
        run_id=run_id,
        step_number=step_number,
        model_id=model_id,
        provider=provider,
        finish_reason=finish_reason,
        tool_calls_count=len(tool_calls),
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
    )


async def run_agent_round(
    *,
    conversation_id: str,
    task_id: str,
    run_id: str,
    step_number: int,
    model_id: str,
    provider: str,
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    should_use_reasoning: bool,
    call_kwargs: dict,
    accumulated_usage: Usage,
    step_context: Any,
    llm_call_fn: Callable[..., Awaitable[Any]],
    stream_round_fn: Callable[..., Awaitable[StreamRoundResult]],
    log_round_summary_fn: Callable[..., None],
    assistant_message_id: str | None = None,
    emitter: Any | None = None,
    on_context_updated: Callable[[ContextUsage], None] | None = None,
    defer_output: bool = False,
) -> AgentRoundResult:
    try:
        context_plan = await prepare_context(
            messages=messages,
            model_id=model_id,
            litellm_model=litellm_model,
            call_kwargs=call_kwargs,
        )
    except ContextManagementError as error:
        error_context = build_context_usage(error.plan, round_index=step_number)
        if on_context_updated is not None:
            on_context_updated(error_context)
        await emit_context_status(emitter, phase="error", context=error_context)
        observation = _create_agent_round_observation(
            context_plan=error.plan,
            conversation_id=conversation_id,
            run_id=run_id,
            step_number=step_number,
            step_id=step_context.step_id,
            model_id=model_id,
            provider=provider,
            litellm_model=litellm_model,
            call_kwargs=call_kwargs,
            assistant_message_id=assistant_message_id,
            estimator_status="context_manager_error",
        )
        observation.start()
        await observation.finish_error(error)
        raise
    effective_messages = context_plan.messages
    estimated_context = build_context_usage(context_plan, round_index=step_number)
    if on_context_updated is not None:
        on_context_updated(estimated_context)
    await emit_context_status(emitter, phase="estimated", context=estimated_context)
    observation = _create_agent_round_observation(
        context_plan=context_plan,
        conversation_id=conversation_id,
        run_id=run_id,
        step_number=step_number,
        step_id=step_context.step_id,
        model_id=model_id,
        provider=provider,
        litellm_model=litellm_model,
        call_kwargs=call_kwargs,
        assistant_message_id=assistant_message_id,
    )
    observation.start()
    try:
        stream_result = await collect_agent_round_stream(
            conversation_id=conversation_id,
            task_id=task_id,
            run_id=run_id,
            litellm_model=litellm_model,
            litellm_kwargs=litellm_kwargs,
            messages=effective_messages,
            should_use_reasoning=should_use_reasoning,
            call_kwargs=call_kwargs,
            step_context=step_context,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            observation=observation,
            defer_output=defer_output,
        )
    except BaseException as exc:
        await observation.finish_error(exc)
        raise
    reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = stream_result
    final_context = build_context_usage(context_plan, usage_data, round_index=step_number)
    if on_context_updated is not None:
        on_context_updated(final_context)
    await emit_context_status(emitter, phase="final", context=final_context)
    await observation.finish_success(usage=usage_data, finish_reason=finish_reason)
    log_agent_round_summary(
        conversation_id=conversation_id,
        run_id=run_id,
        step_number=step_number,
        model_id=model_id,
        provider=provider,
        stream_result=stream_result,
        log_round_summary_fn=log_round_summary_fn,
    )
    return AgentRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        accumulated_usage=accumulate_usage(accumulated_usage, usage_data),
        context=final_context,
        announced_tool_names=_announced_tool_names(call_kwargs),
        output_deferred=defer_output and _accepts_keyword(stream_round_fn, "defer_output"),
    )
