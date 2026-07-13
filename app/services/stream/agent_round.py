"""Agent loop 单轮普通 LLM 调用编排。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.ai.llm_round_observability import create_llm_round_observation
from app.schemas.chat import Usage
from app.services.chat.context_manager import ContextManagementError, ContextPlan, prepare_context


@dataclass(frozen=True)
class AgentRoundResult:
    reasoning_buf: str
    content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    accumulated_usage: Usage


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
) -> StreamRoundResult:
    response = await llm_call_fn(
        litellm_model,
        litellm_kwargs,
        messages,
        **call_kwargs,
    )
    if observation is not None:
        response = observation.wrap_response(response)
    return await stream_round_fn(
        response,
        conversation_id,
        task_id,
        should_use_reasoning,
        step_context.thinking_block_id,
        step_context.text_block_id,
        run_id=run_id,
        step_id=step_context.step_id,
    )


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
) -> AgentRoundResult:
    try:
        context_plan = await prepare_context(
            messages=messages,
            model_id=model_id,
            litellm_model=litellm_model,
            call_kwargs=call_kwargs,
        )
    except ContextManagementError as error:
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
        )
    except BaseException as exc:
        await observation.finish_error(exc)
        raise
    reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = stream_result
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
    )
