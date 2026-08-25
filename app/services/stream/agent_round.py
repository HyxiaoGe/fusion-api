"""Agent loop 单轮普通 LLM 调用编排。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from app.ai import litellm_health
from app.ai.llm_round_observability import create_llm_round_observation
from app.core.logger import app_logger as logger
from app.schemas.chat import ContextUsage, Usage
from app.services.agent.llm_round_detail_recorder import LlmRoundDetailDraft
from app.services.chat.context_manager import ContextManagementError, ContextPlan, prepare_context
from app.services.chat.model_call_language_policy import finalize_model_call_language_policy
from app.services.stream.context_status import build_context_usage, emit_context_status
from app.services.stream.llm_round_lifecycle import LLMRoundLifecycle, accumulate_token_usage


@dataclass(frozen=True)
class AgentRoundResult:
    reasoning_buf: str
    content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    accumulated_usage: Usage
    protocol_reasoning_buf: str | None = None
    protocol_content_buf: str | None = None
    context: ContextUsage | None = None
    announced_tool_names: frozenset[str] | None = None
    output_deferred: bool = False
    allow_deferred_reasoning_output: bool = False
    llm_lifecycle: LLMRoundLifecycle | None = None


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
    return accumulate_token_usage(accumulated_usage, usage_data)


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
    provider: str | None = None,
    model_id: str | None = None,
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
    allow_deferred_reasoning_output: bool = False,
    on_visible_output: Callable[[str], Awaitable[None]] | None = None,
    on_output_candidate: Callable[..., None] | None = None,
    capture_output_candidate_time: Callable[[], float | None] | None = None,
    partial_output: dict[str, str] | None = None,
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
    if provider is not None and _accepts_keyword(stream_round_fn, "provider"):
        stream_kwargs["provider"] = provider
    if model_id is not None and _accepts_keyword(stream_round_fn, "model_id"):
        stream_kwargs["model_id"] = model_id
    if defer_output and _accepts_keyword(stream_round_fn, "defer_output"):
        stream_kwargs["defer_output"] = True
    if allow_deferred_reasoning_output and _accepts_keyword(
        stream_round_fn,
        "allow_deferred_reasoning_output",
    ):
        stream_kwargs["allow_deferred_reasoning_output"] = True
    if on_visible_output is not None and _accepts_keyword(stream_round_fn, "on_visible_output"):
        stream_kwargs["on_visible_output"] = on_visible_output
    if on_output_candidate is not None and _accepts_keyword(stream_round_fn, "on_output_candidate"):
        stream_kwargs["on_output_candidate"] = on_output_candidate
    if capture_output_candidate_time is not None and _accepts_keyword(
        stream_round_fn,
        "capture_output_candidate_time",
    ):
        stream_kwargs["capture_output_candidate_time"] = capture_output_candidate_time
    if partial_output is not None and _accepts_keyword(stream_round_fn, "partial_output"):
        stream_kwargs["partial_output"] = partial_output
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


def _freeze_observation(observation: Any) -> None:
    freeze = getattr(observation, "freeze", None)
    if callable(freeze):
        freeze()


async def _close_round_after_primary_error(
    *, observation: Any, lifecycle: LLMRoundLifecycle | None, error: BaseException
) -> None:
    """terminal sink 是 secondary；不得替换网络异常或取消。"""
    _freeze_observation(observation)
    try:
        await observation.finish_error(error)
    except BaseException as secondary:
        logger.warning("LLM 观测异常收尾失败，保留主异常: error_type=%s", type(secondary).__name__)
    if lifecycle is None:
        return
    try:
        if isinstance(error, asyncio.CancelledError):
            await lifecycle.finish_cancelled(reason="shutdown")
        else:
            await lifecycle.finish_failed(error)
    except BaseException as secondary:
        logger.warning("LLM 生命周期异常收尾失败，保留主异常: error_type=%s", type(secondary).__name__)


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
    allow_deferred_reasoning_output: bool = True,
    llm_round_detail_scheduler: Callable[[LlmRoundDetailDraft], Any] | None = None,
) -> AgentRoundResult:
    finalized_messages = finalize_model_call_language_policy(messages)
    try:
        context_plan = await prepare_context(
            messages=finalized_messages,
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
    lifecycle = await LLMRoundLifecycle.start(
        emitter=emitter,
        observation=observation,
        round_index=step_number,
        model=model_id,
        provider=provider,
        parent_step_id=step_context.step_id,
        conversation_id=conversation_id,
        run_id=run_id,
        message_id=assistant_message_id,
        detail_scheduler=llm_round_detail_scheduler,
    )
    observation.start()
    partial_output: dict[str, str] = {}
    try:
        stream_result = await collect_agent_round_stream(
            conversation_id=conversation_id,
            task_id=task_id,
            run_id=run_id,
            provider=provider,
            model_id=model_id,
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
            allow_deferred_reasoning_output=defer_output and allow_deferred_reasoning_output,
            on_visible_output=(
                lifecycle.publish_visible_output if lifecycle is not None and not defer_output else None
            ),
            on_output_candidate=getattr(observation, "observe_output_candidate", None),
            capture_output_candidate_time=getattr(observation, "capture_output_candidate_time", None),
            partial_output=partial_output,
        )
    except asyncio.CancelledError as exc:
        if lifecycle is not None:
            lifecycle.record_detail(
                reasoning_text=partial_output.get("reasoning_buf", ""),
                content_text=partial_output.get("content_buf", ""),
            )
        await _close_round_after_primary_error(observation=observation, lifecycle=lifecycle, error=exc)
        raise
    except BaseException as exc:
        if lifecycle is not None:
            lifecycle.record_detail(
                reasoning_text=partial_output.get("reasoning_buf", ""),
                content_text=partial_output.get("content_buf", ""),
            )
        await _close_round_after_primary_error(observation=observation, lifecycle=lifecycle, error=exc)
        raise
    try:
        reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = stream_result
        _freeze_observation(observation)
        final_context = build_context_usage(context_plan, usage_data, round_index=step_number)
        if on_context_updated is not None:
            on_context_updated(final_context)
        await emit_context_status(emitter, phase="final", context=final_context)
        await observation.finish_success(usage=usage_data, finish_reason=finish_reason)
        if lifecycle is not None:
            lifecycle.record_detail(reasoning_text=reasoning_buf, content_text=content_buf)
            lifecycle.record_result(usage=usage_data, finish_reason=finish_reason)
            if finish_reason == "cancelled":
                await lifecycle.finish_cancelled(reason="superseded")
            elif not defer_output or tool_calls:
                if tool_calls:
                    await lifecycle.publish_tool_output()
                elif content_buf:
                    await lifecycle.publish_visible_output("content")
                elif reasoning_buf:
                    await lifecycle.publish_visible_output("reasoning")
                await lifecycle.finish_success(output_visible=False)
        if finish_reason != "cancelled":
            litellm_health.record_success(model_id)
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
            protocol_reasoning_buf=getattr(stream_result, "protocol_reasoning_buf", reasoning_buf),
            protocol_content_buf=getattr(stream_result, "protocol_content_buf", content_buf),
            content_buf=content_buf,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            accumulated_usage=accumulate_usage(accumulated_usage, usage_data),
            context=final_context,
            announced_tool_names=_announced_tool_names(call_kwargs),
            output_deferred=defer_output and _accepts_keyword(stream_round_fn, "defer_output"),
            allow_deferred_reasoning_output=(
                defer_output
                and allow_deferred_reasoning_output
                and _accepts_keyword(stream_round_fn, "allow_deferred_reasoning_output")
            ),
            llm_lifecycle=(lifecycle if lifecycle is not None and not lifecycle.terminal_emitted else None),
        )
    except asyncio.CancelledError as exc:
        await _close_round_after_primary_error(observation=observation, lifecycle=lifecycle, error=exc)
        raise
    except BaseException as exc:
        await _close_round_after_primary_error(observation=observation, lifecycle=lifecycle, error=exc)
        raise
