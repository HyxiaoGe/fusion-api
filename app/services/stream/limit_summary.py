"""Agent 触顶后的强制总结 step 编排。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from app.ai.llm_round_observability import create_llm_round_observation
from app.ai.prompts.agent_loop import LIMIT_SUMMARY_PROMPT as _LIMIT_SUMMARY_PROMPT
from app.ai.prompts.agent_loop import (
    NO_PROGRESS_SUMMARY_PROMPT,
    PLAN_REPAIR_SUMMARY_PROMPT,
    PLAN_SYNTHESIS_PROMPT,
    RESEARCH_EVIDENCE_SUMMARY_PROMPT,
    SUMMARY_NON_DISCLOSURE_PROMPT,
    get_limit_summary_prompt,
)
from app.core.logger import app_logger as logger
from app.schemas.chat import ContextUsage, KnowledgeEvidenceBlock, TextBlock, ThinkingBlock, Usage
from app.services.chat.context_manager import ContextManagementError, ContextPlan, prepare_context
from app.services.chat.model_call_language_policy import finalize_model_call_language_policy
from app.services.final_answer_evidence import build_used_final_answer_evidence
from app.services.knowledge.chat_grounding import (
    KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT,
    validate_grounded_answer,
)
from app.services.stream.context_status import build_context_usage, emit_context_status
from app.services.stream.llm_round_lifecycle import LLMRoundLifecycle, accumulate_token_usage
from app.services.stream.reasoning_policy import configure_reasoning_call_kwargs
from app.services.stream.research_evidence import (
    ResearchEvidenceWorkset,
    build_research_repair_prompt,
    validate_research_completion,
)
from app.services.stream_state_service import StreamWriteTerminalError, append_chunk

LIMIT_SUMMARY_PROMPT = _LIMIT_SUMMARY_PROMPT


def _accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


@dataclass(frozen=True)
class LimitSummaryOutcome:
    accumulated_usage: Usage
    context: ContextUsage | None = None
    incomplete: bool = False


@dataclass(frozen=True)
class LimitSummaryRoundResult:
    reasoning_buf: str
    content_buf: str
    usage_data: Usage | None
    context: ContextUsage | None = None
    tool_calls: tuple[dict, ...] = ()
    finish_reason: str = "stop"
    llm_lifecycle: LLMRoundLifecycle | None = None


@dataclass(frozen=True)
class LimitSummaryStepRequest:
    conversation_id: str
    task_id: str
    run_id: str
    step_number: int
    model_id: str
    provider: str
    litellm_model: str
    litellm_kwargs: dict
    messages: list[dict]
    should_use_reasoning: bool
    content_blocks: list
    call_kwargs: dict
    accumulated_usage: Usage
    emitter: Any
    session_cache: Any
    total_timeout_s: int
    run_start: float
    start_step_fn: Callable[..., Awaitable[Any]]
    complete_step_fn: Callable[..., Awaitable[Any]]
    llm_call_fn: Callable[..., Awaitable[Any]]
    stream_round_fn: Callable[..., Awaitable[tuple[str, str, list[dict], str, Usage | None]]]
    log_round_summary_fn: Callable[..., None]
    warning_fn: Callable[[str], None] | None = None
    clock: Callable[[], float] = time.time
    on_step_started: Callable[[str], None] | None = None
    on_context_updated: Callable[[ContextUsage], None] | None = None
    assistant_message_id: str | None = None
    summary_finish_reason: str = "limit_summary"
    task_mode: str = "standard"
    evidence_policy: str = "standard"
    research_workset: ResearchEvidenceWorkset | None = None
    defer_output: bool = True


def _streams_standard_plan_synthesis(request: LimitSummaryStepRequest) -> bool:
    """普通计划最终综合直接透传正文；深度研究仍需缓存全文完成校验。"""

    return request.summary_finish_reason == "plan_synthesis" and request.task_mode != "deep_research"


def _should_defer_summary_output(request: LimitSummaryStepRequest) -> bool:
    if _streams_standard_plan_synthesis(request):
        return False
    return request.defer_output or request.task_mode == "deep_research"


def build_limit_summary_call_kwargs(call_kwargs: dict) -> dict:
    return {key: value for key, value in call_kwargs.items() if key not in ("tools", "tool_choice")}


def compute_summary_timeout(*, total_timeout_s: int, run_start: float, clock: Callable[[], float]) -> float:
    return max(10, total_timeout_s - (clock() - run_start))


def append_limit_summary_prompt(
    messages: list[dict],
    *,
    summary_finish_reason: str = "limit_summary",
    task_mode: str = "standard",
) -> None:
    if summary_finish_reason == "plan_synthesis":
        prompt = PLAN_SYNTHESIS_PROMPT
    elif summary_finish_reason == "no_progress_summary":
        prompt = NO_PROGRESS_SUMMARY_PROMPT
    elif summary_finish_reason == "plan_repair_exhausted":
        prompt = PLAN_REPAIR_SUMMARY_PROMPT
    elif summary_finish_reason == "research_evidence_repair_exhausted":
        prompt = RESEARCH_EVIDENCE_SUMMARY_PROMPT
    else:
        prompt = get_limit_summary_prompt()
        if SUMMARY_NON_DISCLOSURE_PROMPT not in prompt:
            prompt = f"{prompt}\n\n{SUMMARY_NON_DISCLOSURE_PROMPT}"
    if task_mode == "deep_research" and RESEARCH_EVIDENCE_SUMMARY_PROMPT not in prompt:
        prompt = f"{prompt}\n\n{RESEARCH_EVIDENCE_SUMMARY_PROMPT}"
    messages.append({"role": "system", "content": prompt})


def remove_conflicting_tool_usage_contract(
    messages: list[dict],
    *,
    task_mode: str = "standard",
    final_synthesis: bool = False,
) -> None:
    """收尾总结移除会继续诱发工具协议的旧契约与事务历史。"""

    del final_synthesis  # 终局总结统一清理控制契约，不再按结束原因分叉。
    terminal_control_markers = (
        "【自主联网判断规则】",
        "【工具调用一致性规则】",
        "【执行计划控制规则】",
        "【可核验证据计划规则】",
        "【计划控制修正】",
        "【计划执行修正】",
    )
    deep_research_control_markers = (
        "【深度研究执行约束】",
        "【深度研究阶段控制】",
        "【深度研究完成校验】",
    )
    strip_tool_transactions = task_mode == "deep_research" or _only_recoverable_tool_transactions(messages)
    filtered: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system" and any(marker in content for marker in terminal_control_markers):
            continue
        if (
            task_mode == "deep_research"
            and role == "system"
            and any(marker in content for marker in deep_research_control_markers)
        ):
            continue
        if strip_tool_transactions and role == "assistant" and message.get("tool_calls"):
            continue
        if strip_tool_transactions and role == "tool":
            continue
        filtered.append(message)
    messages[:] = filtered


def _only_recoverable_tool_transactions(messages: list[dict]) -> bool:
    """仅当全部事务都有服务端安全投影时，才从普通总结上下文移除原始协议。"""

    recoverable_tool_names = {"update_plan", "web_search", "url_read"}
    tool_call_names: list[str] = []
    tool_call_ids: set[str] = set()
    tool_message_ids: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if not tool_call_id:
                return False
            tool_message_ids.add(tool_call_id)
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return False
            function = tool_call.get("function")
            if isinstance(function, dict):
                tool_name = str(function.get("name") or "")
            else:
                tool_name = str(tool_call.get("name") or "")
            if not tool_name:
                return False
            tool_call_id = str(tool_call.get("id") or "")
            if not tool_call_id:
                return False
            tool_call_ids.add(tool_call_id)
            tool_call_names.append(tool_name)

    if not tool_call_names:
        return False
    return tool_call_ids == tool_message_ids and all(
        tool_name in recoverable_tool_names for tool_name in tool_call_names
    )


SUMMARY_TOOL_PROTOCOL_RETRY_PROMPT = (
    "请立即基于已有资料直接输出面向用户的最终答复。当前不能调用任何工具；"
    "不要输出任何工具调用、DSML/XML 协议、函数名、参数或内部规划，只输出自然语言答案。"
)

SUMMARY_PROTOCOL_FALLBACK_TEXT = "当前未能生成可靠的最终答复，请稍后重试。"
DEEP_RESEARCH_INCOMPLETE_TEXT = (
    "本次研究尚未完成，当前取得的可核验依据不足，暂时无法给出可靠结论。你可以稍后重试，或缩小研究范围后重新发起。"
)


def _create_limit_summary_observation(
    *,
    request: LimitSummaryStepRequest,
    context_plan: ContextPlan,
    step_id: str,
    call_kwargs: dict,
    round_index: int | None = None,
    estimator_status: str | None = None,
) -> Any:
    round_kind = "plan_synthesis" if request.summary_finish_reason == "plan_synthesis" else "limit_summary"
    return create_llm_round_observation(
        conversation_id=request.conversation_id,
        run_id=request.run_id,
        round_index=round_index or request.step_number,
        step_id=step_id,
        round_kind=round_kind,
        model_id=request.model_id,
        provider=request.provider,
        litellm_model=request.litellm_model,
        messages=context_plan.messages,
        call_kwargs=call_kwargs,
        assistant_message_id=request.assistant_message_id,
        context_management=context_plan.telemetry(),
        estimated_prompt_tokens=context_plan.estimated_tokens_after,
        estimator_status=estimator_status,
    )


async def call_limit_summary_round(
    *,
    request: LimitSummaryStepRequest,
    thinking_block_id: str,
    text_block_id: str,
    step_id: str,
    partial_output: dict[str, str] | None = None,
    round_index: int | None = None,
) -> LimitSummaryRoundResult:
    final_call_kwargs = configure_reasoning_call_kwargs(
        build_limit_summary_call_kwargs(request.call_kwargs),
        provider=request.provider,
        should_use_reasoning=request.should_use_reasoning,
    )
    finalized_messages = finalize_model_call_language_policy(request.messages)
    try:
        context_plan = await prepare_context(
            messages=finalized_messages,
            model_id=request.model_id,
            litellm_model=request.litellm_model,
            call_kwargs=final_call_kwargs,
        )
    except ContextManagementError as error:
        error_context = build_context_usage(error.plan, round_index=request.step_number)
        if request.on_context_updated is not None:
            request.on_context_updated(error_context)
        await emit_context_status(request.emitter, phase="error", context=error_context)
        observation = _create_limit_summary_observation(
            request=request,
            context_plan=error.plan,
            step_id=step_id,
            call_kwargs=final_call_kwargs,
            round_index=round_index,
            estimator_status="context_manager_error",
        )
        observation.start()
        await observation.finish_error(error)
        raise
    effective_messages = context_plan.messages
    estimated_context = build_context_usage(context_plan, round_index=request.step_number)
    if request.on_context_updated is not None:
        request.on_context_updated(estimated_context)
    await emit_context_status(request.emitter, phase="estimated", context=estimated_context)
    observation = _create_limit_summary_observation(
        request=request,
        context_plan=context_plan,
        step_id=step_id,
        call_kwargs=final_call_kwargs,
        round_index=round_index,
    )
    lifecycle = await LLMRoundLifecycle.start(
        emitter=request.emitter,
        observation=observation,
        round_index=round_index or request.step_number,
        model=request.model_id,
        provider=request.provider,
        parent_step_id=step_id,
    )
    observation.start()
    try:
        response = await request.llm_call_fn(
            request.litellm_model,
            request.litellm_kwargs,
            effective_messages,
            **final_call_kwargs,
        )
        response = observation.wrap_response(response)
        stream_kwargs = {"run_id": request.run_id, "step_id": step_id}
        if _accepts_keyword(request.stream_round_fn, "provider"):
            stream_kwargs["provider"] = request.provider
        if _accepts_keyword(request.stream_round_fn, "defer_output"):
            stream_kwargs["defer_output"] = _should_defer_summary_output(request)
        if (
            request.should_use_reasoning
            and _should_defer_summary_output(request)
            and request.evidence_policy != "knowledge_grounded_v1"
            and _accepts_keyword(request.stream_round_fn, "allow_deferred_reasoning_output")
        ):
            stream_kwargs["allow_deferred_reasoning_output"] = True
        if partial_output is not None and _accepts_keyword(request.stream_round_fn, "partial_output"):
            stream_kwargs["partial_output"] = partial_output
        visible_callback_supported = _accepts_keyword(request.stream_round_fn, "on_visible_output")
        if (
            lifecycle is not None
            and not _should_defer_summary_output(request)
            and visible_callback_supported
        ):
            stream_kwargs["on_visible_output"] = lifecycle.publish_visible_output
        candidate_callback = getattr(observation, "observe_output_candidate", None)
        if callable(candidate_callback) and _accepts_keyword(request.stream_round_fn, "on_output_candidate"):
            stream_kwargs["on_output_candidate"] = candidate_callback
        capture_candidate_time = getattr(observation, "capture_output_candidate_time", None)
        if callable(capture_candidate_time) and _accepts_keyword(
            request.stream_round_fn,
            "capture_output_candidate_time",
        ):
            stream_kwargs["capture_output_candidate_time"] = capture_candidate_time
        reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = await request.stream_round_fn(
            response,
            request.conversation_id,
            request.task_id,
            request.should_use_reasoning,
            thinking_block_id,
            text_block_id,
            **stream_kwargs,
        )
    except asyncio.CancelledError as exc:
        await _close_summary_round_after_primary_error(
            observation=observation,
            lifecycle=lifecycle,
            error=exc,
        )
        raise
    except BaseException as exc:
        await _close_summary_round_after_primary_error(
            observation=observation,
            lifecycle=lifecycle,
            error=exc,
        )
        raise
    try:
        freeze = getattr(observation, "freeze", None)
        if callable(freeze):
            freeze()
        final_context = build_context_usage(context_plan, usage_data, round_index=request.step_number)
        if request.on_context_updated is not None:
            request.on_context_updated(final_context)
        await emit_context_status(request.emitter, phase="final", context=final_context)
        await observation.finish_success(usage=usage_data, finish_reason=finish_reason)
        if lifecycle is not None:
            lifecycle.record_result(usage=usage_data, finish_reason=finish_reason)
            if finish_reason == "cancelled":
                await lifecycle.finish_cancelled(reason="superseded")
            elif _should_defer_summary_output(request) and tool_calls:
                await lifecycle.publish_tool_output()
            elif not _should_defer_summary_output(request):
                if tool_calls:
                    await lifecycle.publish_tool_output()
                elif content_buf:
                    await lifecycle.publish_visible_output("content")
                elif reasoning_buf:
                    await lifecycle.publish_visible_output("reasoning")
                await lifecycle.finish_success(output_visible=False)
        request.log_round_summary_fn(
            conversation_id=request.conversation_id,
            run_id=request.run_id,
            step_number=request.step_number,
            model_id=request.model_id,
            provider=request.provider,
            finish_reason=request.summary_finish_reason,
            tool_calls_count=len(tool_calls),
            reasoning_buf=reasoning_buf,
            content_buf=content_buf,
        )
        return LimitSummaryRoundResult(
            reasoning_buf=reasoning_buf,
            content_buf=content_buf,
            usage_data=usage_data,
            context=final_context,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            llm_lifecycle=(lifecycle if lifecycle is not None and not lifecycle.terminal_emitted else None),
        )
    except asyncio.CancelledError as exc:
        await _close_summary_round_after_primary_error(
            observation=observation,
            lifecycle=lifecycle,
            error=exc,
        )
        raise
    except BaseException as exc:
        await _close_summary_round_after_primary_error(
            observation=observation,
            lifecycle=lifecycle,
            error=exc,
        )
        raise


def accumulate_summary_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage:
    return accumulate_token_usage(accumulated_usage, usage_data)


async def _close_summary_round_after_primary_error(
    *, observation: Any, lifecycle: LLMRoundLifecycle | None, error: BaseException
) -> None:
    freeze = getattr(observation, "freeze", None)
    if callable(freeze):
        freeze()
    try:
        await observation.finish_error(error)
    except BaseException as secondary:
        logger.warning("总结 LLM 观测异常收尾失败，保留主异常: error_type=%s", type(secondary).__name__)
    if lifecycle is None:
        return
    try:
        if isinstance(error, asyncio.CancelledError):
            await lifecycle.finish_cancelled(reason="shutdown")
        else:
            await lifecycle.finish_failed(error)
    except BaseException as secondary:
        logger.warning("总结 LLM 生命周期异常收尾失败，保留主异常: error_type=%s", type(secondary).__name__)


def append_summary_content_blocks(
    *,
    content_blocks: list,
    content_buf: str,
    text_block_id: str,
) -> None:
    if content_buf:
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))


async def complete_limit_summary_step(
    *,
    summary_context: Any,
    emitter: Any,
    session_cache: Any,
    complete_step_fn: Callable[..., Awaitable[Any]],
    clock: Callable[[], float],
) -> None:
    await complete_step_fn(
        context=summary_context,
        emitter=emitter,
        session_cache=session_cache,
        tool_names=[],
        tool_call_count=0,
        clock=clock,
    )


async def start_limit_summary_step(*, request: LimitSummaryStepRequest) -> Any:
    return await request.start_step_fn(
        emitter=request.emitter,
        session_cache=request.session_cache,
        run_id=request.run_id,
        step_number=request.step_number,
        clock=request.clock,
        on_step_started=request.on_step_started,
    )


async def run_summary_round_with_timeout(
    *,
    request: LimitSummaryStepRequest,
    summary_context: Any,
    thinking_block_id: str,
    text_block_id: str,
    remaining: float,
) -> LimitSummaryRoundResult:
    started_at = time.monotonic()
    next_round_index = request.step_number
    first_partial: dict[str, str] = {}
    try:
        first_result = await asyncio.wait_for(
            call_limit_summary_round(
                request=request,
                thinking_block_id=thinking_block_id,
                text_block_id=text_block_id,
                step_id=summary_context.step_id,
                partial_output=first_partial if _streams_standard_plan_synthesis(request) else None,
                round_index=next_round_index,
            ),
            timeout=remaining,
        )
        next_round_index += 1
    except asyncio.TimeoutError:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(f"触顶总结超出剩余预算: conv_id={request.conversation_id}, budget={remaining}s")
        return _build_timeout_partial_result(first_partial)
    except StreamWriteTerminalError:
        raise
    except Exception as error:
        partial_result = _build_stream_error_partial_result(first_partial)
        if partial_result is None:
            raise
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(
            "流式计划综合异常中止，保留已发送的安全片段: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, "
            f"step={request.step_number}, error_type={type(error).__name__}"
        )
        return partial_result

    if not _is_summary_tool_protocol_violation(first_result):
        result = first_result
    elif _streams_standard_plan_synthesis(request) and first_result.content_buf:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(
            "流式计划综合返回了工具协议，保留已发送的安全正文并终止综合: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
        )
        result = LimitSummaryRoundResult(
            reasoning_buf=first_result.reasoning_buf,
            content_buf=first_result.content_buf,
            usage_data=first_result.usage_data,
            context=first_result.context,
            tool_calls=(),
            finish_reason="protocol_fallback",
            llm_lifecycle=first_result.llm_lifecycle,
        )
    else:
        await _finish_summary_round_lifecycle(first_result, model_output_visible=False)
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(
            "无工具收尾总结返回了工具协议，执行一次无工具重试: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
        )
        request.messages.append({"role": "system", "content": SUMMARY_TOOL_PROTOCOL_RETRY_PROMPT})
        retry_remaining = remaining - (time.monotonic() - started_at)
        if retry_remaining <= 0:
            return _build_streamed_retry_failure(
                request=request,
                first_result=first_result,
            )

        retry_partial: dict[str, str] = {}
        try:
            retry_result = await asyncio.wait_for(
                call_limit_summary_round(
                    request=request,
                    thinking_block_id=thinking_block_id,
                    text_block_id=text_block_id,
                    step_id=summary_context.step_id,
                    partial_output=retry_partial if _streams_standard_plan_synthesis(request) else None,
                    round_index=next_round_index,
                ),
                timeout=retry_remaining,
            )
            next_round_index += 1
        except asyncio.TimeoutError:
            warning(
                "无工具收尾重试超出剩余预算，使用安全失败文案: "
                f"conv_id={request.conversation_id}, budget={retry_remaining}s"
            )
            return _build_streamed_retry_failure(
                request=request,
                first_result=first_result,
                retry_partial=retry_partial,
            )
        except StreamWriteTerminalError:
            raise
        except Exception as error:
            if not _streams_standard_plan_synthesis(request):
                raise
            partial_result = _build_streamed_retry_failure(
                request=request,
                first_result=first_result,
                retry_partial=retry_partial,
                finish_reason="stream_error_partial",
            )
            if not partial_result.reasoning_buf and not partial_result.content_buf:
                raise
            warning(
                "流式计划综合重试异常中止，保留已发送的安全片段: "
                f"conv_id={request.conversation_id}, run_id={request.run_id}, "
                f"step={request.step_number}, error_type={type(error).__name__}"
            )
            return partial_result

        usage_data = _combine_optional_usage(first_result.usage_data, retry_result.usage_data)
        if _is_summary_tool_protocol_violation(retry_result):
            await _finish_summary_round_lifecycle(retry_result, model_output_visible=False)
            warning(
                "无工具收尾重试仍返回工具协议，使用安全失败文案: "
                f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
            )
            return _build_streamed_retry_failure(
                request=request,
                first_result=first_result,
                retry_result=retry_result,
                usage_data=usage_data,
            )
        if _streams_standard_plan_synthesis(request):
            result = LimitSummaryRoundResult(
                reasoning_buf=first_result.reasoning_buf + retry_result.reasoning_buf,
                content_buf=first_result.content_buf + retry_result.content_buf,
                usage_data=usage_data,
                context=retry_result.context,
                tool_calls=(),
                finish_reason=retry_result.finish_reason,
                llm_lifecycle=retry_result.llm_lifecycle,
            )
        else:
            result = LimitSummaryRoundResult(
                reasoning_buf=first_result.reasoning_buf + retry_result.reasoning_buf,
                content_buf=retry_result.content_buf,
                usage_data=usage_data,
                context=retry_result.context,
                tool_calls=(),
                finish_reason=retry_result.finish_reason,
                llm_lifecycle=retry_result.llm_lifecycle,
            )

    return await _repair_deep_research_summary_citations(
        request=request,
        summary_context=summary_context,
        thinking_block_id=thinking_block_id,
        text_block_id=text_block_id,
        result=result,
        remaining=remaining - (time.monotonic() - started_at),
        round_index=next_round_index,
    )


async def _repair_deep_research_summary_citations(
    *,
    request: LimitSummaryStepRequest,
    summary_context: Any,
    thinking_block_id: str,
    text_block_id: str,
    result: LimitSummaryRoundResult,
    remaining: float,
    round_index: int,
) -> LimitSummaryRoundResult:
    """证据充足但最终引用缺失或越界时，允许一次无工具引用修正。"""

    if request.task_mode != "deep_research":
        return result
    workset = request.research_workset or ResearchEvidenceWorkset()
    validation = validate_research_completion(workset, result.content_buf)
    if validation.is_valid or validation.reason not in {"missing_citation", "invalid_citation"}:
        return result

    warning = request.warning_fn if request.warning_fn is not None else logger.warning
    warning(
        "深度研究收尾引用校验未通过，执行一次无工具引用修正: "
        f"conv_id={request.conversation_id}, run_id={request.run_id}, "
        f"step={request.step_number}, reason={validation.reason}"
    )
    if remaining <= 0:
        return result
    request.messages.append(
        {
            "role": "system",
            "content": build_research_repair_prompt(validation.reason, workset),
        }
    )
    await _finish_summary_round_lifecycle(result, model_output_visible=False)
    try:
        repaired = await asyncio.wait_for(
            call_limit_summary_round(
                request=request,
                thinking_block_id=thinking_block_id,
                text_block_id=text_block_id,
                step_id=summary_context.step_id,
                round_index=round_index,
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        warning(f"深度研究收尾引用修正超出剩余预算: conv_id={request.conversation_id}, budget={remaining}s")
        return result
    if _is_summary_tool_protocol_violation(repaired):
        await _finish_summary_round_lifecycle(repaired, model_output_visible=False)
        warning(
            "深度研究收尾引用修正返回工具协议，保留原候选等待安全门禁: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
        )
        return result
    return LimitSummaryRoundResult(
        reasoning_buf=repaired.reasoning_buf,
        content_buf=repaired.content_buf,
        usage_data=_combine_optional_usage(result.usage_data, repaired.usage_data),
        context=repaired.context,
        tool_calls=(),
        finish_reason=repaired.finish_reason,
        llm_lifecycle=repaired.llm_lifecycle,
    )


def _combine_optional_usage(*items: Usage | None) -> Usage | None:
    present = [item for item in items if item is not None]
    if not present:
        return None
    combined = present[0]
    for item in present[1:]:
        combined = accumulate_token_usage(combined, item)
    return combined


def _is_summary_tool_protocol_violation(result: LimitSummaryRoundResult) -> bool:
    return bool(result.tool_calls) or result.finish_reason == "tool_protocol_error"


async def _finish_summary_round_lifecycle(
    result: LimitSummaryRoundResult,
    *,
    model_output_visible: bool,
) -> None:
    lifecycle = result.llm_lifecycle
    if lifecycle is None:
        return
    if model_output_visible:
        await lifecycle.publish_visible_output("content")
    await lifecycle.finish_success(output_visible=False)


def _build_summary_protocol_fallback(
    result: LimitSummaryRoundResult,
    *,
    usage_data: Usage | None = None,
) -> LimitSummaryRoundResult:
    return LimitSummaryRoundResult(
        reasoning_buf=result.reasoning_buf,
        content_buf=SUMMARY_PROTOCOL_FALLBACK_TEXT,
        usage_data=result.usage_data if usage_data is None else usage_data,
        context=result.context,
        tool_calls=(),
        finish_reason="protocol_fallback",
        llm_lifecycle=result.llm_lifecycle,
    )


def _build_timeout_partial_result(partial_output: dict[str, str]) -> LimitSummaryRoundResult:
    reasoning_buf = partial_output.get("reasoning_buf", "")
    content_buf = partial_output.get("content_buf", "")
    return LimitSummaryRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        usage_data=None,
        finish_reason="timeout_partial" if reasoning_buf or content_buf else "timeout",
    )


def _build_stream_error_partial_result(
    partial_output: dict[str, str],
) -> LimitSummaryRoundResult | None:
    reasoning_buf = partial_output.get("reasoning_buf", "")
    content_buf = partial_output.get("content_buf", "")
    if not reasoning_buf and not content_buf:
        return None
    return LimitSummaryRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        usage_data=None,
        finish_reason="stream_error_partial",
    )


def _build_streamed_retry_failure(
    *,
    request: LimitSummaryStepRequest,
    first_result: LimitSummaryRoundResult,
    retry_partial: dict[str, str] | None = None,
    retry_result: LimitSummaryRoundResult | None = None,
    usage_data: Usage | None = None,
    finish_reason: str = "protocol_fallback",
) -> LimitSummaryRoundResult:
    if not _streams_standard_plan_synthesis(request):
        fallback_result = retry_result or first_result
        if retry_result is not None:
            fallback_result = LimitSummaryRoundResult(
                reasoning_buf=first_result.reasoning_buf + retry_result.reasoning_buf,
                content_buf=retry_result.content_buf,
                usage_data=retry_result.usage_data,
                context=retry_result.context,
                tool_calls=(),
                finish_reason=retry_result.finish_reason,
                llm_lifecycle=retry_result.llm_lifecycle,
            )
        return _build_summary_protocol_fallback(
            fallback_result,
            usage_data=usage_data,
        )
    retry_reasoning = (
        retry_result.reasoning_buf if retry_result is not None else (retry_partial or {}).get("reasoning_buf", "")
    )
    retry_content = (
        retry_result.content_buf if retry_result is not None else (retry_partial or {}).get("content_buf", "")
    )
    reasoning_buf = first_result.reasoning_buf + retry_reasoning
    content_buf = first_result.content_buf + retry_content
    if not reasoning_buf and not content_buf:
        return LimitSummaryRoundResult(
            reasoning_buf="",
            content_buf="",
            usage_data=usage_data if usage_data is not None else first_result.usage_data,
            context=(retry_result or first_result).context,
            tool_calls=(),
            finish_reason=finish_reason,
            llm_lifecycle=None,
        )
    return LimitSummaryRoundResult(
        reasoning_buf=reasoning_buf,
        content_buf=content_buf,
        usage_data=usage_data if usage_data is not None else first_result.usage_data,
        context=(retry_result or first_result).context,
        tool_calls=(),
        finish_reason=finish_reason,
        llm_lifecycle=None,
    )


async def run_limit_summary_step(
    *,
    request: LimitSummaryStepRequest,
) -> LimitSummaryOutcome:
    summary_context = await start_limit_summary_step(request=request)

    remove_conflicting_tool_usage_contract(
        request.messages,
        task_mode=request.task_mode,
        final_synthesis=True,
    )
    append_limit_summary_prompt(
        request.messages,
        summary_finish_reason=request.summary_finish_reason,
        task_mode=request.task_mode,
    )
    thinking_block_id = summary_context.thinking_block_id
    text_block_id = summary_context.text_block_id
    remaining = compute_summary_timeout(
        total_timeout_s=request.total_timeout_s,
        run_start=request.run_start,
        clock=request.clock,
    )

    round_result = await run_summary_round_with_timeout(
        request=request,
        summary_context=summary_context,
        thinking_block_id=thinking_block_id,
        text_block_id=text_block_id,
        remaining=remaining,
    )

    next_usage = accumulate_summary_usage(request.accumulated_usage, round_result.usage_data)
    try:
        incomplete = await _commit_limit_summary_result(
            request=request,
            round_result=round_result,
            summary_context=summary_context,
            thinking_block_id=thinking_block_id,
            text_block_id=text_block_id,
        )
    finally:
        await _finish_summary_round_lifecycle(round_result, model_output_visible=False)
    await complete_limit_summary_step(
        summary_context=summary_context,
        emitter=request.emitter,
        session_cache=request.session_cache,
        complete_step_fn=request.complete_step_fn,
        clock=request.clock,
    )
    return LimitSummaryOutcome(
        accumulated_usage=next_usage,
        context=round_result.context,
        incomplete=incomplete,
    )


async def _commit_limit_summary_result(
    *,
    request: LimitSummaryStepRequest,
    round_result: LimitSummaryRoundResult,
    summary_context: Any,
    thinking_block_id: str,
    text_block_id: str,
) -> bool:
    if round_result.reasoning_buf:
        request.content_blocks.append(
            ThinkingBlock(
                type="thinking",
                id=thinking_block_id,
                thinking=round_result.reasoning_buf,
            )
        )

    if request.task_mode == "deep_research":
        return await _complete_deep_research_summary(
            request=request,
            round_result=round_result,
            thinking_block_id=thinking_block_id,
            text_block_id=text_block_id,
            step_id=summary_context.step_id,
        )
    elif request.evidence_policy == "knowledge_grounded_v1":
        evidence_block = next(
            (block for block in reversed(request.content_blocks) if isinstance(block, KnowledgeEvidenceBlock)),
            None,
        )
        candidate = round_result.content_buf.strip()
        valid = evidence_block is not None and validate_grounded_answer(candidate, evidence_block)
        answer = candidate if valid else KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT
        await append_chunk(
            request.conversation_id,
            "answering",
            answer,
            text_block_id,
            task_id=request.task_id,
            run_id=request.run_id,
            step_id=summary_context.step_id,
        )
        if valid:
            await _finish_summary_round_lifecycle(round_result, model_output_visible=True)
        request.content_blocks.append(TextBlock(type="text", id=text_block_id, text=answer))
        await _emit_knowledge_summary_used_evidence(request=request, answer_text=answer)
        return not valid
    elif request.summary_finish_reason == "plan_synthesis":
        has_answer = bool(round_result.content_buf.strip())
        has_streamed_content = _streams_standard_plan_synthesis(request) and bool(round_result.content_buf.strip())
        incomplete = (
            round_result.finish_reason
            in {
                "protocol_fallback",
                "timeout",
                "timeout_partial",
                "stream_error_partial",
            }
            or not has_answer
        )
        answer = round_result.content_buf if has_streamed_content else SUMMARY_PROTOCOL_FALLBACK_TEXT
        if not has_streamed_content:
            await append_chunk(
                request.conversation_id,
                "answering",
                answer,
                text_block_id,
                task_id=request.task_id,
                run_id=request.run_id,
                step_id=summary_context.step_id,
            )
        if answer:
            request.content_blocks.append(TextBlock(type="text", id=text_block_id, text=answer))
        return incomplete
    else:
        answer = round_result.content_buf.strip()
        incomplete = round_result.finish_reason == "protocol_fallback" or not answer
        if not answer:
            answer = SUMMARY_PROTOCOL_FALLBACK_TEXT
        if request.defer_output:
            await append_chunk(
                request.conversation_id,
                "answering",
                answer,
                text_block_id,
                task_id=request.task_id,
                run_id=request.run_id,
                step_id=summary_context.step_id,
            )
            if round_result.content_buf.strip():
                await _finish_summary_round_lifecycle(round_result, model_output_visible=True)
        if round_result.content_buf.strip():
            append_summary_content_blocks(
                content_blocks=request.content_blocks,
                content_buf=round_result.content_buf,
                text_block_id=text_block_id,
            )
        else:
            request.content_blocks.append(TextBlock(type="text", id=text_block_id, text=answer))
        return incomplete


async def _complete_deep_research_summary(
    *,
    request: LimitSummaryStepRequest,
    round_result: LimitSummaryRoundResult,
    thinking_block_id: str,
    text_block_id: str,
    step_id: str,
) -> bool:
    workset = request.research_workset or ResearchEvidenceWorkset()
    validation = validate_research_completion(workset, round_result.content_buf)
    answer = round_result.content_buf.strip() if validation.is_valid else DEEP_RESEARCH_INCOMPLETE_TEXT
    if not validation.is_valid:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(
            "深度研究收尾未通过安全门禁: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, "
            f"step={request.step_number}, research_validation_reason={validation.reason}"
        )
    await append_chunk(
        request.conversation_id,
        "answering",
        answer,
        text_block_id,
        task_id=request.task_id,
        run_id=request.run_id,
        step_id=step_id,
    )
    if validation.is_valid:
        await _finish_summary_round_lifecycle(round_result, model_output_visible=True)
    append_summary_content_blocks(
        content_blocks=request.content_blocks,
        content_buf=answer,
        text_block_id=text_block_id,
    )
    if not validation.is_valid:
        return True
    await _emit_deep_summary_used_evidence(
        request=request,
        answer_text=answer,
        workset=workset,
    )
    return False


async def _emit_knowledge_summary_used_evidence(
    *,
    request: LimitSummaryStepRequest,
    answer_text: str,
) -> None:
    emit = getattr(request.emitter, "evidence_item_upserted", None)
    if emit is None:
        return
    try:
        evidence_items = build_used_final_answer_evidence(
            content_blocks=request.content_blocks,
            answer_text=answer_text,
            evidence_policy="knowledge_grounded_v1",
        )
        for evidence in evidence_items:
            await emit(tool_call_id=None, evidence=evidence)
    except StreamWriteTerminalError:
        raise
    except Exception as error:  # noqa: BLE001 — used evidence 观测失败不能覆盖安全收尾
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(f"知识库总结 used evidence 发送失败: error_type={type(error).__name__}")


async def _emit_deep_summary_used_evidence(
    *,
    request: LimitSummaryStepRequest,
    answer_text: str,
    workset: ResearchEvidenceWorkset,
) -> None:
    emit = getattr(request.emitter, "evidence_item_upserted", None)
    if emit is None:
        return
    try:
        evidence_items = build_used_final_answer_evidence(
            content_blocks=request.content_blocks,
            answer_text=answer_text,
            evidence_policy=request.evidence_policy,
            allowed_citation_indexes=workset.valid_citation_indexes,
        )
        for evidence in evidence_items:
            await emit(tool_call_id=None, evidence=evidence)
    except StreamWriteTerminalError:
        raise
    except Exception as error:  # noqa: BLE001 — used evidence 观测失败不能覆盖安全收尾
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(f"深度研究总结 used evidence 发送失败: error_type={type(error).__name__}")
