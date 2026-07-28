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
    RESEARCH_EVIDENCE_SUMMARY_PROMPT,
    SUMMARY_NON_DISCLOSURE_PROMPT,
    get_limit_summary_prompt,
)
from app.core.logger import app_logger as logger
from app.schemas.chat import ContextUsage, TextBlock, ThinkingBlock, Usage
from app.services.chat.context_manager import ContextManagementError, ContextPlan, prepare_context
from app.services.final_answer_evidence import build_used_final_answer_evidence
from app.services.stream.context_status import build_context_usage, emit_context_status
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
    if summary_finish_reason == "no_progress_summary":
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
) -> None:
    """收尾总结移除会继续诱发工具协议的旧契约与事务历史。"""

    deep_research_control_markers = (
        "【自主联网判断规则】",
        "【工具调用一致性规则】",
        "【执行计划控制规则】",
        "【深度研究执行约束】",
        "【深度研究阶段控制】",
        "【深度研究完成校验】",
        "【计划控制修正】",
    )
    filtered: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system" and "【工具调用一致性规则】" in content:
            continue
        if task_mode == "deep_research":
            if role in {"assistant", "tool"}:
                continue
            if role == "system" and any(marker in content for marker in deep_research_control_markers):
                continue
        filtered.append(message)
    messages[:] = filtered


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
    estimator_status: str | None = None,
) -> Any:
    return create_llm_round_observation(
        conversation_id=request.conversation_id,
        run_id=request.run_id,
        round_index=request.step_number,
        step_id=step_id,
        round_kind="limit_summary",
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
) -> LimitSummaryRoundResult:
    final_call_kwargs = build_limit_summary_call_kwargs(request.call_kwargs)
    try:
        context_plan = await prepare_context(
            messages=request.messages,
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
        if request.task_mode == "deep_research" and _accepts_keyword(request.stream_round_fn, "defer_output"):
            stream_kwargs["defer_output"] = True
        reasoning_buf, content_buf, tool_calls, finish_reason, usage_data = await request.stream_round_fn(
            response,
            request.conversation_id,
            request.task_id,
            request.should_use_reasoning,
            thinking_block_id,
            text_block_id,
            **stream_kwargs,
        )
    except BaseException as exc:
        await observation.finish_error(exc)
        raise
    final_context = build_context_usage(context_plan, usage_data, round_index=request.step_number)
    if request.on_context_updated is not None:
        request.on_context_updated(final_context)
    await emit_context_status(request.emitter, phase="final", context=final_context)
    await observation.finish_success(usage=usage_data, finish_reason=finish_reason)
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
    )


def accumulate_summary_usage(accumulated_usage: Usage, usage_data: Usage | None) -> Usage:
    if not usage_data:
        return accumulated_usage
    return Usage(
        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
    )


def append_summary_content_blocks(
    *,
    content_blocks: list,
    reasoning_buf: str,
    content_buf: str,
    thinking_block_id: str,
    text_block_id: str,
) -> None:
    if content_buf:
        if reasoning_buf:
            content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
        return

    if reasoning_buf:
        content_blocks.append(TextBlock(type="text", id=text_block_id, text=reasoning_buf))


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
    try:
        first_result = await asyncio.wait_for(
            call_limit_summary_round(
                request=request,
                thinking_block_id=thinking_block_id,
                text_block_id=text_block_id,
                step_id=summary_context.step_id,
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(f"触顶总结超出剩余预算: conv_id={request.conversation_id}, budget={remaining}s")
        return LimitSummaryRoundResult(reasoning_buf="", content_buf="", usage_data=None)

    if not _is_summary_tool_protocol_violation(first_result):
        result = first_result
    else:
        warning = request.warning_fn if request.warning_fn is not None else logger.warning
        warning(
            "无工具收尾总结返回了工具协议，执行一次无工具重试: "
            f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
        )
        request.messages.append({"role": "system", "content": SUMMARY_TOOL_PROTOCOL_RETRY_PROMPT})
        retry_remaining = remaining - (time.monotonic() - started_at)
        if retry_remaining <= 0:
            return _build_summary_protocol_fallback(first_result)

        try:
            retry_result = await asyncio.wait_for(
                call_limit_summary_round(
                    request=request,
                    thinking_block_id=thinking_block_id,
                    text_block_id=text_block_id,
                    step_id=summary_context.step_id,
                ),
                timeout=retry_remaining,
            )
        except asyncio.TimeoutError:
            warning(
                "无工具收尾重试超出剩余预算，使用安全失败文案: "
                f"conv_id={request.conversation_id}, budget={retry_remaining}s"
            )
            return _build_summary_protocol_fallback(first_result)

        usage_data = _combine_optional_usage(first_result.usage_data, retry_result.usage_data)
        if _is_summary_tool_protocol_violation(retry_result):
            warning(
                "无工具收尾重试仍返回工具协议，使用安全失败文案: "
                f"conv_id={request.conversation_id}, run_id={request.run_id}, step={request.step_number}"
            )
            return _build_summary_protocol_fallback(retry_result, usage_data=usage_data)
        result = LimitSummaryRoundResult(
            reasoning_buf=retry_result.reasoning_buf,
            content_buf=retry_result.content_buf,
            usage_data=usage_data,
            context=retry_result.context,
            tool_calls=(),
            finish_reason=retry_result.finish_reason,
        )

    return await _repair_deep_research_summary_citations(
        request=request,
        summary_context=summary_context,
        thinking_block_id=thinking_block_id,
        text_block_id=text_block_id,
        result=result,
        remaining=remaining - (time.monotonic() - started_at),
    )


async def _repair_deep_research_summary_citations(
    *,
    request: LimitSummaryStepRequest,
    summary_context: Any,
    thinking_block_id: str,
    text_block_id: str,
    result: LimitSummaryRoundResult,
    remaining: float,
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
    try:
        repaired = await asyncio.wait_for(
            call_limit_summary_round(
                request=request,
                thinking_block_id=thinking_block_id,
                text_block_id=text_block_id,
                step_id=summary_context.step_id,
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        warning(
            "深度研究收尾引用修正超出剩余预算: "
            f"conv_id={request.conversation_id}, budget={remaining}s"
        )
        return result
    if _is_summary_tool_protocol_violation(repaired):
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
    )


def _combine_optional_usage(*items: Usage | None) -> Usage | None:
    present = [item for item in items if item is not None]
    if not present:
        return None
    return Usage(
        input_tokens=sum(item.input_tokens for item in present),
        output_tokens=sum(item.output_tokens for item in present),
    )


def _is_summary_tool_protocol_violation(result: LimitSummaryRoundResult) -> bool:
    return bool(result.tool_calls) or result.finish_reason == "tool_protocol_error"


def _build_summary_protocol_fallback(
    result: LimitSummaryRoundResult,
    *,
    usage_data: Usage | None = None,
) -> LimitSummaryRoundResult:
    return LimitSummaryRoundResult(
        reasoning_buf="",
        content_buf=SUMMARY_PROTOCOL_FALLBACK_TEXT,
        usage_data=result.usage_data if usage_data is None else usage_data,
        context=result.context,
        tool_calls=(),
        finish_reason="protocol_fallback",
    )


async def run_limit_summary_step(
    *,
    request: LimitSummaryStepRequest,
) -> LimitSummaryOutcome:
    summary_context = await start_limit_summary_step(request=request)

    remove_conflicting_tool_usage_contract(
        request.messages,
        task_mode=request.task_mode,
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
    incomplete = False
    if request.task_mode == "deep_research":
        incomplete = await _complete_deep_research_summary(
            request=request,
            round_result=round_result,
            text_block_id=text_block_id,
            step_id=summary_context.step_id,
        )
    else:
        append_summary_content_blocks(
            content_blocks=request.content_blocks,
            reasoning_buf=round_result.reasoning_buf,
            content_buf=round_result.content_buf,
            thinking_block_id=thinking_block_id,
            text_block_id=text_block_id,
        )
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


async def _complete_deep_research_summary(
    *,
    request: LimitSummaryStepRequest,
    round_result: LimitSummaryRoundResult,
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
    request.content_blocks.append(TextBlock(type="text", id=text_block_id, text=answer))
    if not validation.is_valid:
        return True
    await _emit_deep_summary_used_evidence(
        request=request,
        answer_text=answer,
        workset=workset,
    )
    return False


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
