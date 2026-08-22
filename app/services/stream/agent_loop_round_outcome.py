"""Agent loop round outcome 分发。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logger import app_logger as logger
from app.schemas.chat import KnowledgeEvidenceBlock, ThinkingBlock
from app.services.final_answer_evidence import build_used_final_answer_evidence
from app.services.knowledge.chat_grounding import (
    KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT,
    validate_grounded_answer,
)
from app.services.mcp.amap_product_tools import AMAP_PRODUCT_TOOL_NAMES
from app.services.mcp.flyai_travel_tools import FLYAI_TRAVEL_TOOL_NAMES
from app.services.stream.agent_loop_outcome import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_tool_round_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.product_answer_validator import (
    repair_unsupported_product_answer,
    validate_product_answer,
)
from app.services.stream.product_result_answer import (
    build_grounded_product_answer,
    build_product_tool_failure_answer,
    build_tool_repair_clarification,
    has_product_result_blocks,
    neutralize_product_provider_mentions,
)
from app.services.stream.research_evidence import (
    build_research_repair_prompt,
    validate_research_completion,
)
from app.services.stream.round_completion import append_round_content_blocks, complete_text_response_step
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundOutcome
from app.services.stream_state_service import StreamWriteTerminalError, append_chunk

PLAN_REQUIRED_RETRY_PROMPT = (
    "【计划控制修正】当前为强制计划模式，上一轮未建立有效计划，不能直接回答。"
    "必须先调用计划控制工具创建 2 至 6 个步骤，再继续回答或调用外部工具。"
    "只静默修正，不要向用户解释这条内部规则。"
)
PLAN_EXECUTION_REQUIRED_RETRY_PROMPT = (
    "【计划执行修正】当前执行计划仍有未完成的工具步骤。不要输出最终回答；"
    "请只调用这些步骤声明的真实工具，并为每次调用填写对应的 _plan_item_id。"
)


@dataclass(frozen=True)
class AgentRoundOutcomeRequest:
    db: object
    messages: list[dict]
    state: AgentLoopState
    runtime: AgentLoopRuntime
    step_number: int
    step_context: AgentStepContext
    round_result: AgentRoundResult


async def handle_agent_round_outcome(
    *,
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome | None:
    primary_error: BaseException | None = None
    try:
        return await _handle_agent_round_outcome(request=request)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        lifecycle = request.round_result.llm_lifecycle
        if lifecycle is not None:
            try:
                await lifecycle.finish_success(output_visible=False)
            except BaseException as secondary_error:
                if primary_error is None:
                    raise
                try:
                    logger.warning(
                        "Agent deferred LLM 生命周期收尾失败，保留主异常: "
                        "error_type=%s error_code=deferred_terminal_failure",
                        type(secondary_error).__name__,
                    )
                except BaseException:
                    pass


async def _handle_agent_round_outcome(
    *,
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome | None:
    if _requires_deep_synthesis_protocol_summary(request):
        return await _complete_deep_synthesis_protocol_round(request)

    finish_reason = request.round_result.finish_reason
    if finish_reason == "stop":
        if _requires_plan_before_stop(request):
            return await _repair_missing_required_plan(request)
        if _requires_execution_before_stop(request):
            await _repair_incomplete_execution(request)
            return None
        if _requires_research_completion_repair(request):
            return await _repair_research_completion(request)
        if _requires_plan_synthesis(request):
            await _complete_round_before_plan_synthesis(request)
            return None
        if _needs_empty_answer_summary(request):
            await _complete_empty_round_before_summary(request)
            return AgentLoopOutcome(exit=AgentLoopExit.SUMMARY_REQUIRED)
        await _complete_text_round(request)
        return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

    if finish_reason == "cancelled":
        _append_round_blocks(request)
        return AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED, error_msg="被新请求取代")

    if finish_reason == "tool_calls" and request.round_result.tool_calls:
        return await _handle_tool_calls_round(request)

    if _requires_plan_before_stop(request):
        return await _repair_missing_required_plan(request)
    if _requires_execution_before_stop(request):
        await _repair_incomplete_execution(request)
        return None
    if _requires_research_completion_repair(request):
        return await _repair_research_completion(request)
    if _requires_plan_synthesis(request):
        await _complete_round_before_plan_synthesis(request)
        return None
    if finish_reason == "tool_protocol_error":
        return await _complete_tool_protocol_error_round(request)

    await _complete_unknown_round(request)
    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


def _requires_deep_synthesis_protocol_summary(request: AgentRoundOutcomeRequest) -> bool:
    """零公告工具的深研综合轮若仍吐出工具协议，直接进入无工具安全收口。"""

    workset = request.state.research_workset
    return (
        request.runtime.task_mode == "deep_research"
        and workset.successful_searches >= 1
        and len(workset.successful_read_urls) >= 2
        and request.round_result.announced_tool_names == frozenset()
        and (bool(request.round_result.tool_calls) or request.round_result.finish_reason == "tool_protocol_error")
    )


async def _complete_deep_synthesis_protocol_round(
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome:
    request.runtime.warning_fn(
        "深度研究综合阶段返回未公告工具协议，切换到无工具证据收口: "
        f"conv_id={request.runtime.conversation_id}, run_id={request.runtime.run_id}, "
        f"step={request.step_number}"
    )
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.state.clear_current_step()
    return AgentLoopOutcome(
        exit=AgentLoopExit.SUMMARY_REQUIRED,
        summary_finish_reason="research_evidence_repair_exhausted",
    )


async def _complete_tool_protocol_error_round(
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome:
    """协议前缀不是最终答案，统一交给无工具总结做有界重试。"""

    request.runtime.warning_fn(
        "模型工具协议无法解析，切换到无工具安全收口: "
        f"conv_id={request.runtime.conversation_id}, run_id={request.runtime.run_id}, "
        f"step={request.step_number}"
    )
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.state.clear_current_step()
    if _has_product_answer_context(request.state):
        return AgentLoopOutcome(exit=AgentLoopExit.PRODUCT_RESULT_READY)
    summary_finish_reason = (
        "plan_synthesis" if request.state.plan_coordinator.has_valid_model_plan else "tool_protocol_error"
    )
    return AgentLoopOutcome(
        exit=AgentLoopExit.SUMMARY_REQUIRED,
        summary_finish_reason=summary_finish_reason,
    )


def _requires_plan_before_stop(request: AgentRoundOutcomeRequest) -> bool:
    return request.runtime.plan_mode == "on" and not request.state.plan_coordinator.has_valid_model_plan


def _requires_execution_before_stop(request: AgentRoundOutcomeRequest) -> bool:
    coordinator = request.state.plan_coordinator
    return (
        coordinator.has_valid_model_plan
        and not coordinator.synthesis_started
        and not coordinator.execution_items_terminal()
    )


def _requires_plan_synthesis(request: AgentRoundOutcomeRequest) -> bool:
    coordinator = request.state.plan_coordinator
    return (
        coordinator.has_valid_model_plan
        and not coordinator.synthesis_started
        and request.state.ready_for_plan_synthesis()
    )


async def _complete_round_before_plan_synthesis(request: AgentRoundOutcomeRequest) -> None:
    """计划执行完成后的普通回合只负责收 step，正文统一交给显式综合阶段。"""

    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.state.clear_current_step()


async def _repair_incomplete_execution(request: AgentRoundOutcomeRequest) -> None:
    """计划执行未终态时丢弃正文，下一轮只允许继续执行既定工具步骤。"""

    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    pending_items = request.state.plan_coordinator.pending_execution_items()
    pending_summary = [
        {
            "id": item.get("id"),
            "planned_tools": list(item.get("planned_tools") or []),
        }
        for item in pending_items
    ]
    request.messages.append(
        {
            "role": "system",
            "content": f"{PLAN_EXECUTION_REQUIRED_RETRY_PROMPT}\n待执行步骤：{pending_summary}",
        }
    )
    request.state.clear_current_step()


def _requires_research_completion_repair(request: AgentRoundOutcomeRequest) -> bool:
    if request.runtime.task_mode != "deep_research" or not request.state.research_network_required:
        return False
    result = validate_research_completion(
        request.state.research_workset,
        request.round_result.content_buf,
    )
    return not result.is_valid


async def _repair_research_completion(
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome | None:
    result = validate_research_completion(
        request.state.research_workset,
        request.round_result.content_buf,
    )
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.messages.append(
        {
            "role": "system",
            "content": build_research_repair_prompt(result.reason, request.state.research_workset),
        }
    )
    request.state.clear_current_step()
    if request.state.record_research_repair():
        return AgentLoopOutcome(
            exit=AgentLoopExit.SUMMARY_REQUIRED,
            summary_finish_reason="research_evidence_repair_exhausted",
        )
    return None


async def _complete_plan_required_round(request: AgentRoundOutcomeRequest) -> None:
    """丢弃未经过强制计划门禁的正文，并给下一轮加入内部修正指令。"""

    _append_round_blocks(request)
    _persist_visible_plan_reasoning_checkpoint(request)
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    if not any(
        message.get("role") == "system" and PLAN_REQUIRED_RETRY_PROMPT in str(message.get("content", ""))
        for message in request.messages
    ):
        request.messages.append({"role": "system", "content": PLAN_REQUIRED_RETRY_PROMPT})
    request.state.clear_current_step()


def _remove_plan_required_retry_prompt(messages: list[dict]) -> None:
    """兜底计划生效后移除已经过期的强制建计划指令。"""

    messages[:] = [
        message
        for message in messages
        if not (message.get("role") == "system" and message.get("content") == PLAN_REQUIRED_RETRY_PROMPT)
    ]


def _persist_visible_plan_reasoning_checkpoint(request: AgentRoundOutcomeRequest) -> None:
    if (
        not request.round_result.output_deferred
        or not request.round_result.allow_deferred_reasoning_output
        or not request.round_result.reasoning_buf
    ):
        return
    request.state.content_blocks.append(
        ThinkingBlock(
            type="thinking",
            id=request.step_context.thinking_block_id,
            thinking=request.round_result.reasoning_buf,
        )
    )
    persistence_kwargs = (
        {"sequence": request.runtime.assistant_message_sequence}
        if request.runtime.assistant_message_sequence is not None
        else {}
    )
    request.runtime.persist_message_fn(
        request.db,
        request.runtime.assistant_message_id,
        request.runtime.conversation_id,
        request.runtime.model_id,
        request.state.content_blocks,
        partial=True,
        **persistence_kwargs,
    )


async def _repair_missing_required_plan(
    request: AgentRoundOutcomeRequest,
) -> AgentLoopOutcome | None:
    await _complete_plan_required_round(request)
    repair_result = request.state.plan_coordinator.record_repair_round_with_fallback()
    if repair_result.fallback is not None and repair_result.fallback.snapshot is not None:
        _remove_plan_required_retry_prompt(request.messages)
        await request.runtime.emitter.plan_snapshot(**repair_result.fallback.snapshot)
    if repair_result.exhausted:
        return AgentLoopOutcome(
            exit=AgentLoopExit.SUMMARY_REQUIRED,
            summary_finish_reason="plan_repair_exhausted",
        )
    return None


def _append_round_blocks(request: AgentRoundOutcomeRequest) -> None:
    if request.round_result.output_deferred:
        return
    append_round_content_blocks(
        request.state.content_blocks,
        request.round_result.reasoning_buf,
        request.round_result.content_buf,
        request.step_context.thinking_block_id,
        request.step_context.text_block_id,
    )


async def _complete_text_round(request: AgentRoundOutcomeRequest) -> None:
    request = await _commit_deferred_answer(request)
    _append_round_blocks(request)
    await _emit_final_answer_used_evidence(request)
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.state.clear_current_step()


async def _commit_deferred_answer(
    request: AgentRoundOutcomeRequest,
) -> AgentRoundOutcomeRequest:
    if request.runtime.evidence_policy == "knowledge_grounded_v1":
        return await _commit_deferred_knowledge_answer(request)

    clarification = build_tool_repair_clarification(request.state.pending_tool_repairs)
    if clarification:
        grounded_answer = build_grounded_product_answer(request.state.content_blocks)
        answer = "\n\n".join(part for part in (grounded_answer, clarification) if part)
        await _append_committed_answer(request, answer)
        return _with_replaced_answer(request, answer)

    if not request.round_result.output_deferred:
        return request

    if request.runtime.task_mode == "deep_research" or not _has_product_answer_context(request.state):
        answer = request.round_result.content_buf.strip()
        if answer:
            await _append_committed_answer(request, answer, model_output_visible=True)
        return _with_replaced_answer(request, answer)

    return await _commit_deferred_product_answer(request)


async def _commit_deferred_knowledge_answer(
    request: AgentRoundOutcomeRequest,
) -> AgentRoundOutcomeRequest:
    """知识库回答只有通过显式引用校验后才写入用户可见流。"""

    evidence_block = next(
        (block for block in reversed(request.state.content_blocks) if isinstance(block, KnowledgeEvidenceBlock)),
        None,
    )
    candidate = request.round_result.content_buf.strip()
    if evidence_block is not None and validate_grounded_answer(candidate, evidence_block):
        answer = candidate
        model_output_visible = True
    else:
        request.runtime.warning_fn(
            "知识库回答缺少有效引用，使用确定性兜底: "
            f"conv_id={request.runtime.conversation_id} run_id={request.runtime.run_id} "
            f"step={request.step_number}"
        )
        answer = KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT
        model_output_visible = False
    await _append_committed_answer(request, answer, model_output_visible=model_output_visible)
    return _with_replaced_answer(request, answer)


async def _commit_deferred_product_answer(
    request: AgentRoundOutcomeRequest,
) -> AgentRoundOutcomeRequest:
    candidate = neutralize_product_provider_mentions(
        request.round_result.content_buf.strip(),
        request.state.content_blocks,
    )
    validation = validate_product_answer(
        candidate,
        request.state.content_blocks,
        messages=request.messages,
    )
    if validation.is_valid:
        answer = candidate
        model_output_visible = True
    else:
        repaired_answer, repair_reason_code = repair_unsupported_product_answer(
            candidate,
            request.state.content_blocks,
            messages=request.messages,
        )
        if repaired_answer is not None:
            request.runtime.warning_fn(
                "产品结果模型回答含越界分句，已安全修整: "
                f"conv_id={request.runtime.conversation_id} run_id={request.runtime.run_id} "
                f"step={request.step_number} reason_code={validation.reason_code}"
            )
            answer = repaired_answer
            model_output_visible = True
        else:
            request.runtime.warning_fn(
                "产品结果模型回答校验未通过，使用确定性兜底: "
                f"conv_id={request.runtime.conversation_id} run_id={request.runtime.run_id} "
                f"step={request.step_number} reason_code={validation.reason_code} "
                f"repair_reason_code={repair_reason_code}"
            )
            answer = build_grounded_product_answer(request.state.content_blocks)
            if answer:
                completed_answer, _ = repair_unsupported_product_answer(
                    answer,
                    request.state.content_blocks,
                    messages=request.messages,
                )
                if completed_answer is not None:
                    answer = completed_answer
            if not answer and request.state.product_tool_attempted:
                answer = build_product_tool_failure_answer(request.messages)
            if not answer:
                answer = "已展示本次查询的结构化结果，请以卡片信息为准。"
            model_output_visible = False
    answer = neutralize_product_provider_mentions(answer, request.state.content_blocks)
    if answer:
        await _append_committed_answer(
            request,
            answer,
            model_output_visible=model_output_visible,
        )
    return _with_replaced_answer(request, answer)


def _has_product_answer_context(state: AgentLoopState) -> bool:
    return (
        has_product_result_blocks(state.content_blocks)
        or state.product_tool_attempted
        or bool(state.pending_tool_repairs)
    )


async def _append_committed_answer(
    request: AgentRoundOutcomeRequest,
    answer: str,
    *,
    model_output_visible: bool = False,
) -> None:
    snapshot = request.state.plan_coordinator.begin_synthesis()
    emit_snapshot = getattr(request.runtime.emitter, "plan_snapshot", None)
    if snapshot is not None and emit_snapshot is not None:
        await emit_snapshot(**snapshot)
    await append_chunk(
        request.runtime.conversation_id,
        "answering",
        answer,
        request.step_context.text_block_id,
        task_id=request.runtime.task_id,
        run_id=request.runtime.run_id,
        step_id=request.step_context.step_id,
    )
    lifecycle = request.round_result.llm_lifecycle
    if model_output_visible and lifecycle is not None:
        await lifecycle.publish_visible_output("content")


def _with_replaced_answer(
    request: AgentRoundOutcomeRequest,
    answer: str,
) -> AgentRoundOutcomeRequest:
    visible_reasoning = request.round_result.reasoning_buf
    if request.round_result.output_deferred and not request.round_result.allow_deferred_reasoning_output:
        visible_reasoning = ""
    return AgentRoundOutcomeRequest(
        db=request.db,
        messages=request.messages,
        state=request.state,
        runtime=request.runtime,
        step_number=request.step_number,
        step_context=request.step_context,
        round_result=AgentRoundResult(
            reasoning_buf=visible_reasoning,
            protocol_reasoning_buf=request.round_result.protocol_reasoning_buf,
            protocol_content_buf=request.round_result.protocol_content_buf,
            content_buf=answer,
            tool_calls=request.round_result.tool_calls,
            finish_reason=request.round_result.finish_reason,
            accumulated_usage=request.round_result.accumulated_usage,
            context=request.round_result.context,
            announced_tool_names=request.round_result.announced_tool_names,
            output_deferred=False,
            allow_deferred_reasoning_output=request.round_result.allow_deferred_reasoning_output,
            llm_lifecycle=request.round_result.llm_lifecycle,
        ),
    )


def _needs_empty_answer_summary(request: AgentRoundOutcomeRequest) -> bool:
    if request.round_result.output_deferred and _has_product_answer_context(request.state):
        return False
    return (
        request.state.total_tool_calls > 0
        and not request.round_result.content_buf
        and not request.round_result.reasoning_buf
        and not request.round_result.tool_calls
    )


async def _complete_empty_round_before_summary(request: AgentRoundOutcomeRequest) -> None:
    request.runtime.warning_fn(
        "工具结果后模型返回空终态，切换到无工具收尾总结: "
        f"conv_id={request.runtime.conversation_id} run_id={request.runtime.run_id} "
        f"step={request.step_number} model_id={request.runtime.model_id}"
    )
    await complete_text_response_step(
        context=request.step_context,
        emitter=request.runtime.emitter,
        session_cache=request.runtime.session_cache,
        complete_step_fn=request.runtime.complete_step_fn,
        completed_tool_calls=request.state.total_tool_calls,
        max_tool_calls=request.runtime.limits.max_tool_calls,
        clock=request.runtime.clock,
    )
    request.state.clear_current_step()


async def _handle_tool_calls_round(request: AgentRoundOutcomeRequest) -> AgentLoopOutcome | None:
    await _discard_streamed_tool_round_content(request)
    outcome = await request.runtime.handle_tool_calls_round_fn(
        request=build_tool_round_request(
            db=request.db,
            messages=request.messages,
            state=request.state,
            runtime=request.runtime,
            step_number=request.step_number,
            step_context=request.step_context,
            round_result=request.round_result,
        ),
    )
    if isinstance(outcome, ToolRoundOutcome):
        request.state.record_no_progress_search_results(outcome.no_progress_search_results)
        request.state.record_product_tool_attempt(
            outcome.tool_call_count > 0
            and any(
                str(tool_call.get("name", "")) in AMAP_PRODUCT_TOOL_NAMES | FLYAI_TRAVEL_TOOL_NAMES
                for tool_call in request.round_result.tool_calls
            )
        )
        if outcome.control_repair_exhausted:
            request.state.clear_current_step()
            return AgentLoopOutcome(
                exit=AgentLoopExit.SUMMARY_REQUIRED,
                summary_finish_reason="plan_repair_exhausted",
            )
    request.state.clear_current_step()
    if _requires_user_input(request.state):
        return AgentLoopOutcome(exit=AgentLoopExit.PRODUCT_RESULT_READY)
    if isinstance(outcome, ToolRoundOutcome) and outcome.product_result_count > 0:
        return None
    unavailable_only = (
        isinstance(outcome, ToolRoundOutcome)
        and outcome.tool_call_count == 0
        and outcome.unavailable_tool_call_count > 0
        and request.state.plan_coordinator.has_valid_model_plan
    )
    if (
        unavailable_only
        and request.state.plan_coordinator.execution_items_terminal()
        and _has_product_answer_context(request.state)
    ):
        return AgentLoopOutcome(exit=AgentLoopExit.PRODUCT_RESULT_READY)
    if unavailable_only and request.state.ready_for_plan_synthesis():
        request.runtime.warning_fn(
            "计划执行完成后模型返回未公告工具，切换到无工具综合: "
            f"conv_id={request.runtime.conversation_id}, run_id={request.runtime.run_id}, "
            f"step={request.step_number}"
        )
        return AgentLoopOutcome(
            exit=AgentLoopExit.SUMMARY_REQUIRED,
            summary_finish_reason="plan_synthesis",
        )
    should_summarize = request.state.should_summarize_no_progress_search()
    if should_summarize:
        request.runtime.warning_fn(
            "连续搜索未取得新进展，切换到无工具收尾总结: "
            f"conv_id={request.runtime.conversation_id} run_id={request.runtime.run_id} "
            f"step={request.step_number} finish_reason=no_progress_summary"
        )
        return AgentLoopOutcome(
            exit=AgentLoopExit.SUMMARY_REQUIRED,
            summary_finish_reason="no_progress_summary",
        )
    return None


def _requires_user_input(state: AgentLoopState) -> bool:
    return any(
        isinstance(repair, dict) and repair.get("requires_user_input") is True
        for repair in state.pending_tool_repairs.values()
    )


async def _discard_streamed_tool_round_content(request: AgentRoundOutcomeRequest) -> None:
    """工具决策前的正文只是过程性话术，工具调用成立后精确撤回。"""

    if request.round_result.output_deferred or not request.round_result.content_buf:
        return
    discard = getattr(request.runtime.emitter, "content_block_discarded", None)
    if discard is None:
        return
    await discard(block_id=request.step_context.text_block_id)


async def _complete_unknown_round(request: AgentRoundOutcomeRequest) -> None:
    request.state.mark_unknown_terminated()
    await _complete_text_round(request)


async def _emit_final_answer_used_evidence(request: AgentRoundOutcomeRequest) -> None:
    emit = getattr(request.runtime.emitter, "evidence_item_upserted", None)
    if emit is None:
        return
    try:
        evidence_items = build_used_final_answer_evidence(
            content_blocks=request.state.content_blocks,
            answer_text=request.round_result.content_buf,
            evidence_policy=request.runtime.evidence_policy,
            allowed_citation_indexes=request.state.research_workset.valid_citation_indexes,
        )
        for evidence in evidence_items:
            await emit(tool_call_id=None, evidence=evidence)
    except StreamWriteTerminalError:
        raise
    except Exception as exc:  # noqa: BLE001 — 非写入故障的 used 判定不能阻断主回答完成
        request.runtime.warning_fn(f"发送最终回答 used evidence 失败: {exc}")
