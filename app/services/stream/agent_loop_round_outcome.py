"""Agent loop round outcome 分发。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.final_answer_evidence import build_used_final_answer_evidence
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
    if _requires_deep_synthesis_protocol_summary(request):
        return await _complete_deep_synthesis_protocol_round(request)

    finish_reason = request.round_result.finish_reason
    if finish_reason == "stop":
        if _requires_plan_before_stop(request):
            return await _repair_missing_required_plan(request)
        if _requires_research_completion_repair(request):
            return await _repair_research_completion(request)
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
    if _requires_research_completion_repair(request):
        return await _repair_research_completion(request)

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
        and bool(request.round_result.tool_calls)
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


def _requires_plan_before_stop(request: AgentRoundOutcomeRequest) -> bool:
    return request.runtime.plan_mode == "on" and not request.state.plan_coordinator.has_valid_model_plan


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
    request = await _replace_deferred_product_answer(request)
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
    _sync_plan_items(request)
    request.state.clear_current_step()


async def _replace_deferred_product_answer(
    request: AgentRoundOutcomeRequest,
) -> AgentRoundOutcomeRequest:
    clarification = build_tool_repair_clarification(request.state.pending_tool_repairs)
    if clarification:
        grounded_answer = build_grounded_product_answer(request.state.content_blocks)
        answer = "\n\n".join(part for part in (grounded_answer, clarification) if part)
        await _append_committed_answer(request, answer)
        return _with_replaced_answer(request, answer)

    if not request.round_result.output_deferred:
        return request

    if request.runtime.task_mode == "deep_research":
        answer = request.round_result.content_buf.strip()
        if answer:
            await _append_committed_answer(request, answer)
        return _with_replaced_answer(request, answer)

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
    answer = neutralize_product_provider_mentions(answer, request.state.content_blocks)
    if answer:
        await _append_committed_answer(request, answer)
    return _with_replaced_answer(request, answer)


async def _append_committed_answer(
    request: AgentRoundOutcomeRequest,
    answer: str,
) -> None:
    snapshot = request.state.plan_coordinator.commit_answer_started()
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


def _with_replaced_answer(
    request: AgentRoundOutcomeRequest,
    answer: str,
) -> AgentRoundOutcomeRequest:
    return AgentRoundOutcomeRequest(
        db=request.db,
        messages=request.messages,
        state=request.state,
        runtime=request.runtime,
        step_number=request.step_number,
        step_context=request.step_context,
        round_result=AgentRoundResult(
            reasoning_buf="",
            protocol_reasoning_buf=request.round_result.protocol_reasoning_buf,
            content_buf=answer,
            tool_calls=request.round_result.tool_calls,
            finish_reason=request.round_result.finish_reason,
            accumulated_usage=request.round_result.accumulated_usage,
            context=request.round_result.context,
            announced_tool_names=request.round_result.announced_tool_names,
            output_deferred=False,
        ),
    )


def _needs_empty_answer_summary(request: AgentRoundOutcomeRequest) -> bool:
    if request.round_result.output_deferred:
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
    _sync_plan_items(request)
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
            _sync_plan_items(request)
            request.state.clear_current_step()
            return AgentLoopOutcome(
                exit=AgentLoopExit.SUMMARY_REQUIRED,
                summary_finish_reason="plan_repair_exhausted",
            )
    _sync_plan_items(request)
    request.state.clear_current_step()
    if _requires_user_input(request.state):
        return AgentLoopOutcome(exit=AgentLoopExit.PRODUCT_RESULT_READY)
    if isinstance(outcome, ToolRoundOutcome) and outcome.product_result_count > 0:
        return None
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


def _sync_plan_items(request: AgentRoundOutcomeRequest) -> None:
    if not request.step_context.plan_items:
        return
    request.state.plan_coordinator.adopt_observed_items(
        list(request.step_context.plan_items.values()),
        reason="legacy_observed",
    )
    if request.state.plan_coordinator.has_valid_model_plan:
        return
    request.state.plan_items = {str(item_id): dict(item) for item_id, item in request.step_context.plan_items.items()}
