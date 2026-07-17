"""Agent loop round outcome 分发。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.final_answer_evidence import build_used_final_answer_evidence
from app.services.stream.agent_loop_outcome import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_tool_round_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.product_result_answer import build_grounded_product_answer
from app.services.stream.round_completion import append_round_content_blocks, complete_text_response_step
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundOutcome
from app.services.stream_state_service import StreamWriteTerminalError, append_chunk


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
    finish_reason = request.round_result.finish_reason
    if finish_reason == "stop":
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

    await _complete_unknown_round(request)
    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


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
    if not request.round_result.output_deferred:
        return request
    answer = (
        build_grounded_product_answer(request.state.content_blocks)
        or "已展示地图服务返回的结构化结果，请以卡片信息为准。"
    )
    if answer:
        await append_chunk(
            request.runtime.conversation_id,
            "answering",
            answer,
            request.step_context.text_block_id,
            task_id=request.runtime.task_id,
            run_id=request.runtime.run_id,
            step_id=request.step_context.step_id,
        )
    return AgentRoundOutcomeRequest(
        db=request.db,
        messages=request.messages,
        state=request.state,
        runtime=request.runtime,
        step_number=request.step_number,
        step_context=request.step_context,
        round_result=AgentRoundResult(
            reasoning_buf="",
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
    _sync_plan_items(request)
    request.state.clear_current_step()
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
    request.state.plan_items = {str(item_id): dict(item) for item_id, item in request.step_context.plan_items.items()}
