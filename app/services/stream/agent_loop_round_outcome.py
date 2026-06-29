"""Agent loop round outcome 分发。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.stream.agent_loop_outcome import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_tool_round_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.round_completion import append_round_content_blocks, complete_text_response_step
from app.services.stream.step_lifecycle import AgentStepContext


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
        await _complete_text_round(request)
        return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

    if finish_reason == "cancelled":
        _append_round_blocks(request)
        return AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED, error_msg="被新请求取代")

    if finish_reason == "tool_calls" and request.round_result.tool_calls:
        await _handle_tool_calls_round(request)
        return None

    await _complete_unknown_round(request)
    return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)


def _append_round_blocks(request: AgentRoundOutcomeRequest) -> None:
    append_round_content_blocks(
        request.state.content_blocks,
        request.round_result.reasoning_buf,
        request.round_result.content_buf,
        request.step_context.thinking_block_id,
        request.step_context.text_block_id,
    )


async def _complete_text_round(request: AgentRoundOutcomeRequest) -> None:
    _append_round_blocks(request)
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


async def _handle_tool_calls_round(request: AgentRoundOutcomeRequest) -> None:
    await request.runtime.handle_tool_calls_round_fn(
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
    _sync_plan_items(request)
    request.state.clear_current_step()


async def _complete_unknown_round(request: AgentRoundOutcomeRequest) -> None:
    request.state.mark_unknown_terminated()
    await _complete_text_round(request)


def _sync_plan_items(request: AgentRoundOutcomeRequest) -> None:
    if not request.step_context.plan_items:
        return
    request.state.plan_items = {
        str(item_id): dict(item) for item_id, item in request.step_context.plan_items.items()
    }
