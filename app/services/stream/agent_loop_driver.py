"""Agent loop 状态机 driver。"""

from __future__ import annotations

from copy import deepcopy
from inspect import Parameter, signature

from app.services.stream.agent_loop_outcome import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_policy import check_agent_loop_limit
from app.services.stream.agent_loop_round_outcome import AgentRoundOutcomeRequest, handle_agent_round_outcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_limit_summary_step_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.product_result_answer import has_product_result_blocks
from app.services.stream.reasoning_policy import configure_reasoning_call_kwargs
from app.services.stream.research_evidence import (
    build_deep_research_stage_prompt,
    build_research_untrusted_context_messages,
    build_research_workset_prompt,
    deep_research_stage_required_tool,
    deep_research_stage_tool_names,
    resolve_deep_research_stage,
)
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

        exhausted_plan_changed = await _reconcile_exhausted_dynamic_tool_owners(
            state=state,
            runtime=runtime,
        )
        if exhausted_plan_changed and state.plan_coordinator.execution_items_terminal():
            if (
                has_product_result_blocks(state.content_blocks)
                or state.product_tool_attempted
                or state.pending_tool_repairs
            ):
                await _complete_product_result_without_llm(
                    db=db,
                    messages=messages,
                    state=state,
                    runtime=runtime,
                )
                break
            if state.ready_for_plan_synthesis():
                state.finish_reason = "plan_synthesis"
                await _run_limit_summary(
                    state=state,
                    runtime=runtime,
                    messages=messages,
                    summary_finish_reason="plan_synthesis",
                )
                break
            state.finish_reason = "dynamic_tool_budget_exhausted"
            state.mark_unknown_terminated()
            await _run_limit_summary(
                state=state,
                runtime=runtime,
                messages=messages,
                summary_finish_reason="dynamic_tool_budget_exhausted",
            )
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
            if state.ready_for_plan_synthesis():
                state.finish_reason = "plan_synthesis"
                await _run_limit_summary(
                    state=state,
                    runtime=runtime,
                    messages=messages,
                    summary_finish_reason="plan_synthesis",
                )
                break
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
        if (
            has_product_result_blocks(state.content_blocks)
            or state.product_tool_attempted
            or state.pending_tool_repairs
        ):
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
    """产品结果或待修参数用结构化状态确定性收口，不再消耗模型调用。"""
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
        elapsed_seconds=state.active_elapsed_seconds(now=runtime.clock(), run_start=runtime.run_start),
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
    if runtime.task_mode != "deep_research" and runtime.plan_mode == "on":
        if not state.plan_coordinator.has_valid_model_plan:
            call_kwargs = _filter_tools_for_research_stage(
                call_kwargs,
                allowed_tool_names=frozenset({"update_plan"}),
            )
            call_kwargs = _require_tool_call(
                call_kwargs,
                preferred_tool_name="update_plan",
                provider=runtime.provider,
            )
        else:
            active_tool_names = state.plan_coordinator.active_plan_tool_names()
            call_kwargs = _filter_tools_for_research_stage(
                call_kwargs,
                allowed_tool_names=frozenset(active_tool_names),
            )
            for tool_name in active_tool_names:
                call_kwargs = _constrain_research_stage_plan_binding(
                    call_kwargs,
                    tool_name=tool_name,
                    active_plan_item_ids=state.plan_coordinator.active_plan_item_ids_for_tool(tool_name),
                )
            call_kwargs = _require_tool_call(call_kwargs, provider=runtime.provider)
    research_stage = None
    plan_repair_tool = None
    state.required_plan_repair_tool = None
    active_plan_item_ids: list[str] = []
    if runtime.task_mode == "deep_research":
        unexecuted_plan_tool_names = state.plan_coordinator.unexecuted_plan_tool_names()
        research_stage = resolve_deep_research_stage(
            state.research_workset,
            has_valid_plan=state.plan_coordinator.has_valid_model_plan,
            unexecuted_plan_tool_names=unexecuted_plan_tool_names,
        )
        required_tool = deep_research_stage_required_tool(research_stage)
        if required_tool:
            active_plan_item_ids = state.plan_coordinator.unexecuted_plan_item_ids_for_tool(required_tool)
            if not active_plan_item_ids:
                active_plan_item_ids = state.plan_coordinator.active_plan_item_ids_for_tool(required_tool)
            if not active_plan_item_ids and research_stage == "read" and state.research_workset.unread_candidate_urls:
                recovery_snapshot = state.plan_coordinator.add_server_recovery_item(required_tool)
                if recovery_snapshot is not None:
                    await runtime.emitter.plan_snapshot(**recovery_snapshot)
                    active_plan_item_ids = state.plan_coordinator.unexecuted_plan_item_ids_for_tool(required_tool)
            if not active_plan_item_ids:
                plan_repair_tool = required_tool
        state.required_plan_repair_tool = plan_repair_tool
        allowed_tool_names = deep_research_stage_tool_names(research_stage)
        if research_stage == "planning" or plan_repair_tool:
            allowed_tool_names = frozenset({"update_plan"})
        call_kwargs = _filter_tools_for_research_stage(
            call_kwargs,
            allowed_tool_names=allowed_tool_names,
        )
        if required_tool and active_plan_item_ids and not plan_repair_tool:
            call_kwargs = _constrain_research_stage_plan_binding(
                call_kwargs,
                tool_name=required_tool,
                active_plan_item_ids=active_plan_item_ids,
            )
        call_kwargs = _require_tool_call(
            call_kwargs,
            preferred_tool_name="update_plan" if research_stage == "planning" or plan_repair_tool else required_tool,
            provider=runtime.provider,
        )
    call_kwargs = configure_reasoning_call_kwargs(
        call_kwargs,
        provider=runtime.provider,
        should_use_reasoning=runtime.should_use_reasoning,
    )
    effective_messages = _messages_with_research_workset(
        messages,
        state=state,
        runtime=runtime,
        research_stage=research_stage,
        plan_repair_tool=plan_repair_tool,
        active_plan_item_ids=active_plan_item_ids,
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
        messages=effective_messages,
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
    must_defer_until_plan = runtime.plan_mode == "on" and not state.plan_coordinator.has_valid_model_plan
    should_defer_output = (
        has_product_result_blocks(state.content_blocks)
        or state.product_tool_attempted
        or state.pending_tool_repairs
        or must_defer_until_plan
        or runtime.plan_mode == "on"
        or state.plan_coordinator.has_valid_model_plan
        or runtime.task_mode == "deep_research"
    )
    if should_defer_output and _accepts_keyword(runtime.run_round_fn, "defer_output"):
        run_round_kwargs["defer_output"] = True
    round_result = await runtime.run_round_fn(**run_round_kwargs)
    state.finish_reason = round_result.finish_reason
    state.update_usage(round_result.accumulated_usage)
    state.update_context(round_result.context)
    return round_result


def _filter_tools_for_research_stage(
    call_kwargs: dict,
    *,
    allowed_tool_names: frozenset[str] | None,
) -> dict:
    if allowed_tool_names is None or not call_kwargs.get("tools"):
        return call_kwargs

    filtered_tools = []
    for tool in call_kwargs["tools"]:
        function = tool.get("function") if isinstance(tool, dict) else None
        tool_name = function.get("name") if isinstance(function, dict) else None
        if tool_name in allowed_tool_names:
            filtered_tools.append(tool)

    filtered_call_kwargs = dict(call_kwargs)
    if filtered_tools:
        filtered_call_kwargs["tools"] = filtered_tools
    else:
        filtered_call_kwargs.pop("tools", None)
        filtered_call_kwargs.pop("tool_choice", None)
    return filtered_call_kwargs


def _require_tool_call(
    call_kwargs: dict,
    *,
    preferred_tool_name: str | None = None,
    provider: str | None = None,
) -> dict:
    """计划与执行阶段由服务端锁定工具选择，禁止模型用正文绕过状态机。"""

    tools = call_kwargs.get("tools")
    if not isinstance(tools, list) or not tools:
        filtered_call_kwargs = dict(call_kwargs)
        filtered_call_kwargs.pop("tool_choice", None)
        return filtered_call_kwargs

    available_names = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            available_names.append(name)

    filtered_call_kwargs = dict(call_kwargs)
    if provider == "moonshot":
        filtered_call_kwargs["tool_choice"] = "required"
    elif preferred_tool_name in available_names:
        filtered_call_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": preferred_tool_name},
        }
    elif len(available_names) == 1:
        filtered_call_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": available_names[0]},
        }
    else:
        filtered_call_kwargs["tool_choice"] = "required"
    return filtered_call_kwargs


def _constrain_research_stage_plan_binding(
    call_kwargs: dict,
    *,
    tool_name: str,
    active_plan_item_ids: list[str],
) -> dict:
    """把阶段工具的计划项参数收窄为当前仍可执行的 ID。"""

    constrained = deepcopy(call_kwargs)
    for tool in constrained.get("tools", []):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            continue
        properties = parameters.setdefault("properties", {})
        if not isinstance(properties, dict):
            continue
        binding = dict(properties.get("_plan_item_id") or {})
        binding.update(
            {
                "type": "string",
                "enum": list(active_plan_item_ids),
                "description": "内部计划步骤 ID，只能选择当前仍可执行的计划项。",
            }
        )
        properties["_plan_item_id"] = binding
        required = parameters.setdefault("required", [])
        if isinstance(required, list) and "_plan_item_id" not in required:
            required.append("_plan_item_id")
    return constrained


async def _filter_exhausted_dynamic_tools(
    *,
    call_kwargs: dict,
    dynamic_tool_handlers: dict[str, object],
) -> dict:
    """在下一轮模型调用前隐藏已耗尽单服务预算的动态工具。"""

    if not dynamic_tool_handlers or not call_kwargs.get("tools"):
        return call_kwargs

    exhausted_aliases = await _exhausted_dynamic_tool_aliases(dynamic_tool_handlers)
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


async def _exhausted_dynamic_tool_aliases(
    dynamic_tool_handlers: dict[str, object] | None,
) -> set[str]:
    if not dynamic_tool_handlers:
        return set()
    exhausted_aliases: set[str] = set()
    for alias, handler in dynamic_tool_handlers.items():
        is_exhausted = getattr(handler, "is_run_budget_exhausted", None)
        if is_exhausted is not None and await is_exhausted():
            exhausted_aliases.add(alias)
    return exhausted_aliases


async def _reconcile_exhausted_dynamic_tool_owners(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
) -> bool:
    """模型调用前先把预算耗尽工具的计划 owner 收口，避免隐藏 schema 后无工具空转。"""

    exhausted_aliases = await _exhausted_dynamic_tool_aliases(runtime.dynamic_tool_handlers)
    if not exhausted_aliases:
        return False
    snapshot = state.plan_coordinator.block_tool_owners(
        exhausted_aliases,
        reason="dynamic_tool_budget_exhausted",
    )
    if snapshot is None:
        return False
    await runtime.emitter.plan_snapshot(**snapshot)
    return True


async def _run_limit_summary(
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    messages: list[dict],
    summary_finish_reason: str = "limit_summary",
) -> None:
    if not state.plan_coordinator.execution_items_terminal():
        blocked_snapshot = state.plan_coordinator.block_pending_execution(
            reason=f"{summary_finish_reason}_execution_blocked"
        )
        if blocked_snapshot is not None:
            await runtime.emitter.plan_snapshot(**blocked_snapshot)
    snapshot = state.plan_coordinator.begin_synthesis()
    if snapshot is not None:
        await runtime.emitter.plan_snapshot(**snapshot)
    summary_outcome = await runtime.run_limit_summary_step_fn(
        request=build_limit_summary_step_request(
            state=state,
            runtime=runtime,
            messages=deepcopy(
                _messages_with_research_workset(
                    messages,
                    state=state,
                    runtime=runtime,
                    include_candidates=runtime.task_mode != "deep_research",
                    terminal_summary=True,
                )
            ),
            summary_finish_reason=summary_finish_reason,
        ),
    )
    state.update_usage(summary_outcome.accumulated_usage)
    state.update_context(summary_outcome.context)
    if summary_outcome.incomplete and state.limit_reason is None:
        state.mark_unknown_terminated()
    if summary_finish_reason == "plan_repair_exhausted":
        state.mark_unknown_terminated()
    state.clear_current_step()


def _messages_with_research_workset(
    messages: list[dict],
    *,
    state: AgentLoopState,
    runtime: AgentLoopRuntime,
    include_candidates: bool = True,
    research_stage: str | None = None,
    plan_repair_tool: str | None = None,
    active_plan_item_ids: list[str] | None = None,
    terminal_summary: bool = False,
) -> list[dict]:
    if runtime.task_mode != "deep_research":
        if not terminal_summary:
            return messages
        untrusted_messages = build_research_untrusted_context_messages(
            state.research_workset,
            include_candidates=True,
        )
        if not untrusted_messages:
            return messages
        insert_at = 0
        while insert_at < len(messages) and messages[insert_at].get("role") == "system":
            insert_at += 1
        return [
            *messages[:insert_at],
            *untrusted_messages,
            *messages[insert_at:],
        ]
    stage_prompt = (
        build_deep_research_stage_prompt(
            research_stage,
            plan_repair_tool=plan_repair_tool,
            active_plan_item_ids=active_plan_item_ids,
        )
        if research_stage
        else ""
    )
    prompt = build_research_workset_prompt(
        state.research_workset,
        include_candidates=include_candidates,
    )
    untrusted_messages = build_research_untrusted_context_messages(
        state.research_workset,
        include_candidates=include_candidates,
    )
    if not stage_prompt and not prompt and not untrusted_messages:
        return messages
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    return [
        *messages[:insert_at],
        *([{"role": "system", "content": stage_prompt}] if stage_prompt else []),
        *([{"role": "system", "content": prompt}] if prompt else []),
        *untrusted_messages,
        *messages[insert_at:],
    ]
