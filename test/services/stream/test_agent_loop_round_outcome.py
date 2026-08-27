import asyncio
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.schemas.chat import (
    PlaceResult,
    PlaceResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    SearchBlock,
    SearchSourceSummary,
    SourceReference,
    Usage,
    WeatherForecastDay,
    WeatherResultsBlock,
)
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.stream.agent_loop_outcome import AgentLoopExit
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_round_outcome import (
    PLAN_REQUIRED_RETRY_PROMPT,
    AgentRoundOutcomeRequest,
    handle_agent_round_outcome,
)
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundOutcome


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


def _runtime(**overrides):
    values = {
        "conversation_id": "conv-outcome",
        "task_id": "task-outcome",
        "run_id": "run-outcome",
        "user_id": "user-outcome",
        "model_id": "gpt-4",
        "provider": "openai",
        "litellm_model": "openai/gpt-4",
        "litellm_kwargs": {},
        "should_use_reasoning": True,
        "call_kwargs": {},
        "assistant_message_id": "msg-outcome",
        "run_start": 0.0,
        "limits": AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        "emitter": object(),
        "session_cache": object(),
        "network_budget": object(),
        "start_step_fn": _unused_async,
        "complete_step_fn": _unused_async,
        "run_round_fn": _unused_async,
        "handle_tool_calls_round_fn": _unused_async,
        "run_limit_summary_step_fn": _unused_async,
        "llm_call_fn": _unused_async,
        "stream_round_fn": _unused_async,
        "execute_tools_fn": _unused_async,
        "persist_message_fn": _unused_sync,
        "log_round_summary_fn": lambda **_kwargs: None,
        "warning_fn": lambda _message: None,
        "clock": lambda: 1.0,
    }
    values.update(overrides)
    return AgentLoopRuntime(**values)


def _step_context(step_id="step-outcome"):
    return AgentStepContext(
        step_id=step_id,
        step_number=1,
        started_at=1.0,
        thinking_block_id=f"{step_id}-thinking",
        text_block_id=f"{step_id}-text",
    )


def _synthesis_state(*, run_id: str, step_id: str, **state_kwargs) -> AgentLoopState:
    coordinator = PlanCoordinator(run_id=run_id, mode="on")
    update = coordinator.apply_model_update(
        {
            "reason": "先执行查询，再整理结论",
            "items": [
                {
                    "id": "query",
                    "title": "执行查询",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search"],
                },
                {
                    "id": "answer",
                    "title": "整理结论",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["query"],
                    "planned_tools": [],
                },
            ],
        }
    )
    if not update.accepted:
        raise AssertionError(f"综合阶段计划 fixture 无效: {update.reason}")
    coordinator.mark_tool_results({"query": "completed"})
    if coordinator.begin_synthesis() is None:
        raise AssertionError("综合阶段计划 fixture 未进入 synthesis")
    state = AgentLoopState(plan_coordinator=coordinator, **state_kwargs)
    state.mark_current_step(step_id)
    return state


class AgentLoopRoundOutcomeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _deferred_lifecycle_request(lifecycle) -> AgentRoundOutcomeRequest:
        return AgentRoundOutcomeRequest(
            db="db",
            messages=[{"role": "user", "content": "测试 deferred commit"}],
            state=AgentLoopState(),
            runtime=_runtime(),
            step_number=1,
            step_context=_step_context("step-deferred-terminal"),
            round_result=AgentRoundResult(
                reasoning_buf="",
                content_buf="候选答案",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=1, output_tokens=1),
                output_deferred=True,
                llm_lifecycle=lifecycle,
            ),
        )

    async def test_deferred_outcome_terminal_secondary_preserves_primary_base_exception(self):
        error_factories = (
            ("runtime", lambda role: RuntimeError(f"{role} secret")),
            ("cancel", lambda role: asyncio.CancelledError(f"{role} secret")),
        )

        for primary_name, make_primary in error_factories:
            for secondary_name, make_secondary in error_factories:
                with self.subTest(primary=primary_name, secondary=secondary_name):
                    primary = make_primary("primary")
                    secondary = make_secondary("secondary")
                    finish_success = AsyncMock(side_effect=secondary)
                    lifecycle = SimpleNamespace(finish_success=finish_success)
                    warning = Mock()
                    cancelling_before = asyncio.current_task().cancelling()

                    with (
                        patch(
                            "app.services.stream.agent_loop_round_outcome._handle_agent_round_outcome",
                            new=AsyncMock(side_effect=primary),
                        ),
                        patch(
                            "app.services.stream.agent_loop_round_outcome.logger",
                            new=SimpleNamespace(warning=warning),
                            create=True,
                        ),
                    ):
                        with self.assertRaises(type(primary)) as raised:
                            await handle_agent_round_outcome(
                                request=self._deferred_lifecycle_request(lifecycle),
                            )

                    self.assertIs(raised.exception, primary)
                    self.assertEqual(asyncio.current_task().cancelling(), cancelling_before)
                    finish_success.assert_awaited_once_with(output_visible=False)
                    warning.assert_called_once()
                    logged = repr(warning.call_args)
                    self.assertIn("error_code=deferred_terminal_failure", logged)
                    self.assertIn(type(secondary).__name__, logged)
                    self.assertNotIn(str(primary), logged)
                    self.assertNotIn(str(secondary), logged)

    async def test_deferred_outcome_terminal_failure_without_primary_remains_fail_closed(self):
        for secondary in (RuntimeError("secondary secret"), asyncio.CancelledError("secondary secret")):
            with self.subTest(secondary=type(secondary).__name__):
                finish_success = AsyncMock(side_effect=secondary)
                lifecycle = SimpleNamespace(finish_success=finish_success)
                warning = Mock()
                cancelling_before = asyncio.current_task().cancelling()

                with (
                    patch(
                        "app.services.stream.agent_loop_round_outcome._handle_agent_round_outcome",
                        new=AsyncMock(return_value=None),
                    ),
                    patch(
                        "app.services.stream.agent_loop_round_outcome.logger",
                        new=SimpleNamespace(warning=warning),
                        create=True,
                    ),
                ):
                    with self.assertRaises(type(secondary)) as raised:
                        await handle_agent_round_outcome(
                            request=self._deferred_lifecycle_request(lifecycle),
                        )

                self.assertIs(raised.exception, secondary)
                self.assertEqual(asyncio.current_task().cancelling(), cancelling_before)
                finish_success.assert_awaited_once_with(output_visible=False)
                warning.assert_not_called()

    async def test_deep_synthesis_unannounced_tool_protocol_goes_directly_to_safe_summary(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-synthesis", mode="on"))
        state.configure_research_mode(network_required=True)
        state.research_workset.successful_searches = 1
        state.research_workset.successful_read_urls = {
            "https://example.com/a",
            "https://example.com/b",
        }
        state.plan_coordinator.source = "model"
        state.plan_coordinator.revision = 1
        state.plan_coordinator.items = [
            {
                "id": "answer",
                "title": "综合回答",
                "status": "running",
                "kind": "answer",
                "depends_on": [],
                "planned_tools": [],
            }
        ]
        state.mark_current_step("step-synthesis")
        complete_step = AsyncMock()
        warnings = []

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "深度调研"}],
                state=state,
                runtime=_runtime(
                    complete_step_fn=complete_step,
                    task_mode="deep_research",
                    evidence_policy="deep_research_v1",
                    plan_mode="on",
                    warning_fn=warnings.append,
                ),
                step_number=4,
                step_context=_step_context("step-synthesis"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[
                        {
                            "id": "dsml-step-synthesis-1",
                            "name": "update_plan",
                            "arguments": "{}",
                        }
                    ],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    announced_tool_names=frozenset(),
                    output_deferred=True,
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.SUMMARY_REQUIRED)
        self.assertEqual(outcome.summary_finish_reason, "research_evidence_repair_exhausted")
        self.assertEqual(state.total_tool_calls, 0)
        self.assertIsNone(state.current_step_id)
        complete_step.assert_awaited_once()
        self.assertTrue(any("未公告工具协议" in warning for warning in warnings))

    async def test_deep_research_unknown_terminal_cannot_bypass_evidence_gate(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-unknown", mode="on"))
        state.configure_research_mode(network_required=True)
        state.plan_coordinator.source = "model"
        state.plan_coordinator.revision = 1
        state.plan_coordinator.items = [{"id": "research", "status": "running"}]
        state.mark_current_step("step-unknown")

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "深度调研"}],
                state=state,
                runtime=_runtime(
                    complete_step_fn=AsyncMock(),
                    task_mode="deep_research",
                    evidence_policy="deep_research_v1",
                    plan_mode="on",
                ),
                step_number=1,
                step_context=_step_context("step-unknown"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="没有证据的终态。",
                    tool_calls=[],
                    finish_reason="length",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    output_deferred=True,
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertFalse(state.unknown_terminated)
        self.assertEqual(state.research_repair_attempts, 1)
        self.assertEqual(state.content_blocks, [])

    async def test_deep_research_defers_invalid_answer_and_exhausts_after_two_repairs(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-research", mode="on"))
        state.configure_research_mode(network_required=True)
        messages = [{"role": "user", "content": "深度调研"}]
        complete_step = AsyncMock()
        runtime = _runtime(
            complete_step_fn=complete_step,
            task_mode="deep_research",
            evidence_policy="deep_research_v1",
            plan_mode="on",
        )
        state.plan_coordinator.source = "model"
        state.plan_coordinator.revision = 1
        state.plan_coordinator.items = [{"id": "research", "status": "running"}]

        outcomes = []
        for index in range(3):
            state.mark_current_step(f"step-research-{index}")
            outcomes.append(
                await handle_agent_round_outcome(
                    request=AgentRoundOutcomeRequest(
                        db="db",
                        messages=messages,
                        state=state,
                        runtime=runtime,
                        step_number=index + 1,
                        step_context=_step_context(f"step-research-{index}"),
                        round_result=AgentRoundResult(
                            reasoning_buf="",
                            content_buf="没有真实来源的回答。",
                            tool_calls=[],
                            finish_reason="stop",
                            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                            output_deferred=True,
                        ),
                    )
                )
            )

        self.assertIsNone(outcomes[0])
        self.assertIsNone(outcomes[1])
        self.assertEqual(outcomes[2].exit, AgentLoopExit.SUMMARY_REQUIRED)
        self.assertEqual(outcomes[2].summary_finish_reason, "research_evidence_repair_exhausted")
        self.assertEqual(state.research_repair_attempts, 3)
        self.assertEqual(state.content_blocks, [])
        self.assertIn("至少完成一次有效搜索", messages[-1]["content"])

    async def test_deep_research_with_files_still_requires_network_evidence(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-file", mode="on"))
        state.configure_research_mode(network_required=True)
        state.plan_coordinator.source = "model"
        state.plan_coordinator.revision = 1
        state.plan_coordinator.items = [{"id": "file", "status": "running"}]
        state.mark_current_step("step-file")
        messages = [{"role": "user", "content": "总结附件"}]

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=messages,
                state=state,
                runtime=_runtime(
                    complete_step_fn=AsyncMock(),
                    task_mode="deep_research",
                    evidence_policy="deep_research_v1",
                    plan_mode="on",
                ),
                step_number=1,
                step_context=_step_context("step-file"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="附件的核心结论如下。",
                    tool_calls=[],
                    finish_reason="stop",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    output_deferred=True,
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertEqual(state.content_blocks, [])
        self.assertIn("至少完成一次有效搜索", messages[-1]["content"])

    async def test_on_mode_hidden_stop_without_plan_retries_then_uses_plan_repair_summary(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-plan", mode="on"))
        messages = [{"role": "user", "content": "制定一个计划"}]
        complete_step = AsyncMock(return_value=1)
        runtime = _runtime(complete_step_fn=complete_step, plan_mode="on")

        outcomes = []
        for index in range(3):
            state.mark_current_step(f"step-plan-{index}")
            outcomes.append(
                await handle_agent_round_outcome(
                    request=AgentRoundOutcomeRequest(
                        db="db",
                        messages=messages,
                        state=state,
                        runtime=runtime,
                        step_number=index + 1,
                        step_context=_step_context(f"step-plan-{index}"),
                        round_result=AgentRoundResult(
                            reasoning_buf="忽略计划",
                            content_buf="直接回答",
                            tool_calls=[],
                            finish_reason="stop",
                            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                            output_deferred=True,
                        ),
                    )
                )
            )

        self.assertIsNone(outcomes[0])
        self.assertIsNone(outcomes[1])
        self.assertEqual(outcomes[2].exit, AgentLoopExit.SUMMARY_REQUIRED)
        self.assertEqual(outcomes[2].summary_finish_reason, "plan_repair_exhausted")
        self.assertEqual(state.plan_coordinator.repair_attempt_count, 3)
        self.assertEqual(state.content_blocks, [])
        self.assertFalse(any(message.get("role") == "assistant" for message in messages))
        self.assertIn("必须先调用计划控制工具", messages[-1]["content"])
        self.assertEqual(complete_step.await_count, 3)

    async def test_on_mode_plan_gate_persists_streamed_reasoning_but_discards_answering(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-plan", mode="on"))
        messages = [{"role": "user", "content": "制定一个计划"}]
        persist_message = Mock()

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=messages,
                state=state,
                runtime=_runtime(
                    complete_step_fn=AsyncMock(return_value=1),
                    persist_message_fn=persist_message,
                    plan_mode="on",
                ),
                step_number=1,
                step_context=_step_context("step-plan-visible-reasoning"),
                round_result=AgentRoundResult(
                    reasoning_buf="先拆解任务，再建立计划。",
                    content_buf="这段正文不能提前展示。",
                    tool_calls=[],
                    finish_reason="stop",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    output_deferred=True,
                    allow_deferred_reasoning_output=True,
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertEqual(len(state.content_blocks), 1)
        self.assertEqual(state.content_blocks[0].type, "thinking")
        self.assertEqual(state.content_blocks[0].thinking, "先拆解任务，再建立计划。")
        self.assertFalse(any(block.type == "text" for block in state.content_blocks))
        self.assertIn("必须先调用计划控制工具", messages[-1]["content"])
        persist_message.assert_called_once()
        self.assertTrue(persist_message.call_args.kwargs["partial"])

    async def test_plan_gate_does_not_discard_answering_block_that_was_never_streamed(self):
        emitter = AsyncMock()
        handle_tool_calls = AsyncMock(
            return_value=ToolRoundOutcome(
                tool_call_count=0,
                tool_names=[],
                product_result_count=0,
            )
        )
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-plan", mode="on"))

        await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "制定计划"}],
                state=state,
                runtime=_runtime(
                    emitter=emitter,
                    handle_tool_calls_round_fn=handle_tool_calls,
                    plan_mode="on",
                ),
                step_number=1,
                step_context=_step_context("step-plan-tool"),
                round_result=AgentRoundResult(
                    reasoning_buf="正在创建计划。",
                    content_buf="过程性正文",
                    tool_calls=[
                        {
                            "id": "call-plan",
                            "name": "update_plan",
                            "arguments": '{"plan":[]}',
                        }
                    ],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    output_deferred=True,
                    allow_deferred_reasoning_output=True,
                ),
            )
        )

        emitter.content_block_discarded.assert_not_awaited()

    async def test_deep_research_hidden_stop_adopts_fallback_plan_after_retries(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-research", mode="on"))
        state.plan_coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 2,
            }
        )
        messages = [{"role": "user", "content": "调研数据库升级风险"}]
        complete_step = AsyncMock(return_value=1)
        emitter = AsyncMock()
        runtime = _runtime(
            complete_step_fn=complete_step,
            emitter=emitter,
            plan_mode="on",
            task_mode="deep_research",
            evidence_policy="deep_research_v1",
        )

        outcomes = []
        for index in range(3):
            state.mark_current_step(f"step-research-{index}")
            outcomes.append(
                await handle_agent_round_outcome(
                    request=AgentRoundOutcomeRequest(
                        db="db",
                        messages=messages,
                        state=state,
                        runtime=runtime,
                        step_number=index + 1,
                        step_context=_step_context(f"step-research-{index}"),
                        round_result=AgentRoundResult(
                            reasoning_buf="忽略计划",
                            content_buf="直接回答",
                            tool_calls=[],
                            finish_reason="stop",
                            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                            output_deferred=True,
                        ),
                    )
                )
            )

        self.assertEqual(outcomes, [None, None, None])
        self.assertTrue(state.plan_coordinator.has_valid_model_plan)
        self.assertEqual(state.plan_coordinator.reason, "system_fallback")
        self.assertEqual(state.plan_coordinator.repair_attempt_count, 0)
        self.assertEqual(
            [item["planned_tools"] for item in state.plan_coordinator.items],
            [["web_search"], ["url_read"], ["url_read"], []],
        )
        self.assertFalse(
            any(
                message.get("role") == "system" and message.get("content") == PLAN_REQUIRED_RETRY_PROMPT
                for message in messages
            )
        )
        emitter.plan_snapshot.assert_awaited_once()

    async def test_on_mode_repairs_any_non_cancelled_terminal_without_valid_plan(self):
        for finish_reason in ("tool_protocol_error", "length", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id=f"run-{finish_reason}", mode="on"))
                messages = [{"role": "user", "content": "制定计划"}]
                complete_step = AsyncMock(return_value=1)

                outcome = await handle_agent_round_outcome(
                    request=AgentRoundOutcomeRequest(
                        db="db",
                        messages=messages,
                        state=state,
                        runtime=_runtime(complete_step_fn=complete_step, plan_mode="on"),
                        step_number=1,
                        step_context=_step_context(f"step-{finish_reason}"),
                        round_result=AgentRoundResult(
                            reasoning_buf="协议异常",
                            content_buf="不应展示",
                            tool_calls=[],
                            finish_reason=finish_reason,
                            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                            output_deferred=True,
                        ),
                    )
                )

                self.assertIsNone(outcome)
                self.assertEqual(state.plan_coordinator.repair_attempt_count, 1)
                self.assertEqual(state.content_blocks, [])
                self.assertFalse(state.unknown_terminated)
                complete_step.assert_awaited_once()

    async def test_product_tool_round_returns_to_driver_for_next_model_decision(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-tool")
        state.consecutive_no_progress_search_results = 1
        tool_call = {"id": "tc-place", "name": "local_place_search", "arguments": '{"query":"咖啡"}'}

        async def handle_tool_calls_round_fn(**kwargs):
            request = kwargs["request"]
            request.on_tools_executed(1)
            return ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["local_place_search"],
                no_progress_search_results=(True,),
                product_result_count=1,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "咖啡店和附近桌球"}],
                state=state,
                runtime=_runtime(handle_tool_calls_round_fn=handle_tool_calls_round_fn),
                step_number=1,
                step_context=_step_context("step-product-tool"),
                round_result=AgentRoundResult(
                    reasoning_buf="先查咖啡店",
                    content_buf="",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertEqual(state.total_tool_calls, 1)
        self.assertEqual(state.consecutive_no_progress_search_results, 2)
        self.assertIsNone(state.current_step_id)

    async def test_failed_product_tool_attempt_is_recorded_for_next_round_guard(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-failed")

        async def handle_tool_calls_round_fn(**kwargs):
            kwargs["request"].on_tools_executed(1)
            return ToolRoundOutcome(
                tool_call_count=1,
                tool_names=[],
                product_result_count=0,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "比较通勤路线"}],
                state=state,
                runtime=_runtime(handle_tool_calls_round_fn=handle_tool_calls_round_fn),
                step_number=1,
                step_context=_step_context("step-product-failed"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[{"id": "tc-route", "name": "route_compare", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertTrue(state.product_tool_attempted)

    async def test_required_user_input_stops_before_another_model_round(self):
        state = AgentLoopState()
        state.mark_current_step("step-weather-ambiguous")

        async def handle_tool_calls_round_fn(**kwargs):
            request = kwargs["request"]
            request.on_tools_executed(1)
            request.agent_state.pending_tool_repairs["repair-weather"] = {
                "required_fields": ["location"],
                "retryable": False,
                "requires_user_input": True,
            }
            return ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["weather_forecast"],
                product_result_count=0,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "南山区明天天气如何？"}],
                state=state,
                runtime=_runtime(handle_tool_calls_round_fn=handle_tool_calls_round_fn),
                step_number=1,
                step_context=_step_context("step-weather-ambiguous"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[
                        {
                            "id": "tc-weather",
                            "name": "weather_forecast",
                            "arguments": '{"location":"南山区"}',
                        }
                    ],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.PRODUCT_RESULT_READY)
        self.assertEqual(state.total_tool_calls, 1)
        self.assertIsNone(state.current_step_id)

    async def test_failed_travel_tool_attempt_is_recorded_for_next_round_guard(self):
        state = AgentLoopState()
        state.mark_current_step("step-travel-failed")

        async def handle_tool_calls_round_fn(**kwargs):
            kwargs["request"].on_tools_executed(1)
            return ToolRoundOutcome(tool_call_count=1, tool_names=[], product_result_count=0)

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "查询航班"}],
                state=state,
                runtime=_runtime(handle_tool_calls_round_fn=handle_tool_calls_round_fn),
                step_number=1,
                step_context=_step_context("step-travel-failed"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[{"id": "tc-flight", "name": "search_flights", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertTrue(state.product_tool_attempted)

    async def test_failed_product_tool_deferred_answer_uses_safe_failure_message(self):
        state = AgentLoopState(product_tool_attempted=True)
        state.mark_current_step("step-product-failure-answer")
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "比较通勤路线"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-product-failure-answer"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="4号线直达，早高峰约30分钟。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=9),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("本次未取得可用", emitted_answer)
        self.assertNotIn("高德", emitted_answer)
        self.assertIn("稍后重试", emitted_answer)
        self.assertNotIn("4号线", emitted_answer)
        self.assertNotIn("30分钟", emitted_answer)

    async def test_pending_location_repair_deterministically_asks_for_complete_location(self):
        state = AgentLoopState(
            product_tool_attempted=True,
            pending_tool_repairs={
                "weather_forecast": {
                    "required_fields": ["location"],
                    "requires_user_input": True,
                }
            },
        )
        state.mark_current_step("step-weather-city-clarification")
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "南山区明天天气如何？"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-weather-city-clarification"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="深圳南山区明天多云。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=9),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("请补充包含城市的完整地点", emitted_answer)
        self.assertIn("不会猜测", emitted_answer)
        self.assertNotIn("深圳南山区明天多云", emitted_answer)
        self.assertNotIn("高德", emitted_answer)

    async def test_non_product_protocol_repair_overrides_model_answer_without_asking_for_missing_info(self):
        state = AgentLoopState(
            pending_tool_repairs={
                "repair_protocol": {
                    "required_fields": [],
                    "retryable": False,
                    "requires_user_input": False,
                    "retry_exhausted": True,
                }
            },
        )
        state.mark_current_step("step-protocol-repair")
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "帮我查一下资料"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-protocol-repair"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="工具已经成功返回了结果。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=9),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("请重试当前请求", emitted_answer)
        self.assertNotIn("补充更明确", emitted_answer)
        self.assertNotIn("工具已经成功", emitted_answer)

    async def test_location_context_timeout_uses_location_specific_safe_failure_message(self):
        state = AgentLoopState(product_tool_attempted=True)
        state.mark_current_step("step-location-timeout-answer")
        append_chunk = AsyncMock()

        messages = [
            {"role": "user", "content": "从我当前位置到深圳市民中心"},
            {
                "role": "tool",
                "tool_call_id": "tc-route",
                "content": (
                    '{"status":"unavailable","error_code":"context_required_not_provided",'
                    '"context_type":"geolocation","context_status":"timeout",'
                    '"reason":"geolocation_timeout"}'
                ),
            },
        ]
        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=messages,
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-location-timeout-answer"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="高德接口失败，请稍后重试。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=9),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("未能获取当前位置", emitted_answer)
        self.assertIn("浏览器或系统定位权限", emitted_answer)
        self.assertIn("依赖当前位置的查询尚未执行", emitted_answer)
        self.assertNotIn("路线查询尚未执行", emitted_answer)

    async def test_weather_location_context_denied_uses_neutral_safe_failure_message(self):
        state = AgentLoopState(product_tool_attempted=True)
        state.mark_current_step("step-weather-location-denied")
        append_chunk = AsyncMock()
        messages = [
            {"role": "user", "content": "我这里未来几天天气怎么样"},
            {
                "role": "tool",
                "tool_call_id": "tc-weather",
                "content": (
                    '{"status":"unavailable","error_code":"context_required_not_provided",'
                    '"context_type":"geolocation","context_status":"denied",'
                    '"reason":"permission_denied"}'
                ),
            },
        ]

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=messages,
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-weather-location-denied"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="当前温度30度。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=9),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("未能获取当前位置", emitted_answer)
        self.assertIn("依赖当前位置的查询尚未执行", emitted_answer)
        self.assertNotIn("当前温度30度", emitted_answer)
        self.assertNotIn("起点", emitted_answer)
        self.assertNotIn("高德", emitted_answer)

    async def test_empty_deferred_model_answer_still_completes_from_product_result(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-empty")
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="咖啡",
                status="success",
                result_count=1,
                places=[PlaceResult(name="示例咖啡")],
            )
        )
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "附近咖啡"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-product-empty"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=0),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertIn("示例咖啡", append_chunk.await_args.args[2])

    async def test_non_k3_deferred_answer_refresh_history_has_no_thinking_block(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-valid")
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="烤肉",
                near="深圳民治",
                status="success",
                result_count=1,
                places=[PlaceResult(name="炭火一号", rating=4.7)],
                limitations=["不包含实时排队或空位信息"],
            )
        )
        model_answer = (
            "结论：如果更看重本次返回的评分，可以优先看炭火一号。实时排队和空位本次无法确认，建议到店前核实。"
        )
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()
        warnings: list[str] = []
        llm_lifecycle = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "找一家烤肉店"}],
                    state=state,
                    runtime=_runtime(
                        complete_step_fn=complete_step_fn,
                        warning_fn=warnings.append,
                    ),
                    step_number=2,
                    step_context=_step_context("step-product-valid"),
                    round_result=AgentRoundResult(
                        reasoning_buf="只按实际字段总结",
                        content_buf=model_answer,
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=20),
                        output_deferred=True,
                        llm_lifecycle=llm_lifecycle,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(append_chunk.await_args.args[2], model_answer)
        self.assertEqual(state.content_blocks[-1].text, model_answer)
        self.assertEqual(
            [block.type for block in state.content_blocks],
            ["place_results", "text"],
        )
        self.assertEqual(warnings, [])
        complete_step_fn.assert_awaited_once()
        llm_lifecycle.publish_visible_output.assert_awaited_once_with("content")
        llm_lifecycle.finish_success.assert_awaited_once_with(output_visible=False)

    async def test_plan_mode_deferred_web_research_answer_bypasses_product_validation(self):
        coordinator = PlanCoordinator(run_id="run-web-research-answer", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先联网检索，再整理结论",
                    "items": [
                        {
                            "id": "search",
                            "title": "检索官方资料",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理研究结论",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search": "completed"})
        self.assertIsNotNone(coordinator.begin_synthesis())
        state = AgentLoopState(plan_coordinator=coordinator)
        state.mark_current_step("step-web-research-answer")
        state.content_blocks.append(
            SearchBlock(
                type="search",
                id="blk-web-research",
                query="Redis 版本更新",
                sources=[SearchSourceSummary(title="Redis 官方文档", url="https://redis.io/docs")],
                source_refs=[
                    SourceReference(kind="search", title="Redis 官方文档", url="https://redis.io/docs")
                ],
                source_count=1,
            )
        )
        model_answer = "Redis 的最新版本变更应以官方文档为准；本次检索到的依据见来源。[1]"
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()

        with (
            patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk),
            patch(
                "app.services.stream.agent_loop_round_outcome.validate_product_answer",
                side_effect=AssertionError("普通联网研究正文不应进入产品结果校验器"),
            ) as validate_product_answer,
        ):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "调研 Redis 版本更新"}],
                    state=state,
                    runtime=_runtime(
                        complete_step_fn=complete_step_fn,
                        plan_mode="on",
                        model_id="kimi-k3",
                        provider="moonshot",
                        litellm_model="moonshot/kimi-k3",
                    ),
                    step_number=3,
                    step_context=_step_context("step-web-research-answer"),
                    round_result=AgentRoundResult(
                        reasoning_buf="先核对官方来源，再整理结论。",
                        content_buf=model_answer,
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=5, output_tokens=18),
                        output_deferred=True,
                        allow_deferred_reasoning_output=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        validate_product_answer.assert_not_called()
        self.assertEqual(append_chunk.await_args.args[2], model_answer)
        self.assertEqual(state.content_blocks[-1].text, model_answer)
        self.assertEqual(
            [block.type for block in state.content_blocks],
            ["search", "thinking", "text"],
        )
        self.assertEqual(state.content_blocks[-2].thinking, "先核对官方来源，再整理结论。")
        complete_step_fn.assert_awaited_once()

    async def test_plan_synthesis_protocol_error_discards_safe_prefix_and_requests_summary(self):
        coordinator = PlanCoordinator(run_id="run-synthesis-protocol-error", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先联网检索，再整理结论",
                    "items": [
                        {
                            "id": "search",
                            "title": "检索官方资料",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理研究结论",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search": "completed"})
        self.assertIsNotNone(coordinator.begin_synthesis())
        state = AgentLoopState(plan_coordinator=coordinator)
        state.mark_current_step("step-synthesis-protocol-error")
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()

        with (
            patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk),
            patch(
                "app.services.stream.agent_loop_round_outcome.validate_product_answer",
            ) as validate_product_answer,
        ):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "调研 Redis 版本更新"}],
                    state=state,
                    runtime=_runtime(
                        complete_step_fn=complete_step_fn,
                        plan_mode="on",
                    ),
                    step_number=3,
                    step_context=_step_context("step-synthesis-protocol-error"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="我先整理一下检索结果：",
                        tool_calls=[],
                        finish_reason="tool_protocol_error",
                        accumulated_usage=Usage(input_tokens=5, output_tokens=5),
                        output_deferred=True,
                    ),
                )
            )

        complete_step_fn.assert_awaited_once()
        self.assertIsNone(state.current_step_id)
        self.assertEqual(outcome.exit, AgentLoopExit.SUMMARY_REQUIRED)
        self.assertEqual(outcome.summary_finish_reason, "plan_synthesis")
        validate_product_answer.assert_not_called()
        append_chunk.assert_not_awaited()
        self.assertEqual(state.content_blocks, [])
        self.assertFalse(state.unknown_terminated)

    async def test_unannounced_control_after_terminal_execution_goes_to_complete_plan_synthesis(self):
        coordinator = PlanCoordinator(run_id="run-unannounced-control", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先检索，再整理结论",
                    "items": [
                        {
                            "id": "search",
                            "title": "检索可靠资料",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理最终结论",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search": "completed"})
        state = AgentLoopState(plan_coordinator=coordinator)
        state.mark_current_step("step-unannounced-control")

        async def handle_tool_calls_round_fn(**_kwargs):
            return ToolRoundOutcome(
                tool_call_count=0,
                tool_names=[],
                unavailable_tool_call_count=1,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "整理已有检索结果"}],
                state=state,
                runtime=_runtime(
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                    plan_mode="on",
                ),
                step_number=3,
                step_context=_step_context("step-unannounced-control"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[
                        {
                            "id": "tc-stale-plan",
                            "name": "update_plan",
                            "arguments": {"step": "错误的单步状态更新"},
                        }
                    ],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    announced_tool_names=frozenset(),
                    output_deferred=True,
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.SUMMARY_REQUIRED)
        self.assertEqual(outcome.summary_finish_reason, "plan_synthesis")
        self.assertIsNone(state.current_step_id)
        self.assertFalse(state.unknown_terminated)

    async def test_unannounced_control_does_not_bypass_deep_research_evidence_gate(self):
        coordinator = PlanCoordinator(run_id="run-deep-unannounced-control", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "检索并核验来源后回答",
                    "items": [
                        {
                            "id": "search",
                            "title": "检索候选来源",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "read",
                            "title": "核验关键来源",
                            "status": "pending",
                            "kind": "read",
                            "depends_on": ["search"],
                            "planned_tools": ["url_read"],
                        },
                        {
                            "id": "answer",
                            "title": "整理研究结论",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["read"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search": "failed", "read": "blocked"})
        state = AgentLoopState(plan_coordinator=coordinator)
        state.configure_research_mode(network_required=True)
        state.mark_current_step("step-deep-unannounced-control")

        async def handle_tool_calls_round_fn(**_kwargs):
            return ToolRoundOutcome(
                tool_call_count=0,
                tool_names=[],
                unavailable_tool_call_count=1,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "深入调研监管要求"}],
                state=state,
                runtime=_runtime(
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                    plan_mode="on",
                    task_mode="deep_research",
                ),
                step_number=4,
                step_context=_step_context("step-deep-unannounced-control"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[{"id": "tc-stale-plan", "name": "update_plan", "arguments": {}}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    announced_tool_names=frozenset(),
                    output_deferred=True,
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertIsNone(state.current_step_id)
        self.assertFalse(state.ready_for_plan_synthesis())
        self.assertFalse(state.unknown_terminated)

    async def test_unannounced_control_with_existing_product_context_uses_product_result_path(self):
        coordinator = PlanCoordinator(run_id="run-product-unannounced-control", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "查询地点后回答",
                    "items": [
                        {
                            "id": "place",
                            "title": "查询地点",
                            "status": "running",
                            "kind": "other",
                            "depends_on": [],
                            "planned_tools": ["local_place_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理推荐",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["place"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"place": "completed"})
        state = AgentLoopState(
            plan_coordinator=coordinator,
            product_tool_attempted=True,
        )
        state.mark_current_step("step-product-unannounced-control")

        async def handle_tool_calls_round_fn(**_kwargs):
            return ToolRoundOutcome(
                tool_call_count=0,
                tool_names=[],
                unavailable_tool_call_count=1,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "推荐附近地点"}],
                state=state,
                runtime=_runtime(
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                    plan_mode="on",
                ),
                step_number=3,
                step_context=_step_context("step-product-unannounced-control"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[{"id": "tc-stale-plan", "name": "update_plan", "arguments": {}}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    announced_tool_names=frozenset(),
                    output_deferred=True,
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.PRODUCT_RESULT_READY)
        self.assertIsNone(state.current_step_id)
        self.assertFalse(state.unknown_terminated)

    async def test_unannounced_control_does_not_skip_pending_product_execution(self):
        coordinator = PlanCoordinator(run_id="run-pending-product-unannounced-control", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先查地点和天气，再整理推荐",
                    "items": [
                        {
                            "id": "place",
                            "title": "查询地点",
                            "status": "running",
                            "kind": "other",
                            "depends_on": [],
                            "planned_tools": ["local_place_search"],
                        },
                        {
                            "id": "weather",
                            "title": "查询天气",
                            "status": "pending",
                            "kind": "other",
                            "depends_on": ["place"],
                            "planned_tools": ["weather_forecast"],
                        },
                        {
                            "id": "answer",
                            "title": "整理推荐",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["weather"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"place": "completed"})
        state = AgentLoopState(
            plan_coordinator=coordinator,
            product_tool_attempted=True,
        )
        state.mark_current_step("step-pending-product-unannounced-control")

        async def handle_tool_calls_round_fn(**_kwargs):
            return ToolRoundOutcome(
                tool_call_count=0,
                tool_names=[],
                unavailable_tool_call_count=1,
            )

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "结合地点和天气给出推荐"}],
                state=state,
                runtime=_runtime(
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                    plan_mode="on",
                ),
                step_number=3,
                step_context=_step_context("step-pending-product-unannounced-control"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="",
                    tool_calls=[{"id": "tc-stale-plan", "name": "update_plan", "arguments": {}}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    announced_tool_names=frozenset({"weather_forecast"}),
                    output_deferred=True,
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertFalse(state.plan_coordinator.execution_items_terminal())
        self.assertEqual(
            state.plan_coordinator.active_plan_tool_names(),
            {"weather_forecast"},
        )
        self.assertIsNone(state.current_step_id)

    async def test_plan_synthesis_protocol_error_with_product_block_returns_product_result_ready(self):
        state = _synthesis_state(
            run_id="run-product-block-protocol-error",
            step_id="step-product-block-protocol-error",
        )
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="咖啡",
                status="success",
                result_count=1,
                places=[PlaceResult(name="示例咖啡")],
            )
        )
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "附近有什么咖啡店"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=complete_step_fn, plan_mode="on"),
                    step_number=3,
                    step_context=_step_context("step-product-block-protocol-error"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="我先整理一下查询结果：",
                        tool_calls=[],
                        finish_reason="tool_protocol_error",
                        accumulated_usage=Usage(input_tokens=5, output_tokens=5),
                        output_deferred=True,
                    ),
                )
            )

        complete_step_fn.assert_awaited_once()
        self.assertIsNone(state.current_step_id)
        self.assertEqual(outcome.exit, AgentLoopExit.PRODUCT_RESULT_READY)
        append_chunk.assert_not_awaited()
        self.assertEqual([block.type for block in state.content_blocks], ["place_results"])

    async def test_plan_synthesis_protocol_error_after_product_attempt_returns_product_result_ready(self):
        state = _synthesis_state(
            run_id="run-product-attempt-protocol-error",
            step_id="step-product-attempt-protocol-error",
            product_tool_attempted=True,
        )
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "比较通勤路线"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=complete_step_fn, plan_mode="on"),
                    step_number=3,
                    step_context=_step_context("step-product-attempt-protocol-error"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="我先整理一下查询结果：",
                        tool_calls=[],
                        finish_reason="tool_protocol_error",
                        accumulated_usage=Usage(input_tokens=5, output_tokens=5),
                        output_deferred=True,
                    ),
                )
            )

        complete_step_fn.assert_awaited_once()
        self.assertIsNone(state.current_step_id)
        self.assertEqual(outcome.exit, AgentLoopExit.PRODUCT_RESULT_READY)
        append_chunk.assert_not_awaited()
        self.assertEqual(state.content_blocks, [])

    async def test_plan_synthesis_protocol_error_with_pending_repair_returns_product_result_ready(self):
        state = _synthesis_state(
            run_id="run-pending-repair-protocol-error",
            step_id="step-pending-repair-protocol-error",
            pending_tool_repairs={
                "weather_forecast": {
                    "required_fields": ["location"],
                    "requires_user_input": True,
                }
            },
        )
        append_chunk = AsyncMock()
        complete_step_fn = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "我这里明天天气如何"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=complete_step_fn, plan_mode="on"),
                    step_number=3,
                    step_context=_step_context("step-pending-repair-protocol-error"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="我先整理一下查询结果：",
                        tool_calls=[],
                        finish_reason="tool_protocol_error",
                        accumulated_usage=Usage(input_tokens=5, output_tokens=5),
                        output_deferred=True,
                    ),
                )
            )

        complete_step_fn.assert_awaited_once()
        self.assertIsNone(state.current_step_id)
        self.assertEqual(outcome.exit, AgentLoopExit.PRODUCT_RESULT_READY)
        append_chunk.assert_not_awaited()
        self.assertEqual(state.content_blocks, [])

    async def test_deferred_answer_commits_plan_snapshot_before_answering_chunk(self):
        coordinator = PlanCoordinator(run_id="run-deferred-plan", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先查询再回答",
                    "items": [
                        {
                            "id": "search",
                            "title": "查询地点",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["local_place_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理建议",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search": "completed"})
        state = AgentLoopState(plan_coordinator=coordinator)
        state.mark_current_step("step-deferred-plan")
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="烤肉",
                status="success",
                result_count=1,
                places=[PlaceResult(name="炭火一号", rating=4.7)],
                limitations=["不包含实时排队或空位信息"],
            )
        )
        events: list[tuple[str, dict | None]] = []
        emitter = AsyncMock()

        async def record_snapshot(**snapshot):
            events.append(("plan_snapshot", snapshot))

        async def record_answer(*_args, **_kwargs):
            events.append(("answering", None))

        emitter.plan_snapshot.side_effect = record_snapshot
        with patch(
            "app.services.stream.agent_loop_round_outcome.append_chunk",
            side_effect=record_answer,
        ):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "找一家烤肉店"}],
                    state=state,
                    runtime=_runtime(
                        complete_step_fn=AsyncMock(),
                        emitter=emitter,
                    ),
                    step_number=2,
                    step_context=_step_context("step-deferred-plan"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="可以优先查看炭火一号；实时排队和空位本次无法确认。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=12),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual([event[0] for event in events], ["plan_snapshot", "answering"])
        self.assertEqual(
            {item["id"]: item["status"] for item in events[0][1]["items"]},
            {"search": "completed", "answer": "running"},
        )

    async def test_valid_deferred_product_answer_neutralizes_provider_attribution_but_keeps_place_name(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-provider-neutral")
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="商场",
                near="深圳市民中心",
                status="success",
                result_count=1,
                places=[PlaceResult(name="高德置地广场")],
            )
        )
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "附近有什么商场"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-product-provider-neutral"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="根据高德返回的结果，可以优先查看高德置地广场。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=12),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("根据本次查询返回的结果", emitted_answer)
        self.assertIn("高德置地广场", emitted_answer)
        self.assertNotIn("根据高德返回", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)

    async def test_cancelled_deferred_model_output_is_not_persisted(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-cancelled")

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "路线"}],
                state=state,
                runtime=_runtime(),
                step_number=2,
                step_context=_step_context("step-product-cancelled"),
                round_result=AgentRoundResult(
                    reasoning_buf="准备补充停车建议",
                    content_buf="停车方便",
                    tool_calls=[],
                    finish_reason="cancelled",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=1),
                    output_deferred=True,
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.SUPERSEDED)
        self.assertEqual(state.content_blocks, [])

    async def test_deferred_product_answer_replaces_model_prose_before_emitting_and_persisting(self):
        state = AgentLoopState()
        state.mark_current_step("step-product")
        state.content_blocks.append(
            PlaceResultsBlock(
                type="place_results",
                schema_version=1,
                provider="amap",
                query="烤肉",
                near="深圳民治",
                status="success",
                result_count=1,
                places=[PlaceResult(name="炭火一号")],
                limitations=["不包含实时排队或空位信息"],
            )
        )
        complete_step_fn = AsyncMock()
        append_chunk = AsyncMock()
        step_context = _step_context("step-product")
        warnings: list[str] = []
        llm_lifecycle = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "找一家不用排队的烤肉店"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=complete_step_fn, warning_fn=warnings.append),
                    step_number=2,
                    step_context=step_context,
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="方便停车，也不会排队。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                        output_deferred=True,
                        llm_lifecycle=llm_lifecycle,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("本次查询返回 1 个", emitted_answer)
        self.assertNotIn("高德", emitted_answer)
        self.assertIn("不包含实时排队或空位信息", emitted_answer)
        self.assertNotIn("停车", emitted_answer)
        self.assertNotIn("不会排队", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual([block.type for block in state.content_blocks], ["place_results", "text"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("reason_code=unsupported_claim", warnings[0])
        self.assertNotIn("停车", warnings[0])
        llm_lifecycle.publish_visible_output.assert_not_awaited()
        llm_lifecycle.finish_success.assert_awaited_once_with(output_visible=False)
        self.assertNotIn("排队", warnings[0])
        complete_step_fn.assert_awaited_once()

    async def test_grounded_fallback_also_completes_missing_place_relation_caveat(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-fallback-caveat")
        state.content_blocks.extend(
            [
                PlaceResultsBlock(
                    type="place_results",
                    schema_version=1,
                    provider="amap",
                    query="烤肉",
                    near="深圳民治",
                    status="success",
                    result_count=1,
                    places=[PlaceResult(name="炭火一号", rating=4.7)],
                ),
                PlaceResultsBlock(
                    type="place_results",
                    schema_version=1,
                    provider="amap",
                    query="桌球",
                    near="深圳民治",
                    status="success",
                    result_count=1,
                    places=[PlaceResult(name="金杆桌球", rating=4.1)],
                ),
            ]
        )
        append_chunk = AsyncMock()

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[
                        {
                            "role": "user",
                            "content": "想吃烤肉，吃完去桌球厅，不想走太远，请给组合建议",
                        }
                    ],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock()),
                    step_number=2,
                    step_context=_step_context("step-product-fallback-caveat"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="推荐未返回的火星烤肉店，吃完步行即达桌球厅。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=10),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("炭火一号", emitted_answer)
        self.assertIn("金杆桌球", emitted_answer)
        self.assertNotIn("火星烤肉店", emitted_answer)
        self.assertIn("地点之间的距离和步行时间", emitted_answer)
        self.assertIn("另行查询路线", emitted_answer)

    async def test_deferred_product_answer_repairs_unsafe_clause_and_keeps_safe_prose(self):
        state = AgentLoopState()
        state.mark_current_step("step-product-repair")
        state.content_blocks.append(
            RouteResultsBlock(
                type="route_results",
                schema_version=1,
                provider="amap",
                status="success",
                origin=RouteEndpoint(label="民治站"),
                destination=RouteEndpoint(label="雅宝站"),
                routes=[
                    RouteOption(mode="driving", duration_s=840, distance_m=6200),
                    RouteOption(mode="transit", duration_s=1920, walking_distance_m=420),
                ],
            )
        )
        model_answer = "结论：驾车约14分钟，是本次用时最短的方案。高峰期可能拥堵。地铁约32分钟，适合能接受换乘的情况。"
        append_chunk = AsyncMock()
        warnings: list[str] = []

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "比较通勤路线"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock(), warning_fn=warnings.append),
                    step_number=2,
                    step_context=_step_context("step-product-repair"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf=model_answer,
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=20),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("驾车约14分钟", emitted_answer)
        self.assertIn("地铁约32分钟", emitted_answer)
        self.assertNotIn("高峰期可能拥堵", emitted_answer)
        self.assertIn("本次查询结果无法确认", emitted_answer)
        self.assertNotIn("高德", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual(len(warnings), 1)
        self.assertIn("已安全修整", warnings[0])

    async def test_deferred_weather_activity_answer_is_always_deterministic(self):
        state = AgentLoopState()
        state.mark_current_step("step-weather-activity-repair")
        state.content_blocks.append(
            WeatherResultsBlock(
                type="weather_results",
                schema_version=1,
                provider="amap",
                status="degraded",
                query="南山区",
                resolved_location="南山区",
                day_count=1,
                forecast_days=[
                    WeatherForecastDay(
                        date=date(2026, 7, 24),
                        weekday=5,
                        day_weather="雷阵雨",
                        night_weather="多云",
                        high_c=31,
                        low_c=26,
                    )
                ],
                fetched_at=datetime(2026, 7, 23, 8, tzinfo=timezone.utc),
                limitations=["天气预报按行政区提供，不代表具体建筑物"],
            )
        )
        model_answer = (
            "7月24日（周五）南山区白天雷阵雨、夜间多云，26–31℃。"
            "如果你的条件是上午骑行时避开降雨，本次预报不满足这一条件。"
            "本次预报只有白天和夜间粒度，无法确认上午这一细分时段。"
            "从避雨角度看这一天的白天时段存在被淋雨的可能。"
            "夜间转为多云，如果你计划调整到晚上活动，天气条件相对更宽松一些。"
        )
        append_chunk = AsyncMock()
        warnings: list[str] = []

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "7月24日南山区天气怎么样，适合上午骑行吗？"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock(), warning_fn=warnings.append),
                    step_number=2,
                    step_context=_step_context("step-weather-activity-repair"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf=model_answer,
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=30),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("不满足这一条件", emitted_answer)
        self.assertIn("只有白天和夜间粒度，无法确认上午这一细分时段", emitted_answer)
        self.assertNotIn("存在被淋雨的可能", emitted_answer)
        self.assertNotIn("调整到晚上活动", emitted_answer)
        self.assertNotIn("天气条件相对更宽松", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual(len(warnings), 1)
        self.assertIn("天气活动条件使用确定性回答", warnings[0])

    async def test_deferred_mixed_travel_answer_is_always_deterministic(self):
        state = AgentLoopState()
        state.mark_current_step("step-mixed-travel-answer")
        state.content_blocks.extend(
            [
                {
                    "type": "flight_results",
                    "id": "flight-out",
                    "origin": "北京",
                    "destination": "上海",
                    "departure_date": "2026-08-29",
                    "flights": [
                        {
                            "id": "flight-1",
                            "flight_no": "MU5101",
                            "duration_s": 8100,
                            "price": {"currency": "CNY", "amount_minor": 76000},
                            "departure": {"station_name": "北京首都国际机场", "scheduled_at": "2026-08-29T07:00:00"},
                            "arrival": {"station_name": "上海浦东国际机场", "scheduled_at": "2026-08-29T09:15:00"},
                        }
                    ],
                    "limitations": ["班次与参考价格仅代表本次查询时刻"],
                },
                {
                    "type": "train_results",
                    "id": "train-out",
                    "origin": "北京",
                    "destination": "上海",
                    "departure_date": "2026-08-29",
                    "trains": [
                        {
                            "id": "train-1",
                            "train_no": "G1",
                            "duration_s": 17640,
                            "price": {"currency": "CNY", "amount_minor": 66100},
                            "departure": {"station_name": "北京南站", "scheduled_at": "2026-08-29T06:30:00"},
                            "arrival": {"station_name": "上海虹桥站", "scheduled_at": "2026-08-29T11:24:00"},
                        }
                    ],
                    "limitations": ["本次结果不包含余票或准点率"],
                },
            ]
        )
        model_answer = (
            "航班都优于高铁。若希望落地更接近市区，可选虹桥机场。"
            "G1 是兼顾早到与耗时的选择。"
        )
        append_chunk = AsyncMock()
        warnings: list[str] = []

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "北京到上海，高铁和飞机都查，比较最省钱和最快方案"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock(), warning_fn=warnings.append),
                    step_number=3,
                    step_context=_step_context("step-mixed-travel-answer"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf=model_answer,
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=30),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("同时返回北京到上海", emitted_answer)
        self.assertIn("MU5101", emitted_answer)
        self.assertIn("G1", emitted_answer)
        self.assertNotIn("都优于", emitted_answer)
        self.assertNotIn("更接近市区", emitted_answer)
        self.assertNotIn("兼顾早到", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual(len(warnings), 1)
        self.assertIn("混合出行比较使用确定性回答", warnings[0])

    async def test_deferred_single_train_comparison_is_always_deterministic(self):
        state = AgentLoopState()
        state.mark_current_step("step-single-train-comparison")
        state.content_blocks.append(
            {
                "type": "train_results",
                "id": "train-out",
                "origin": "北京",
                "destination": "上海",
                "departure_date": "2026-08-29",
                "trains": [
                    {
                        "train_no": "G737",
                        "duration_s": 21900,
                        "price": {"currency": "CNY", "amount_minor": 59800},
                    },
                    {
                        "train_no": "G37",
                        "duration_s": 16980,
                        "price": {"currency": "CNY", "amount_minor": 66100},
                    },
                ],
                "limitations": ["班次与参考价格仅代表本次查询时刻"],
            }
        )
        append_chunk = AsyncMock()
        warnings: list[str] = []

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[
                        {
                            "role": "user",
                            "content": "请查询北京到上海的高铁，告诉我本次返回中最便宜和最快的车次。",
                        }
                    ],
                    state=state,
                    runtime=_runtime(complete_step_fn=AsyncMock(), warning_fn=warnings.append),
                    step_number=2,
                    step_context=_step_context("step-single-train-comparison"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="G737 最省事，G37 兼顾价格和时间。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=10),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("参考价最低的是G737", emitted_answer)
        self.assertIn("计划行程时长最短的是G37", emitted_answer)
        self.assertNotIn("最省事", emitted_answer)
        self.assertNotIn("兼顾", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual(len(warnings), 1)
        self.assertIn("单一出行比较使用确定性回答", warnings[0])

    async def test_final_answer_evidence_does_not_swallow_stream_write_unavailable(self):
        from app.services.stream_state_service import StreamWriteUnavailableError

        state = AgentLoopState()
        state.mark_current_step("step-write-failed")
        state.content_blocks.append(
            SearchBlock(
                type="search",
                id="blk-search",
                query="Redis",
                sources=[SearchSourceSummary(title="官方文档", url="https://redis.io/docs")],
                source_refs=[SourceReference(kind="search", title="官方文档", url="https://redis.io/docs")],
                source_count=1,
            )
        )
        emitter = SimpleNamespace(
            evidence_item_upserted=AsyncMock(side_effect=StreamWriteUnavailableError("Redis write failed"))
        )

        with self.assertRaises(StreamWriteUnavailableError):
            await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "hi"}],
                    state=state,
                    runtime=_runtime(emitter=emitter, complete_step_fn=AsyncMock()),
                    step_number=1,
                    step_context=_step_context("step-write-failed"),
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="参考官方文档。[1]",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                    ),
                )
            )

    async def test_stop_round_appends_blocks_completes_step_and_returns_completed(self):
        state = AgentLoopState()
        state.mark_current_step("step-stop")
        completed_steps = []
        step_context = _step_context("step-stop")

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "hi"}],
                state=state,
                runtime=_runtime(complete_step_fn=complete_step_fn),
                step_number=1,
                step_context=step_context,
                round_result=AgentRoundResult(
                    reasoning_buf="思考",
                    content_buf="回答",
                    tool_calls=[],
                    finish_reason="stop",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(completed_steps, ["step-stop"])
        self.assertEqual(state.current_step_id, None)
        self.assertEqual([block.type for block in state.content_blocks], ["thinking", "text"])

    async def test_stop_round_marks_final_answer_used_evidence_before_completion(self):
        state = AgentLoopState()
        state.mark_current_step("step-used")
        step_context = _step_context("step-used")
        state.content_blocks.append(
            SearchBlock(
                type="search",
                id="blk-search",
                query="OpenAI 产品更新",
                sources=[
                    SearchSourceSummary(title="官方公告", url="https://openai.com/news/product"),
                    SearchSourceSummary(title="媒体报道", url="https://example.com/media"),
                ],
                source_refs=[
                    SourceReference(kind="search", title="官方公告", url="https://openai.com/news/product"),
                    SourceReference(kind="search", title="媒体报道", url="https://example.com/media"),
                ],
                source_count=2,
            )
        )
        emitter = SimpleNamespace(evidence_item_upserted=AsyncMock())
        calls = []

        async def complete_step_fn(**kwargs):
            calls.append(("complete", kwargs["context"].step_id))

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "hi"}],
                state=state,
                runtime=_runtime(emitter=emitter, complete_step_fn=complete_step_fn),
                step_number=1,
                step_context=step_context,
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="最终回答使用官方公告。[1]",
                    tool_calls=[],
                    finish_reason="stop",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(calls, [("complete", "step-used")])
        emitter.evidence_item_upserted.assert_awaited_once()
        event = emitter.evidence_item_upserted.await_args.kwargs
        self.assertIsNone(event["tool_call_id"])
        self.assertEqual(event["evidence"]["status"], "used")
        self.assertTrue(event["evidence"]["used_by_final_answer"])
        self.assertEqual(event["evidence"]["url"], "https://openai.com/news/product")

    async def test_cancelled_round_appends_partial_blocks_and_returns_superseded(self):
        state = AgentLoopState()
        state.mark_current_step("step-cancelled")
        step_context = _step_context("step-cancelled")

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "hi"}],
                state=state,
                runtime=_runtime(),
                step_number=1,
                step_context=step_context,
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="半截回答",
                    tool_calls=[],
                    finish_reason="cancelled",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.SUPERSEDED)
        self.assertEqual(outcome.error_msg, "被新请求取代")
        self.assertEqual(state.current_step_id, "step-cancelled")
        self.assertEqual([block.type for block in state.content_blocks], ["text"])

    async def test_tool_calls_round_delegates_and_requests_loop_continue(self):
        state = AgentLoopState()
        state.mark_current_step("step-tool")
        messages = [{"role": "user", "content": "hi"}]
        tool_requests = []
        step_context = _step_context("step-tool")

        async def handle_tool_calls_round_fn(**kwargs):
            tool_requests.append(kwargs["request"])
            kwargs["request"].on_tools_executed(len(kwargs["request"].tool_calls))

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=messages,
                state=state,
                runtime=_runtime(handle_tool_calls_round_fn=handle_tool_calls_round_fn),
                step_number=1,
                step_context=step_context,
                round_result=AgentRoundResult(
                    reasoning_buf="需要工具",
                    content_buf="",
                    tool_calls=[{"id": "tc-1", "name": "web_search", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertEqual(state.total_tool_calls, 1)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(tool_requests[0].db, "db")
        self.assertIs(tool_requests[0].messages, messages)

    async def test_tool_call_round_discards_streamed_preamble_before_execution(self):
        state = AgentLoopState()
        state.mark_current_step("step-tool-preamble")
        emitter = SimpleNamespace(content_block_discarded=AsyncMock())
        execution_started = False

        async def handle_tool_calls_round_fn(**kwargs):
            nonlocal execution_started
            self.assertEqual(emitter.content_block_discarded.await_count, 1)
            execution_started = True
            kwargs["request"].on_tools_executed(1)

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "比较通勤路线"}],
                state=state,
                runtime=_runtime(
                    emitter=emitter,
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                ),
                step_number=1,
                step_context=_step_context("step-tool-preamble"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="好的，我先调用路线工具。",
                    tool_calls=[{"id": "tc-route", "name": "route_compare", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=8),
                ),
            )
        )

        self.assertIsNone(outcome)
        self.assertTrue(execution_started)
        emitter.content_block_discarded.assert_awaited_once_with(block_id="step-tool-preamble-text")

    async def test_plan_mode_tool_round_uses_deferred_content_without_discard_fallback(self):
        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-deferred-tool", mode="on"))
        self.assertTrue(
            state.plan_coordinator.apply_model_update(
                {
                    "reason": "先查询再回答",
                    "items": [
                        {
                            "id": "route",
                            "title": "查询路线",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["route_compare"],
                        },
                        {
                            "id": "answer",
                            "title": "整理建议",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["route"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        state.mark_current_step("step-deferred-tool")
        emitter = SimpleNamespace(content_block_discarded=AsyncMock())

        async def handle_tool_calls_round_fn(**kwargs):
            kwargs["request"].on_tools_executed(1)

        await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "继续查询"}],
                state=state,
                runtime=_runtime(
                    emitter=emitter,
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                    plan_mode="on",
                ),
                step_number=2,
                step_context=_step_context("step-deferred-tool"),
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="继续查询。",
                    tool_calls=[{"id": "tc-route", "name": "route_compare", "arguments": "{}"}],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=4),
                    output_deferred=True,
                ),
            )
        )

        emitter.content_block_discarded.assert_not_awaited()

    async def test_unknown_round_marks_unknown_and_completes_text_step(self):
        state = AgentLoopState()
        state.mark_current_step("step-unknown")
        completed_steps = []
        step_context = _step_context("step-unknown")

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)

        outcome = await handle_agent_round_outcome(
            request=AgentRoundOutcomeRequest(
                db="db",
                messages=[{"role": "user", "content": "hi"}],
                state=state,
                runtime=_runtime(complete_step_fn=complete_step_fn),
                step_number=1,
                step_context=step_context,
                round_result=AgentRoundResult(
                    reasoning_buf="",
                    content_buf="退化回答",
                    tool_calls=[],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=1, output_tokens=2),
                ),
            )
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertTrue(state.unknown_terminated)
        self.assertEqual(completed_steps, ["step-unknown"])
        self.assertEqual(state.current_step_id, None)


if __name__ == "__main__":
    unittest.main()
