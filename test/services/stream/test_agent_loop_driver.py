import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from app.schemas.chat import PlaceResult, PlaceResultsBlock, TextBlock, Usage
from app.services.stream.agent_loop_driver import AgentLoopExit, run_agent_loop
from app.services.stream.agent_loop_policy import AgentLoopLimits, map_run_terminal_state
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.limit_summary import LimitSummaryOutcome
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_round import ToolRoundOutcome


@dataclass
class DummyEmitter:
    limit_reasons: list[str]

    async def run_limit_reached(self, *, reason):
        self.limit_reasons.append(reason)


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


def _runtime(**overrides):
    values = {
        "conversation_id": "conv-driver",
        "task_id": "task-driver",
        "run_id": "run-driver",
        "user_id": "user-driver",
        "model_id": "gpt-4",
        "provider": "openai",
        "litellm_model": "openai/gpt-4",
        "litellm_kwargs": {},
        "should_use_reasoning": True,
        "call_kwargs": {},
        "assistant_message_id": "msg-driver",
        "run_start": 0.0,
        "limits": AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        "emitter": DummyEmitter(limit_reasons=[]),
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


class AgentLoopDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_product_result_allows_sequential_multi_intent_tools_before_grounded_completion(self):
        state = AgentLoopState()
        started_steps: list[int] = []
        llm_steps: list[int] = []
        defer_output_flags: list[bool] = []
        tool_calls = [
            {"id": "tc-cafe", "name": "local_place_search", "arguments": '{"query":"咖啡店"}'},
            {"id": "tc-pool", "name": "local_place_search", "arguments": '{"query":"桌球"}'},
        ]

        async def start_step_fn(**kwargs):
            started_steps.append(kwargs["step_number"])
            context = AgentStepContext(
                step_id=f"step-{kwargs['step_number']}",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id=f"thinking-{kwargs['step_number']}",
                text_block_id=f"text-{kwargs['step_number']}",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**kwargs):
            llm_steps.append(kwargs["step_number"])
            defer_output_flags.append(bool(kwargs.get("defer_output")))
            step_number = kwargs["step_number"]
            if step_number <= 2:
                return AgentRoundResult(
                    reasoning_buf="先查咖啡店，再查附近桌球" if step_number == 1 else "继续完成桌球意图",
                    content_buf="",
                    tool_calls=[tool_calls[step_number - 1]],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=step_number * 2, output_tokens=step_number * 3),
                    output_deferred=bool(kwargs.get("defer_output")),
                )
            return AgentRoundResult(
                reasoning_buf="模型可能生成未验证组合距离",
                content_buf="模型自由文本：两家店步行五分钟。",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=7, output_tokens=9),
                output_deferred=bool(kwargs.get("defer_output")),
            )

        async def handle_tool_calls_round_fn(**kwargs):
            request = kwargs["request"]
            request.on_tools_executed(1)
            tool_call = request.tool_calls[0]
            query = "咖啡店" if tool_call["id"] == "tc-cafe" else "桌球"
            name = "示例咖啡" if tool_call["id"] == "tc-cafe" else "示例桌球馆"
            request.content_blocks.append(
                PlaceResultsBlock(
                    type="place_results",
                    schema_version=1,
                    provider="amap",
                    query=query,
                    status="success",
                    result_count=1,
                    places=[PlaceResult(name=name)],
                )
            )
            return ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["local_place_search"],
                product_result_count=1,
            )

        append_chunk = AsyncMock()
        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await run_agent_loop(
                db=object(),
                messages=[{"role": "user", "content": "附近咖啡"}],
                state=state,
                runtime=_runtime(
                    start_step_fn=start_step_fn,
                    complete_step_fn=AsyncMock(),
                    run_round_fn=run_round_fn,
                    handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                ),
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(llm_steps, [1, 2, 3])
        self.assertEqual(started_steps, [1, 2, 3])
        self.assertEqual(defer_output_flags, [False, True, True])
        grounded_answer = append_chunk.await_args.args[2]
        self.assertIn("示例咖啡", grounded_answer)
        self.assertIn("示例桌球馆", grounded_answer)
        self.assertNotIn("步行五分钟", grounded_answer)
        self.assertEqual(state.total_tool_calls, 2)
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=7, output_tokens=9))
        self.assertEqual([block.type for block in state.content_blocks], ["place_results", "place_results", "text"])

    async def test_existing_product_result_defers_next_round_model_output(self):
        state = AgentLoopState(
            content_blocks=[
                PlaceResultsBlock(
                    type="place_results",
                    schema_version=1,
                    provider="amap",
                    query="咖啡",
                    status="success",
                    result_count=1,
                    places=[PlaceResult(name="示例咖啡")],
                )
            ]
        )

        async def start_step_fn(**kwargs):
            context = AgentStepContext(
                step_id="step-product-final",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id="thinking-product-final",
                text_block_id="text-product-final",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**kwargs):
            self.assertTrue(kwargs["defer_output"])
            return AgentRoundResult(
                reasoning_buf="",
                content_buf="最终回答",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=2, output_tokens=3),
            )

        outcome = await run_agent_loop(
            db=object(),
            messages=[{"role": "user", "content": "附近咖啡"}],
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                complete_step_fn=AsyncMock(),
                run_round_fn=run_round_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)

    async def test_product_result_limit_uses_grounded_completion_without_limit_summary(self):
        cases = (
            {
                "name": "max_steps",
                "step": 8,
                "tool_calls": 1,
                "limits": AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
                "clock": 1.0,
                "finish_reason": "tool_calls",
            },
            {
                "name": "max_tool_calls",
                "step": 1,
                "tool_calls": 20,
                "limits": AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
                "clock": 1.0,
                "finish_reason": "tool_calls",
            },
            {
                "name": "timeout",
                "step": 1,
                "tool_calls": 1,
                "limits": AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=30),
                "clock": 31.0,
                "finish_reason": "timeout",
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                state = AgentLoopState(
                    content_blocks=[
                        PlaceResultsBlock(
                            type="place_results",
                            schema_version=1,
                            provider="amap",
                            query="咖啡店",
                            status="success",
                            result_count=1,
                            places=[PlaceResult(name="真实咖啡店")],
                        )
                    ],
                    step=case["step"],
                    total_tool_calls=case["tool_calls"],
                )
                emitter = DummyEmitter(limit_reasons=[])
                started_steps: list[int] = []

                async def start_step_fn(**kwargs):
                    started_steps.append(kwargs["step_number"])
                    context = AgentStepContext(
                        step_id=f"step-{kwargs['step_number']}",
                        step_number=kwargs["step_number"],
                        started_at=kwargs["clock"](),
                        thinking_block_id=f"thinking-{kwargs['step_number']}",
                        text_block_id=f"text-{kwargs['step_number']}",
                    )
                    kwargs["on_step_started"](context.step_id)
                    return context

                async def unsafe_limit_summary(**kwargs):
                    request = kwargs["request"]
                    request.content_blocks.append(TextBlock(type="text", id="unsafe", text="停车肯定方便"))
                    request.on_step_started("unsafe-summary")
                    return LimitSummaryOutcome(accumulated_usage=Usage(input_tokens=9, output_tokens=9))

                summary = AsyncMock(side_effect=unsafe_limit_summary)
                append_chunk = AsyncMock()
                with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
                    outcome = await run_agent_loop(
                        db=object(),
                        messages=[{"role": "user", "content": "咖啡店和附近桌球"}],
                        state=state,
                        runtime=_runtime(
                            emitter=emitter,
                            limits=case["limits"],
                            clock=lambda value=case["clock"]: value,
                            start_step_fn=start_step_fn,
                            complete_step_fn=AsyncMock(),
                            run_limit_summary_step_fn=summary,
                        ),
                    )

                summary.assert_not_awaited()
                self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
                self.assertEqual(emitter.limit_reasons, [case["name"]])
                self.assertEqual(state.limit_reason, case["name"])
                self.assertEqual(state.finish_reason, case["finish_reason"])
                terminal_state = map_run_terminal_state(
                    unknown_terminated=state.unknown_terminated,
                    limit_reason=state.limit_reason,
                )
                self.assertEqual(terminal_state.run_finish_reason, "limit_reached")
                self.assertEqual(terminal_state.session_status, "limit_reached")
                grounded_answer = append_chunk.await_args.args[2]
                self.assertIn("真实咖啡店", grounded_answer)
                self.assertNotIn("停车肯定方便", grounded_answer)
                self.assertEqual([block.type for block in state.content_blocks], ["place_results", "text"])
                self.assertEqual(started_steps, [case["step"] + 1])

    async def test_two_no_progress_search_results_run_one_summary_without_limit_event(self):
        state = AgentLoopState()
        emitter = DummyEmitter(limit_reasons=[])
        started_steps: list[int] = []
        tool_round_calls = 0
        summary_calls = []

        async def start_step_fn(**kwargs):
            started_steps.append(kwargs["step_number"])
            context = AgentStepContext(
                step_id=f"step-{kwargs['step_number']}",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id=f"thinking-{kwargs['step_number']}",
                text_block_id=f"text-{kwargs['step_number']}",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**_kwargs):
            step_number = len(started_steps)
            return AgentRoundResult(
                reasoning_buf="继续搜索",
                content_buf="",
                tool_calls=[
                    {
                        "id": f"tc-{step_number}",
                        "name": "web_search",
                        "arguments": f'{{"query":"x-{step_number}"}}',
                    },
                ],
                finish_reason="tool_calls",
                accumulated_usage=Usage(input_tokens=2, output_tokens=3),
            )

        async def handle_tool_calls_round_fn(**kwargs):
            nonlocal tool_round_calls
            tool_round_calls += 1
            request = kwargs["request"]
            request.on_tools_executed(1)
            return ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["web_search"],
                no_progress_search_results=(True,),
            )

        async def run_limit_summary_step_fn(**kwargs):
            summary_calls.append(kwargs["request"])
            request = kwargs["request"]
            request.content_blocks.append(TextBlock(type="text", id="summary-text", text="根据已有结果总结"))
            request.on_step_started("summary-step")
            return LimitSummaryOutcome(accumulated_usage=Usage(input_tokens=5, output_tokens=8))

        outcome = await run_agent_loop(
            db="db",
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                emitter=emitter,
                start_step_fn=start_step_fn,
                run_round_fn=run_round_fn,
                handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                run_limit_summary_step_fn=run_limit_summary_step_fn,
            ),
        )

        terminal_state = map_run_terminal_state(
            unknown_terminated=state.unknown_terminated,
            limit_reason=state.limit_reason,
        )
        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started_steps, [1, 2])
        self.assertEqual(tool_round_calls, 2)
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(summary_calls[0].summary_finish_reason, "no_progress_summary")
        self.assertEqual(emitter.limit_reasons, [])
        self.assertIsNone(state.limit_reason)
        self.assertEqual(state.finish_reason, "no_progress_summary")
        self.assertEqual(terminal_state.session_status, "completed")
        self.assertEqual(terminal_state.run_finish_reason, "stop")

    async def test_stop_round_completes_text_step_and_returns_completed(self):
        state = AgentLoopState()
        started_steps: list[int] = []
        completed_steps: list[str] = []

        async def start_step_fn(**kwargs):
            started_steps.append(kwargs["step_number"])
            context = AgentStepContext(
                step_id="step-1",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id="thinking-1",
                text_block_id="text-1",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)
            return 25

        async def run_round_fn(**kwargs):
            self.assertEqual(kwargs["step_number"], 1)
            return AgentRoundResult(
                reasoning_buf="思考",
                content_buf="回答",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=3, output_tokens=5),
            )

        outcome = await run_agent_loop(
            db=object(),
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                run_round_fn=run_round_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started_steps, [1])
        self.assertEqual(completed_steps, ["step-1"])
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=3, output_tokens=5))
        self.assertEqual([block.type for block in state.content_blocks], ["thinking", "text"])
        self.assertEqual(state.content_blocks[0].thinking, "思考")
        self.assertEqual(state.content_blocks[1].text, "回答")

    async def test_immediate_limit_runs_summary_step_and_returns_completed(self):
        state = AgentLoopState()
        state.step = 8
        emitter = DummyEmitter(limit_reasons=[])
        summary_calls = []

        async def run_limit_summary_step_fn(**kwargs):
            summary_calls.append(kwargs)
            request = kwargs["request"]
            request.content_blocks.append(TextBlock(type="text", id="summary-text", text="总结回答"))
            request.on_step_started("summary-step")
            return LimitSummaryOutcome(accumulated_usage=Usage(input_tokens=11, output_tokens=13))

        outcome = await run_agent_loop(
            db=object(),
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                emitter=emitter,
                run_limit_summary_step_fn=run_limit_summary_step_fn,
                clock=lambda: 1.0,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(emitter.limit_reasons, ["max_steps"])
        self.assertEqual(state.limit_reason, "max_steps")
        self.assertEqual(state.finish_reason, "tool_calls")
        self.assertEqual(state.step, 9)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=11, output_tokens=13))
        self.assertEqual(state.content_blocks[-1].text, "总结回答")
        self.assertEqual(summary_calls[0]["request"].step_number, 9)
        self.assertEqual(summary_calls[0]["request"].messages[0], {"role": "user", "content": "hi"})

    async def test_immediate_timeout_marks_timeout_and_runs_summary_step(self):
        state = AgentLoopState()
        emitter = DummyEmitter(limit_reasons=[])
        summary_calls = []

        async def run_limit_summary_step_fn(**kwargs):
            summary_calls.append(kwargs)
            kwargs["request"].on_step_started("summary-timeout-step")
            return LimitSummaryOutcome(accumulated_usage=Usage(input_tokens=1, output_tokens=1))

        outcome = await run_agent_loop(
            db=object(),
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                emitter=emitter,
                limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=30),
                run_limit_summary_step_fn=run_limit_summary_step_fn,
                clock=lambda: 31.0,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(emitter.limit_reasons, ["timeout"])
        self.assertEqual(state.limit_reason, "timeout")
        self.assertEqual(state.finish_reason, "timeout")
        self.assertEqual(state.step, 1)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(summary_calls[0]["request"].step_number, 1)

    async def test_tool_calls_round_delegates_and_continues_to_stop(self):
        state = AgentLoopState()
        messages = [{"role": "user", "content": "hi"}]
        tool_call = {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'}
        started_steps: list[int] = []
        tool_round_calls = []
        completed_steps: list[str] = []
        call_kwargs = {}
        emitter = DummyEmitter(limit_reasons=[])
        session_cache = object()
        network_budget = object()

        def clock():
            return 1.0

        async def start_step_fn(**kwargs):
            step_number = kwargs["step_number"]
            started_steps.append(step_number)
            context = AgentStepContext(
                step_id=f"step-{step_number}",
                step_number=step_number,
                started_at=kwargs["clock"](),
                thinking_block_id=f"thinking-{step_number}",
                text_block_id=f"text-{step_number}",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**kwargs):
            if kwargs["step_number"] == 1:
                self.assertEqual(kwargs["accumulated_usage"], Usage(input_tokens=0, output_tokens=0))
                return AgentRoundResult(
                    reasoning_buf="需要搜索",
                    content_buf="",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                )
            self.assertEqual(kwargs["step_number"], 2)
            self.assertEqual(kwargs["accumulated_usage"], Usage(input_tokens=2, output_tokens=3))
            return AgentRoundResult(
                reasoning_buf="",
                content_buf="最终回答",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=5, output_tokens=8),
            )

        async def handle_tool_calls_round_fn(**kwargs):
            request = kwargs["request"]
            tool_round_calls.append(request)
            request.on_tools_executed(len(request.tool_calls))
            request.messages.append({"role": "tool", "tool_call_id": "tc-1", "content": "搜索结果"})

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)
            return 25

        runtime = _runtime(
            start_step_fn=start_step_fn,
            complete_step_fn=complete_step_fn,
            run_round_fn=run_round_fn,
            handle_tool_calls_round_fn=handle_tool_calls_round_fn,
            call_kwargs=call_kwargs,
            emitter=emitter,
            session_cache=session_cache,
            network_budget=network_budget,
            clock=clock,
        )
        outcome = await run_agent_loop(
            db="db",
            messages=messages,
            state=state,
            runtime=runtime,
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started_steps, [1, 2])
        self.assertEqual(len(tool_round_calls), 1)
        self.assertEqual(tool_round_calls[0].db, "db")
        self.assertEqual(tool_round_calls[0].step_number, 1)
        self.assertEqual(tool_round_calls[0].tool_calls, [tool_call])
        self.assertEqual(tool_round_calls[0].reasoning_buf, "需要搜索")
        self.assertIs(tool_round_calls[0].messages, messages)
        self.assertIs(tool_round_calls[0].content_blocks, state.content_blocks)
        self.assertIs(tool_round_calls[0].call_kwargs, call_kwargs)
        self.assertIs(tool_round_calls[0].emitter, emitter)
        self.assertIs(tool_round_calls[0].session_cache, session_cache)
        self.assertIs(tool_round_calls[0].network_budget, network_budget)
        self.assertIs(tool_round_calls[0].clock, clock)
        self.assertEqual(completed_steps, ["step-2"])
        self.assertEqual(state.total_tool_calls, 1)
        self.assertEqual(state.step, 2)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=5, output_tokens=8))
        self.assertEqual([block.type for block in state.content_blocks], ["text"])
        self.assertEqual(state.content_blocks[0].text, "最终回答")
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "tc-1", "content": "搜索结果"})

    async def test_empty_stop_after_tools_runs_summary_step(self):
        state = AgentLoopState()
        messages = [{"role": "user", "content": "hi"}]
        tool_call = {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'}
        started_steps: list[int] = []
        completed_steps: list[str] = []
        summary_calls = []

        async def start_step_fn(**kwargs):
            step_number = kwargs["step_number"]
            started_steps.append(step_number)
            context = AgentStepContext(
                step_id=f"step-{step_number}",
                step_number=step_number,
                started_at=kwargs["clock"](),
                thinking_block_id=f"thinking-{step_number}",
                text_block_id=f"text-{step_number}",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**kwargs):
            if kwargs["step_number"] == 1:
                return AgentRoundResult(
                    reasoning_buf="需要搜索",
                    content_buf="",
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                )
            return AgentRoundResult(
                reasoning_buf="",
                content_buf="",
                tool_calls=[],
                finish_reason="stop",
                accumulated_usage=Usage(input_tokens=5, output_tokens=8),
            )

        async def handle_tool_calls_round_fn(**kwargs):
            request = kwargs["request"]
            request.on_tools_executed(len(request.tool_calls))
            request.messages.append({"role": "tool", "tool_call_id": "tc-1", "content": "搜索结果"})

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)

        async def run_limit_summary_step_fn(**kwargs):
            summary_calls.append(kwargs["request"])
            request = kwargs["request"]
            request.content_blocks.append(TextBlock(type="text", id="summary-text", text="总结回答"))
            request.on_step_started("summary-step")
            return LimitSummaryOutcome(accumulated_usage=Usage(input_tokens=9, output_tokens=13))

        outcome = await run_agent_loop(
            db="db",
            messages=messages,
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                run_round_fn=run_round_fn,
                handle_tool_calls_round_fn=handle_tool_calls_round_fn,
                run_limit_summary_step_fn=run_limit_summary_step_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started_steps, [1, 2])
        self.assertEqual(completed_steps, ["step-2"])
        self.assertEqual(len(summary_calls), 1)
        self.assertEqual(summary_calls[0].step_number, 3)
        self.assertEqual(state.finish_reason, "empty_answer_summary")
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=9, output_tokens=13))
        self.assertEqual([block.type for block in state.content_blocks], ["text"])
        self.assertEqual(state.content_blocks[0].text, "总结回答")
        self.assertEqual(state.current_step_id, None)

    async def test_unknown_tool_calls_without_tool_list_marks_incomplete_and_completes_step(self):
        state = AgentLoopState()
        completed_steps: list[str] = []

        async def start_step_fn(**kwargs):
            context = AgentStepContext(
                step_id="step-unknown",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id="thinking-unknown",
                text_block_id="text-unknown",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**_kwargs):
            return AgentRoundResult(
                reasoning_buf="",
                content_buf="退化回答",
                tool_calls=[],
                finish_reason="tool_calls",
                accumulated_usage=Usage(input_tokens=7, output_tokens=9),
            )

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)
            return 25

        outcome = await run_agent_loop(
            db=object(),
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                run_round_fn=run_round_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(completed_steps, ["step-unknown"])
        self.assertEqual(state.finish_reason, "tool_calls")
        self.assertTrue(state.unknown_terminated)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=7, output_tokens=9))
        self.assertEqual([block.type for block in state.content_blocks], ["text"])
        self.assertEqual(state.content_blocks[0].text, "退化回答")

    async def test_cancelled_round_returns_superseded_without_terminal_side_effects(self):
        state = AgentLoopState()
        persist_calls = []

        async def start_step_fn(**kwargs):
            context = AgentStepContext(
                step_id="step-cancelled",
                step_number=kwargs["step_number"],
                started_at=kwargs["clock"](),
                thinking_block_id="thinking-cancelled",
                text_block_id="text-cancelled",
            )
            kwargs["on_step_started"](context.step_id)
            return context

        async def run_round_fn(**_kwargs):
            return AgentRoundResult(
                reasoning_buf="",
                content_buf="半截回答",
                tool_calls=[],
                finish_reason="cancelled",
                accumulated_usage=Usage(input_tokens=2, output_tokens=1),
            )

        def persist_message_fn(db, message_id, conversation_id, model_id, content_blocks, usage_data=None):
            persist_calls.append(
                {
                    "db": db,
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "model_id": model_id,
                    "block_types": [block.type for block in content_blocks],
                    "usage_data": usage_data,
                }
            )

        outcome = await run_agent_loop(
            db="db",
            messages=[{"role": "user", "content": "hi"}],
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                run_round_fn=run_round_fn,
                persist_message_fn=persist_message_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.SUPERSEDED)
        self.assertEqual(outcome.error_msg, "被新请求取代")
        self.assertEqual([block.type for block in state.content_blocks], ["text"])
        self.assertEqual(state.content_blocks[0].text, "半截回答")
        self.assertEqual(state.final_usage(), Usage(input_tokens=2, output_tokens=1))
        self.assertEqual(state.current_step_id, "step-cancelled")
        self.assertEqual(persist_calls, [])
        self.assertFalse(state.terminal_emitted)


if __name__ == "__main__":
    unittest.main()
