import unittest
from dataclasses import dataclass

from app.schemas.chat import TextBlock, Usage
from app.services.stream.agent_loop_driver import AgentLoopExit, run_agent_loop
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.limit_summary import LimitSummaryOutcome
from app.services.stream.step_lifecycle import AgentStepContext


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
            tool_round_calls.append(kwargs)
            kwargs["on_tools_executed"](len(kwargs["tool_calls"]))
            kwargs["messages"].append({"role": "tool", "tool_call_id": "tc-1", "content": "搜索结果"})

        async def complete_step_fn(**kwargs):
            completed_steps.append(kwargs["context"].step_id)
            return 25

        outcome = await run_agent_loop(
            db="db",
            messages=messages,
            state=state,
            runtime=_runtime(
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                run_round_fn=run_round_fn,
                handle_tool_calls_round_fn=handle_tool_calls_round_fn,
            ),
        )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        self.assertEqual(started_steps, [1, 2])
        self.assertEqual(len(tool_round_calls), 1)
        self.assertEqual(tool_round_calls[0]["db"], "db")
        self.assertEqual(tool_round_calls[0]["step_number"], 1)
        self.assertEqual(tool_round_calls[0]["tool_calls"], [tool_call])
        self.assertEqual(tool_round_calls[0]["reasoning_buf"], "需要搜索")
        self.assertIs(tool_round_calls[0]["messages"], messages)
        self.assertEqual(completed_steps, ["step-2"])
        self.assertEqual(state.total_tool_calls, 1)
        self.assertEqual(state.step, 2)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=5, output_tokens=8))
        self.assertEqual([block.type for block in state.content_blocks], ["text"])
        self.assertEqual(state.content_blocks[0].text, "最终回答")
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "tc-1", "content": "搜索结果"})

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
