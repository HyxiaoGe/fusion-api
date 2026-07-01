import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.chat import SearchBlock, SearchSourceSummary, SourceReference, Usage
from app.services.stream.agent_loop_outcome import AgentLoopExit
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_round_outcome import AgentRoundOutcomeRequest, handle_agent_round_outcome
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.step_lifecycle import AgentStepContext


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


class AgentLoopRoundOutcomeTests(unittest.IsolatedAsyncioTestCase):
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
