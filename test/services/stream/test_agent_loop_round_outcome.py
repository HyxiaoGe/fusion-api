import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.chat import PlaceResult, PlaceResultsBlock, SearchBlock, SearchSourceSummary, SourceReference, Usage
from app.services.stream.agent_loop_outcome import AgentLoopExit
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_round_outcome import AgentRoundOutcomeRequest, handle_agent_round_outcome
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


class AgentLoopRoundOutcomeTests(unittest.IsolatedAsyncioTestCase):
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

        with patch("app.services.stream.agent_loop_round_outcome.append_chunk", append_chunk):
            outcome = await handle_agent_round_outcome(
                request=AgentRoundOutcomeRequest(
                    db="db",
                    messages=[{"role": "user", "content": "找一家不用排队的烤肉店"}],
                    state=state,
                    runtime=_runtime(complete_step_fn=complete_step_fn),
                    step_number=2,
                    step_context=step_context,
                    round_result=AgentRoundResult(
                        reasoning_buf="",
                        content_buf="方便停车，也不会排队。",
                        tool_calls=[],
                        finish_reason="stop",
                        accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                        output_deferred=True,
                    ),
                )
            )

        self.assertEqual(outcome.exit, AgentLoopExit.COMPLETED)
        emitted_answer = append_chunk.await_args.args[2]
        self.assertIn("高德返回 1 个", emitted_answer)
        self.assertIn("不包含实时排队或空位信息", emitted_answer)
        self.assertNotIn("停车", emitted_answer)
        self.assertNotIn("不会排队", emitted_answer)
        self.assertEqual(state.content_blocks[-1].text, emitted_answer)
        self.assertEqual([block.type for block in state.content_blocks], ["place_results", "text"])
        complete_step_fn.assert_awaited_once()

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
