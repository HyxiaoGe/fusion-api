import unittest
from functools import partial

from app.schemas.chat import TextBlock, Usage
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_runtime import AgentLoopRuntime
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.agent_loop_step_requests import build_limit_summary_step_request, build_tool_round_request
from app.services.stream.agent_round import AgentRoundResult
from app.services.stream.step_lifecycle import AgentStepContext


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


def _runtime(**overrides):
    values = {
        "conversation_id": "conv-req",
        "task_id": "task-req",
        "run_id": "run-req",
        "user_id": "user-req",
        "model_id": "gpt-4",
        "provider": "openai",
        "litellm_model": "openai/gpt-4",
        "litellm_kwargs": {"metadata": {"trace": "x"}},
        "should_use_reasoning": True,
        "call_kwargs": {"tools": ["web_search"]},
        "assistant_message_id": "msg-req",
        "run_start": 100.0,
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
        "clock": lambda: 123.0,
    }
    values.update(overrides)
    return AgentLoopRuntime(**values)


class AgentLoopStepRequestTests(unittest.TestCase):
    def test_build_tool_round_request_copies_runtime_state_and_round_fields(self):
        state = AgentLoopState()
        content_block = TextBlock(type="text", id="text-existing", text="已有内容")
        state.content_blocks.append(content_block)
        messages = [{"role": "user", "content": "hi"}]
        step_context = AgentStepContext(
            step_id="step-tool",
            step_number=2,
            started_at=100.0,
            thinking_block_id="thinking-tool",
            text_block_id="text-tool",
        )
        round_result = AgentRoundResult(
            reasoning_buf="需要 联网搜索",
            protocol_reasoning_buf="需要 web_search",
            protocol_content_buf="<think>需要 web_search</think>",
            content_buf="",
            tool_calls=[{"id": "tc-1", "name": "web_search", "arguments": "{}"}],
            finish_reason="tool_calls",
            accumulated_usage=Usage(input_tokens=3, output_tokens=4),
            announced_tool_names=frozenset({"web_search"}),
            output_deferred=True,
            allow_deferred_reasoning_output=True,
        )
        runtime = _runtime()

        request = build_tool_round_request(
            db="db",
            messages=messages,
            state=state,
            runtime=runtime,
            step_number=2,
            step_context=step_context,
            round_result=round_result,
        )

        self.assertEqual(request.db, "db")
        self.assertEqual(request.assistant_message_id, "msg-req")
        self.assertEqual(request.conversation_id, "conv-req")
        self.assertEqual(request.user_id, "user-req")
        self.assertIs(request.content_blocks, state.content_blocks)
        self.assertIs(request.messages, messages)
        self.assertIs(request.call_kwargs, runtime.call_kwargs)
        self.assertIs(request.emitter, runtime.emitter)
        self.assertIs(request.session_cache, runtime.session_cache)
        self.assertIs(request.network_budget, runtime.network_budget)
        self.assertEqual(request.tool_calls, round_result.tool_calls)
        self.assertEqual(request.reasoning_buf, "需要 联网搜索")
        self.assertEqual(request.protocol_reasoning_buf, "需要 web_search")
        self.assertEqual(request.protocol_content_buf, "<think>需要 web_search</think>")
        self.assertEqual(request.announced_tool_names, frozenset({"web_search"}))
        self.assertEqual(request.task_id, "task-req")
        self.assertTrue(request.output_deferred)
        self.assertTrue(request.allow_deferred_reasoning_output)
        self.assertIs(request.agent_state, state)
        self.assertIs(request.on_tools_executed.__self__, state)
        self.assertIs(request.on_tools_executed.__func__, state.record_executed_tool_calls.__func__)

    def test_build_limit_summary_step_request_advances_step_and_wires_runtime_dependencies(self):
        state = AgentLoopState(
            accumulated_usage=Usage(input_tokens=5, output_tokens=7),
            context_wait_seconds=45.0,
        )
        state.step = 3
        state.content_blocks.append(TextBlock(type="text", id="text-existing", text="已有内容"))
        messages = [{"role": "user", "content": "hi"}]
        runtime = _runtime()

        request = build_limit_summary_step_request(state=state, runtime=runtime, messages=messages)

        self.assertEqual(request.step_number, 4)
        self.assertEqual(state.step, 4)
        self.assertEqual(request.conversation_id, "conv-req")
        self.assertEqual(request.task_id, "task-req")
        self.assertEqual(request.run_id, "run-req")
        self.assertEqual(request.messages, messages)
        self.assertIs(request.content_blocks, state.content_blocks)
        self.assertIs(request.call_kwargs, runtime.call_kwargs)
        self.assertEqual(request.accumulated_usage, Usage(input_tokens=5, output_tokens=7))
        self.assertEqual(request.total_timeout_s, 300)
        self.assertEqual(request.run_start, 145.0)
        self.assertIs(request.start_step_fn, runtime.start_step_fn)
        self.assertIs(request.complete_step_fn, runtime.complete_step_fn)
        self.assertIs(request.llm_call_fn, runtime.llm_call_fn)
        self.assertIsInstance(request.stream_round_fn, partial)
        self.assertIs(request.stream_round_fn.func, runtime.stream_round_fn)
        self.assertEqual(request.stream_round_fn.keywords["model_id"], "gpt-4")
        self.assertIs(request.log_round_summary_fn, runtime.log_round_summary_fn)
        self.assertIs(request.warning_fn, runtime.warning_fn)
        self.assertIs(request.clock, runtime.clock)
        self.assertEqual(request.task_mode, "standard")
        self.assertEqual(request.evidence_policy, "standard")
        self.assertTrue(request.defer_output)
        self.assertIs(request.research_workset, state.research_workset)
        self.assertIs(request.on_step_started.__self__, state)
        self.assertIs(request.on_step_started.__func__, state.mark_current_step.__func__)

    def test_limit_summary_binds_k3_and_non_k3_model_ids_to_stream_transport(self):
        for model_id in ("moonshot/kimi-k3", "moonshot/kimi-k2.7-code"):
            with self.subTest(model_id=model_id):
                runtime = _runtime(model_id=model_id, provider="moonshot")

                request = build_limit_summary_step_request(
                    state=AgentLoopState(),
                    runtime=runtime,
                    messages=[{"role": "user", "content": "总结"}],
                )

                self.assertIsInstance(request.stream_round_fn, partial)
                self.assertEqual(request.stream_round_fn.keywords["model_id"], model_id)

    def test_limit_summary_keeps_legacy_stream_callable_without_model_id_keyword(self):
        async def legacy_stream_round(response):
            return response

        runtime = _runtime(stream_round_fn=legacy_stream_round)

        request = build_limit_summary_step_request(
            state=AgentLoopState(),
            runtime=runtime,
            messages=[],
        )

        self.assertIs(request.stream_round_fn, legacy_stream_round)


if __name__ == "__main__":
    unittest.main()
