import unittest

from app.schemas.chat import Usage
from app.services.stream.agent_round import accumulate_usage, collect_agent_round_stream, run_agent_round
from app.services.stream.step_lifecycle import AgentStepContext


class AgentRoundUsageTests(unittest.TestCase):
    def test_accumulate_usage_adds_usage_data(self):
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        usage_data = Usage(input_tokens=5, output_tokens=7)

        result = accumulate_usage(accumulated_usage, usage_data)

        self.assertEqual(result, Usage(input_tokens=7, output_tokens=10))

    def test_accumulate_usage_keeps_original_usage_without_usage_data(self):
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)

        result = accumulate_usage(accumulated_usage, None)

        self.assertIs(result, accumulated_usage)


class AgentRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_agent_round_stream_calls_llm_then_streams_with_step_ids(self):
        messages = [{"role": "user", "content": "你好"}]
        step_context = AgentStepContext(
            step_id="step-collect",
            step_number=4,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        events = []

        async def llm_call_fn(litellm_model, litellm_kwargs, call_messages, **call_kwargs):
            events.append(("llm", litellm_model, litellm_kwargs, call_messages, call_kwargs))
            return "response"

        async def stream_round_fn(
            response,
            conversation_id,
            task_id,
            should_use_reasoning,
            thinking_block_id,
            text_block_id,
            *,
            run_id,
            step_id,
        ):
            events.append(
                (
                    "stream",
                    response,
                    conversation_id,
                    task_id,
                    should_use_reasoning,
                    thinking_block_id,
                    text_block_id,
                    run_id,
                    step_id,
                )
            )
            return "推理", "正文", [{"id": "tool-1"}], "tool_calls", Usage(input_tokens=5, output_tokens=7)

        result = await collect_agent_round_stream(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"metadata": {"trace": "x"}},
            messages=messages,
            should_use_reasoning=True,
            call_kwargs={"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
            step_context=step_context,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
        )

        self.assertEqual(
            result, ("推理", "正文", [{"id": "tool-1"}], "tool_calls", Usage(input_tokens=5, output_tokens=7))
        )
        self.assertEqual([event[0] for event in events], ["llm", "stream"])
        self.assertEqual(
            events[0],
            (
                "llm",
                "openai/gpt-4",
                {"metadata": {"trace": "x"}},
                messages,
                {"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
            ),
        )
        self.assertEqual(
            events[1],
            (
                "stream",
                "response",
                "conv-1",
                "task-1",
                True,
                "blk-thinking",
                "blk-text",
                "run-1",
                "step-collect",
            ),
        )

    async def test_run_agent_round_records_success_and_accumulates_usage(self):
        messages = [{"role": "user", "content": "你好"}]
        step_context = AgentStepContext(
            step_id="step-1",
            step_number=3,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        events = []

        async def llm_call_fn(litellm_model, litellm_kwargs, call_messages, **call_kwargs):
            events.append(("llm", litellm_model, litellm_kwargs, call_messages, call_kwargs))
            return "response"

        async def stream_round_fn(
            response,
            conversation_id,
            task_id,
            should_use_reasoning,
            thinking_block_id,
            text_block_id,
            *,
            run_id,
            step_id,
        ):
            events.append(
                (
                    "stream",
                    response,
                    conversation_id,
                    task_id,
                    should_use_reasoning,
                    thinking_block_id,
                    text_block_id,
                    run_id,
                    step_id,
                )
            )
            return "推理", "正文", [{"id": "tool-1"}], "tool_calls", Usage(input_tokens=5, output_tokens=7)

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        result = await run_agent_round(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=3,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"metadata": {"trace": "x"}},
            messages=messages,
            should_use_reasoning=True,
            call_kwargs={"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
            step_context=step_context,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=log_round_summary_fn,
        )

        self.assertEqual(result.reasoning_buf, "推理")
        self.assertEqual(result.content_buf, "正文")
        self.assertEqual(result.tool_calls, [{"id": "tool-1"}])
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.accumulated_usage, Usage(input_tokens=7, output_tokens=10))
        self.assertEqual([event[0] for event in events], ["llm", "stream", "log"])
        self.assertEqual(
            events[0],
            (
                "llm",
                "openai/gpt-4",
                {"metadata": {"trace": "x"}},
                messages,
                {"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
            ),
        )
        self.assertEqual(
            events[1],
            (
                "stream",
                "response",
                "conv-1",
                "task-1",
                True,
                "blk-thinking",
                "blk-text",
                "run-1",
                "step-1",
            ),
        )
        self.assertEqual(
            events[2],
            (
                "log",
                {
                    "conversation_id": "conv-1",
                    "run_id": "run-1",
                    "step_number": 3,
                    "model_id": "gpt-4",
                    "provider": "openai",
                    "finish_reason": "tool_calls",
                    "tool_calls_count": 1,
                    "reasoning_buf": "推理",
                    "content_buf": "正文",
                },
            ),
        )

    async def test_run_agent_round_keeps_accumulated_usage_without_usage_data(self):
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        step_context = AgentStepContext(
            step_id="step-1",
            step_number=3,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        async def llm_call_fn(*_args, **_kwargs):
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "正文", [], "stop", None

        def log_round_summary_fn(**_kwargs):
            return None

        result = await run_agent_round(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=3,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=[],
            should_use_reasoning=False,
            call_kwargs={},
            accumulated_usage=accumulated_usage,
            step_context=step_context,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=log_round_summary_fn,
        )

        self.assertIs(result.accumulated_usage, accumulated_usage)
