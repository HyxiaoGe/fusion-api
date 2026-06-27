import asyncio
import unittest

from app.schemas.chat import Usage
from app.services.stream.limit_summary import (
    LIMIT_SUMMARY_PROMPT,
    build_limit_summary_call_kwargs,
    compute_summary_timeout,
    run_limit_summary_step,
)
from app.services.stream.step_lifecycle import AgentStepContext


class LimitSummaryHelpersTests(unittest.TestCase):
    def test_build_limit_summary_call_kwargs_copies_and_removes_tool_controls(self):
        tools = [{"function": {"name": "web_search"}}]
        call_kwargs = {
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "extra_body": {"thinking": {"type": "disabled"}},
        }

        result = build_limit_summary_call_kwargs(call_kwargs)

        self.assertIsNot(result, call_kwargs)
        self.assertEqual(
            result,
            {
                "temperature": 0.2,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        )
        self.assertEqual(call_kwargs["tools"], tools)
        self.assertEqual(call_kwargs["tool_choice"], "auto")

    def test_compute_summary_timeout_uses_remaining_budget(self):
        timeout = compute_summary_timeout(
            total_timeout_s=300,
            run_start=100.0,
            clock=lambda: 125.5,
        )

        self.assertEqual(timeout, 274.5)

    def test_compute_summary_timeout_has_10s_floor(self):
        timeout = compute_summary_timeout(
            total_timeout_s=300,
            run_start=100.0,
            clock=lambda: 450.0,
        )

        self.assertEqual(timeout, 10)


class LimitSummaryStepTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_limit_summary_step_appends_prompt_and_records_success(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        emitter = object()
        session_cache = object()
        events = []

        async def start_step_fn(*, emitter, session_cache, run_id, step_number, clock, on_step_started):
            events.append(("start", emitter, session_cache, run_id, step_number, clock))
            on_step_started("step-summary")
            return AgentStepContext(
                step_id="step-summary",
                step_number=step_number,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(litellm_model, litellm_kwargs, call_messages, **call_kwargs):
            events.append(("llm", litellm_model, litellm_kwargs, list(call_messages), call_kwargs))
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
            return "推理", "总结正文", [], "stop", Usage(input_tokens=5, output_tokens=7)

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        async def complete_step_fn(*, context, emitter, session_cache, tool_names, tool_call_count, clock):
            events.append(("complete", context, emitter, session_cache, tool_names, tool_call_count, clock))

        warnings = []
        marked_step_ids = []

        def clock():
            return 120.0

        outcome = await run_limit_summary_step(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=4,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"metadata": {"trace": "x"}},
            messages=messages,
            should_use_reasoning=True,
            content_blocks=content_blocks,
            call_kwargs={"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto", "temperature": 0.1},
            accumulated_usage=accumulated_usage,
            emitter=emitter,
            session_cache=session_cache,
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=complete_step_fn,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=log_round_summary_fn,
            warning_fn=warnings.append,
            clock=clock,
            on_step_started=marked_step_ids.append,
        )

        self.assertEqual(marked_step_ids, ["step-summary"])
        self.assertEqual(messages[-1], {"role": "system", "content": LIMIT_SUMMARY_PROMPT})
        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=7, output_tokens=10))
        self.assertEqual(len(content_blocks), 2)
        self.assertEqual(content_blocks[0].type, "thinking")
        self.assertEqual(content_blocks[0].id, "blk-thinking")
        self.assertEqual(content_blocks[0].thinking, "推理")
        self.assertEqual(content_blocks[1].type, "text")
        self.assertEqual(content_blocks[1].id, "blk-text")
        self.assertEqual(content_blocks[1].text, "总结正文")
        self.assertEqual(warnings, [])

        self.assertEqual([event[0] for event in events], ["start", "llm", "stream", "log", "complete"])
        self.assertEqual(events[1][4], {"temperature": 0.1})
        self.assertEqual(
            events[2],
            (
                "stream",
                "response",
                "conv-1",
                "task-1",
                True,
                "blk-thinking",
                "blk-text",
                "run-1",
                "step-summary",
            ),
        )
        self.assertEqual(events[3][1]["finish_reason"], "limit_summary")
        self.assertEqual(events[3][1]["tool_calls_count"], 0)
        self.assertEqual(events[3][1]["reasoning_buf"], "推理")
        self.assertEqual(events[3][1]["content_buf"], "总结正文")
        self.assertEqual(events[4][1].step_id, "step-summary")
        self.assertEqual(events[4][4], [])
        self.assertEqual(events[4][5], 0)

    async def test_run_limit_summary_step_swallows_timeout_and_completes_step(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        emitter = object()
        session_cache = object()
        events = []

        async def start_step_fn(*, emitter, session_cache, run_id, step_number, clock, on_step_started):
            on_step_started("step-timeout")
            events.append(("start", step_number))
            return AgentStepContext(
                step_id="step-timeout",
                step_number=step_number,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(*_args, **_kwargs):
            events.append(("llm",))
            raise asyncio.TimeoutError

        async def stream_round_fn(*_args, **_kwargs):
            events.append(("stream",))
            return "", "", [], "stop", None

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        async def complete_step_fn(*, context, emitter, session_cache, tool_names, tool_call_count, clock):
            events.append(("complete", context.step_id, tool_names, tool_call_count))

        warnings = []
        marked_step_ids = []

        outcome = await run_limit_summary_step(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=5,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=messages,
            should_use_reasoning=False,
            content_blocks=content_blocks,
            call_kwargs={"tools": [], "tool_choice": "auto"},
            accumulated_usage=accumulated_usage,
            emitter=emitter,
            session_cache=session_cache,
            total_timeout_s=2,
            run_start=0.0,
            start_step_fn=start_step_fn,
            complete_step_fn=complete_step_fn,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=log_round_summary_fn,
            warning_fn=warnings.append,
            clock=lambda: 10.0,
            on_step_started=marked_step_ids.append,
        )

        self.assertEqual(marked_step_ids, ["step-timeout"])
        self.assertEqual(messages[-1], {"role": "system", "content": LIMIT_SUMMARY_PROMPT})
        self.assertEqual(outcome.accumulated_usage, accumulated_usage)
        self.assertEqual(content_blocks, [])
        self.assertEqual([event[0] for event in events], ["start", "llm", "complete"])
        self.assertEqual(events[-1], ("complete", "step-timeout", [], 0))
        self.assertEqual(len(warnings), 1)
        self.assertIn("触顶总结超出剩余预算", warnings[0])
        self.assertIn("conv_id=conv-1", warnings[0])
