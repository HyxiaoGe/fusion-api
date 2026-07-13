import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import ContextUsage, Usage
from app.services.chat.context_manager import ContextBudgetExceededError, ContextPlan
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
    async def test_run_agent_round_emits_estimated_and_final_context_status(self):
        emitter = AsyncMock()
        step_context = AgentStepContext(
            step_id="step-context",
            step_number=2,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        context_plan = ContextPlan(
            messages=[{"role": "user", "content": "有效快照"}],
            status="trimmed",
            context_window_tokens=100_000,
            context_window_source="registry",
            context_window_status="known",
            estimated_tokens_before=90_000,
            estimated_tokens_after=70_000,
            removed_turns=1,
            removed_messages=2,
            removed_tool_transactions=0,
        )

        async def stream_round_fn(*_args, **_kwargs):
            return "", "正文", [], "stop", Usage(input_tokens=69_500, output_tokens=10)

        with patch(
            "app.services.stream.agent_round.prepare_context",
            new=AsyncMock(return_value=context_plan),
        ):
            result = await run_agent_round(
                conversation_id="conv-1",
                task_id="task-1",
                run_id="run-1",
                step_number=2,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=[{"role": "user", "content": "问题"}],
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(input_tokens=10, output_tokens=5),
                step_context=step_context,
                llm_call_fn=AsyncMock(return_value="response"),
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                emitter=emitter,
            )

        self.assertEqual(result.accumulated_usage, Usage(input_tokens=69_510, output_tokens=15))
        self.assertEqual(
            result.context,
            ContextUsage(
                status="trimmed",
                round_index=2,
                window_tokens=100_000,
                estimated_tokens_before=90_000,
                estimated_tokens_after=70_000,
                actual_prompt_tokens=69_500,
                removed_turns=1,
                removed_messages=2,
                removed_tool_transactions=0,
            ),
        )
        self.assertEqual(emitter.context_status_updated.await_count, 2)
        first = emitter.context_status_updated.await_args_list[0].kwargs
        final = emitter.context_status_updated.await_args_list[1].kwargs
        self.assertEqual(first["phase"], "estimated")
        self.assertIsNone(first["actual_prompt_tokens"])
        self.assertEqual(final["phase"], "final")
        self.assertEqual(final["actual_prompt_tokens"], 69_500)

    async def test_context_budget_error_emits_safe_error_status(self):
        emitter = AsyncMock()
        context_updates = []
        step_context = AgentStepContext(
            step_id="step-budget",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        plan = ContextPlan(
            messages=[{"role": "user", "content": "不可泄露的过长正文"}],
            status="required_context_over_budget",
            context_window_tokens=100,
            context_window_source="private-source",
            context_window_status="known",
            estimated_tokens_before=120,
            estimated_tokens_after=120,
        )

        with patch(
            "app.services.stream.agent_round.prepare_context",
            new=AsyncMock(side_effect=ContextBudgetExceededError(plan)),
        ):
            with self.assertRaises(ContextBudgetExceededError):
                await run_agent_round(
                    conversation_id="conv-1",
                    task_id="task-1",
                    run_id="run-1",
                    step_number=1,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=plan.messages,
                    should_use_reasoning=False,
                    call_kwargs={},
                    accumulated_usage=Usage(),
                    step_context=step_context,
                    llm_call_fn=AsyncMock(),
                    stream_round_fn=AsyncMock(),
                    log_round_summary_fn=lambda **_kwargs: None,
                    emitter=emitter,
                    on_context_updated=context_updates.append,
                )

        payload = emitter.context_status_updated.await_args.kwargs
        self.assertEqual(payload["phase"], "error")
        self.assertEqual(payload["status"], "required_context_over_budget")
        self.assertNotIn("messages", payload)
        self.assertNotIn("context_window_source", payload)
        self.assertEqual(context_updates[-1].status, "required_context_over_budget")

    async def test_llm_failure_keeps_estimated_context_for_failed_finalization(self):
        emitter = AsyncMock()
        context_updates = []
        plan = ContextPlan(
            messages=[{"role": "user", "content": "问题"}],
            status="no_op_fast_path",
            context_window_tokens=128_000,
            context_window_source="registry",
            context_window_status="known",
        )
        step_context = AgentStepContext(
            step_id="step-failure",
            step_number=3,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        with patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=plan)):
            with self.assertRaises(RuntimeError):
                await run_agent_round(
                    conversation_id="conv-1",
                    task_id="task-1",
                    run_id="run-1",
                    step_number=3,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=plan.messages,
                    should_use_reasoning=False,
                    call_kwargs={},
                    accumulated_usage=Usage(),
                    step_context=step_context,
                    llm_call_fn=AsyncMock(side_effect=RuntimeError("provider failed")),
                    stream_round_fn=AsyncMock(),
                    log_round_summary_fn=lambda **_kwargs: None,
                    emitter=emitter,
                    on_context_updated=context_updates.append,
                )

        self.assertEqual(context_updates[-1].round_index, 3)
        self.assertEqual(context_updates[-1].status, "no_op_fast_path")
        self.assertIsNone(context_updates[-1].actual_prompt_tokens)

    async def test_context_budget_error_is_observed_before_llm_call(self):
        step_context = AgentStepContext(
            step_id="step-budget",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        plan = ContextPlan(
            messages=[{"role": "user", "content": "过长正文"}],
            status="required_context_over_budget",
            context_window_tokens=100,
            context_window_source="test",
            context_window_status="known",
            estimated_tokens_before=120,
            estimated_tokens_after=120,
        )
        error = ContextBudgetExceededError(plan)
        observation = MagicMock()
        observation.finish_error = AsyncMock()
        llm_call = AsyncMock()

        with (
            patch(
                "app.services.stream.agent_round.prepare_context",
                new=AsyncMock(side_effect=error),
            ),
            patch(
                "app.services.stream.agent_round.create_llm_round_observation",
                return_value=observation,
            ) as create_observation,
        ):
            with self.assertRaises(ContextBudgetExceededError):
                await run_agent_round(
                    conversation_id="conv-1",
                    task_id="task-1",
                    run_id="run-1",
                    step_number=1,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=plan.messages,
                    should_use_reasoning=False,
                    call_kwargs={},
                    accumulated_usage=Usage(input_tokens=0, output_tokens=0),
                    step_context=step_context,
                    llm_call_fn=llm_call,
                    stream_round_fn=AsyncMock(),
                    log_round_summary_fn=lambda **_kwargs: None,
                )

        llm_call.assert_not_awaited()
        observation.start.assert_called_once_with()
        observation.finish_error.assert_awaited_once_with(error)
        self.assertEqual(create_observation.call_args.kwargs["estimator_status"], "context_manager_error")
        self.assertEqual(
            create_observation.call_args.kwargs["context_management"]["context_management_status"],
            "required_context_over_budget",
        )

    async def test_run_agent_round_records_only_current_round_usage(self):
        step_context = AgentStepContext(
            step_id="step-obs",
            step_number=2,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response

        async def llm_call_fn(*_args, **_kwargs):
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "正文", [], "stop", Usage(input_tokens=11, output_tokens=13)

        context_plan = MagicMock(
            messages=[{"role": "user", "content": "有效快照"}],
            estimated_tokens_after=8,
        )
        context_plan.telemetry.return_value = {"context_management_status": "trimmed"}
        with (
            patch(
                "app.services.stream.agent_round.create_llm_round_observation",
                return_value=observation,
            ) as create_observation,
            patch(
                "app.services.stream.agent_round.prepare_context",
                new=AsyncMock(return_value=context_plan),
            ),
        ):
            result = await run_agent_round(
                conversation_id="conv-1",
                task_id="task-1",
                run_id="run-1",
                step_number=2,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=[{"role": "user", "content": "你好"}],
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(input_tokens=5, output_tokens=7),
                step_context=step_context,
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
            )

        self.assertEqual(result.accumulated_usage, Usage(input_tokens=16, output_tokens=20))
        self.assertEqual(create_observation.call_args.kwargs["round_kind"], "agent")
        self.assertEqual(create_observation.call_args.kwargs["round_index"], 2)
        self.assertEqual(create_observation.call_args.kwargs["messages"], context_plan.messages)
        self.assertEqual(
            create_observation.call_args.kwargs["context_management"],
            {"context_management_status": "trimmed"},
        )
        self.assertEqual(create_observation.call_args.kwargs["estimated_prompt_tokens"], 8)
        observation.start.assert_called_once_with()
        observation.finish_success.assert_awaited_once_with(
            usage=Usage(input_tokens=11, output_tokens=13),
            finish_reason="stop",
        )

    async def test_run_agent_round_sends_effective_snapshot_without_mutating_canonical(self):
        canonical = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "最新问题"},
        ]
        effective = [canonical[-1]]
        step_context = AgentStepContext(
            step_id="step-context",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        observed_messages = []

        async def llm_call_fn(_model, _kwargs, messages, **_call_kwargs):
            observed_messages.extend(messages)
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "正文", [], "stop", None

        context_plan = MagicMock(messages=effective, estimated_tokens_after=10)
        context_plan.telemetry.return_value = {"context_management_status": "trimmed"}

        with patch(
            "app.services.stream.agent_round.prepare_context",
            new=AsyncMock(return_value=context_plan),
        ) as prepare:
            await run_agent_round(
                conversation_id="conv-1",
                task_id="task-1",
                run_id="run-1",
                step_number=1,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=canonical,
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(input_tokens=0, output_tokens=0),
                step_context=step_context,
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
            )

        prepare.assert_awaited_once()
        self.assertEqual(observed_messages, effective)
        self.assertEqual(len(canonical), 3)
        self.assertEqual(canonical[0]["content"], "旧问题")

    async def test_run_agent_round_records_error_without_swallowing_it(self):
        step_context = AgentStepContext(
            step_id="step-error",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        error = RuntimeError("provider echoed private prompt")

        async def llm_call_fn(*_args, **_kwargs):
            raise error

        with patch(
            "app.services.stream.agent_round.create_llm_round_observation",
            return_value=observation,
        ):
            with self.assertRaises(RuntimeError) as raised:
                await run_agent_round(
                    conversation_id="conv-1",
                    task_id="task-1",
                    run_id="run-1",
                    step_number=1,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=[],
                    should_use_reasoning=False,
                    call_kwargs={},
                    accumulated_usage=Usage(input_tokens=0, output_tokens=0),
                    step_context=step_context,
                    llm_call_fn=llm_call_fn,
                    stream_round_fn=AsyncMock(),
                    log_round_summary_fn=lambda **_kwargs: None,
                )

        self.assertIs(raised.exception, error)
        observation.finish_error.assert_awaited_once_with(error)

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
