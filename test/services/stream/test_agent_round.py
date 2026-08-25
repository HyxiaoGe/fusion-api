import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.prompts.agent_loop import VISIBLE_RESPONSE_LANGUAGE_PROMPT
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

    def test_accumulate_usage_preserves_optional_cache_token_totals(self):
        result = accumulate_usage(
            Usage(input_tokens=2, output_tokens=3, cache_read_tokens=4),
            Usage(input_tokens=5, output_tokens=7, cache_read_tokens=6, cache_write_tokens=8),
        )

        self.assertEqual(result.cache_read_tokens, 10)
        self.assertEqual(result.cache_write_tokens, 8)

    def test_accumulate_usage_preserves_optional_reasoning_token_totals(self):
        result = accumulate_usage(
            Usage(input_tokens=2, output_tokens=3, reasoning_tokens=4),
            Usage(input_tokens=5, output_tokens=7, reasoning_tokens=6),
        )

        self.assertEqual(result.reasoning_tokens, 10)


class AgentRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_timing_excludes_started_and_context_sink_delay(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [0.0]
        calls = []
        emitter = AsyncMock()

        async def started(**_kwargs):
            calls.append("started")
            now[0] += 5.0

        async def context_status(**kwargs):
            if kwargs["phase"] == "final":
                now[0] += 7.0

        emitter.llm_round_started.side_effect = started
        emitter.context_status_updated.side_effect = context_status
        context_plan = ContextPlan(
            messages=[],
            status="no_op",
            context_window_tokens=1000,
            context_window_source="test",
            context_window_status="known",
            estimated_tokens_after=10,
        )
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (1000, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 5.1
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(content="答案", reasoning_content=None, tool_calls=None))]
            )
            now[0] = 5.2

        async def llm_call_fn(*_args, **_kwargs):
            calls.append("network")
            return response()

        async def stream_round_fn(observed, *_args, **kwargs):
            async for _chunk in observed:
                kwargs["on_output_candidate"]("content")
                await kwargs["on_visible_output"]("content")
            return "", "答案", [], "stop", Usage(input_tokens=1, output_tokens=1)

        with (
            patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
            patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
        ):
            await run_agent_round(
                conversation_id="conv",
                task_id="task",
                run_id="run",
                step_number=1,
                model_id="model",
                provider="provider",
                litellm_model="test/model",
                litellm_kwargs={},
                messages=[],
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(),
                step_context=AgentStepContext("step", 1, 0.0, "thinking", "text"),
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                emitter=emitter,
            )

        self.assertEqual(calls, ["started", "network"])
        completed = emitter.llm_round_completed.await_args.kwargs
        self.assertEqual(completed["ttft_ms"], 100)
        self.assertEqual(completed["duration_ms"], 200)

    async def test_run_agent_round_emits_llm_lifecycle_in_order(self):
        calls: list[str] = []
        emitter = AsyncMock()
        emitter.llm_round_started.side_effect = lambda **_kwargs: calls.append("started")
        emitter.llm_round_first_output_delta.side_effect = lambda **_kwargs: calls.append("first")
        emitter.llm_round_completed.side_effect = lambda **_kwargs: calls.append("completed")
        observation = MagicMock(first_output_delta_kind="content", first_output_delta_ms=125, duration_ms=400)
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response
        context_plan = MagicMock(messages=[], estimated_tokens_after=10)
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}

        async def llm_call_fn(*_args, **_kwargs):
            calls.append("network")
            return "response"

        async def stream_round_fn(*_args, **kwargs):
            await kwargs["on_visible_output"]("content")
            return "", "回答", [], "stop", Usage(input_tokens=3, output_tokens=2)

        with (
            patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
            patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
        ):
            result = await run_agent_round(
                conversation_id="conv-life",
                task_id="task-life",
                run_id="run-life",
                step_number=1,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=[],
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(),
                step_context=AgentStepContext("step-life", 1, 0.0, "thinking", "text"),
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                emitter=emitter,
            )

        self.assertEqual(calls, ["started", "network", "first", "completed"])
        self.assertIsNone(result.llm_lifecycle)
        completed = emitter.llm_round_completed.await_args.kwargs
        self.assertEqual(completed["total_tokens"], 5)
        self.assertEqual(completed["ttft_ms"], 125)

    async def test_cancelled_llm_call_emits_cancelled_and_reraises(self):
        emitter = AsyncMock()
        emitter.llm_round_cancelled.side_effect = RuntimeError("terminal sink failed")
        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        context_plan = MagicMock(messages=[], estimated_tokens_after=10)
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}

        with (
            patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
            patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_agent_round(
                    conversation_id="conv-cancel",
                    task_id="task-cancel",
                    run_id="run-cancel",
                    step_number=1,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=[],
                    should_use_reasoning=False,
                    call_kwargs={},
                    accumulated_usage=Usage(),
                    step_context=AgentStepContext("step-cancel", 1, 0.0, "thinking", "text"),
                    llm_call_fn=AsyncMock(side_effect=asyncio.CancelledError),
                    stream_round_fn=AsyncMock(),
                    log_round_summary_fn=lambda **_kwargs: None,
                    emitter=emitter,
                )

        emitter.llm_round_cancelled.assert_awaited_once()
        self.assertEqual(emitter.llm_round_cancelled.await_args.kwargs["reason"], "shutdown")
        emitter.llm_round_failed.assert_not_awaited()

    async def test_stream_failure_persists_visible_partial_round_detail(self):
        emitter = AsyncMock()
        detail_scheduler = MagicMock()
        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response
        context_plan = MagicMock(messages=[], estimated_tokens_after=10)
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}

        async def stream_round_fn(*_args, partial_output, **_kwargs):
            partial_output["reasoning_buf"] = "部分推理"
            partial_output["content_buf"] = "部分回答"
            raise RuntimeError("stream failed")

        with (
            patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
            patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
        ):
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await run_agent_round(
                    conversation_id="conv-partial",
                    task_id="task-partial",
                    run_id="run-partial",
                    step_number=1,
                    model_id="deepseek-chat",
                    provider="deepseek",
                    litellm_model="deepseek/deepseek-chat",
                    litellm_kwargs={},
                    messages=[],
                    should_use_reasoning=True,
                    call_kwargs={},
                    accumulated_usage=Usage(),
                    step_context=AgentStepContext("step-partial", 1, 0.0, "thinking", "text"),
                    llm_call_fn=AsyncMock(return_value="response"),
                    stream_round_fn=stream_round_fn,
                    log_round_summary_fn=lambda **_kwargs: None,
                    emitter=emitter,
                    llm_round_detail_scheduler=detail_scheduler,
                )

        emitter.llm_round_failed.assert_awaited_once()
        detail_scheduler.assert_called_once()
        draft = detail_scheduler.call_args.args[0]
        self.assertEqual(draft.reasoning_text, "部分推理")
        self.assertEqual(draft.content_text, "部分回答")

    async def test_deferred_agent_round_keeps_lifecycle_open_until_visibility_decision(self):
        emitter = AsyncMock()
        observation = MagicMock(first_output_delta_kind="content", first_output_delta_ms=80, duration_ms=200)
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response
        context_plan = MagicMock(messages=[], estimated_tokens_after=10)
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}

        stream_round_fn = AsyncMock(return_value=("", "候选", [], "stop", Usage()))
        with (
            patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
            patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
        ):
            result = await run_agent_round(
                conversation_id="conv-deferred",
                task_id="task-deferred",
                run_id="run-deferred",
                step_number=2,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=[],
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(),
                step_context=AgentStepContext("step-deferred", 2, 0.0, "thinking", "text"),
                llm_call_fn=AsyncMock(return_value="response"),
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                emitter=emitter,
                defer_output=True,
            )

        self.assertIsNotNone(result.llm_lifecycle)
        self.assertNotIn("on_visible_output", stream_round_fn.await_args.kwargs)
        emitter.llm_round_first_output_delta.assert_not_awaited()
        emitter.llm_round_completed.assert_not_awaited()

        await result.llm_lifecycle.finish_success(output_visible=False)
        emitter.llm_round_first_output_delta.assert_not_awaited()
        self.assertIsNone(emitter.llm_round_completed.await_args.kwargs["ttft_ms"])

    async def test_post_stream_context_error_and_cancel_close_llm_round(self):
        for primary in (RuntimeError("final context failed"), asyncio.CancelledError()):
            with self.subTest(primary=type(primary).__name__):
                emitter = AsyncMock()
                observation = MagicMock(first_output_delta_kind=None, duration_ms=10)
                observation.finish_success = AsyncMock()
                observation.finish_error = AsyncMock()
                observation.wrap_response.side_effect = lambda response: response
                context_plan = MagicMock(messages=[], estimated_tokens_after=10)
                context_plan.telemetry.return_value = {"context_management_status": "no_op"}

                def build_context(_plan, usage=None, **_kwargs):
                    if usage is None:
                        return ContextUsage(status="no_op")
                    raise primary

                with (
                    patch("app.services.stream.agent_round.prepare_context", new=AsyncMock(return_value=context_plan)),
                    patch("app.services.stream.agent_round.create_llm_round_observation", return_value=observation),
                    patch("app.services.stream.agent_round.build_context_usage", side_effect=build_context),
                ):
                    with self.assertRaises(type(primary)) as raised:
                        await run_agent_round(
                            conversation_id="conv-post",
                            task_id="task-post",
                            run_id="run-post",
                            step_number=1,
                            model_id="model",
                            provider="provider",
                            litellm_model="test/model",
                            litellm_kwargs={},
                            messages=[],
                            should_use_reasoning=False,
                            call_kwargs={},
                            accumulated_usage=Usage(),
                            step_context=AgentStepContext("step-post", 1, 0.0, "thinking", "text"),
                            llm_call_fn=AsyncMock(return_value="response"),
                            stream_round_fn=AsyncMock(
                                return_value=("", "answer", [], "stop", Usage(input_tokens=1, output_tokens=1))
                            ),
                            log_round_summary_fn=lambda **_kwargs: None,
                            emitter=emitter,
                        )

                self.assertIs(raised.exception, primary)
                if isinstance(primary, asyncio.CancelledError):
                    emitter.llm_round_cancelled.assert_awaited_once()
                    emitter.llm_round_failed.assert_not_awaited()
                else:
                    emitter.llm_round_failed.assert_awaited_once()
                    emitter.llm_round_cancelled.assert_not_awaited()

    async def test_run_agent_round_finalizes_language_contract_before_budget_and_model_call(self):
        step_context = AgentStepContext(
            step_id="step-language",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        prepared_messages = []
        sent_messages = []

        async def prepare_context_fn(**kwargs):
            prepared_messages.extend(kwargs["messages"])
            return ContextPlan(
                messages=kwargs["messages"],
                status="no_op_fast_path",
                context_window_tokens=128_000,
                context_window_source="registry",
                context_window_status="known",
            )

        async def llm_call_fn(_model, _kwargs, call_messages, **_call_kwargs):
            sent_messages.extend(call_messages)
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "中文思考", "中文回答", [], "stop", None

        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response
        messages = [
            {"role": "system", "content": "身份规则"},
            {"role": "user", "content": "分析具身智能产业"},
            {"role": "system", "content": "当前执行搜索阶段"},
        ]

        with (
            patch("app.services.stream.agent_round.prepare_context", new=prepare_context_fn),
            patch(
                "app.services.stream.agent_round.create_llm_round_observation",
                return_value=observation,
            ),
        ):
            await run_agent_round(
                conversation_id="conv-language",
                task_id="task-language",
                run_id="run-language",
                step_number=1,
                model_id="deepseek-v4",
                provider="deepseek",
                litellm_model="deepseek/deepseek-v4",
                litellm_kwargs={},
                messages=messages,
                should_use_reasoning=True,
                call_kwargs={},
                accumulated_usage=Usage(),
                step_context=step_context,
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
            )

        self.assertEqual(sent_messages, prepared_messages)
        self.assertTrue(sent_messages[-1]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertEqual(
            sum(str(item.get("content") or "").count(VISIBLE_RESPONSE_LANGUAGE_PROMPT) for item in sent_messages),
            1,
        )

    async def test_run_agent_round_marks_and_forwards_deferred_output(self):
        step_context = AgentStepContext(
            step_id="step-product",
            step_number=2,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        received_kwargs = {}

        async def stream_round_fn(*_args, **kwargs):
            received_kwargs.update(kwargs)
            return "", "模型自由文本", [], "stop", None

        context_plan = MagicMock(
            messages=[{"role": "tool", "content": "结构化地点结果"}],
            estimated_tokens_after=10,
        )
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}
        with patch(
            "app.services.stream.agent_round.prepare_context",
            new=AsyncMock(return_value=context_plan),
        ):
            result = await run_agent_round(
                conversation_id="conv-product",
                task_id="task-product",
                run_id="run-product",
                step_number=2,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=context_plan.messages,
                should_use_reasoning=False,
                call_kwargs={},
                accumulated_usage=Usage(),
                step_context=step_context,
                llm_call_fn=AsyncMock(return_value="response"),
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                defer_output=True,
            )

        self.assertTrue(received_kwargs["defer_output"])
        self.assertTrue(received_kwargs["allow_deferred_reasoning_output"])
        self.assertEqual(received_kwargs["provider"], "openai")
        self.assertNotIn("on_answer_started", received_kwargs)
        self.assertTrue(result.output_deferred)
        self.assertTrue(result.allow_deferred_reasoning_output)

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
        self.assertEqual(result.announced_tool_names, frozenset())
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

    @patch("app.services.stream.agent_round.litellm_health.record_success")
    async def test_run_agent_round_records_error_without_swallowing_it(self, record_success):
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
        emitter = AsyncMock()
        emitter.llm_round_failed.side_effect = RuntimeError("terminal sink failed")

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
                    emitter=emitter,
                )

        self.assertIs(raised.exception, error)
        observation.finish_error.assert_awaited_once_with(error)
        emitter.llm_round_failed.assert_awaited_once()
        self.assertEqual(emitter.llm_round_failed.await_args.kwargs["error_code"], "provider_error")
        self.assertIsNone(emitter.llm_round_failed.await_args.kwargs["message"])
        record_success.assert_not_called()

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
            model_id,
            allow_deferred_reasoning_output,
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
                    model_id,
                    allow_deferred_reasoning_output,
                    run_id,
                    step_id,
                )
            )
            return "推理", "正文", [{"id": "tool-1"}], "tool_calls", Usage(input_tokens=5, output_tokens=7)

        result = await collect_agent_round_stream(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            model_id="kimi-k3",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"metadata": {"trace": "x"}},
            messages=messages,
            should_use_reasoning=True,
            call_kwargs={"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
            step_context=step_context,
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            allow_deferred_reasoning_output=True,
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
                "kimi-k3",
                True,
                "run-1",
                "step-collect",
            ),
        )

    @patch("app.services.stream.agent_round.litellm_health.record_success")
    async def test_run_agent_round_records_success_and_accumulates_usage(self, record_success):
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

        record_success.assert_called_once_with("gpt-4")
        self.assertEqual(result.reasoning_buf, "推理")
        self.assertEqual(result.content_buf, "正文")
        self.assertEqual(result.tool_calls, [{"id": "tool-1"}])
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.accumulated_usage, Usage(input_tokens=7, output_tokens=10))
        self.assertEqual(result.announced_tool_names, frozenset({"web_search"}))
        self.assertEqual([event[0] for event in events], ["llm", "stream", "log"])
        self.assertEqual(events[0][:3], ("llm", "openai/gpt-4", {"metadata": {"trace": "x"}}))
        self.assertEqual(events[0][3][-1], messages[0])
        self.assertEqual(events[0][3][0]["content"], VISIBLE_RESPONSE_LANGUAGE_PROMPT)
        self.assertEqual(
            events[0][4],
            {"temperature": 0.1, "tools": [{"function": {"name": "web_search"}}]},
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

    async def test_run_agent_round_does_not_record_cancelled_stream_as_healthy(self):
        step_context = AgentStepContext(
            step_id="step-cancelled",
            step_number=1,
            started_at=100.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        async def llm_call_fn(*_args, **_kwargs):
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "", [], "cancelled", None

        emitter = AsyncMock()
        with patch("app.services.stream.agent_round.litellm_health.record_success") as record_success:
            result = await run_agent_round(
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
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=lambda **_kwargs: None,
                emitter=emitter,
            )

        self.assertEqual(result.finish_reason, "cancelled")
        emitter.llm_round_cancelled.assert_awaited_once()
        self.assertEqual(emitter.llm_round_cancelled.await_args.kwargs["reason"], "superseded")
        record_success.assert_not_called()

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
