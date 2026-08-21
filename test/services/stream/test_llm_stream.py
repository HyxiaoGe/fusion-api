import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.chat import Usage
from app.services.stream import llm_stream as llm_stream_module


def make_chunk(*, delta=None, finish_reason=None, usage=None, choices=True):
    if not choices:
        return SimpleNamespace(choices=[], usage=usage)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta or SimpleNamespace(content=None), finish_reason=finish_reason)],
        usage=usage,
    )


def make_tool_delta(*, index=0, tool_call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                index=index,
                id=tool_call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ],
        content=None,
    )


async def async_response(chunks):
    for chunk in chunks:
        yield chunk


class LLMStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_filtered_candidates_keep_reclassified_and_late_visible_times(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [30.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 30.1
            yield make_chunk(delta=SimpleNamespace(content="<think>隐藏推理</think>"))
            now[0] = 30.5
            yield make_chunk(delta=SimpleNamespace(content="可见正文"), finish_reason="stop")

        observation.start()
        with (
            patch("app.services.stream.llm_stream.append_chunk", new=AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                True,
                "thinking",
                "text",
                reasoning_transport_mode="delta",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
            )

        self.assertEqual(result[:2], ("隐藏推理", "可见正文"))
        self.assertEqual(observation.output_delta_ms("reasoning"), 100)
        self.assertEqual(observation.output_delta_ms("content"), 500)
        self.assertIsNone(observation.output_delta_ms("tool_call"))

    async def test_malformed_dsml_does_not_create_visible_or_tool_candidate(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [40.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 40.1
            yield make_chunk(delta=SimpleNamespace(content="<｜｜DSML｜｜tool_calls>garbage"), finish_reason="stop")

        observation.start()
        with patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                False,
                "thinking",
                "text",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
            )

        self.assertEqual(result[3], "tool_protocol_error")
        self.assertIsNone(observation.output_delta_ms("content"))
        self.assertIsNone(observation.output_delta_ms("tool_call"))

    async def test_probing_flush_uses_earliest_contributing_chunk_time(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [50.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            for observed_at, text, finish_reason in (
                (50.1, "第一段缓冲推理", None),
                (50.3, "第二段缓冲推理", None),
                (50.5, "第三段缓冲推理", "stop"),
            ):
                now[0] = observed_at
                yield make_chunk(
                    delta=SimpleNamespace(content=None, reasoning_content=text),
                    finish_reason=finish_reason,
                )

        observation.start()
        with (
            patch("app.services.stream.llm_stream.append_chunk", new=AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                True,
                "thinking",
                "text",
                reasoning_transport_mode="probing",
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[0], "第一段缓冲推理第二段缓冲推理第三段缓冲推理")
        self.assertEqual(observation.output_delta_ms("reasoning"), 100)

    async def test_discarded_probing_buffer_never_creates_candidate(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [60.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 60.1
            yield make_chunk(
                delta=SimpleNamespace(content=None, reasoning_content="<｜｜DSML"),
                finish_reason=None,
            )
            now[0] = 60.4
            yield make_chunk(
                delta=SimpleNamespace(content=None, reasoning_content="｜｜tool_calls>garbage"),
                finish_reason="stop",
            )

        observation.start()
        with patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                True,
                "thinking",
                "text",
                reasoning_transport_mode="probing",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[0], "")
        self.assertIsNone(observation.output_delta_ms("reasoning"))

    async def test_split_update_plan_uses_first_content_fragment_without_crossing_reasoning_kind(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [70.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            for observed_at, text, finish_reason in (
                (70.1, "<think>隐藏推理</think>", None),
                (70.4, "up", None),
                (70.5, "date_plan 已完成", "stop"),
            ):
                now[0] = observed_at
                yield make_chunk(delta=SimpleNamespace(content=text), finish_reason=finish_reason)

        observation.start()
        with (
            patch("app.services.stream.llm_stream.append_chunk", new=AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                True,
                "thinking",
                "text",
                reasoning_transport_mode="delta",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[:2], ("隐藏推理", "更新计划已完成"))
        self.assertEqual(observation.output_delta_ms("reasoning"), 100)
        self.assertEqual(observation.output_delta_ms("content"), 400)

    async def test_split_mcp_alias_uses_first_content_fragment_time(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [80.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 80.1
            yield make_chunk(delta=SimpleNamespace(content="mcp_"))
            now[0] = 80.6
            yield make_chunk(
                delta=SimpleNamespace(content="a" * 43),
                finish_reason="stop",
            )

        observation.start()
        with (
            patch("app.services.stream.llm_stream.append_chunk", new=AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                False,
                "thinking",
                "text",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[1], "外部工具")
        self.assertEqual(observation.output_delta_ms("content"), 100)

    async def test_finish_flush_uses_first_buffered_content_fragment_time(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [90.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 90.1
            yield make_chunk(delta=SimpleNamespace(content="upda"))
            now[0] = 90.7
            yield make_chunk(delta=SimpleNamespace(content=None), finish_reason="stop")

        observation.start()
        with (
            patch("app.services.stream.llm_stream.append_chunk", new=AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                False,
                "thinking",
                "text",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[1], "更新计划")
        self.assertEqual(observation.output_delta_ms("content"), 100)

    async def test_fragmented_dsml_discard_does_not_create_content_candidate(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        now = [100.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 100.1
            yield make_chunk(delta=SimpleNamespace(content="<｜｜DSML"))
            now[0] = 100.4
            yield make_chunk(
                delta=SimpleNamespace(content="｜｜tool_calls>garbage"),
                finish_reason="stop",
            )

        observation.start()
        with patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)):
            result = await llm_stream_module.stream_round(
                observation.wrap_response(response()),
                "conv",
                "task",
                False,
                "thinking",
                "text",
                defer_output=True,
                on_output_candidate=observation.observe_output_candidate,
                capture_output_candidate_time=observation.capture_output_candidate_time,
            )

        self.assertEqual(result[1], "")
        self.assertEqual(result[3], "tool_protocol_error")
        self.assertIsNone(observation.output_delta_ms("content"))

    async def test_leading_whitespace_dsml_only_never_becomes_visible_content(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        valid_protocol = (
            "<｜｜DSML｜｜tool_calls>"
            '<｜｜DSML｜｜invoke name="web_search"></｜｜DSML｜｜invoke>'
            "</｜｜DSML｜｜tool_calls>"
        )
        cases = (
            ("valid", valid_protocol, "tool_calls"),
            ("malformed", "<｜｜DSML｜｜tool_calls>garbage", "tool_protocol_error"),
        )

        for case_index, (name, protocol, finish_reason) in enumerate(cases, 1):
            with self.subTest(name=name):
                now = [110.0 + case_index]
                observation = LLMRoundObservation(
                    metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
                    litellm_model="test/model",
                    messages=[],
                    call_kwargs={},
                    clock=lambda: now[0],
                    token_estimator=lambda *_args, **_kwargs: 1,
                    context_window_resolver=lambda _model_id: (4096, "test", "known"),
                    run_context_in_thread=False,
                )
                append_chunk = AsyncMock()
                on_visible_output = AsyncMock()

                async def response():
                    start = 110.0 + case_index
                    for offset, content, terminal in (
                        (0.1, "\n", None),
                        (0.2, "\r\n", None),
                        (0.3, protocol[:9], None),
                        (0.4, protocol[9:], "stop"),
                    ):
                        now[0] = start + offset
                        yield make_chunk(delta=SimpleNamespace(content=content), finish_reason=terminal)

                observation.start()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
                    patch("app.services.stream.llm_stream.logger.warning") as warning,
                ):
                    result = await llm_stream_module.stream_round(
                        observation.wrap_response(response()),
                        "conv",
                        "task",
                        False,
                        "thinking",
                        "text",
                        on_visible_output=on_visible_output,
                        on_output_candidate=observation.observe_output_candidate,
                        capture_output_candidate_time=observation.capture_output_candidate_time,
                    )

                self.assertEqual(result[1], "")
                self.assertEqual(result[3], finish_reason)
                append_chunk.assert_not_awaited()
                on_visible_output.assert_not_awaited()
                self.assertIsNone(observation.output_delta_ms("content"))
                self.assertFalse(
                    any(
                        "answering 内容过滤出现非单调输出" in str(call.args[0])
                        for call in warning.call_args_list
                    )
                )

    async def test_normal_whitespace_content_stays_visible(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        cases = (
            (
                "whitespace_only",
                ((0.1, "\n", None), (0.2, "\r\n", "stop")),
                "\n\r\n",
            ),
            (
                "whitespace_then_body",
                ((0.1, "\n", None), (0.2, "\r\n", None), (0.4, "普通正文", "stop")),
                "\n\r\n普通正文",
            ),
        )

        for case_index, (name, chunks, expected) in enumerate(cases, 1):
            with self.subTest(name=name):
                now = [120.0 + case_index]
                observation = LLMRoundObservation(
                    metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
                    litellm_model="test/model",
                    messages=[],
                    call_kwargs={},
                    clock=lambda: now[0],
                    token_estimator=lambda *_args, **_kwargs: 1,
                    context_window_resolver=lambda _model_id: (4096, "test", "known"),
                    run_context_in_thread=False,
                )
                append_chunk = AsyncMock()
                on_visible_output = AsyncMock()

                async def response():
                    start = 120.0 + case_index
                    for offset, content, finish_reason in chunks:
                        now[0] = start + offset
                        yield make_chunk(delta=SimpleNamespace(content=content), finish_reason=finish_reason)

                observation.start()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
                ):
                    result = await llm_stream_module.stream_round(
                        observation.wrap_response(response()),
                        "conv",
                        "task",
                        False,
                        "thinking",
                        "text",
                        on_visible_output=on_visible_output,
                        on_output_candidate=observation.observe_output_candidate,
                        capture_output_candidate_time=observation.capture_output_candidate_time,
                    )

                self.assertEqual(result[1], expected)
                self.assertEqual(
                    [call.args[2] for call in append_chunk.await_args_list],
                    [expected],
                )
                on_visible_output.assert_awaited_once_with("content")
                self.assertEqual(observation.output_delta_ms("content"), 100)

    async def test_whitespace_after_visible_body_is_not_delayed_until_finish(self):
        append_chunk = AsyncMock()
        append_counts_before_finish = []
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-visible-body",
            task_id="task-visible-body",
            should_use_reasoning=False,
            thinking_block_id="thinking",
            text_block_id="text",
        )

        async def response():
            yield make_chunk(delta=SimpleNamespace(content="普通正文"))
            yield make_chunk(delta=SimpleNamespace(content="\n"))
            append_counts_before_finish.append(append_chunk.await_count)
            yield make_chunk(delta=SimpleNamespace(content=None), finish_reason="stop")

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.consume_stream_round(response(), request)

        self.assertEqual(result.content_buf, "普通正文\n")
        self.assertEqual(append_counts_before_finish, [2])
        self.assertEqual(
            [call.args[2] for call in append_chunk.await_args_list],
            ["普通正文", "\n"],
        )

    async def test_final_flush_clears_pending_candidate_state_on_error_and_cancel(self):
        cases = (
            ("success", None),
            ("error", RuntimeError("redis failed")),
            ("cancel", asyncio.CancelledError()),
        )

        for name, error in cases:
            with self.subTest(name=name):
                state = llm_stream_module.LLMStreamState(
                    raw_content_buf="\n",
                    pending_reasoning_candidate_time=1.0,
                    pending_content_candidate_time=2.0,
                )
                request = llm_stream_module.LLMStreamRequest(
                    conversation_id="conv-flush-cleanup",
                    task_id="task-flush-cleanup",
                    should_use_reasoning=False,
                    thinking_block_id="thinking",
                    text_block_id="text",
                )
                append_chunk = AsyncMock(side_effect=error)

                with patch("app.services.stream.llm_stream.append_chunk", append_chunk):
                    if error is None:
                        await llm_stream_module.flush_pending_internal_mcp_aliases(
                            request=request,
                            state=state,
                        )
                    else:
                        with self.assertRaises(type(error)) as raised:
                            await llm_stream_module.flush_pending_internal_mcp_aliases(
                                request=request,
                                state=state,
                            )
                        self.assertIs(raised.exception, error)

                self.assertIsNone(state.pending_reasoning_candidate_time)
                self.assertIsNone(state.pending_content_candidate_time)

    async def test_leading_whitespace_is_not_published_when_stream_errors_or_is_cancelled(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        cases = (
            ("error", RuntimeError("stream failed")),
            ("cancel", asyncio.CancelledError()),
        )

        for case_index, (name, error) in enumerate(cases, 1):
            with self.subTest(name=name):
                now = [130.0 + case_index]
                observation = LLMRoundObservation(
                    metadata=RoundMetadata("conv", "run", 1, "step", "agent", "model", "provider"),
                    litellm_model="test/model",
                    messages=[],
                    call_kwargs={},
                    clock=lambda: now[0],
                    token_estimator=lambda *_args, **_kwargs: 1,
                    context_window_resolver=lambda _model_id: (4096, "test", "known"),
                    run_context_in_thread=False,
                )
                append_chunk = AsyncMock()
                on_visible_output = AsyncMock()

                async def response():
                    start = 130.0 + case_index
                    now[0] = start + 0.1
                    yield make_chunk(delta=SimpleNamespace(content="\n"))
                    now[0] = start + 0.2
                    yield make_chunk(delta=SimpleNamespace(content="\r\n"))
                    raise error

                observation.start()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
                ):
                    with self.assertRaises(type(error)) as raised:
                        await llm_stream_module.stream_round(
                            observation.wrap_response(response()),
                            "conv",
                            "task",
                            False,
                            "thinking",
                            "text",
                            on_visible_output=on_visible_output,
                            on_output_candidate=observation.observe_output_candidate,
                            capture_output_candidate_time=observation.capture_output_candidate_time,
                        )

                self.assertIs(raised.exception, error)
                append_chunk.assert_not_awaited()
                on_visible_output.assert_not_awaited()
                self.assertIsNone(observation.output_delta_ms("content"))

    def test_extract_usage_supports_cache_tokens_and_rejects_invalid_values(self):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            cache_read_input_tokens=6,
            cache_creation_input_tokens=2,
        )

        result = llm_stream_module.extract_usage(SimpleNamespace(usage=usage))

        self.assertEqual(
            result,
            Usage(
                input_tokens=10,
                output_tokens=4,
                cache_read_tokens=6,
                cache_write_tokens=2,
            ),
        )

        invalid = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            cache_read_tokens=True,
            cache_read_input_tokens=-1,
            cache_write_tokens=float("inf"),
        )
        invalid_result = llm_stream_module.extract_usage(SimpleNamespace(usage=invalid))
        self.assertIsNone(invalid_result.cache_read_tokens)
        self.assertIsNone(invalid_result.cache_write_tokens)

    def test_extract_usage_supports_nested_prompt_cache_details(self):
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 7},
            "cache_creation_input_tokens": 3,
        }

        result = llm_stream_module.extract_usage(SimpleNamespace(usage=usage))

        self.assertEqual(result.cache_read_tokens, 7)
        self.assertEqual(result.cache_write_tokens, 3)

    async def test_plan_mode_tool_round_buffers_content_without_answering_or_preview(self):
        events: list[str] = []

        async def append_chunk(_conversation_id, chunk_type, *_args, **_kwargs):
            events.append(chunk_type)

        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-late-tool",
            task_id="task-late-tool",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            defer_output=True,
        )

        with (
            patch("app.services.stream.llm_stream.append_chunk", side_effect=append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="先给出草稿"), finish_reason=None),
                        make_chunk(
                            delta=make_tool_delta(
                                tool_call_id="tc-late",
                                name="web_search",
                                arguments='{"query":"补充事实"}',
                            ),
                            finish_reason="tool_calls",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(events, [])
        self.assertEqual(outcome.content_buf, "先给出草稿")
        self.assertEqual(outcome.finish_reason, "tool_calls")
        self.assertEqual(outcome.tool_calls[0]["name"], "web_search")

    async def test_non_deferred_round_streams_each_answer_chunk(self):
        events: list[str] = []

        async def append_chunk(*_args, **_kwargs):
            events.append("answering")

        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-answer-started",
            task_id="task-answer-started",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        with (
            patch("app.services.stream.llm_stream.append_chunk", side_effect=append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="第一段"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="第二段"), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(events, ["answering", "answering"])

    def test_llm_retry_does_not_match_rate_substring_in_plain_message(self):
        self.assertFalse(
            llm_stream_module._is_llm_error_retryable(
                RuntimeError("moderate policy rejection"),
            )
        )

    def test_llm_retry_does_not_parse_status_from_secret_bearing_message(self):
        error = RuntimeError("Authorization: Bearer sk-secret-value; upstream 503")

        self.assertFalse(llm_stream_module._is_llm_error_retryable(error))

    def test_llm_retry_accepts_structured_retryable_status(self):
        class UpstreamUnavailableError(RuntimeError):
            status_code = 503

        self.assertTrue(
            llm_stream_module._is_llm_error_retryable(
                UpstreamUnavailableError("Authorization: Bearer sk-secret-value"),
            )
        )

    def test_llm_retry_log_never_contains_exception_message(self):
        class UpstreamUnavailableError(RuntimeError):
            status_code = 503

        error = UpstreamUnavailableError("Authorization: Bearer sk-secret-value")
        with patch("app.services.stream.llm_stream.logger.warning") as warning:
            llm_stream_module._log_llm_retry(
                {
                    "tries": 1,
                    "wait": 2,
                    "exception": error,
                }
            )

        rendered = " ".join(str(value) for value in warning.call_args.args)
        self.assertNotIn("sk-secret-value", rendered)
        self.assertNotIn(str(error), rendered)
        self.assertIn("http_503", rendered)

    async def test_k3_streams_sanitized_reasoning_while_answer_is_deferred(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-product",
            task_id="task-product",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="kimi-k3",
            allow_deferred_reasoning_output=True,
            run_id="run-product",
            step_id="step-product",
            defer_output=True,
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="先调用 weather_"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content="方便停车，驾车更合适。",
                                reasoning_content="forecast，再核对路线事实。",
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "先调用天气查询，再核对路线事实。")
        self.assertEqual(outcome.raw_reasoning_buf, "先调用 weather_forecast，再核对路线事实。")
        self.assertEqual(outcome.content_buf, "方便停车，驾车更合适。")
        self.assertEqual(
            [call.args[1] for call in append_chunk.await_args_list],
            ["reasoning"],
        )

    async def test_non_k3_streams_reasoning_while_answer_is_deferred(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-product-non-k3",
            task_id="task-product-non-k3",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="openai",
            model_id="gpt-4",
            allow_deferred_reasoning_output=True,
            defer_output=True,
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="先调用 weather_"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content="方便停车，驾车更合适。",
                                reasoning_content="forecast，再核对路线事实。",
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "先调用天气查询，再核对路线事实。")
        self.assertEqual(outcome.raw_reasoning_buf, "先调用 weather_forecast，再核对路线事实。")
        self.assertEqual(outcome.content_buf, "方便停车，驾车更合适。")
        self.assertEqual(
            [call.args[1] for call in append_chunk.await_args_list],
            ["reasoning", "reasoning"],
        )

    async def test_all_reasoning_content_models_share_deferred_reasoning_transport(self):
        model_ids = (
            "deepseek-chat",
            "deepseek-reasoner",
            "qwen-max-latest",
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "grok-4.5",
            "kimi-k2.5",
            "kimi-k2.7-code",
            "kimi-k2.7-code-highspeed",
            "kimi-k3",
            "mimo-v2.5-pro",
            "mimo-v2.5-pro-ultraspeed",
            "qwen3.6-plus",
            "doubao-seed-2-0-pro-260215",
        )

        for model_id in model_ids:
            with self.subTest(model_id=model_id):
                request = llm_stream_module.LLMStreamRequest(
                    conversation_id=f"conv-{model_id}",
                    task_id=f"task-{model_id}",
                    should_use_reasoning=True,
                    thinking_block_id="blk-thinking",
                    text_block_id="blk-text",
                    model_id=model_id,
                    reasoning_transport_mode="delta",
                    defer_output=True,
                    allow_deferred_reasoning_output=True,
                )
                append_chunk = AsyncMock()

                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
                ):
                    outcome = await llm_stream_module.consume_stream_round(
                        async_response(
                            [
                                make_chunk(
                                    delta=SimpleNamespace(
                                        content="最终正文",
                                        reasoning_content="可见推理",
                                    ),
                                    finish_reason="stop",
                                )
                            ]
                        ),
                        request,
                    )

                self.assertEqual(outcome.reasoning_buf, "可见推理")
                self.assertEqual(outcome.content_buf, "最终正文")
                self.assertEqual(
                    [(call.args[1], call.args[2]) for call in append_chunk.await_args_list],
                    [("reasoning", "可见推理")],
                )

    async def test_k3_deferred_reasoning_requires_call_level_opt_in(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-k3-no-opt-in",
            task_id="task-k3-no-opt-in",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="kimi-k3",
            defer_output=True,
        )
        append_chunk = AsyncMock()
        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="候选综合推理"),
                            finish_reason="stop",
                        )
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "候选综合推理")
        append_chunk.assert_not_awaited()

    async def test_reasoning_never_streams_split_tool_protocol(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-deferred-protocol",
            task_id="task-deferred-protocol",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            defer_output=False,
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="先核对公开资料。"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="<｜｜DSML"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="｜｜tool_calls><｜｜DSML｜｜invoke",
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(emitted_reasoning, "先核对公开资料。")
        self.assertEqual(outcome.reasoning_buf, "先核对公开资料。")
        self.assertIn("DSML", outcome.raw_reasoning_buf)

    async def test_reasoning_hides_split_internal_control_instructions_but_keeps_raw(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-deferred-control-instructions",
            task_id="task-deferred-control-instructions",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="moonshot/kimi-k3",
            defer_output=False,
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="先比较两种授权方案的风险和成本。\n\n",
                            ),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="According to the autonomous web search ",
                            ),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content=(
                                    "rules: this should not be searched. 本轮启用了强制计划模式，必须先创建执行计划。"
                                ),
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        raw_reasoning = (
            "先比较两种授权方案的风险和成本。\n\n"
            "According to the autonomous web search rules: this should not be searched. "
            "本轮启用了强制计划模式，必须先创建执行计划。"
        )
        self.assertEqual(outcome.reasoning_buf, "先比较两种授权方案的风险和成本。\n\n")
        self.assertEqual(outcome.raw_reasoning_buf, raw_reasoning)
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertNotIn("autonomous web search", emitted_reasoning)
        self.assertNotIn("强制计划模式", emitted_reasoning)

    async def test_consume_stream_round_keeps_reasoning_raw_in_protocol_but_redacts_visible_copies(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-mcp",
            task_id="task-mcp",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        alias = "mcp__Kk9Rl3y2tic_MqcwCyHgNX8oGM-5DtPdWal8L1leVU"
        reasoning = f"调用 {alias} 获取官方资料。"
        answer = f"结果来自 {alias}，下面给出结论。"
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=reasoning[:15]),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=reasoning[15:]),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=answer[:18], reasoning_content=None),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=answer[18:], reasoning_content=None),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "调用 外部工具 获取官方资料。")
        self.assertEqual(outcome.raw_reasoning_buf, reasoning)
        self.assertEqual(outcome.content_buf, "结果来自 外部工具，下面给出结论。")
        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        emitted_answer = "".join(call.args[2] for call in append_chunk.await_args_list if call.args[1] == "answering")
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertEqual(emitted_answer, outcome.content_buf)
        self.assertNotIn(alias, emitted_reasoning)
        self.assertNotIn(alias, emitted_answer)

    async def test_consume_stream_round_sanitizes_internal_tool_names_in_visible_reasoning(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-tools",
            task_id="task-tools",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="我会调用 route_"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="compare 工具，并用 local_place_",
                            ),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="search 查找地点。"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content="已通过 web_", reasoning_content=None),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content="search 和 url_", reasoning_content=None),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content="read 核对。", reasoning_content=None),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(
            outcome.reasoning_buf,
            "我会调用路线比较工具，并用地点搜索查找地点。",
        )
        self.assertEqual(
            outcome.raw_reasoning_buf,
            "我会调用 route_compare 工具，并用 local_place_search 查找地点。",
        )
        self.assertEqual(outcome.content_buf, "已通过 web_search 和 url_read 核对。")
        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        for internal_name in ("route_compare", "local_place_search"):
            self.assertNotIn(internal_name, emitted_reasoning)

    async def test_consume_stream_round_keeps_reasoning_monotonic_for_character_chunks(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-character-tools",
            task_id="task-character-tools",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        raw_reasoning = "我直接调用 route_compare 工具来获取路线信息。"
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=character),
                            finish_reason="stop" if index == len(raw_reasoning) - 1 else None,
                        )
                        for index, character in enumerate(raw_reasoning)
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(outcome.reasoning_buf, "我直接调用路线比较工具来获取路线信息。")
        self.assertEqual(outcome.raw_reasoning_buf, raw_reasoning)
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertEqual(emitted_reasoning.count("工具来获取路线信息"), 1)

    async def test_answering_rewrites_split_and_repeated_update_plan_monotonically(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-update-plan",
            task_id="task-update-plan",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="调用 up"), finish_reason=None),
                        make_chunk(
                            delta=SimpleNamespace(content="date_plan 后，再调用 update_plan 完成。"),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted = "".join(call.args[2] for call in append_chunk.await_args_list if call.args[1] == "answering")
        self.assertEqual(outcome.content_buf, "调用更新计划后，再调用更新计划完成。")
        self.assertEqual(emitted, outcome.content_buf)
        self.assertNotIn("update_plan", emitted)

    async def test_answering_buffers_trailing_space_before_character_split_update_plan(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-update-plan-characters",
            task_id="task-update-plan-characters",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()
        raw_chunks = ["调用 ", "u", *"pdate_plan 完成"]

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=chunk),
                            finish_reason="stop" if index == len(raw_chunks) - 1 else None,
                        )
                        for index, chunk in enumerate(raw_chunks)
                    ]
                ),
                request,
            )

        emitted = "".join(call.args[2] for call in append_chunk.await_args_list if call.args[1] == "answering")
        self.assertEqual(outcome.content_buf, "调用更新计划完成")
        self.assertEqual(emitted, outcome.content_buf)
        self.assertNotIn("update_plan", emitted)

    def test_final_short_update_plan_prefix_is_preserved_but_unique_long_prefix_is_productized(self):
        sanitize = llm_stream_module.sanitize_internal_mcp_aliases

        self.assertEqual(sanitize("正常结尾 ", final=True), "正常结尾 ")
        self.assertEqual(sanitize("调用 up", final=True), "调用 up")
        self.assertEqual(sanitize("调用 upd", final=True), "调用 upd")
        self.assertEqual(sanitize("调用 upda", final=True), "调用更新计划")
        self.assertEqual(
            sanitize("update_plan 与 update_plan 完成", final=True),
            "更新计划与更新计划完成",
        )

    async def test_consume_stream_round_productizes_plan_binding_names_across_answer_chunks(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-plan-binding-sanitizer",
            task_id="task-plan-binding-sanitizer",
            should_use_reasoning=False,
            thinking_block_id=None,
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()
        raw_chunks = [
            "内部字段 _plan_",
            "item_id、planned_",
            "tools 和 plan_item_",
            "id 不应展示，route_compare 可保留。",
        ]

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=chunk),
                            finish_reason="stop" if index == len(raw_chunks) - 1 else None,
                        )
                        for index, chunk in enumerate(raw_chunks)
                    ]
                ),
                request,
            )

        emitted = "".join(call.args[2] for call in append_chunk.await_args_list if call.args[1] == "answering")
        self.assertEqual(emitted, outcome.content_buf)
        self.assertNotIn("_plan_item_id", emitted)
        self.assertNotIn("planned_tools", emitted)
        self.assertNotIn("plan_item_id", emitted)
        self.assertIn("对应计划步骤", emitted)
        self.assertIn("预计使用的工具", emitted)
        self.assertIn("route_compare", emitted)

    async def test_consume_stream_round_merges_cumulative_reasoning_snapshots_once(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-reasoning-snapshot",
            task_id="task-reasoning-snapshot",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="  candidate/moonshot/KIMI-K3  ",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="先核对用户提供的出发日期"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="先核对用户提供的出发日期和目的地"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="先核对用户提供的出发日期和目的地，再查询天气",
                            ),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="先核对用户提供的出发日期和目的地，再查询天气并比较路线。",
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        expected = "先核对用户提供的出发日期和目的地，再查询天气并比较路线。"
        self.assertEqual(outcome.raw_reasoning_buf, expected)
        self.assertEqual(outcome.reasoning_buf, expected)
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)

    async def test_consume_stream_round_keeps_standard_reasoning_deltas_even_when_one_extends_previous(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-standard-reasoning-delta",
            task_id="task-standard-reasoning-delta",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="openai",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="哈"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="哈哈"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="。"),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(outcome.raw_reasoning_buf, "哈哈哈。")
        self.assertEqual(outcome.reasoning_buf, "哈哈哈。")
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)

    async def test_consume_stream_round_falls_back_to_moonshot_deltas_after_prefix_like_probe(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-moonshot-delta",
            task_id="task-moonshot-delta",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="kimi-k3",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="用户"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="用户在"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="检查行程约束"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="。"),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(outcome.raw_reasoning_buf, "用户用户在检查行程约束。")
        self.assertEqual(outcome.reasoning_buf, outcome.raw_reasoning_buf)
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)

    async def test_consume_stream_round_deduplicates_repeated_moonshot_reasoning_snapshots(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-repeated-reasoning-snapshot",
            task_id="task-repeated-reasoning-snapshot",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="kimi-k3",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="准备调用 route_compare"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="准备调用 route_compare"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="准备调用 route_compare 工具"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="准备调用 route_compare 工具。"),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        reasoning_calls = [call for call in append_chunk.await_args_list if call.args[1] == "reasoning"]
        emitted_reasoning = "".join(call.args[2] for call in reasoning_calls)
        self.assertEqual(outcome.raw_reasoning_buf, "准备调用 route_compare 工具。")
        self.assertEqual(outcome.reasoning_buf, "准备调用路线比较工具。")
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertNotIn("route_compare", emitted_reasoning)

    async def test_consume_stream_round_uses_latest_buffered_moonshot_snapshot(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-revised-reasoning-snapshot",
            task_id="task-revised-reasoning-snapshot",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="moonshot/kimi-k3",
        )
        append_chunk = AsyncMock()
        snapshots = [
            "a" * 20,
            "a" * 40 + "x",
            "a" * 40 + "yz",
            "a" * 40 + "yz!",
            "a" * 40 + "zz!",
        ]

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=snapshot),
                            finish_reason="stop" if index == len(snapshots) - 1 else None,
                        )
                        for index, snapshot in enumerate(snapshots)
                    ]
                ),
                request,
            )

        reasoning_calls = [call for call in append_chunk.await_args_list if call.args[1] == "reasoning"]
        emitted_reasoning = "".join(call.args[2] for call in reasoning_calls)
        self.assertEqual(outcome.raw_reasoning_buf, snapshots[-1])
        self.assertEqual(outcome.reasoning_buf, snapshots[-2])
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertNotIn("zz", emitted_reasoning)

    async def test_consume_stream_round_sanitizes_split_weather_tool_name_in_reasoning(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-weather-tool",
            task_id="task-weather-tool",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="需要调用"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="weather"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="_"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="fore"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="cast"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="工具查询天气。"),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        emitted_reasoning = "".join(
            call.args[2] for call in append_chunk.await_args_list if call.args[1] == "reasoning"
        )
        self.assertEqual(outcome.reasoning_buf, "需要调用天气查询工具查询天气。")
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertNotIn("weather_forecast", emitted_reasoning)

    async def test_non_k3_moonshot_model_never_enters_cumulative_snapshot_probe(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-moonshot-non-k3",
            task_id="task-moonshot-non-k3",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="moonshot",
            model_id="moonshot/kimi-k2.7-code",
        )
        chunks = ["a" * 20, "a" * 24, "a" * 28, "a" * 32]

        with (
            patch("app.services.stream.llm_stream.append_chunk", AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=chunk),
                            finish_reason="stop" if index == len(chunks) - 1 else None,
                        )
                        for index, chunk in enumerate(chunks)
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.raw_reasoning_buf, "".join(chunks))
        self.assertEqual(outcome.reasoning_buf, outcome.raw_reasoning_buf)

    async def test_explicit_snapshot_transport_mode_can_enable_non_k3_snapshot_stream(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-explicit-snapshot",
            task_id="task-explicit-snapshot",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="custom",
            model_id="custom/reasoning-model",
            reasoning_transport_mode="snapshot",
        )
        snapshots = ["先分析需求", "先分析需求并核对约束", "先分析需求并核对约束。"]

        with (
            patch("app.services.stream.llm_stream.append_chunk", AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content=snapshot),
                            finish_reason="stop" if index == len(snapshots) - 1 else None,
                        )
                        for index, snapshot in enumerate(snapshots)
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.raw_reasoning_buf, snapshots[-1])
        self.assertEqual(outcome.reasoning_buf, snapshots[-1])

    async def test_consume_stream_round_hides_mixed_prefix_dsml_without_executing_ambiguous_protocol(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            step_id="step-2",
        )
        raw_content = (
            "我再查一下。"
            '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="web_search">'
            '<｜｜DSML｜｜parameter name="query" string="true">北京天气</｜｜DSML｜｜parameter>'
            "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content=raw_content[:10]), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content=raw_content[10:]), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "我再查一下。")
        self.assertEqual(outcome.tool_calls, [])
        self.assertEqual(outcome.finish_reason, "tool_protocol_error")
        emitted = "".join(call.args[2] for call in append_chunk.await_args_list)
        self.assertEqual(emitted, "我再查一下。")
        self.assertNotIn("DSML", emitted)

    async def test_consume_stream_round_rejects_malformed_dsml_without_leaking_or_partial_execution(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            step_id="step-2",
        )
        malformed = (
            '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="web_search">'
            '<｜｜DSML｜｜parameter name="query" string="true">北京天气</｜｜DSML｜｜parameter>'
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response([make_chunk(delta=SimpleNamespace(content=malformed), finish_reason="stop")]),
                request,
            )

        self.assertEqual(outcome.content_buf, "")
        self.assertEqual(outcome.tool_calls, [])
        self.assertEqual(outcome.finish_reason, "tool_protocol_error")
        append_chunk.assert_not_awaited()

    async def test_consume_stream_round_rejects_mimo_xml_tool_protocol_without_leaking(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-mimo",
            task_id="task-mimo",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            step_id="step-mimo",
        )
        protocol = (
            "<tool_call> <function=web_search> "
            "<parameter=query>React 19 稳定版 2026 生产环境 "
            "<parameter=intent>freshness "
            "<parameter=recency_days>90 "
            "<parameter=对应计划步骤>2 </tool_call>"
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content=protocol[:7]), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content=protocol[7:]), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "")
        self.assertEqual(outcome.tool_calls, [])
        self.assertEqual(outcome.finish_reason, "tool_protocol_error")
        append_chunk.assert_not_awaited()

    async def test_consume_stream_round_holds_complete_mimo_protocol_prefix_across_chunks(self):
        protocol_cases = (
            (
                "<tool_call><function=web_search><parameter=query>React 19</tool_call>",
                len("<tool_call"),
            ),
            (
                "<tool_call ><function=web_search><parameter=query>React 19</tool_call>",
                len("<tool_call "),
            ),
            (
                "<tool_call id=next><function=web_search><parameter=query>React 19</tool_call>",
                len("<tool_call id=next"),
            ),
        )

        for case_index, (protocol, split_at) in enumerate(protocol_cases, 1):
            with self.subTest(protocol=protocol):
                request = llm_stream_module.LLMStreamRequest(
                    conversation_id=f"conv-mimo-{case_index}",
                    task_id=f"task-mimo-{case_index}",
                    should_use_reasoning=False,
                    thinking_block_id="blk-thinking",
                    text_block_id="blk-text",
                    step_id=f"step-mimo-{case_index}",
                )
                append_chunk = AsyncMock()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
                ):
                    outcome = await llm_stream_module.consume_stream_round(
                        async_response(
                            [
                                make_chunk(delta=SimpleNamespace(content=protocol[:split_at]), finish_reason=None),
                                make_chunk(delta=SimpleNamespace(content=protocol[split_at:]), finish_reason="stop"),
                            ]
                        ),
                        request,
                    )

                self.assertEqual(outcome.content_buf, "")
                self.assertEqual(outcome.finish_reason, "tool_protocol_error")
                append_chunk.assert_not_awaited()

    async def test_consume_stream_round_keeps_literal_tool_call_tags_in_code_text(self):
        literal_cases = (
            "解释 `<tool_call>` 标签与普通 XML 标签的区别。",
            "示例代码：\n```xml\n<tool_call><function=demo></tool_call>\n```",
            "`<tool_call>`",
            "```xml\n<tool_call>\n```",
        )

        for case_index, literal in enumerate(literal_cases, 1):
            with self.subTest(literal=literal):
                request = llm_stream_module.LLMStreamRequest(
                    conversation_id=f"conv-literal-{case_index}",
                    task_id=f"task-literal-{case_index}",
                    should_use_reasoning=False,
                    thinking_block_id="blk-thinking",
                    text_block_id="blk-text",
                    step_id=f"step-literal-{case_index}",
                )
                append_chunk = AsyncMock()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
                ):
                    outcome = await llm_stream_module.consume_stream_round(
                        async_response([make_chunk(delta=SimpleNamespace(content=literal), finish_reason="stop")]),
                        request,
                    )

                self.assertEqual(outcome.content_buf, literal)
                self.assertEqual(outcome.finish_reason, "stop")
                self.assertEqual("".join(call.args[2] for call in append_chunk.await_args_list), literal)

    async def test_consume_stream_round_rejects_final_bare_tool_call_tag_outside_code(self):
        cases = (
            ("<tool_call>", ""),
            ("先核对资料。<tool_call>", "先核对资料。"),
        )

        for case_index, (raw_content, visible_content) in enumerate(cases, 1):
            with self.subTest(raw_content=raw_content):
                request = llm_stream_module.LLMStreamRequest(
                    conversation_id=f"conv-bare-tool-call-{case_index}",
                    task_id=f"task-bare-tool-call-{case_index}",
                    should_use_reasoning=False,
                    thinking_block_id="blk-thinking",
                    text_block_id="blk-text",
                    step_id=f"step-bare-tool-call-{case_index}",
                )
                append_chunk = AsyncMock()
                with (
                    patch("app.services.stream.llm_stream.append_chunk", append_chunk),
                    patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
                ):
                    outcome = await llm_stream_module.consume_stream_round(
                        async_response([make_chunk(delta=SimpleNamespace(content=raw_content), finish_reason="stop")]),
                        request,
                    )

                self.assertEqual(outcome.content_buf, visible_content)
                self.assertEqual(outcome.tool_calls, [])
                self.assertEqual(outcome.finish_reason, "tool_protocol_error")
                self.assertEqual(
                    "".join(call.args[2] for call in append_chunk.await_args_list),
                    visible_content,
                )

    async def test_consume_stream_round_prefers_native_tool_calls_without_leaking_parallel_dsml_content(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            step_id="step-2",
        )
        dsml = (
            '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="web_search">'
            '<｜｜DSML｜｜parameter name="query" string="true">重复</｜｜DSML｜｜parameter>'
            "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
        )
        native_tool_call = {"id": "native-1", "name": "web_search", "arguments": '{"query":"北京"}'}
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content=dsml), finish_reason=None),
                        make_chunk(
                            delta=make_tool_delta(
                                tool_call_id=native_tool_call["id"],
                                name=native_tool_call["name"],
                                arguments=native_tool_call["arguments"],
                            ),
                            finish_reason="tool_calls",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "")
        self.assertEqual(outcome.tool_calls, [native_tool_call])
        self.assertEqual(outcome.finish_reason, "tool_calls")
        append_chunk.assert_not_awaited()

    def test_parse_dsml_tool_calls_rejects_unconsumed_protocol_garbage(self):
        malformed = (
            "<｜｜DSML｜｜tool_calls>garbage"
            '<｜｜DSML｜｜invoke name="web_search">'
            '<｜｜DSML｜｜parameter name="query" string="true">北京天气</｜｜DSML｜｜parameter>'
            "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"
        )

        self.assertEqual(llm_stream_module.parse_dsml_tool_calls(malformed, id_prefix="step-1"), [])

    async def test_consume_stream_round_converts_split_dsml_tool_protocol_without_leaking_answer_chunks(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            run_id="run-1",
            step_id="step-4",
        )
        raw_tool_protocol = (
            "<｜｜DSML｜｜tool_calls>"
            '<｜｜DSML｜｜invoke name="web_search">'
            '<｜｜DSML｜｜parameter name="query" string="true">北京周末天气</｜｜DSML｜｜parameter>'
            '<｜｜DSML｜｜parameter name="recency_days" string="false">60</｜｜DSML｜｜parameter>'
            "</｜｜DSML｜｜invoke>"
            '<｜｜DSML｜｜invoke name="url_read">'
            '<｜｜DSML｜｜parameter name="url" string="true">https://example.com/a</｜｜DSML｜｜parameter>'
            "</｜｜DSML｜｜invoke>"
            "</｜｜DSML｜｜tool_calls>"
        )
        chunks = [raw_tool_protocol[:4], raw_tool_protocol[4:29], raw_tool_protocol[29:103], raw_tool_protocol[103:]]
        response = [
            make_chunk(delta=SimpleNamespace(content=chunk), finish_reason="stop" if index == 3 else None)
            for index, chunk in enumerate(chunks)
        ]
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(async_response(response), request)

        self.assertEqual(outcome.content_buf, "")
        self.assertEqual(outcome.finish_reason, "tool_calls")
        self.assertEqual(
            outcome.tool_calls,
            [
                {
                    "id": "dsml-step-4-1",
                    "name": "web_search",
                    "arguments": '{"query": "北京周末天气", "recency_days": 60}',
                },
                {
                    "id": "dsml-step-4-2",
                    "name": "url_read",
                    "arguments": '{"url": "https://example.com/a"}',
                },
            ],
        )
        append_chunk.assert_not_awaited()

    async def test_consume_stream_round_keeps_normal_angle_bracket_content_visible(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="<"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="普通正文>"), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "<普通正文>")
        self.assertEqual(outcome.finish_reason, "stop")
        self.assertEqual(
            "".join(call.args[2] for call in append_chunk.await_args_list),
            "<普通正文>",
        )

    async def test_consume_stream_round_accumulates_usage_content_reasoning_and_tool_calls(self):
        request_cls = getattr(llm_stream_module, "LLMStreamRequest")
        consume_stream_round = getattr(llm_stream_module, "consume_stream_round")
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
        response = [
            make_chunk(delta=SimpleNamespace(content=None, reasoning_content="想一想"), finish_reason=None),
            make_chunk(delta=SimpleNamespace(content="答案", reasoning_content=None), finish_reason="stop"),
            make_chunk(delta=make_tool_delta(index=1, tool_call_id="tc-2", name="url_read", arguments='{"url"')),
            make_chunk(delta=make_tool_delta(index=1, arguments=':"https://example.com"}'), finish_reason="tool_calls"),
            make_chunk(
                delta=make_tool_delta(index=0, tool_call_id="tc-1", name="web_search", arguments='{"query":"x"}')
            ),
            make_chunk(choices=False, usage=usage),
        ]
        append_calls = []

        async def append_chunk(*args, **kwargs):
            append_calls.append((args, kwargs))

        request = request_cls(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            run_id="run-1",
            step_id="step-1",
        )
        check_lock_owner = AsyncMock(return_value=True)

        with (
            patch("app.services.stream.llm_stream.append_chunk", side_effect=append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", check_lock_owner),
        ):
            outcome = await consume_stream_round(async_response(response), request)

        self.assertEqual(outcome.reasoning_buf, "想一想")
        self.assertEqual(outcome.content_buf, "答案")
        self.assertEqual(outcome.finish_reason, "tool_calls")
        self.assertEqual(outcome.usage_data, Usage(input_tokens=5, output_tokens=7))
        self.assertEqual(
            outcome.tool_calls,
            [
                {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'},
                {"id": "tc-2", "name": "url_read", "arguments": '{"url":"https://example.com"}'},
            ],
        )
        self.assertEqual(
            append_calls,
            [
                (
                    ("conv-1", "reasoning", "想一想", "blk-thinking"),
                    {"task_id": "task-1", "run_id": "run-1", "step_id": "step-1"},
                ),
                (
                    ("conv-1", "answering", "答案", "blk-text"),
                    {"task_id": "task-1", "run_id": "run-1", "step_id": "step-1"},
                ),
            ],
        )
        check_lock_owner.assert_not_awaited()

    async def test_consume_stream_round_uses_model_extra_reasoning_and_dedupes_content(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        delta = SimpleNamespace(content="推理", model_extra={"reasoning_content": "推理"})
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response([make_chunk(delta=delta, finish_reason="stop")]),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "推理")
        self.assertEqual(outcome.content_buf, "")
        self.assertEqual(outcome.finish_reason, "stop")
        append_chunk.assert_awaited_once_with(
            "conv-1",
            "reasoning",
            "推理",
            "blk-thinking",
            task_id="task-1",
            run_id=None,
            step_id=None,
        )

    async def test_consume_stream_round_sanitizes_reasoning_alias_and_dedupes_mirrored_answer(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-mcp-mirrored",
            task_id="task-mcp-mirrored",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        alias = "mcp__Kk9Rl3y2tic_MqcwCyHgNX8oGM-5DtPdWal8L1leVU"
        mirrored = f"调用 {alias} 获取资料"
        delta = SimpleNamespace(content=mirrored, reasoning_content=mirrored)
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response([make_chunk(delta=delta, finish_reason="stop")]),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "调用 外部工具 获取资料")
        self.assertEqual(outcome.raw_reasoning_buf, mirrored)
        self.assertEqual(outcome.content_buf, "")
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[1], "reasoning")

    async def test_consume_stream_round_strips_reasoning_tags_from_answering_content(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [make_chunk(delta=SimpleNamespace(content="<think>内部思考</think>可见正文"), finish_reason="stop")]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "可见正文")
        append_chunk.assert_awaited_once_with(
            "conv-1",
            "answering",
            "可见正文",
            "blk-text",
            task_id="task-1",
            run_id=None,
            step_id=None,
        )

    async def test_consume_stream_round_strips_split_reasoning_tags_before_emitting(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="<thi"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="nk>内部"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="思考</thi"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="nk>可见正文"), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.content_buf, "可见正文")
        append_chunk.assert_awaited_once_with(
            "conv-1",
            "answering",
            "可见正文",
            "blk-text",
            task_id="task-1",
            run_id=None,
            step_id=None,
        )

    async def test_minimax_split_think_tags_stream_reasoning_and_keep_answer_deferred(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-minimax",
            task_id="task-minimax",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="minimax",
            model_id="MiniMax-M2.7",
            defer_output=True,
            allow_deferred_reasoning_output=True,
        )
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response(
                    [
                        make_chunk(delta=SimpleNamespace(content="<thi"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="nk>先比较"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="两边。</thi"), finish_reason=None),
                        make_chunk(delta=SimpleNamespace(content="nk>结论：324 更大。"), finish_reason="stop"),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "先比较两边。")
        self.assertEqual(outcome.raw_reasoning_buf, "先比较两边。")
        self.assertEqual(outcome.content_buf, "结论：324 更大。")
        self.assertEqual(
            [(call.args[1], call.args[2]) for call in append_chunk.await_args_list],
            [("reasoning", "先比较"), ("reasoning", "两边。")],
        )

    async def test_minimax_think_tag_raw_content_is_kept_for_tool_protocol_replay(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-minimax-tool",
            task_id="task-minimax-tool",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="minimax",
            model_id="MiniMax-M2.7",
            defer_output=True,
            allow_deferred_reasoning_output=True,
        )
        raw_content = "<think>需要查询天气。</think>"

        with (
            patch("app.services.stream.llm_stream.append_chunk", AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response([make_chunk(delta=SimpleNamespace(content=raw_content), finish_reason="stop")]),
                request,
            )

        self.assertEqual(outcome.raw_content_buf, raw_content)

    async def test_reasoning_is_not_dropped_when_provider_combines_it_with_tool_call_delta(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-combined-tool",
            task_id="task-combined-tool",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
            provider="minimax",
            model_id="MiniMax-M2.7",
            defer_output=True,
            allow_deferred_reasoning_output=True,
        )
        delta = make_tool_delta(
            tool_call_id="tc-weather",
            name="get_weather",
            arguments='{"city":"北京"}',
        )
        delta.content = "<think>先查询实时天气。</think>"
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            outcome = await llm_stream_module.consume_stream_round(
                async_response([make_chunk(delta=delta, finish_reason="tool_calls")]),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "先查询实时天气。")
        self.assertEqual(outcome.raw_content_buf, "<think>先查询实时天气。</think>")
        self.assertEqual(outcome.tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            [(call.args[1], call.args[2]) for call in append_chunk.await_args_list],
            [("reasoning", "先查询实时天气。")],
        )

    async def test_consume_stream_round_cancels_when_lock_owner_is_lost_after_interval(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        response = [
            make_chunk(delta=SimpleNamespace(content=str(index)), finish_reason=None)
            for index in range(llm_stream_module.LOCK_CHECK_INTERVAL)
        ]
        append_chunk = AsyncMock()
        check_lock_owner = AsyncMock(return_value=False)

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", check_lock_owner),
        ):
            outcome = await llm_stream_module.consume_stream_round(async_response(response), request)

        self.assertEqual(
            outcome.content_buf, "".join(str(index) for index in range(llm_stream_module.LOCK_CHECK_INTERVAL))
        )
        self.assertEqual(outcome.finish_reason, "cancelled")
        self.assertEqual(append_chunk.await_count, llm_stream_module.LOCK_CHECK_INTERVAL)
        check_lock_owner.assert_awaited_once_with("conv-1", "task-1")

    async def test_consume_stream_round_does_not_count_usage_only_chunks_for_lock_checks(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-1",
            task_id="task-1",
            should_use_reasoning=False,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4)
        response = [
            *[
                make_chunk(delta=SimpleNamespace(content="x"), finish_reason=None)
                for _ in range(llm_stream_module.LOCK_CHECK_INTERVAL - 1)
            ],
            make_chunk(choices=False, usage=usage),
            make_chunk(delta=SimpleNamespace(content="y"), finish_reason=None),
        ]
        append_chunk = AsyncMock()
        check_lock_owner = AsyncMock(return_value=False)

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", check_lock_owner),
        ):
            outcome = await llm_stream_module.consume_stream_round(async_response(response), request)

        self.assertEqual(outcome.content_buf, "x" * (llm_stream_module.LOCK_CHECK_INTERVAL - 1) + "y")
        self.assertEqual(outcome.usage_data, Usage(input_tokens=3, output_tokens=4))
        self.assertEqual(outcome.finish_reason, "cancelled")
        self.assertEqual(append_chunk.await_count, llm_stream_module.LOCK_CHECK_INTERVAL)
        check_lock_owner.assert_awaited_once_with("conv-1", "task-1")

    async def test_stream_round_keeps_legacy_tuple_contract(self):
        append_chunk = AsyncMock()

        with (
            patch("app.services.stream.llm_stream.append_chunk", append_chunk),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                async_response([make_chunk(delta=SimpleNamespace(content="答案"), finish_reason="stop")]),
                "conv-1",
                "task-1",
                False,
                "blk-thinking",
                "blk-text",
                run_id="run-1",
                step_id="step-1",
            )

        self.assertEqual(result, ("", "答案", [], "stop", None))
        append_chunk.assert_awaited_once_with(
            "conv-1",
            "answering",
            "答案",
            "blk-text",
            task_id="task-1",
            run_id="run-1",
            step_id="step-1",
        )

    async def test_stream_round_exposes_raw_reasoning_without_expanding_legacy_tuple(self):
        with (
            patch("app.services.stream.llm_stream.append_chunk", AsyncMock()),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
        ):
            result = await llm_stream_module.stream_round(
                async_response(
                    [
                        make_chunk(
                            delta=SimpleNamespace(content=None, reasoning_content="调用 route_compare"),
                            finish_reason=None,
                        ),
                        make_chunk(delta=SimpleNamespace(content="答案"), finish_reason="stop"),
                    ]
                ),
                "conv-1",
                "task-1",
                True,
                "blk-thinking",
                "blk-text",
            )

        self.assertEqual(result, ("调用路线比较", "答案", [], "stop", None))
        self.assertEqual(len(result), 5)
        self.assertEqual(result.protocol_reasoning_buf, "调用 route_compare")

    async def test_llm_call_with_retry_attaches_chat_stream_tags(self):
        response = object()
        with patch("app.services.stream.llm_stream.litellm.acompletion", new=AsyncMock(return_value=response)) as call:
            result = await llm_stream_module.llm_call_with_retry(
                "openai/deepseek-chat",
                {"api_key": "test-key"},
                [{"role": "user", "content": "hello"}],
            )

        self.assertIs(result, response)
        self.assertEqual(
            call.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:chat_stream"]}},
        )

    async def test_visible_output_callback_runs_after_redis_append_for_each_kind(self):
        events = []

        async def append_chunk(*args, **_kwargs):
            events.append(("append", args[1]))
            return "entry-id"

        async def on_visible_output(kind):
            events.append(("first", kind))

        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv",
            task_id="task",
            should_use_reasoning=True,
            thinking_block_id="thinking",
            text_block_id="text",
            on_visible_output=on_visible_output,
        )
        state = llm_stream_module.LLMStreamState()

        with patch("app.services.stream.llm_stream.append_chunk", new=append_chunk):
            await llm_stream_module.append_reasoning_and_content(
                request=request,
                state=state,
                reasoning_delta="推理",
                content_delta="正文",
            )

        self.assertEqual(
            events,
            [
                ("append", "reasoning"),
                ("first", "reasoning"),
                ("append", "answering"),
                ("first", "content"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
