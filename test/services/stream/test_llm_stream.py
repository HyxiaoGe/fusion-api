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
    async def test_consume_stream_round_defers_all_model_output_for_product_result_guard(self):
        request = llm_stream_module.LLMStreamRequest(
            conversation_id="conv-product",
            task_id="task-product",
            should_use_reasoning=True,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
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
                            delta=SimpleNamespace(content=None, reasoning_content="核对路线事实"),
                            finish_reason=None,
                        ),
                        make_chunk(
                            delta=SimpleNamespace(content="方便停车，驾车更合适。", reasoning_content=None),
                            finish_reason="stop",
                        ),
                    ]
                ),
                request,
            )

        self.assertEqual(outcome.reasoning_buf, "核对路线事实")
        self.assertEqual(outcome.content_buf, "方便停车，驾车更合适。")
        append_chunk.assert_not_awaited()

    async def test_consume_stream_round_redacts_split_internal_mcp_alias_from_reasoning_and_answer(self):
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
        self.assertEqual(outcome.content_buf, "结果来自 外部工具，下面给出结论。")
        emitted = "".join(call.args[2] for call in append_chunk.await_args_list)
        self.assertNotIn(alias, emitted)
        self.assertNotIn("mcp_", emitted)
        self.assertEqual(emitted, "调用 外部工具 获取官方资料。结果来自 外部工具，下面给出结论。")

    async def test_consume_stream_round_rewrites_split_internal_tool_names_as_product_language(self):
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

        self.assertEqual(outcome.reasoning_buf, "我会调用路线比较工具，并用地点搜索查找地点。")
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
        self.assertEqual(emitted_reasoning, outcome.reasoning_buf)
        self.assertEqual(emitted_reasoning.count("工具来获取路线信息"), 1)

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

    async def test_consume_stream_round_dedupes_mirrored_reasoning_before_mcp_alias_redaction(self):
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


if __name__ == "__main__":
    unittest.main()
