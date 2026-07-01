import unittest
from unittest.mock import AsyncMock, Mock

from app.schemas.chat import SearchSource, TextBlock
from app.services.source_evidence_ledger import stable_web_evidence_id
from app.services.stream import tool_round as tool_round_module
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_round import (
    ToolRoundOutcome,
    build_assistant_tool_message,
    handle_tool_calls_round,
    restore_reasoning_after_tool_decision,
)
from app.services.tool_handlers.base import ToolResult


class ToolRoundTests(unittest.IsolatedAsyncioTestCase):
    def test_build_assistant_tool_message_preserves_tool_calls_and_reasoning(self):
        tool_calls = [
            {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'},
            {"id": "tc-2", "name": "url_read", "arguments": '{"url":"https://example.com"}'},
        ]

        message = build_assistant_tool_message(
            tool_calls=tool_calls,
            reasoning_buf="需要搜索",
            should_use_reasoning=True,
        )

        self.assertEqual(
            message,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                    },
                    {
                        "id": "tc-2",
                        "type": "function",
                        "function": {"name": "url_read", "arguments": '{"url":"https://example.com"}'},
                    },
                ],
                "reasoning_content": "需要搜索",
            },
        )

        no_reasoning = build_assistant_tool_message(
            tool_calls=tool_calls,
            reasoning_buf="需要搜索",
            should_use_reasoning=False,
        )
        self.assertNotIn("reasoning_content", no_reasoning)

        empty_reasoning = build_assistant_tool_message(
            tool_calls=tool_calls,
            reasoning_buf="",
            should_use_reasoning=True,
        )
        self.assertNotIn("reasoning_content", empty_reasoning)

    def test_restore_reasoning_after_tool_decision_removes_disabled_extra_body(self):
        call_kwargs = {
            "tools": [{"function": {"name": "web_search"}}],
            "extra_body": {"thinking": {"type": "disabled"}},
        }

        restore_reasoning_after_tool_decision(call_kwargs)

        self.assertNotIn("extra_body", call_kwargs)

    async def test_handle_tool_calls_round_accepts_request_and_preserves_sequence(self):
        request_cls = getattr(tool_round_module, "ToolRoundRequest")
        tool_call = {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'}
        content_block = TextBlock(type="text", id="blk_tool", text="工具摘要")
        handler = Mock()
        handler.format_llm_context.return_value = "LLM 可见工具上下文"
        handler.build_content_block.return_value = content_block
        record = ToolExecutionRecord(
            tool_call=tool_call,
            result=ToolResult(status="success", duration_ms=12),
            handler=handler,
            block_id="blk_tool",
            log_id="log-1",
        )
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        call_kwargs = {"extra_body": {"thinking": {"type": "disabled"}}}
        emitter = object()
        session_cache = object()
        network_budget = object()
        step_context = AgentStepContext(
            step_id="step-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )
        order = []

        def persist_message_fn(db, msg_id, conv_id, model_id, blocks, usage_data=None, partial=False):
            order.append(("persist", partial, [getattr(block, "id", None) for block in blocks], usage_data))

        async def execute_tools_fn(tool_calls, conversation_id, user_id, model_id, provider, **kwargs):
            order.append(("execute", tool_calls, conversation_id, user_id, model_id, provider, kwargs))
            return [record]

        async def complete_step_fn(
            *,
            context,
            emitter,
            session_cache,
            tool_names,
            tool_call_count,
            completed_tool_calls,
            max_tool_calls,
            clock,
        ):
            order.append(
                (
                    "complete",
                    tool_names,
                    tool_call_count,
                    completed_tool_calls,
                    max_tool_calls,
                    clock(),
                    "extra_body" in call_kwargs,
                )
            )

        def on_tools_executed(tool_call_count):
            order.append(("record", tool_call_count))

        request = request_cls(
            db="db",
            assistant_message_id="msg-1",
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            content_blocks=content_blocks,
            messages=messages,
            tool_calls=[tool_call],
            reasoning_buf="需要搜索",
            should_use_reasoning=True,
            step_context=step_context,
            step_number=3,
            run_id="run-1",
            emitter=emitter,
            session_cache=session_cache,
            network_budget=network_budget,
            call_kwargs=call_kwargs,
            persist_message_fn=persist_message_fn,
            execute_tools_fn=execute_tools_fn,
            complete_step_fn=complete_step_fn,
            on_tools_executed=on_tools_executed,
            clock=Mock(return_value=10.5),
        )

        outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(outcome, ToolRoundOutcome(tool_call_count=1, tool_names=["web_search"]))
        self.assertEqual([entry[0] for entry in order], ["persist", "execute", "record", "persist", "complete"])
        self.assertEqual(order[0], ("persist", True, ["blk_thinking"], None))
        self.assertEqual(order[2], ("record", 1))
        self.assertEqual(order[3], ("persist", True, ["blk_thinking", "blk_tool"], None))
        self.assertEqual(order[4], ("complete", ["web_search"], 1, None, None, 10.5, True))
        self.assertNotIn("extra_body", call_kwargs)
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "tc-1", "content": "LLM 可见工具上下文"})

    async def test_handle_tool_calls_round_marks_plan_running_before_execute_tools(self):
        tool_call = {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'}
        handler = Mock()
        handler.format_llm_context.return_value = "LLM 可见工具上下文"
        handler.build_content_block.return_value = None
        record = ToolExecutionRecord(
            tool_call=tool_call,
            result=ToolResult(status="success", duration_ms=12),
            handler=handler,
            block_id="blk_tool",
            log_id="log-1",
        )
        order = []
        original_mark_tool_round_started = getattr(tool_round_module, "mark_tool_round_started", None)

        async def mark_tool_round_started(**kwargs):
            order.append(
                (
                    "plan",
                    kwargs["context"].run_id,
                    kwargs["tool_call_count"],
                    kwargs["tool_names"],
                    kwargs["tool_arguments"],
                    kwargs["completed_tool_calls"],
                    kwargs["max_tool_calls"],
                )
            )

        async def execute_tools_fn(*args, **kwargs):
            order.append(("execute", kwargs["trace_id"], kwargs["network_budget"].max_tool_calls))
            return [record]

        async def complete_step_fn(**kwargs):
            order.append(("complete", kwargs["tool_names"], kwargs["tool_call_count"]))

        tool_round_module.mark_tool_round_started = mark_tool_round_started
        try:
            await handle_tool_calls_round(
                request=tool_round_module.ToolRoundRequest(
                    db="db",
                    assistant_message_id="msg-1",
                    conversation_id="conv-1",
                    user_id="user-1",
                    model_id="gpt-4",
                    provider="openai",
                    content_blocks=[],
                    messages=[{"role": "user", "content": "hi"}],
                    tool_calls=[tool_call],
                    reasoning_buf="",
                    should_use_reasoning=True,
                    step_context=AgentStepContext(
                        step_id="step-1",
                        run_id="run-1",
                        step_number=1,
                        started_at=10.0,
                        thinking_block_id="blk_thinking",
                        text_block_id="blk_text",
                    ),
                    step_number=1,
                    run_id="run-1",
                    emitter=object(),
                    session_cache=object(),
                    network_budget=Mock(max_tool_calls=20, completed_tool_calls=0),
                    call_kwargs={},
                    persist_message_fn=Mock(),
                    execute_tools_fn=execute_tools_fn,
                    complete_step_fn=complete_step_fn,
                )
            )
        finally:
            if original_mark_tool_round_started is None:
                delattr(tool_round_module, "mark_tool_round_started")
            else:
                tool_round_module.mark_tool_round_started = original_mark_tool_round_started

        self.assertEqual(
            order,
            [
                ("plan", "run-1", 1, ["web_search"], [{"query": "x"}], 0, 20),
                ("execute", "run-1", 20),
                ("complete", ["web_search"], 1),
            ],
        )

    async def test_handle_tool_calls_round_preserves_order_and_mutates_state(self):
        tool_call = {"id": "tc-1", "name": "web_search", "arguments": '{"query":"x"}'}
        content_block = TextBlock(type="text", id="blk_tool", text="工具摘要")
        handler = Mock()
        handler.format_llm_context.return_value = "LLM 可见工具上下文"
        handler.build_content_block.return_value = content_block
        record = ToolExecutionRecord(
            tool_call=tool_call,
            result=ToolResult(status="success", duration_ms=12),
            handler=handler,
            block_id="blk_tool",
            log_id="log-1",
        )
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        call_kwargs = {"extra_body": {"thinking": {"type": "disabled"}}}
        emitter = object()
        session_cache = object()
        network_budget = object()
        step_context = AgentStepContext(
            step_id="step-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )
        order = []

        def persist_message_fn(db, msg_id, conv_id, model_id, blocks, usage_data=None, partial=False):
            order.append(
                (
                    "persist",
                    msg_id,
                    conv_id,
                    model_id,
                    partial,
                    [getattr(block, "id", None) for block in blocks],
                    usage_data,
                )
            )

        async def execute_tools_fn(tool_calls, conversation_id, user_id, model_id, provider, **kwargs):
            order.append(
                (
                    "execute",
                    tool_calls,
                    conversation_id,
                    user_id,
                    model_id,
                    provider,
                    kwargs,
                )
            )
            return [record]

        async def complete_step_fn(
            *,
            context,
            emitter,
            session_cache,
            tool_names,
            tool_call_count,
            completed_tool_calls,
            max_tool_calls,
            clock,
        ):
            order.append(
                (
                    "complete",
                    context,
                    emitter,
                    session_cache,
                    tool_names,
                    tool_call_count,
                    completed_tool_calls,
                    max_tool_calls,
                    clock(),
                    "extra_body" in call_kwargs,
                )
            )

        def on_tools_executed(tool_call_count):
            order.append(("record", tool_call_count))

        outcome = await handle_tool_calls_round(
            request=tool_round_module.ToolRoundRequest(
                db="db",
                assistant_message_id="msg-1",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                content_blocks=content_blocks,
                messages=messages,
                tool_calls=[tool_call],
                reasoning_buf="需要搜索",
                should_use_reasoning=True,
                step_context=step_context,
                step_number=3,
                run_id="run-1",
                emitter=emitter,
                session_cache=session_cache,
                network_budget=network_budget,
                call_kwargs=call_kwargs,
                persist_message_fn=persist_message_fn,
                execute_tools_fn=execute_tools_fn,
                complete_step_fn=complete_step_fn,
                on_tools_executed=on_tools_executed,
                clock=Mock(return_value=10.5),
            ),
        )

        self.assertEqual(outcome, ToolRoundOutcome(tool_call_count=1, tool_names=["web_search"]))
        self.assertEqual([entry[0] for entry in order], ["persist", "execute", "record", "persist", "complete"])
        self.assertEqual(order[0], ("persist", "msg-1", "conv-1", "gpt-4", True, ["blk_thinking"], None))
        self.assertEqual(order[2], ("record", 1))
        self.assertEqual(order[3], ("persist", "msg-1", "conv-1", "gpt-4", True, ["blk_thinking", "blk_tool"], None))
        execute_kwargs = order[1][6]
        self.assertEqual(order[1][:6], ("execute", [tool_call], "conv-1", "user-1", "gpt-4", "openai"))
        self.assertEqual(execute_kwargs["message_id"], "msg-1")
        self.assertEqual(execute_kwargs["trace_id"], "run-1")
        self.assertEqual(execute_kwargs["step_number"], 3)
        self.assertIs(execute_kwargs["emitter"], emitter)
        self.assertIs(execute_kwargs["network_budget"], network_budget)
        self.assertEqual(
            order[4],
            ("complete", step_context, emitter, session_cache, ["web_search"], 1, None, None, 10.5, True),
        )
        self.assertNotIn("extra_body", call_kwargs)

        self.assertEqual(len(content_blocks), 2)
        self.assertEqual(content_blocks[0].type, "thinking")
        self.assertEqual(content_blocks[0].id, "blk_thinking")
        self.assertEqual(content_blocks[0].thinking, "需要搜索")
        self.assertIs(content_blocks[1], content_block)
        self.assertEqual(
            messages[1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"x"}'},
                    }
                ],
                "reasoning_content": "需要搜索",
            },
        )
        self.assertEqual(
            messages[2],
            {"role": "tool", "tool_call_id": "tc-1", "content": "LLM 可见工具上下文"},
        )
        handler.format_llm_context.assert_called_once_with(record.result)
        handler.build_content_block.assert_called_once_with(record.result, "blk_tool", "log-1")

    def test_append_tool_round_messages_adds_round_level_source_selection_guidance(self):
        tool_call_1 = {"id": "tc-search-1", "name": "web_search", "arguments": '{"query":"news"}'}
        tool_call_2 = {"id": "tc-search-2", "name": "web_search", "arguments": '{"query":"official"}'}
        handler_1 = Mock()
        handler_1.format_llm_context.return_value = "第一个搜索上下文"
        handler_1.build_content_block.return_value = None
        handler_2 = Mock()
        handler_2.format_llm_context.return_value = "第二个搜索上下文"
        handler_2.build_content_block.return_value = None

        record_1 = ToolExecutionRecord(
            tool_call=tool_call_1,
            result=ToolResult(
                status="success",
                data={
                    "query": "OpenAI GPT-5.6 Sol 发布 2026年6月 新闻",
                    "intent": "freshness",
                    "search_budget": "freshness",
                    "sources": [
                        SearchSource(
                            title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                            url="https://openai.com/index/previewing-gpt-5-6-sol",
                            description="OpenAI official announcement.",
                        ),
                        SearchSource(
                            title="不過，這次GPT-5.6 並未全面開放。應美國政府要求，OpenAI 先以 ...",
                            url="https://threads.com/@kufutw/post/DaJ2_DLD8cS",
                            description="Social repost.",
                        ),
                    ],
                },
            ),
            handler=handler_1,
            block_id="blk-search-1",
            log_id="log-search-1",
        )
        record_2 = ToolExecutionRecord(
            tool_call=tool_call_2,
            result=ToolResult(
                status="success",
                data={
                    "query": "OpenAI GPT-5.6 Sol 2026年6月 官方公告",
                    "intent": "official_source",
                    "search_budget": "official_source",
                    "sources": [
                        SearchSource(
                            title="OpenAI releases powerful new GPT-5.6 model - Axios",
                            url="https://axios.com/2026/06/26/openai-gpt-sol-terra-luna-trump",
                            description="Axios reports on OpenAI GPT-5.6 release restrictions.",
                        ),
                        SearchSource(
                            title="[PDF] GPT-5.6 Preview System Card - Deployment Safety Hub",
                            url="https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf",
                            description="Official system card PDF.",
                        ),
                    ],
                },
            ),
            handler=handler_2,
            block_id="blk-search-2",
            log_id="log-search-2",
        )
        messages = [{"role": "user", "content": "请搜索"}]

        tool_round_module.append_tool_round_messages(
            tool_round_module.ToolRoundRequest(
                db="db",
                assistant_message_id="msg-1",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                content_blocks=[],
                messages=messages,
                tool_calls=[tool_call_1, tool_call_2],
                reasoning_buf="",
                should_use_reasoning=True,
                step_context=AgentStepContext(
                    step_id="step-1",
                    run_id="run-1",
                    step_number=1,
                    started_at=10.0,
                    thinking_block_id="blk_thinking",
                    text_block_id="blk_text",
                ),
                step_number=1,
                run_id="run-1",
                emitter=object(),
                session_cache=object(),
                network_budget=object(),
                call_kwargs={},
                persist_message_fn=Mock(),
                execute_tools_fn=Mock(),
                complete_step_fn=Mock(),
            ),
            [record_1, record_2],
        )

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["tool_call_id"], "tc-search-1")
        self.assertEqual(messages[2]["content"], "第一个搜索上下文")
        self.assertEqual(messages[3]["tool_call_id"], "tc-search-2")
        self.assertIn("第二个搜索上下文", messages[3]["content"])
        self.assertIn("结构化来源选择建议", messages[3]["content"])
        self.assertIn("建议深读最多 3 个来源", messages[3]["content"])
        self.assertIn("Previewing GPT-5.6 Sol", messages[3]["content"])
        self.assertIn("GPT-5.6 Preview System Card", messages[3]["content"])
        self.assertIn("Axios", messages[3]["content"])
        self.assertIn("低优先级候选", messages[3]["content"])
        self.assertIn("threads.com", messages[3]["content"])

    async def test_handle_tool_calls_round_emits_selected_evidence_for_ranker_recommendations(self):
        tool_call = {"id": "tc-search", "name": "web_search", "arguments": '{"query":"OpenAI GPT-5.6"}'}
        handler = Mock()
        handler.format_llm_context.return_value = "搜索上下文"
        handler.build_content_block.return_value = None
        record = ToolExecutionRecord(
            tool_call=tool_call,
            result=ToolResult(
                status="success",
                data={
                    "query": "OpenAI GPT-5.6 官方公告",
                    "sources": [
                        SearchSource(
                            title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                            url="https://openai.com/index/previewing-gpt-5-6-sol?utm_source=feed",
                            description="OpenAI official announcement.",
                        ),
                        SearchSource(
                            title="社交平台转述",
                            url="https://threads.com/@example/post/1",
                            description="Social repost.",
                        ),
                    ],
                },
            ),
            handler=handler,
            block_id="blk-search",
            log_id="log-search",
        )
        emitter = Mock()
        emitter.evidence_item_upserted = AsyncMock()
        original_mark_tool_round_started = getattr(tool_round_module, "mark_tool_round_started", None)

        async def mark_tool_round_started(**kwargs):
            return None

        async def execute_tools_fn(*args, **kwargs):
            return [record]

        async def complete_step_fn(**kwargs):
            return None

        tool_round_module.mark_tool_round_started = mark_tool_round_started
        try:
            await handle_tool_calls_round(
                request=tool_round_module.ToolRoundRequest(
                    db="db",
                    assistant_message_id="msg-1",
                    conversation_id="conv-1",
                    user_id="user-1",
                    model_id="gpt-4",
                    provider="openai",
                    content_blocks=[],
                    messages=[{"role": "user", "content": "请搜索"}],
                    tool_calls=[tool_call],
                    reasoning_buf="",
                    should_use_reasoning=True,
                    step_context=AgentStepContext(
                        step_id="step-1",
                        run_id="run-1",
                        step_number=1,
                        started_at=10.0,
                        thinking_block_id="blk_thinking",
                        text_block_id="blk_text",
                    ),
                    step_number=1,
                    run_id="run-1",
                    emitter=emitter,
                    session_cache=object(),
                    network_budget=object(),
                    call_kwargs={},
                    persist_message_fn=Mock(),
                    execute_tools_fn=execute_tools_fn,
                    complete_step_fn=complete_step_fn,
                )
            )
        finally:
            if original_mark_tool_round_started is None:
                delattr(tool_round_module, "mark_tool_round_started")
            else:
                tool_round_module.mark_tool_round_started = original_mark_tool_round_started

        emitter.evidence_item_upserted.assert_awaited()
        selected_calls = [
            call
            for call in emitter.evidence_item_upserted.await_args_list
            if call.kwargs["evidence"]["status"] == "selected"
        ]
        selected_events = [call.kwargs["evidence"] for call in selected_calls]
        self.assertEqual(len(selected_events), 1)
        self.assertEqual(selected_calls[0].kwargs["tool_call_id"], "tc-search")
        self.assertEqual(
            selected_events[0]["id"],
            stable_web_evidence_id("https://openai.com/index/previewing-gpt-5-6-sol", fallback="unused"),
        )
        self.assertIn("建议深读", selected_events[0]["claim"])

    async def test_tool_round_selected_evidence_count_follows_quick_fact_read_limit(self):
        tool_call = {"id": "tc-search", "name": "web_search", "arguments": '{"query":"OpenAI GPT-5.6 是什么"}'}
        handler = Mock()
        handler.format_llm_context.return_value = "搜索上下文"
        handler.build_content_block.return_value = None
        record = ToolExecutionRecord(
            tool_call=tool_call,
            result=ToolResult(
                status="success",
                data={
                    "query": "OpenAI GPT-5.6 是什么 2026年",
                    "intent": "quick_fact",
                    "search_budget": "quick_fact",
                    "sources": [
                        SearchSource(
                            title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                            url="https://openai.com/index/previewing-gpt-5-6-sol",
                            description="OpenAI official announcement.",
                        ),
                        SearchSource(
                            title="[PDF] GPT-5.6 Preview System Card",
                            url="https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf",
                            description="Official system card PDF.",
                        ),
                        SearchSource(
                            title="OpenAI releases powerful new GPT-5.6 model - Axios",
                            url="https://axios.com/2026/06/26/openai-gpt-sol-terra-luna-trump",
                            description="Axios report.",
                        ),
                    ],
                },
            ),
            handler=handler,
            block_id="blk-search",
            log_id="log-search",
        )
        emitter = Mock()
        emitter.evidence_item_upserted = AsyncMock()

        await tool_round_module.emit_selected_source_evidence(
            tool_round_module.ToolRoundRequest(
                db="db",
                assistant_message_id="msg-1",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                content_blocks=[],
                messages=[{"role": "user", "content": "请搜索"}],
                tool_calls=[tool_call],
                reasoning_buf="",
                should_use_reasoning=True,
                step_context=AgentStepContext(
                    step_id="step-1",
                    run_id="run-1",
                    step_number=1,
                    started_at=10.0,
                    thinking_block_id="blk_thinking",
                    text_block_id="blk_text",
                ),
                step_number=1,
                run_id="run-1",
                emitter=emitter,
                session_cache=object(),
                network_budget=object(),
                call_kwargs={},
                persist_message_fn=Mock(),
                execute_tools_fn=Mock(),
                complete_step_fn=Mock(),
            ),
            [record],
        )

        selected_events = [
            call.kwargs["evidence"]
            for call in emitter.evidence_item_upserted.await_args_list
            if call.kwargs["evidence"]["status"] == "selected"
        ]
        self.assertEqual(len(selected_events), 1)
