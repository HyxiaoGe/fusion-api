import unittest
from unittest.mock import Mock

from app.schemas.chat import TextBlock
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
                ("plan", "run-1", 1, ["web_search"], 0, 20),
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
