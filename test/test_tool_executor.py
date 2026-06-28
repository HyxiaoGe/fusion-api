"""tool_executor 重试策略测试"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.stream import tool_executor as tool_executor_module
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_executor import (
    AgentEventCompositeWriter,
    _should_retry_tool_result,
    execute_tools_parallel,
)
from app.services.tool_handlers.base import ToolResult


class ToolRetryPolicyTests(unittest.TestCase):
    def test_degraded_result_is_terminal_and_not_retried(self):
        """degraded 是已接受的降级结果，不应再触发 backoff 重试/放弃日志"""
        result = ToolResult(status="degraded", error_message="reader-service 暂时未返回内容")

        self.assertFalse(_should_retry_tool_result(result))


class AgentEventCompositeWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_redis_before_recorder(self):
        calls = []

        class RedisWriter:
            async def append_chunk(self, conversation_id, chunk_type, payload):
                calls.append(("redis", conversation_id, chunk_type, payload))

        class Recorder:
            def record_chunk(self, conversation_id, chunk_type, payload):
                calls.append(("recorder", conversation_id, chunk_type, payload))

        writer = AgentEventCompositeWriter(redis_writer=RedisWriter(), recorder=Recorder())

        await writer.append_chunk("c1", "agent_event", {"type": "run_progress_updated"})

        self.assertEqual(
            calls,
            [
                ("redis", "c1", "agent_event", {"type": "run_progress_updated"}),
                ("recorder", "c1", "agent_event", {"type": "run_progress_updated"}),
            ],
        )


class ToolExecutorMessageIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tool_batch_accepts_request_and_preserves_log_context(self):
        request_cls = getattr(tool_executor_module, "ToolExecutionBatchRequest")
        execute_tool_batch = getattr(tool_executor_module, "execute_tool_batch")
        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success", data={"items": []})
        tool_call = {"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}

        request = request_cls(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            trace_id="trace-1",
            step_number=2,
            message_id="assistant-1",
            emitter=None,
            network_budget=None,
        )

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tool_batch(request, [tool_call])

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertIs(record.handler, handler)
        handler.execute.assert_awaited_once_with({"query": "redis stream"})
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")
        self.assertEqual(handler.log.await_args.kwargs["trace_id"], "trace-1")
        self.assertEqual(handler.log.await_args.kwargs["step_number"], 2)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], {"query": "redis stream"})

    async def test_execute_tool_batch_preserves_input_order_when_calls_complete_out_of_order(self):
        request_cls = getattr(tool_executor_module, "ToolExecutionBatchRequest")
        execute_tool_batch = getattr(tool_executor_module, "execute_tool_batch")
        slow_handler = AsyncMock()
        slow_handler.tool_name = "web_search"
        slow_handler.execute.side_effect = lambda _args: ToolResult(status="success", data={"order": "slow"})
        fast_handler = AsyncMock()
        fast_handler.tool_name = "url_read"
        fast_handler.execute.return_value = ToolResult(status="success", data={"order": "fast"})
        slow_call = {"id": "call-slow", "name": "web_search", "arguments": {"query": "redis"}}
        fast_call = {"id": "call-fast", "name": "url_read", "arguments": {"url": "https://example.com"}}
        both_started = asyncio.Event()
        started = []
        completed = []

        async def execute_one_tool_call(_request, tool_call):
            started.append(tool_call["id"])
            if len(started) == 2:
                both_started.set()
            if tool_call is slow_call:
                await asyncio.wait_for(both_started.wait(), timeout=1)
                await asyncio.sleep(0.01)
                completed.append(tool_call["id"])
                return ToolExecutionRecord(
                    tool_call=slow_call,
                    result=ToolResult(status="success", data={"order": "slow"}),
                    handler=slow_handler,
                    block_id="blk_slow",
                    log_id="log_slow",
                )
            await asyncio.wait_for(both_started.wait(), timeout=1)
            completed.append(tool_call["id"])
            return ToolExecutionRecord(
                tool_call=fast_call,
                result=ToolResult(status="success", data={"order": "fast"}),
                handler=fast_handler,
                block_id="blk_fast",
                log_id="log_fast",
            )

        request = request_cls(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
        )
        with patch("app.services.stream.tool_executor.execute_one_tool_call", side_effect=execute_one_tool_call):
            records = await execute_tool_batch(request, [slow_call, fast_call])

        self.assertEqual([record.tool_call for record in records], [slow_call, fast_call])
        self.assertEqual(completed, ["call-fast", "call-slow"])

    async def test_execute_tools_parallel_returns_record_with_named_access_and_handler_helpers(self):
        """成功执行后返回命名记录，并通过记录方法委派 handler 格式化逻辑。"""
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"items": []}))
        handler.log = AsyncMock()
        handler.format_llm_context.return_value = "LLM 可见上下文"
        handler.build_content_block.return_value = {"type": "tool_result"}

        tool_call = {"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertEqual(record.tool_name, "web_search")
        self.assertEqual(record.result.status, "success")
        self.assertIs(record.handler, handler)
        self.assertTrue(record.block_id.startswith("blk_"))
        self.assertIsInstance(record.log_id, str)

        self.assertEqual(record.format_llm_context(), "LLM 可见上下文")
        handler.format_llm_context.assert_called_once_with(record.result)
        self.assertEqual(record.build_content_block(), {"type": "tool_result"})
        handler.build_content_block.assert_called_once_with(record.result, record.block_id, record.log_id)

    async def test_execute_tools_parallel_returns_record_for_unknown_handler_with_safe_fallbacks(self):
        """未知工具仍返回记录，但不会产生 content block，LLM context 使用安全兜底文案。"""
        tool_call = {"id": "call-404", "name": "missing_tool", "arguments": {}}

        with patch("app.services.tool_handlers.get_handler", return_value=None):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertEqual(record.tool_name, "missing_tool")
        self.assertIsNone(record.handler)
        self.assertEqual(record.result.status, "failed")
        self.assertEqual(record.result.error_message, "未知工具: missing_tool")
        self.assertEqual(record.format_llm_context(), "工具未取得可用结果，不能把该工具结果作为依据。")
        self.assertIsNone(record.build_content_block())

    async def test_execute_tools_parallel_unknown_handler_does_not_emit_events(self):
        tool_call = {"id": "call-404", "name": "missing_tool", "arguments": {"query": "x"}}
        emitter = AsyncMock()

        with patch("app.services.tool_handlers.get_handler", return_value=None):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                emitter=emitter,
            )

        self.assertEqual(records[0].result.status, "failed")
        emitter.tool_call_started.assert_not_awaited()
        emitter.tool_call_completed.assert_not_awaited()

    async def test_execute_tools_parallel_passes_message_id_to_handler_log(self):
        """工具调用日志必须关联最终 assistant message"""
        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [{"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")

    async def test_execute_tools_parallel_with_emitter_keeps_events_log_and_record(self):
        """emitter 路径仍发工具事件、写日志，并返回命名记录。"""
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(return_value=ToolResult(status="success"))
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "search", "truncated": False}
        emitter = AsyncMock()
        tool_call = {"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                trace_id="trace-1",
                step_number=2,
                message_id="assistant-1",
                emitter=emitter,
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertIs(record.handler, handler)
        self.assertEqual(record.result.status, "success")
        emitter.tool_call_started.assert_awaited_once_with(
            tool_call_id="call-1",
            tool_name="web_search",
            arguments={"query": "redis stream"},
        )
        emitter.tool_call_completed.assert_awaited_once()
        completed_kwargs = emitter.tool_call_completed.await_args.kwargs
        self.assertEqual(completed_kwargs["tool_call_id"], "call-1")
        self.assertEqual(completed_kwargs["tool_name"], "web_search")
        self.assertEqual(completed_kwargs["status"], "success")
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")
        self.assertEqual(handler.log.await_args.kwargs["trace_id"], "trace-1")
        self.assertEqual(handler.log.await_args.kwargs["step_number"], 2)

    async def test_execute_tools_parallel_with_emitter_emits_tool_digest_and_evidence(self):
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="success",
                data={
                    "sources": [
                        {
                            "title": "官方发布页",
                            "url": "https://example.com/news",
                            "description": "官方页面确认发布时间。",
                            "content": "官方页面确认发布时间，并给出原始公告。",
                        }
                    ]
                },
            )
        )
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {
            "kind": "search",
            "title": "找到 1 条搜索结果",
            "count": 1,
            "truncated": False,
        }
        emitter = AsyncMock()
        tool_call = {"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                trace_id="trace-1",
                step_number=2,
                message_id="assistant-1",
                emitter=emitter,
            )

        emitter.tool_result_digest.assert_awaited_once()
        digest_kwargs = emitter.tool_result_digest.await_args.kwargs
        self.assertEqual(digest_kwargs["tool_call_id"], "call-1")
        self.assertEqual(digest_kwargs["tool_name"], "web_search")
        self.assertEqual(digest_kwargs["status"], "success")
        self.assertEqual(digest_kwargs["title"], "搜索完成")
        self.assertEqual(digest_kwargs["source_refs"], ["ev-call-1-0"])
        self.assertLessEqual(len(digest_kwargs["key_findings"]), 5)

        emitter.evidence_item_upserted.assert_awaited_once()
        evidence_kwargs = emitter.evidence_item_upserted.await_args.kwargs
        self.assertEqual(evidence_kwargs["tool_call_id"], "call-1")
        self.assertEqual(evidence_kwargs["evidence"]["id"], "ev-call-1-0")
        self.assertEqual(evidence_kwargs["evidence"]["title"], "官方发布页")
        self.assertEqual(evidence_kwargs["evidence"]["domain"], "example.com")

    async def test_execute_tools_parallel_uses_network_budget_normalized_args_for_handler_and_log(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success", data={"sources": []})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [
                    {
                        "id": "call-1",
                        "name": "web_search",
                        "arguments": {
                            "query": "redis",
                            "count": 99,
                            "domains": ["https://Redis.io/docs", "bad domain"],
                            "recency_days": 0,
                        },
                    }
                ],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=NetworkToolBudget(),
            )

        normalized = {
            "query": "redis",
            "count": 10,
            "recency_days": 1,
        }
        handler.execute.assert_awaited_once_with(normalized)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], normalized)

    async def test_execute_tools_parallel_returns_degraded_when_network_budget_exhausted_without_handler_execute(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")
        budget = NetworkToolBudget()
        for i in range(4):
            budget.prepare_web_search_args({"query": f"q{i}"})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [{"id": "call-5", "name": "web_search", "arguments": {"query": "q5", "count": 8}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=budget,
            )

        record = results[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertEqual(record.tool_name, "web_search")
        self.assertEqual(record.result.status, "degraded")
        self.assertTrue(record.result.data["budget_limited"])
        self.assertIs(record.handler, handler)
        handler.execute.assert_not_awaited()
        handler.log.assert_awaited_once()

    async def test_execute_tools_parallel_budget_exhausted_with_emitter_emits_events_before_log(self):
        from app.services.stream.network_budget import NetworkToolBudget

        order = []

        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock()
        handler.log = AsyncMock(side_effect=lambda **_kwargs: order.append("log"))
        handler._build_result_summary.return_value = {"kind": "search", "truncated": False}

        emitter = AsyncMock()

        async def record_started(**_kwargs):
            order.append("started")

        async def record_completed(**_kwargs):
            order.append("completed")

        emitter.tool_call_started.side_effect = record_started
        emitter.tool_call_completed.side_effect = record_completed

        budget = NetworkToolBudget()
        for i in range(4):
            budget.prepare_web_search_args({"query": f"q{i}"})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [{"id": "call-5", "name": "web_search", "arguments": {"query": "q5", "count": 8}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                emitter=emitter,
                network_budget=budget,
            )

        record = results[0]
        self.assertEqual(record.result.status, "degraded")
        handler.execute.assert_not_awaited()
        self.assertEqual(order, ["started", "completed", "log"])
        emitter.tool_call_started.assert_awaited_once_with(
            tool_call_id="call-5",
            tool_name="web_search",
            arguments={"query": "q5", "count": 8},
        )
        emitter.tool_call_completed.assert_awaited_once_with(
            tool_call_id="call-5",
            tool_name="web_search",
            status="degraded",
            duration_ms=0,
            result_summary={"kind": "search", "truncated": False},
            error="web_search 已达到本轮联网预算",
        )
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")


if __name__ == "__main__":
    unittest.main()
