"""
stream_handler 单元测试（Redis Stream 架构）

测试 generate_to_redis（后台任务）和 stream_redis_as_sse（SSE 读取器）。
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.stream_handler import StreamHandler, stream_redis_as_sse

# 统一 mock Redis Stream 操作
REDIS_MOCKS = {
    "app.services.stream_handler.init_stream": AsyncMock(),
    "app.services.stream_handler.append_chunk": AsyncMock(return_value="1-0"),
    "app.services.stream_handler.finalize_stream": AsyncMock(),
    "app.services.stream_handler.check_lock_owner": AsyncMock(return_value=True),
}


class GenerateToRedisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = StreamHandler()
        self._redis_patchers = []
        for target, mock_obj in REDIS_MOCKS.items():
            mock_obj.reset_mock()
            p = patch(target, mock_obj)
            p.start()
            self._redis_patchers.append(p)

        # Mock SessionLocal
        self.mock_db = MagicMock()
        self.db_patcher = patch(
            "app.services.stream_handler.SessionLocal",
            return_value=self.mock_db,
        )
        self.db_patcher.start()

    def tearDown(self):
        for p in self._redis_patchers:
            p.stop()
        self.db_patcher.stop()

    async def test_generate_writes_chunks_and_finalizes(self):
        """正常生成：写 chunk 到 Redis + 落库 + finalize"""
        mock_chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello", reasoning_content=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())

            await self.handler.generate_to_redis(
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                messages=[{"role": "user", "content": "hi"}],
                assistant_message_id="msg-1",
                task_id="task-1",
                options={"use_reasoning": False},
            )

        # 验证 init_stream 被调用
        REDIS_MOCKS["app.services.stream_handler.init_stream"].assert_awaited_once()

        # 验证 append_chunk 被调用（content "Hello"）
        REDIS_MOCKS["app.services.stream_handler.append_chunk"].assert_awaited()

        # 验证 finalize_stream 被调用且 success=True
        REDIS_MOCKS["app.services.stream_handler.finalize_stream"].assert_awaited_once_with("conv-1", success=True)

        # 验证 DB 写入
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        self.mock_db.close.assert_called_once()

    async def test_generate_handles_exception(self):
        """LLM 异常时：尝试保存已有内容 + finalize error"""

        async def failing_stream():
            raise RuntimeError("LLM crashed")
            yield  # noqa

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=failing_stream())

            await self.handler.generate_to_redis(
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                messages=[{"role": "user", "content": "hi"}],
                assistant_message_id="msg-1",
                task_id="task-1",
                options={"use_reasoning": False},
            )

        # 验证 finalize 被调用且 success=False
        call_args = REDIS_MOCKS["app.services.stream_handler.finalize_stream"].call_args
        self.assertFalse(call_args[1].get("success", call_args[0][1]))

        # DB session 被关闭
        self.mock_db.close.assert_called_once()


class StreamRedisAsSSETests(unittest.IsolatedAsyncioTestCase):
    async def test_formats_chunks_as_sse(self):
        """read_stream_chunks 的输出被正确格式化为 SSE"""
        mock_chunks = [
            {"entry_id": "1-0", "type": "start", "content": ""},
            {"entry_id": "2-0", "type": "reasoning", "content": "thinking", "block_id": "blk_t"},
            {"entry_id": "3-0", "type": "answering", "content": "answer", "block_id": "blk_c"},
            {"entry_id": "4-0", "type": "done", "content": ""},
        ]

        async def mock_reader(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk

        with patch("app.services.stream_handler.read_stream_chunks", side_effect=mock_reader):
            events = [event async for event in stream_redis_as_sse("conv-1", "msg-1")]

        # 过滤掉 [DONE]
        data_events = [e for e in events if not e.startswith("data: [DONE]")]

        # start 被跳过，应该有 3 个数据事件（reasoning + answering + done）
        self.assertEqual(len(data_events), 3)

        # 验证 reasoning 事件
        first = data_events[0]
        self.assertIn("id: 2-0", first)
        payload = json.loads(first.split("data: ")[1])
        self.assertEqual(payload["id"], "msg-1")
        self.assertEqual(payload["conversation_id"], "conv-1")
        self.assertEqual(payload["choices"][0]["delta"]["content"][0]["type"], "thinking")

        # 验证 done 事件
        last_data = data_events[-1]
        payload = json.loads(last_data.split("data: ")[1])
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

        # 最后一条是 [DONE]
        self.assertEqual(events[-1], "data: [DONE]\n\n")


class HandleToolCallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = StreamHandler()
        self._redis_patchers = []
        for target, mock_obj in REDIS_MOCKS.items():
            mock_obj.reset_mock()
            p = patch(target, mock_obj)
            p.start()
            self._redis_patchers.append(p)

        self.mock_db = MagicMock()

    def tearDown(self):
        for p in self._redis_patchers:
            p.stop()

    async def test_handle_tool_call_logs_success(self):
        """搜索成功时写入 tool_call_log 且 status=success"""
        from app.schemas.chat import SearchSource

        mock_sources = [
            SearchSource(
                title="Result 1",
                url="https://example.com",
                description="desc",
                content="body",
                favicon=None,
            )
        ]

        with (
            patch(
                "app.services.search_client.search_web",
                new_callable=AsyncMock,
                return_value=mock_sources,
            ),
            patch(
                "app.services.stream_handler.log_tool_call",
                new_callable=AsyncMock,
            ) as mock_log,
            patch.object(
                self.handler,
                "_stream_llm_to_redis",
                new_callable=AsyncMock,
                return_value=("", "answer", None),
            ),
            patch.object(self.handler, "_persist_message"),
            patch.object(self.handler, "_extract_user_memories", new_callable=AsyncMock),
        ):
            await self.handler._handle_tool_call(
                db=self.mock_db,
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                messages=[],
                assistant_message_id="msg-1",
                task_id="task-1",
                tool_call_id="tc-1",
                tool_call_name="web_search",
                tool_call_args='{"query": "test"}',
                should_use_reasoning=False,
                thinking_block_id="blk_t",
                text_block_id="blk_c",
            )

            # 等待 asyncio.create_task 调度的日志任务完成
            await asyncio.sleep(0)

            mock_log.assert_awaited_once()
            call_kwargs = mock_log.call_args[1]
            self.assertEqual(call_kwargs["tool_name"], "web_search")
            self.assertEqual(call_kwargs["status"], "success")
            self.assertEqual(call_kwargs["input_params"], {"query": "test"})
            self.assertIsNotNone(call_kwargs["duration_ms"])

    async def test_handle_tool_call_logs_degraded_on_empty_query(self):
        """query 为空时写入 degraded 记录"""
        with (
            patch(
                "app.services.stream_handler.log_tool_call",
                new_callable=AsyncMock,
            ) as mock_log,
            patch.object(self.handler, "_fallback_no_search", new_callable=AsyncMock),
        ):
            await self.handler._handle_tool_call(
                db=self.mock_db,
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                messages=[],
                assistant_message_id="msg-1",
                task_id="task-1",
                tool_call_id="tc-1",
                tool_call_name="web_search",
                tool_call_args='{"query": ""}',
                should_use_reasoning=False,
                thinking_block_id="blk_t",
                text_block_id="blk_c",
            )

            # 等待 asyncio.create_task 调度的日志任务完成
            await asyncio.sleep(0)

            mock_log.assert_awaited_once()
            call_kwargs = mock_log.call_args[1]
            self.assertEqual(call_kwargs["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
