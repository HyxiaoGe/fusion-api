"""
stream_handler 单元测试（Redis Stream 架构）

测试 generate_to_redis（后台任务）和 stream_redis_as_sse（SSE 读取器）。
"""

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


class SseEnvelopeFormatterTests(unittest.TestCase):
    """spec §4.6 SSE 顶层 envelope 形态测试"""

    def test_agent_event_entry_to_envelope(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "agent_event",
            "content": '{"type":"run_started","run_id":"r1","sequence":0}',
            "block_id": "",
        })
        self.assertEqual(env["chunk_type"], "agent_event")
        self.assertEqual(env["data"]["type"], "run_started")
        self.assertEqual(env["data"]["sequence"], 0)
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_reasoning_entry_carries_run_step_ids(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "reasoning",
            "content": "hello",
            "block_id": "b1",
            "run_id": "r1",
            "step_id": "s1",
        })
        self.assertEqual(env["chunk_type"], "reasoning")
        self.assertEqual(env["data"], {
            "block_id": "b1", "delta": "hello",
            "run_id": "r1", "step_id": "s1",
        })

    def test_reasoning_entry_without_run_step_ids(self):
        """旧消息或缺失 run_id/step_id 时，data 不含这两键"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "reasoning",
            "content": "hello",
            "block_id": "b1",
        })
        self.assertEqual(env["data"], {"block_id": "b1", "delta": "hello"})

    def test_answering_entry(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "answering",
            "content": "world",
            "block_id": "b2",
            "run_id": "r1",
            "step_id": "s1",
        })
        self.assertEqual(env["chunk_type"], "answering")
        self.assertEqual(env["data"]["delta"], "world")
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_done_entry_empty_data(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "done", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "done", "data": {}})

    def test_preparing_entry_empty_data(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "preparing", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "preparing", "data": {}})

    def test_thinking_pending_entry(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "thinking_pending", "content": "", "block_id": "b1"})
        self.assertEqual(env, {"chunk_type": "thinking_pending", "data": {"block_id": "b1"}})

    def test_error_entry_byok_structured_promoted(self):
        """BYOK 结构化 error_code: content 是 JSON 时升入 data"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "error",
            "content": '{"code":"provider_offline","message":"offline","retryable":true}',
            "block_id": "",
        })
        self.assertEqual(env["chunk_type"], "error")
        self.assertEqual(env["data"]["code"], "provider_offline")
        self.assertEqual(env["data"]["message"], "offline")
        self.assertEqual(env["data"]["retryable"], True)

    def test_error_entry_non_json_content_empty_data(self):
        """error content 不是 JSON dict 时返回空 data（兼容旧错误形态）"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "error",
            "content": "plain text error",
            "block_id": "",
        })
        self.assertEqual(env, {"chunk_type": "error", "data": {}})

    def test_unknown_type_falls_back_empty_data(self):
        """未知 chunk type 不抛，返回 {chunk_type: <type>, data: {}}"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "future_unknown_type",
            "content": "anything",
            "block_id": "x",
        })
        self.assertEqual(env, {"chunk_type": "future_unknown_type", "data": {}})


if __name__ == "__main__":
    unittest.main()
