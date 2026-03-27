import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.stream_handler import StreamHandler

# 统一 mock 所有 Redis 流状态操作，确保测试不依赖 Redis
REDIS_MOCKS = {
    "app.services.stream_handler.acquire_stream_lock": AsyncMock(return_value="mock-request-id"),
    "app.services.stream_handler.is_lock_owner": AsyncMock(return_value=True),
    "app.services.stream_handler.set_stream_start": AsyncMock(),
    "app.services.stream_handler.append_stream_chunk": AsyncMock(),
    "app.services.stream_handler.set_stream_complete": AsyncMock(),
    "app.services.stream_handler.set_stream_error": AsyncMock(),
}


class StreamHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.memory_service = MagicMock()
        self.handler = StreamHandler(self.db, self.memory_service)
        # 启动所有 Redis mock
        self._redis_patchers = []
        for target, mock_obj in REDIS_MOCKS.items():
            mock_obj.reset_mock()
            p = patch(target, mock_obj)
            p.start()
            self._redis_patchers.append(p)

    def tearDown(self):
        for p in self._redis_patchers:
            p.stop()

    @staticmethod
    def _parse_event(event: str):
        if event == "data: [DONE]\n\n":
            return "[DONE]"
        return json.loads(event.removeprefix("data: ").strip())

    async def test_generate_stream_emits_content_finish_and_done(self):
        """正常流式响应：心跳帧 → 内容帧 → 结束帧 → [DONE]"""
        # 模拟 litellm.acompletion 流式响应
        mock_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel", reasoning_content=None), finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo!", reasoning_content=None), finish_reason=None)],
                usage=None,
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            events = [
                event
                async for event in self.handler.generate_stream(
                    litellm_model="openai/gpt-4o",
                    provider="openai",
                    model_id="gpt-4o",
                    litellm_kwargs={"api_key": "test"},
                    messages=[{"role": "user", "content": "hi"}],
                    conversation_id="conv-1",
                    options={"use_reasoning": False},
                )
            ]

        payloads = [self._parse_event(e) for e in events]
        # 第一帧是心跳
        self.assertIsNone(payloads[0]["choices"][0]["finish_reason"])
        # 中间帧有内容
        text_deltas = []
        for p in payloads[1:-2]:
            if p != "[DONE]":
                blocks = p["choices"][0]["delta"].get("content", [])
                for b in blocks:
                    if b["type"] == "text":
                        text_deltas.append(b["text"])
        self.assertEqual(text_deltas, ["Hel", "lo!"])
        # 结束帧
        self.assertEqual(payloads[-2]["choices"][0]["finish_reason"], "stop")
        # [DONE] 标记
        self.assertEqual(payloads[-1], "[DONE]")
        # 消息已落库
        self.memory_service.create_message.assert_called_once()

    async def test_generate_stream_emits_reasoning_and_content_blocks(self):
        """推理模型：先输出 thinking block，再输出 text block"""
        mock_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="", reasoning_content="think-1"),
                    finish_reason=None,
                )],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="answer-1", reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None,
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            events = [
                event
                async for event in self.handler.generate_stream(
                    litellm_model="openai/qwen-max",
                    provider="qwen",
                    model_id="qwen-max",
                    litellm_kwargs={},
                    messages=[{"role": "user", "content": "hi"}],
                    conversation_id="conv-1",
                    options={},
                )
            ]

        payloads = [self._parse_event(e) for e in events]
        # 所有帧使用相同的 message id
        ids = [p["id"] for p in payloads if p != "[DONE]"]
        self.assertTrue(all(mid == ids[0] for mid in ids))

        # 验证有 thinking 和 text 类型的 block
        block_types = []
        for p in payloads:
            if p == "[DONE]":
                continue
            blocks = p["choices"][0]["delta"].get("content", [])
            for b in blocks:
                block_types.append(b["type"])
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)

    async def test_stream_exception_emits_error_and_done(self):
        """流式过程中异常：发出 error 帧 + [DONE]"""
        async def failing_stream():
            raise RuntimeError("boom")
            yield  # noqa: unreachable - makes this an async generator

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=failing_stream())
            events = [
                event
                async for event in self.handler.generate_stream(
                    litellm_model="openai/gpt-4o",
                    provider="openai",
                    model_id="gpt-4o",
                    litellm_kwargs={},
                    messages=[{"role": "user", "content": "hi"}],
                    conversation_id="conv-1",
                    options={"use_reasoning": False},
                )
            ]

        payloads = [self._parse_event(e) for e in events]
        # 心跳帧
        self.assertIsNone(payloads[0]["choices"][0]["finish_reason"])
        # error 帧
        self.assertEqual(payloads[1]["choices"][0]["finish_reason"], "error")
        # [DONE]
        self.assertEqual(payloads[-1], "[DONE]")

    async def test_duplicate_reasoning_in_content_is_filtered(self):
        """部分 provider 会在 content 和 reasoning_content 重复输出，应去重"""
        mock_chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="dup", reasoning_content="dup"),
                    finish_reason=None,
                )],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="final", reasoning_content=None),
                    finish_reason=None,
                )],
                usage=None,
            ),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch("app.services.stream_handler.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream())
            events = [
                event
                async for event in self.handler.generate_stream(
                    litellm_model="deepseek/deepseek-reasoner",
                    provider="deepseek",
                    model_id="deepseek-reasoner",
                    litellm_kwargs={},
                    messages=[{"role": "user", "content": "hi"}],
                    conversation_id="conv-1",
                    options={},
                )
            ]

        payloads = [self._parse_event(e) for e in events]
        text_values = []
        thinking_values = []
        for p in payloads:
            if p == "[DONE]":
                continue
            for b in p["choices"][0]["delta"].get("content", []):
                if b["type"] == "text":
                    text_values.append(b["text"])
                elif b["type"] == "thinking":
                    thinking_values.append(b["thinking"])
        # reasoning_content="dup" 输出为 thinking，content="dup" 被去重过滤
        self.assertEqual(thinking_values, ["dup"])
        self.assertEqual(text_values, ["final"])


if __name__ == "__main__":
    unittest.main()
