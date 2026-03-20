import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.services.stream_handler import StreamHandler


class StreamHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.memory_service = MagicMock()
        self.handler = StreamHandler(self.db, self.memory_service)

    @staticmethod
    def _parse_event(event: str):
        if event == "data: [DONE]\n\n":
            return "[DONE]"
        return json.loads(event.removeprefix("data: ").strip())

    async def test_normalize_direct_stream_messages_supports_mixed_message_shapes(self):
        messages = [
            {"role": "system", "content": "system"},
            SimpleNamespace(type="human", content="hello"),
            SimpleNamespace(role="assistant", content="world"),
        ]

        normalized = self.handler._normalize_direct_stream_messages(messages)

        self.assertEqual(
            normalized,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )

    async def test_generate_stream_emits_content_finish_and_done_for_normal_flow(self):
        self.memory_service.create_message.return_value = SimpleNamespace(id="assistant-1")
        self.handler.update_stream_response = AsyncMock()
        llm = MagicMock()
        llm.stream.return_value = [
            SimpleNamespace(content="Hel"),
            SimpleNamespace(content="lo"),
            SimpleNamespace(content="!"),
        ]

        with patch("app.services.stream_handler.llm_manager.get_model", return_value=llm):
            events = [
                event
                async for event in self.handler.generate_stream(
                    "openai",
                    "gpt",
                    [SimpleNamespace(content="hi")],
                    "conv-1",
                    {"use_reasoning": False},
                    "turn-1",
                )
            ]

        payloads = [self._parse_event(event) for event in events]
        self.assertEqual(
            [payload["choices"][0]["delta"]["content"] for payload in payloads[:-2]],
            ["Hel", "lo", "!"],
        )
        self.assertEqual(payloads[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payloads[-1], "[DONE]")
        self.handler.update_stream_response.assert_awaited_once_with("assistant-1", "Hello!")

    async def test_generate_stream_emits_reasoning_and_content_with_same_public_message_id(self):
        self.memory_service.create_message.side_effect = [
            SimpleNamespace(id="reasoning-1"),
            SimpleNamespace(id="assistant-1"),
        ]
        self.handler.update_stream_response = AsyncMock()
        llm = MagicMock()
        llm.stream.return_value = [
            SimpleNamespace(additional_kwargs={"reasoning_content": "think-1"}, content=""),
            SimpleNamespace(additional_kwargs={"reasoning_content": "think-2"}, content=""),
            SimpleNamespace(additional_kwargs={}, content="answer-1"),
            SimpleNamespace(additional_kwargs={}, content="answer-2"),
        ]

        with patch("app.services.stream_handler.llm_manager.get_model", return_value=llm):
            events = [
                event
                async for event in self.handler.generate_stream(
                    "qwen",
                    "qwen-max",
                    [SimpleNamespace(content="hi")],
                    "conv-1",
                    {},
                    "turn-1",
                )
            ]

        payloads = [self._parse_event(event) for event in events]
        ids = [payload["id"] for payload in payloads[:-1] if payload != "[DONE]"]
        self.assertTrue(all(message_id == "assistant-1" for message_id in ids))

        reasoning_indexes = [
            index for index, payload in enumerate(payloads)
            if payload != "[DONE]" and "reasoning_content" in payload["choices"][0]["delta"]
        ]
        content_indexes = [
            index for index, payload in enumerate(payloads)
            if payload != "[DONE]" and "content" in payload["choices"][0]["delta"]
        ]
        self.assertTrue(max(reasoning_indexes) < min(content_indexes))
        self.assertEqual(payloads[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payloads[-1], "[DONE]")
        self.handler.update_stream_response.assert_has_awaits(
            [
                call("reasoning-1", "think-1think-2"),
                call("assistant-1", "answer-1answer-2"),
            ]
        )

    async def test_pre_stream_exception_bubbles_up_without_sse_events(self):
        self.memory_service.create_message.return_value = SimpleNamespace(id="assistant-1")

        with patch("app.services.stream_handler.llm_manager.get_model", side_effect=RuntimeError("boom")):
            stream = self.handler.generate_stream(
                "openai",
                "gpt",
                [SimpleNamespace(content="hi")],
                "conv-1",
                {"use_reasoning": False},
                "turn-1",
            )
            with self.assertRaises(RuntimeError):
                await stream.__anext__()

    async def test_stream_internal_exception_before_first_content_emits_error_and_done(self):
        self.memory_service.create_message.return_value = SimpleNamespace(id="assistant-1")
        self.handler.update_stream_response = AsyncMock()
        llm = MagicMock()

        def failing_stream(_messages):
            if False:
                yield None
            raise RuntimeError("boom")

        llm.stream.return_value = failing_stream([SimpleNamespace(content="hi")])

        with patch("app.services.stream_handler.llm_manager.get_model", return_value=llm):
            events = [
                event
                async for event in self.handler.generate_stream(
                    "openai",
                    "gpt",
                    [SimpleNamespace(content="hi")],
                    "conv-1",
                    {"use_reasoning": False},
                    "turn-1",
                )
            ]

        payloads = [self._parse_event(event) for event in events]
        self.assertEqual(payloads[0]["choices"][0]["finish_reason"], "error")
        self.assertEqual(payloads[0]["error"]["message"], "boom")
        self.assertEqual(payloads[1], "[DONE]")
        self.handler.update_stream_response.assert_not_awaited()

    async def test_stream_internal_exception_after_partial_answer_persists_partial_content(self):
        self.memory_service.create_message.return_value = SimpleNamespace(id="assistant-1")
        self.handler.update_stream_response = AsyncMock()
        llm = MagicMock()

        def partial_stream(_messages):
            yield SimpleNamespace(content="a")
            yield SimpleNamespace(content="b")
            raise RuntimeError("boom")

        llm.stream.return_value = partial_stream([SimpleNamespace(content="hi")])

        with patch("app.services.stream_handler.llm_manager.get_model", return_value=llm):
            events = [
                event
                async for event in self.handler.generate_stream(
                    "openai",
                    "gpt",
                    [SimpleNamespace(content="hi")],
                    "conv-1",
                    {"use_reasoning": False},
                    "turn-1",
                )
            ]

        payloads = [self._parse_event(event) for event in events]
        self.assertEqual(
            [payload["choices"][0]["delta"]["content"] for payload in payloads[:2]],
            ["a", "b"],
        )
        self.assertEqual(payloads[2]["choices"][0]["finish_reason"], "error")
        self.assertEqual(payloads[3], "[DONE]")
        self.handler.update_stream_response.assert_awaited_once_with("assistant-1", "ab")

    async def test_deepseek_same_chunk_duplicate_content_is_filtered(self):
        self.memory_service.create_message.side_effect = [
            SimpleNamespace(id="reasoning-1"),
            SimpleNamespace(id="assistant-1"),
        ]
        self.handler.update_stream_response = AsyncMock()
        llm = MagicMock()
        llm.stream.return_value = [
            SimpleNamespace(additional_kwargs={"reasoning_content": "dup"}, content="dup"),
            SimpleNamespace(additional_kwargs={}, content="final"),
        ]

        with patch("app.services.stream_handler.llm_manager.get_model", return_value=llm):
            events = [
                event
                async for event in self.handler.generate_stream(
                    "deepseek",
                    "deepseek-reasoner",
                    [SimpleNamespace(content="hi")],
                    "conv-1",
                    {},
                    "turn-1",
                )
            ]

        payloads = [self._parse_event(event) for event in events]
        content_values = [
            payload["choices"][0]["delta"]["content"]
            for payload in payloads
            if payload != "[DONE]" and "content" in payload["choices"][0]["delta"]
        ]
        reasoning_values = [
            payload["choices"][0]["delta"]["reasoning_content"]
            for payload in payloads
            if payload != "[DONE]" and "reasoning_content" in payload["choices"][0]["delta"]
        ]
        self.assertEqual(reasoning_values, ["dup"])
        self.assertEqual(content_values, ["final"])

    async def test_volcengine_connection_failure_emits_error_and_done(self):
        self.memory_service.create_message.side_effect = [
            SimpleNamespace(id="reasoning-1"),
            SimpleNamespace(id="assistant-1"),
        ]
        self.handler.update_stream_response = AsyncMock()

        failing_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=RuntimeError("connect failed"))
                )
            )
        )

        with patch("app.services.stream_handler.llm_manager._get_model_credentials", return_value={"api_key": "k", "base_url": "https://example.com"}):
            with patch("openai.AsyncOpenAI", return_value=failing_client):
                events = [
                    event
                    async for event in self.handler.generate_stream(
                        "volcengine",
                        "deepseek-r1",
                        [{"role": "user", "content": "hi"}],
                        "conv-1",
                        {},
                        "turn-1",
                    )
                ]

        payloads = [self._parse_event(event) for event in events]
        self.assertEqual(payloads[0]["choices"][0]["finish_reason"], "error")
        self.assertEqual(payloads[0]["error"]["message"], "connect failed")
        self.assertEqual(payloads[1], "[DONE]")
        self.handler.update_stream_response.assert_not_awaited()
