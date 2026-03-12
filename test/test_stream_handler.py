from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.services.stream_handler import StreamHandler


class StreamHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = MagicMock()
        self.memory_service = MagicMock()
        self.handler = StreamHandler(self.db, self.memory_service)

    def test_normalize_direct_stream_messages_supports_mixed_message_shapes(self):
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

    async def test_persist_stream_placeholders_updates_existing_messages(self):
        self.handler.update_stream_response = AsyncMock()

        await self.handler._persist_stream_placeholders(
            assistant_message_id="assistant-1",
            answer_text="final answer",
            reasoning_message_id="reasoning-1",
            reasoning_text="thought process",
        )

        self.handler.update_stream_response.assert_any_await("reasoning-1", "thought process")
        self.handler.update_stream_response.assert_any_await("assistant-1", "final answer")
        self.assertEqual(self.handler.update_stream_response.await_count, 2)


if __name__ == "__main__":
    unittest.main()
