from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, call

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

    def test_build_content_event_includes_message_metadata(self):
        event = self.handler._build_content_event("conv-1", "msg-1", "hello")

        self.assertEqual(
            event,
            'data: {"content": "hello", "conversation_id": "conv-1", "message_id": "msg-1"}\n\n',
        )

    def test_build_done_marker_event_uses_done_content(self):
        event = self.handler._build_done_marker_event("conv-1", "msg-1")

        self.assertEqual(
            event,
            'data: {"content": "[DONE]", "conversation_id": "conv-1", "message_id": "msg-1"}\n\n',
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

    async def test_finalize_reasoning_events_emits_missing_phase_events(self):
        send_event = AsyncMock()

        await self.handler._finalize_reasoning_events(
            send_event,
            reasoning_completed=False,
            answering_started=False,
            reasoning_message_id="reasoning-1",
            assistant_message_id="assistant-1",
        )

        self.assertEqual(
            send_event.await_args_list,
            [
                call("reasoning_complete", message_id="reasoning-1"),
                call("answering_start", message_id="assistant-1"),
                call("answering_complete", message_id="assistant-1"),
            ],
        )

    async def test_finalize_reasoning_events_skips_completed_phases(self):
        send_event = AsyncMock()

        await self.handler._finalize_reasoning_events(
            send_event,
            reasoning_completed=True,
            answering_started=True,
            reasoning_message_id="reasoning-1",
            assistant_message_id="assistant-1",
        )

        self.assertEqual(
            send_event.await_args_list,
            [call("answering_complete", message_id="assistant-1")],
        )


if __name__ == "__main__":
    unittest.main()
