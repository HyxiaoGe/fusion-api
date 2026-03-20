import json
import unittest

from app.services.stream_serializer import StreamSerializer


class StreamSerializerTests(unittest.TestCase):
    @staticmethod
    def _parse_event(event: str):
        if event == "data: [DONE]\n\n":
            return "[DONE]"
        return json.loads(event.removeprefix("data: ").strip())

    def test_content_chunk_uses_delta_content(self):
        payload = self._parse_event(StreamSerializer.content_chunk("msg-1", "conv-1", "hello"))

        self.assertEqual(payload["id"], "msg-1")
        self.assertEqual(payload["conversation_id"], "conv-1")
        self.assertEqual(payload["choices"][0]["delta"]["content"], "hello")
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_reasoning_chunk_uses_delta_reasoning_content(self):
        payload = self._parse_event(StreamSerializer.reasoning_chunk("msg-1", "conv-1", "think"))

        self.assertEqual(payload["choices"][0]["delta"], {"reasoning_content": "think"})
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_finish_chunk_uses_empty_delta_and_finish_reason(self):
        payload = self._parse_event(StreamSerializer.finish_chunk("msg-1", "conv-1", "stop"))

        self.assertEqual(payload["choices"][0]["delta"], {})
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_error_chunk_includes_top_level_error(self):
        payload = self._parse_event(StreamSerializer.error_chunk("msg-1", "conv-1", "boom"))

        self.assertEqual(payload["error"]["message"], "boom")
        self.assertEqual(payload["choices"][0]["delta"], {})
        self.assertEqual(payload["choices"][0]["finish_reason"], "error")

    def test_done_marker_matches_protocol(self):
        self.assertEqual(StreamSerializer.done_marker(), "data: [DONE]\n\n")
