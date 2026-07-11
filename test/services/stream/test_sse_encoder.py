import json
import unittest
from unittest.mock import patch

from app.services.stream.sse_encoder import stream_redis_as_sse


class StreamSseEncoderTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_expected_message_and_task_to_reader(self):
        captured = {}

        async def read_stream_chunks(
            conversation_id,
            last_entry_id,
            *,
            expected_message_id,
            expected_task_id,
        ):
            captured.update(
                conversation_id=conversation_id,
                last_entry_id=last_entry_id,
                expected_message_id=expected_message_id,
                expected_task_id=expected_task_id,
            )
            yield {"entry_id": "2-0", "type": "done", "content": "", "block_id": ""}

        with (
            patch("app.core.redis.get_redis_pool", return_value=object()),
            patch("app.services.stream.sse_encoder.read_stream_chunks", new=read_stream_chunks),
        ):
            frames = [
                frame
                async for frame in stream_redis_as_sse(
                    conversation_id="conv-1",
                    message_id="msg-1",
                    task_id="task-1",
                    last_entry_id="1-0",
                )
            ]

        self.assertEqual(
            captured,
            {
                "conversation_id": "conv-1",
                "last_entry_id": "1-0",
                "expected_message_id": "msg-1",
                "expected_task_id": "task-1",
            },
        )
        self.assertEqual(json.loads(frames[0].split("data: ", 1)[1]), {"chunk_type": "done", "data": {}})
        self.assertEqual(frames[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
