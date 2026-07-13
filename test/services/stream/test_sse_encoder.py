import asyncio
import json
import unittest
from unittest.mock import patch

from app.services.stream.sse_encoder import stream_redis_as_sse


class StreamSseEncoderTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_heartbeat_without_restarting_reader_then_delivers_chunk(self):
        release_chunk = asyncio.Event()
        reader_started = 0
        reader_cancelled = 0

        async def read_stream_chunks(*_args, **_kwargs):
            nonlocal reader_started, reader_cancelled
            reader_started += 1
            try:
                await release_chunk.wait()
            except asyncio.CancelledError:
                reader_cancelled += 1
                raise
            yield {"entry_id": "2-0", "type": "answering", "content": "答案", "block_id": "blk-1"}

        with (
            patch("app.core.redis.get_redis_pool", return_value=object()),
            patch("app.services.stream.sse_encoder.read_stream_chunks", new=read_stream_chunks),
            patch("app.services.stream.sse_encoder.SSE_HEARTBEAT_INTERVAL_SECONDS", 0.01),
        ):
            stream = stream_redis_as_sse("conv-1", "msg-1", "task-1")
            heartbeat = await asyncio.wait_for(anext(stream), timeout=0.2)
            self.assertEqual(heartbeat, ": keepalive\n\n")

            release_chunk.set()
            chunk_frame = await asyncio.wait_for(anext(stream), timeout=0.2)
            done_frame = await asyncio.wait_for(anext(stream), timeout=0.2)

        self.assertEqual(reader_started, 1)
        self.assertEqual(reader_cancelled, 0)
        self.assertEqual(
            chunk_frame,
            'id: 2-0\ndata: {"chunk_type": "answering", "data": {"block_id": "blk-1", "delta": "答案"}}\n\n',
        )
        self.assertEqual(done_frame, "data: [DONE]\n\n")

    async def test_fast_reader_does_not_emit_extra_heartbeat(self):
        async def read_stream_chunks(*_args, **_kwargs):
            yield {"entry_id": "1-0", "type": "start", "content": "", "block_id": ""}
            yield {"entry_id": "2-0", "type": "answering", "content": "快", "block_id": "blk-1"}
            yield {"entry_id": "3-0", "type": "done", "content": "", "block_id": ""}

        with (
            patch("app.core.redis.get_redis_pool", return_value=object()),
            patch("app.services.stream.sse_encoder.read_stream_chunks", new=read_stream_chunks),
            patch("app.services.stream.sse_encoder.SSE_HEARTBEAT_INTERVAL_SECONDS", 0.01),
        ):
            frames = [frame async for frame in stream_redis_as_sse("conv-1", "msg-1", "task-1")]

        self.assertEqual(len(frames), 3)
        self.assertTrue(frames[0].startswith("id: 2-0\n"))
        self.assertTrue(frames[1].startswith("id: 3-0\n"))
        self.assertEqual(frames[2], "data: [DONE]\n\n")
        self.assertNotIn(": keepalive", "".join(frames))

    async def test_closing_sse_stream_cancels_pending_reader_task(self):
        reader_cancelled = asyncio.Event()

        async def read_stream_chunks(*_args, **_kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                reader_cancelled.set()
                raise
            yield  # pragma: no cover - 仅用于保持 async generator 形态

        with (
            patch("app.core.redis.get_redis_pool", return_value=object()),
            patch("app.services.stream.sse_encoder.read_stream_chunks", new=read_stream_chunks),
            patch("app.services.stream.sse_encoder.SSE_HEARTBEAT_INTERVAL_SECONDS", 0.01),
        ):
            stream = stream_redis_as_sse("conv-1", "msg-1", "task-1")
            self.assertEqual(await asyncio.wait_for(anext(stream), timeout=0.2), ": keepalive\n\n")
            await stream.aclose()

        await asyncio.wait_for(reader_cancelled.wait(), timeout=0.2)

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
