"""
流状态缓存服务单元测试（Redis Stream 版本）

使用 FakeRedis mock 模拟 Redis Stream 操作。
"""
import json
import unittest
from unittest.mock import AsyncMock, patch, MagicMock


class FakeRedis:
    """轻量级 Redis mock，模拟 String/Hash/Stream 操作"""

    def __init__(self):
        self._store = {}
        self._streams = {}
        self._seq = 0

    async def get(self, key):
        return self._store.get(key)

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def delete(self, *keys):
        for key in keys:
            self._store.pop(key, None)
            self._streams.pop(key, None)

    async def hset(self, key, field=None, value=None, mapping=None):
        if key not in self._store:
            self._store[key] = {}
        if mapping:
            self._store[key].update(mapping)
        elif field is not None:
            self._store[key][field] = value

    async def hgetall(self, key):
        return self._store.get(key, {})

    async def expire(self, key, seconds):
        pass

    async def xadd(self, key, fields):
        if key not in self._streams:
            self._streams[key] = []
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self._streams[key].append((entry_id, fields))
        return entry_id

    async def xrange(self, key, min=None, count=None):
        entries = self._streams.get(key, [])
        if min and min != "0":
            entries = [(eid, f) for eid, f in entries if eid > min]
        if count:
            entries = entries[:count]
        return entries

    async def xrevrange(self, key, count=None):
        entries = list(reversed(self._streams.get(key, [])))
        if count:
            entries = entries[:count]
        return entries

    async def xread(self, streams, block=None, count=None):
        results = []
        for key, last_id in streams.items():
            entries = self._streams.get(key, [])
            new_entries = [(eid, f) for eid, f in entries if eid > last_id]
            if count:
                new_entries = new_entries[:count]
            if new_entries:
                results.append((key, new_entries))
        return results if results else None

    async def ping(self):
        return True


class TestStreamStateService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        self.patcher = patch(
            "app.services.stream_state_service.get_redis_pool",
            return_value=self.fake_redis,
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    async def test_init_stream_creates_meta_and_start_entry(self):
        from app.services.stream_state_service import init_stream
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")

        meta = await self.fake_redis.hgetall("stream:meta:conv-1")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(meta["user_id"], "user-1")
        self.assertEqual(meta["message_id"], "msg-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1]["type"], "start")

    async def test_append_chunk_writes_to_stream(self):
        from app.services.stream_state_service import init_stream, append_chunk
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "Hello", "blk-1")
        await append_chunk("conv-1", "answering", " World", "blk-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(len(entries), 3)  # start + 2 chunks
        self.assertEqual(entries[1][1]["content"], "Hello")
        self.assertEqual(entries[2][1]["content"], " World")

    async def test_finalize_stream_success_writes_done(self):
        from app.services.stream_state_service import init_stream, finalize_stream
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await finalize_stream("conv-1", success=True)

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(entries[-1][1]["type"], "done")
        meta = await self.fake_redis.hgetall("stream:meta:conv-1")
        self.assertEqual(meta["status"], "done")

    async def test_finalize_stream_error_preserves_content(self):
        from app.services.stream_state_service import init_stream, append_chunk, finalize_stream
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "部分内容", "blk-1")
        await finalize_stream("conv-1", success=False, error_msg="LLM 超时")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        types = [e[1]["type"] for e in entries]
        self.assertIn("answering", types)
        self.assertEqual(types[-1], "error")

    async def test_get_stream_meta_returns_none_when_not_exists(self):
        from app.services.stream_state_service import get_stream_meta
        result = await get_stream_meta("nonexistent")
        self.assertIsNone(result)

    async def test_check_lock_owner(self):
        from app.services.stream_state_service import init_stream, check_lock_owner
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        self.assertTrue(await check_lock_owner("conv-1", "task-1"))
        self.assertFalse(await check_lock_owner("conv-1", "task-other"))

    async def test_read_stream_chunks_yields_entries(self):
        from app.services.stream_state_service import init_stream, append_chunk, finalize_stream, read_stream_chunks
        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "reasoning", "思考中", "blk-t")
        await append_chunk("conv-1", "answering", "回答", "blk-c")
        await finalize_stream("conv-1", success=True)

        chunks = []
        async for chunk in read_stream_chunks("conv-1"):
            chunks.append(chunk)

        types = [c["type"] for c in chunks]
        self.assertIn("start", types)
        self.assertIn("reasoning", types)
        self.assertIn("answering", types)
        self.assertIn("done", types)

    async def test_redis_unavailable_degrades_gracefully(self):
        self.patcher.stop()
        with patch("app.services.stream_state_service.get_redis_pool", return_value=None):
            from app.services.stream_state_service import (
                init_stream, append_chunk, finalize_stream, get_stream_meta,
            )
            await init_stream("conv-1", "user", "model", "msg", "task")
            result = await append_chunk("conv-1", "answering", "test", "blk")
            self.assertIsNone(result)
            await finalize_stream("conv-1", success=True)
            meta = await get_stream_meta("conv-1")
            self.assertIsNone(meta)
        self.patcher.start()


if __name__ == "__main__":
    unittest.main()
