"""
流状态缓存服务单元测试（Redis Stream 版本）

使用 fakeredis 模拟 Redis Stream + Lua 脚本（finalize/cancel 走 Lua eval）。
"""

import unittest
from unittest.mock import patch

import fakeredis.aioredis


class TestStreamStateService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        self.patcher = patch(
            "app.services.stream_state_service.get_redis_pool",
            return_value=self.fake_redis,
        )
        self.patcher.start()

    async def asyncTearDown(self):
        await self.fake_redis.flushall()
        await self.fake_redis.aclose()

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
        from app.services.stream_state_service import append_chunk, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "Hello", "blk-1")
        await append_chunk("conv-1", "answering", " World", "blk-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(len(entries), 3)  # start + 2 chunks
        self.assertEqual(entries[1][1]["content"], "Hello")
        self.assertEqual(entries[2][1]["content"], " World")

    async def test_finalize_stream_success_writes_done(self):
        from app.services.stream_state_service import finalize_stream, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await finalize_stream("conv-1", success=True, task_id="task-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(entries[-1][1]["type"], "done")
        meta = await self.fake_redis.hgetall("stream:meta:conv-1")
        self.assertEqual(meta["status"], "done")

    async def test_finalize_stream_error_preserves_content(self):
        from app.services.stream_state_service import append_chunk, finalize_stream, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "部分内容", "blk-1")
        await finalize_stream("conv-1", success=False, error_msg="LLM 超时", task_id="task-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        types = [e[1]["type"] for e in entries]
        self.assertIn("answering", types)
        self.assertEqual(types[-1], "error")

    async def test_get_stream_meta_returns_none_when_not_exists(self):
        from app.services.stream_state_service import get_stream_meta

        result = await get_stream_meta("nonexistent")
        self.assertIsNone(result)

    async def test_check_lock_owner(self):
        from app.services.stream_state_service import check_lock_owner, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        self.assertTrue(await check_lock_owner("conv-1", "task-1"))
        self.assertFalse(await check_lock_owner("conv-1", "task-other"))

    async def test_read_stream_chunks_yields_entries(self):
        from app.services.stream_state_service import append_chunk, finalize_stream, init_stream, read_stream_chunks

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "reasoning", "思考中", "blk-t")
        await append_chunk("conv-1", "answering", "回答", "blk-c")
        await finalize_stream("conv-1", success=True, task_id="task-1")

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
                append_chunk,
                finalize_stream,
                get_stream_meta,
                init_stream,
            )

            await init_stream("conv-1", "user", "model", "msg", "task")
            result = await append_chunk("conv-1", "answering", "test", "blk")
            self.assertIsNone(result)
            await finalize_stream("conv-1", success=True)
            meta = await get_stream_meta("conv-1")
            self.assertIsNone(meta)
        self.patcher.start()


class AppendChunkExtrasTests(unittest.IsolatedAsyncioTestCase):
    """spec §4.6: append_chunk **extras 写入 Redis Stream entry 额外 hash 字段"""

    async def test_extras_written_to_redis_hash(self):
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "reasoning", "hi", "b1", run_id="r1", step_id="s1")
            # xadd 调用形式：xadd(key, fields)
            args = redis.xadd.call_args.args
            fields = args[1] if len(args) >= 2 else redis.xadd.call_args.kwargs.get("fields", {})
            self.assertEqual(fields["type"], "reasoning")
            self.assertEqual(fields["content"], "hi")
            self.assertEqual(fields["block_id"], "b1")
            self.assertEqual(fields["run_id"], "r1")
            self.assertEqual(fields["step_id"], "s1")

    async def test_extras_none_skipped(self):
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "reasoning", "hi", "b1", run_id=None, step_id="s1")
            args = redis.xadd.call_args.args
            fields = args[1] if len(args) >= 2 else redis.xadd.call_args.kwargs.get("fields", {})
            self.assertNotIn("run_id", fields)
            self.assertEqual(fields["step_id"], "s1")

    async def test_legacy_4_arg_call_still_works(self):
        """旧 4 参数调用方式（无 **extras）应继续工作"""
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "answering", "text", "b2")
            args = redis.xadd.call_args.args
            fields = args[1] if len(args) >= 2 else redis.xadd.call_args.kwargs.get("fields", {})
            self.assertEqual(fields, {"type": "answering", "content": "text", "block_id": "b2"})


if __name__ == "__main__":
    unittest.main()
