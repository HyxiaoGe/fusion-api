"""
流状态缓存服务单元测试（Redis Stream 版本）

使用 fakeredis 模拟 Redis Stream + Lua 脚本（finalize/cancel 走 Lua eval）。
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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

        result = await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")

        self.assertTrue(result.ok)
        self.assertIsNone(result.error_code)

        meta = await self.fake_redis.hgetall("stream:meta:conv-1")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(meta["user_id"], "user-1")
        self.assertEqual(meta["message_id"], "msg-1")
        self.assertEqual(meta["task_id"], "task-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1]["type"], "start")

    async def test_append_chunk_writes_to_stream(self):
        from app.services.stream_state_service import append_chunk, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "Hello", "blk-1", task_id="task-1")
        await append_chunk("conv-1", "answering", " World", "blk-1", task_id="task-1")

        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(len(entries), 3)  # start + 2 chunks
        self.assertEqual(entries[1][1]["content"], "Hello")
        self.assertEqual(entries[2][1]["content"], " World")

    async def test_finalize_stream_success_writes_done(self):
        from app.services.stream_state_service import finalize_stream, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        finalized = await finalize_stream("conv-1", success=True, task_id="task-1")

        self.assertTrue(finalized)
        entries = await self.fake_redis.xrange("stream:chunks:conv-1")
        self.assertEqual(entries[-1][1]["type"], "done")
        meta = await self.fake_redis.hgetall("stream:meta:conv-1")
        self.assertEqual(meta["status"], "done")

    async def test_finalize_stream_error_preserves_content(self):
        from app.services.stream_state_service import append_chunk, finalize_stream, init_stream

        await init_stream("conv-1", "user-1", "gpt-4", "msg-1", "task-1")
        await append_chunk("conv-1", "answering", "部分内容", "blk-1", task_id="task-1")
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
        await append_chunk("conv-1", "reasoning", "思考中", "blk-t", task_id="task-1")
        await append_chunk("conv-1", "answering", "回答", "blk-c", task_id="task-1")
        await finalize_stream("conv-1", success=True, task_id="task-1")

        chunks = []
        async for chunk in read_stream_chunks(
            "conv-1",
            expected_message_id="msg-1",
            expected_task_id="task-1",
        ):
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

            init_result = await init_stream("conv-1", "user", "model", "msg", "task")
            self.assertFalse(init_result.ok)
            self.assertEqual(init_result.error_code, "redis_unavailable")
            result = await append_chunk("conv-1", "answering", "test", "blk", task_id="task")
            self.assertIsNone(result)
            await finalize_stream("conv-1", success=True)
            meta = await get_stream_meta("conv-1")
            self.assertIsNone(meta)
        self.patcher.start()

    async def test_init_stream_returns_failure_and_cleans_partial_state(self):
        from app.core.redis import LUA_INIT_STREAM
        from app.services.stream_state_service import init_stream

        redis = AsyncMock()
        redis.eval.side_effect = RuntimeError("redis write failed")
        with patch("app.services.stream_state_service.get_redis_pool", return_value=redis):
            result = await init_stream("conv-init-fail", "user", "model", "msg", "task")

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "stream_init_failed")
        self.assertEqual(redis.eval.await_count, 2)
        self.assertEqual(redis.eval.await_args_list[0].args[0], LUA_INIT_STREAM)
        redis.delete.assert_not_called()

    async def test_late_init_cleanup_does_not_delete_newer_task_state(self):
        from app.core.redis import LOCK_TTL, LUA_CLEANUP_STREAM_INIT, LUA_INIT_STREAM, STREAM_CHUNK_TTL
        from app.services.stream_state_service import init_stream

        class AmbiguousInitRedis:
            def __init__(self, backing):
                self.backing = backing
                self.first_init = True

            async def eval(self, script, numkeys, *args):
                if script == LUA_INIT_STREAM and self.first_init:
                    self.first_init = False
                    await self.backing.eval(script, numkeys, *args)
                    await self.backing.eval(
                        LUA_INIT_STREAM,
                        3,
                        "stream:lock:conv-race",
                        "stream:chunks:conv-race",
                        "stream:meta:conv-race",
                        "task-new",
                        "user-new",
                        "model-new",
                        "msg-new",
                        "conv-race",
                        "2",
                        str(LOCK_TTL),
                        str(STREAM_CHUNK_TTL),
                    )
                    raise ConnectionError("旧 init 响应丢失")
                if script == LUA_CLEANUP_STREAM_INIT:
                    return await self.backing.eval(script, numkeys, *args)
                raise AssertionError("unexpected script")

        with patch(
            "app.services.stream_state_service.get_redis_pool",
            return_value=AmbiguousInitRedis(self.fake_redis),
        ):
            result = await init_stream("conv-race", "user-old", "model-old", "msg-old", "task-old")

        self.assertFalse(result.ok)
        self.assertEqual(await self.fake_redis.get("stream:lock:conv-race"), "task-new")
        meta = await self.fake_redis.hgetall("stream:meta:conv-race")
        self.assertEqual(meta["message_id"], "msg-new")
        self.assertEqual(meta["task_id"], "task-new")
        entries = await self.fake_redis.xrange("stream:chunks:conv-race")
        self.assertEqual([entry[1]["type"] for entry in entries], ["start"])

    async def test_init_stream_clears_previous_terminal_meta_fields(self):
        from app.services.stream_state_service import init_stream

        await self.fake_redis.hset(
            "stream:meta:conv-reset-meta",
            mapping={
                "status": "error",
                "message_id": "msg-old",
                "task_id": "task-old",
                "error_code": "stream_interrupted",
                "reason": "orphaned_stream",
                "ended_at": "1",
            },
        )

        result = await init_stream("conv-reset-meta", "user", "model", "msg-new", "task-new")

        self.assertTrue(result.ok)
        meta = await self.fake_redis.hgetall("stream:meta:conv-reset-meta")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(meta["message_id"], "msg-new")
        self.assertEqual(meta["task_id"], "task-new")
        self.assertNotIn("error_code", meta)
        self.assertNotIn("reason", meta)
        self.assertNotIn("ended_at", meta)

    async def test_read_stream_chunks_emits_structured_error_when_stream_is_orphaned(self):
        from app.services.stream_state_service import read_stream_chunks

        await self.fake_redis.xadd(
            "stream:chunks:conv-orphan",
            {"type": "start", "content": "", "block_id": ""},
        )
        await self.fake_redis.hset(
            "stream:meta:conv-orphan",
            mapping={
                "status": "streaming",
                "user_id": "user-orphan",
                "message_id": "msg-orphan",
                "task_id": "task-orphan",
            },
        )
        original_xread = self.fake_redis.xread
        self.fake_redis.xread = AsyncMock(return_value=[])
        try:
            chunk = await anext(
                read_stream_chunks(
                    "conv-orphan",
                    expected_message_id="msg-orphan",
                    expected_task_id="task-orphan",
                )
            )
        finally:
            self.fake_redis.xread = original_xread

        self.assertEqual(chunk["type"], "error")
        payload = json.loads(chunk["content"])
        self.assertEqual(payload["code"], "stream_interrupted")
        self.assertEqual(payload["data"]["reason"], "orphaned_stream")
        meta = await self.fake_redis.hgetall("stream:meta:conv-orphan")
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["error_code"], "stream_interrupted")
        self.assertEqual(meta["reason"], "orphaned_stream")
        self.assertFalse(await self.fake_redis.exists("stream:lock:conv-orphan"))

        from app.api.chat import get_stream_status_endpoint

        response = await get_stream_status_endpoint(
            "conv-orphan",
            request=SimpleNamespace(state=SimpleNamespace(request_id="req-1")),
            current_user=SimpleNamespace(id="user-orphan"),
        )
        self.assertEqual(response.data, {"status": "error"})

    async def test_old_reader_reports_replaced_without_mutating_new_stream(self):
        from app.services.stream_state_service import init_stream, read_stream_chunks

        await init_stream("conv-replaced", "user", "model", "msg-old", "task-old")
        await init_stream("conv-replaced", "user", "model", "msg-new", "task-new")

        chunk = await anext(
            read_stream_chunks(
                "conv-replaced",
                expected_message_id="msg-old",
                expected_task_id="task-old",
            )
        )

        payload = json.loads(chunk["content"])
        self.assertEqual(payload["code"], "stream_interrupted")
        self.assertEqual(payload["data"]["reason"], "stream_replaced")
        meta = await self.fake_redis.hgetall("stream:meta:conv-replaced")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(meta["message_id"], "msg-new")
        self.assertEqual(await self.fake_redis.get("stream:lock:conv-replaced"), "task-new")
        entries = await self.fake_redis.xrange("stream:chunks:conv-replaced")
        self.assertEqual([entry[1]["type"] for entry in entries], ["start"])

    async def test_old_reader_with_same_message_id_does_not_mutate_new_continuation(self):
        from app.services.stream_state_service import init_stream, read_stream_chunks

        await init_stream("conv-continued", "user", "model", "msg-shared", "task-old")
        await init_stream("conv-continued", "user", "model", "msg-shared", "task-new")

        chunk = await anext(
            read_stream_chunks(
                "conv-continued",
                expected_message_id="msg-shared",
                expected_task_id="task-old",
            )
        )

        payload = json.loads(chunk["content"])
        self.assertEqual(payload["data"]["reason"], "stream_replaced")
        meta = await self.fake_redis.hgetall("stream:meta:conv-continued")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(meta["message_id"], "msg-shared")
        self.assertEqual(meta["task_id"], "task-new")
        self.assertEqual(await self.fake_redis.get("stream:lock:conv-continued"), "task-new")
        entries = await self.fake_redis.xrange("stream:chunks:conv-continued")
        self.assertEqual([entry[1]["type"] for entry in entries], ["start"])

    async def test_reader_does_not_overwrite_finalize_race(self):
        from app.services.stream_state_service import finalize_stream, init_stream, read_stream_chunks

        await init_stream("conv-finalized", "user", "model", "msg-finalized", "task-finalized")
        self.assertTrue(await finalize_stream("conv-finalized", success=True, task_id="task-finalized"))

        chunks = [
            chunk
            async for chunk in read_stream_chunks(
                "conv-finalized",
                expected_message_id="msg-finalized",
                expected_task_id="task-finalized",
            )
        ]

        self.assertEqual(chunks[-1]["type"], "done")
        meta = await self.fake_redis.hgetall("stream:meta:conv-finalized")
        self.assertEqual(meta["status"], "done")
        entries = await self.fake_redis.xrange("stream:chunks:conv-finalized")
        self.assertEqual([entry[1]["type"] for entry in entries], ["start", "done"])

    async def test_finalize_stream_returns_false_for_replaced_task(self):
        from app.services.stream_state_service import finalize_stream, init_stream

        await init_stream("conv-finalize-cas", "user", "model", "msg-new", "task-new")

        finalized = await finalize_stream("conv-finalize-cas", success=False, task_id="task-old")

        self.assertFalse(finalized)
        meta = await self.fake_redis.hgetall("stream:meta:conv-finalize-cas")
        self.assertEqual(meta["status"], "streaming")
        self.assertEqual(await self.fake_redis.get("stream:lock:conv-finalize-cas"), "task-new")

    async def test_append_chunk_raises_after_consecutive_failures_and_resets_after_success(self):
        from app.services.stream_state_service import (
            StreamWriteUnavailableError,
            _append_failure_counts,
            append_chunk,
            finalize_stream,
        )

        redis = AsyncMock()
        redis.eval.side_effect = [
            RuntimeError("down-1"),
            RuntimeError("down-2"),
            [1, "1-0"],
            RuntimeError("down-3"),
        ]
        with patch("app.services.stream_state_service.get_redis_pool", return_value=redis):
            self.assertIsNone(await append_chunk("conv-flaky", "answering", "a", "blk", task_id="task-flaky"))
            self.assertIsNone(await append_chunk("conv-flaky", "answering", "b", "blk", task_id="task-flaky"))
            self.assertEqual(
                await append_chunk("conv-flaky", "answering", "c", "blk", task_id="task-flaky"),
                "1-0",
            )
            self.assertIsNone(await append_chunk("conv-flaky", "answering", "d", "blk", task_id="task-flaky"))
        self.assertIn(("conv-flaky", "task-flaky"), _append_failure_counts)
        redis.eval.side_effect = RuntimeError("finalize down")
        with patch("app.services.stream_state_service.get_redis_pool", return_value=redis):
            self.assertFalse(await finalize_stream("conv-flaky", success=False, task_id="task-flaky"))
        self.assertNotIn(("conv-flaky", "task-flaky"), _append_failure_counts)

        redis.eval.side_effect = RuntimeError("still down")
        with patch("app.services.stream_state_service.get_redis_pool", return_value=redis):
            self.assertIsNone(await append_chunk("conv-terminal", "answering", "a", "blk", task_id="task-terminal"))
            self.assertIsNone(await append_chunk("conv-terminal", "answering", "b", "blk", task_id="task-terminal"))
            with self.assertRaises(StreamWriteUnavailableError):
                await append_chunk("conv-terminal", "answering", "c", "blk", task_id="task-terminal")
        self.assertNotIn(("conv-terminal", "task-terminal"), _append_failure_counts)

    async def test_append_fencing_rejects_old_task_deltas_and_agent_events(self):
        from app.services.stream.tool_executor import AgentEventRedisWriter
        from app.services.stream_state_service import StreamOwnershipLostError, append_chunk, init_stream

        await init_stream("conv-fenced", "user", "model", "msg-old", "task-old")
        await init_stream("conv-fenced", "user", "model", "msg-new", "task-new")

        with self.assertRaises(StreamOwnershipLostError):
            await append_chunk(
                "conv-fenced",
                "answering",
                "旧回答",
                "blk-old",
                task_id="task-old",
            )
        writer = AgentEventRedisWriter()
        with self.assertRaises(StreamOwnershipLostError):
            await writer.append_chunk(
                "conv-fenced",
                "task-old",
                "agent_event",
                {"type": "run_progress_updated", "text": "旧事件"},
            )

        await append_chunk(
            "conv-fenced",
            "answering",
            "新回答",
            "blk-new",
            task_id="task-new",
        )
        await writer.append_chunk(
            "conv-fenced",
            "task-new",
            "agent_event",
            {"type": "run_progress_updated", "text": "新事件"},
        )

        entries = await self.fake_redis.xrange("stream:chunks:conv-fenced")
        self.assertEqual([entry[1]["type"] for entry in entries], ["start", "answering", "agent_event"])
        self.assertEqual(entries[1][1]["content"], "新回答")
        self.assertIn("新事件", entries[2][1]["content"])


class AppendChunkExtrasTests(unittest.IsolatedAsyncioTestCase):
    """spec §4.6: append_chunk **extras 写入 Redis Stream entry 额外 hash 字段"""

    async def test_extras_written_to_redis_hash(self):
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            redis.eval.return_value = [1, "1-0"]
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "reasoning", "hi", "b1", task_id="task-1", run_id="r1", step_id="s1")
            args = redis.eval.await_args.args
            fields = dict(zip(args[7::2], args[8::2]))
            self.assertEqual(fields["type"], "reasoning")
            self.assertEqual(fields["content"], "hi")
            self.assertEqual(fields["block_id"], "b1")
            self.assertEqual(fields["run_id"], "r1")
            self.assertEqual(fields["step_id"], "s1")

    async def test_extras_none_skipped(self):
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            redis.eval.return_value = [1, "1-0"]
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "reasoning", "hi", "b1", task_id="task-1", run_id=None, step_id="s1")
            args = redis.eval.await_args.args
            fields = dict(zip(args[7::2], args[8::2]))
            self.assertNotIn("run_id", fields)
            self.assertEqual(fields["step_id"], "s1")

    async def test_call_without_extras_still_writes_required_fields(self):
        from unittest.mock import AsyncMock, patch

        with patch("app.services.stream_state_service.get_redis_pool") as gp:
            redis = AsyncMock()
            redis.eval.return_value = [1, "1-0"]
            gp.return_value = redis
            from app.services.stream_state_service import append_chunk

            await append_chunk("c1", "answering", "text", "b2", task_id="task-1")
            args = redis.eval.await_args.args
            fields = dict(zip(args[7::2], args[8::2]))
            self.assertEqual(fields, {"type": "answering", "content": "text", "block_id": "b2"})


if __name__ == "__main__":
    unittest.main()
