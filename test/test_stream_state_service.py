"""
流状态缓存服务单元测试

使用 unittest.mock 模拟 Redis，不依赖真实连接。
"""
import json
import unittest
from unittest.mock import AsyncMock, patch, MagicMock


class FakeRedis:
    """轻量级 Redis mock，模拟 get/set/delete"""

    def __init__(self):
        self._store = {}

    async def get(self, key):
        return self._store.get(key)

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def delete(self, *keys):
        for key in keys:
            self._store.pop(key, None)

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

    async def test_set_stream_start_writes_initial_state(self):
        from app.services.stream_state_service import set_stream_start

        await set_stream_start("conv-123", "user-456", "gpt-4")

        raw = await self.fake_redis.get("stream:conv-123")
        self.assertIsNotNone(raw)
        state = json.loads(raw)
        self.assertEqual(state["status"], "streaming")
        self.assertEqual(state["user_id"], "user-456")
        self.assertEqual(state["model"], "gpt-4")
        self.assertEqual(state["content_blocks"], [])

    async def test_append_stream_chunk_merges_same_type(self):
        from app.services.stream_state_service import set_stream_start, append_stream_chunk

        await set_stream_start("conv-123", "user-456", "gpt-4")
        await append_stream_chunk("conv-123", "answering", "Hello")
        await append_stream_chunk("conv-123", "answering", " World")

        raw = await self.fake_redis.get("stream:conv-123")
        state = json.loads(raw)
        self.assertEqual(len(state["content_blocks"]), 1)
        self.assertEqual(state["content_blocks"][0]["content"], "Hello World")

    async def test_append_stream_chunk_creates_new_block_for_different_type(self):
        from app.services.stream_state_service import set_stream_start, append_stream_chunk

        await set_stream_start("conv-123", "user-456", "gpt-4")
        await append_stream_chunk("conv-123", "reasoning", "思考中")
        await append_stream_chunk("conv-123", "answering", "回答内容")

        raw = await self.fake_redis.get("stream:conv-123")
        state = json.loads(raw)
        self.assertEqual(len(state["content_blocks"]), 2)
        self.assertEqual(state["content_blocks"][0]["type"], "reasoning")
        self.assertEqual(state["content_blocks"][1]["type"], "answering")

    async def test_set_stream_complete_deletes_keys(self):
        from app.services.stream_state_service import set_stream_start, set_stream_complete

        await set_stream_start("conv-123", "user-456", "gpt-4")
        await set_stream_complete("conv-123")

        raw = await self.fake_redis.get("stream:conv-123")
        self.assertIsNone(raw)

    async def test_set_stream_error_preserves_content(self):
        from app.services.stream_state_service import (
            set_stream_start, append_stream_chunk, set_stream_error,
        )

        await set_stream_start("conv-123", "user-456", "gpt-4")
        await append_stream_chunk("conv-123", "answering", "部分内容")
        await set_stream_error("conv-123", "网络中断")

        raw = await self.fake_redis.get("stream:conv-123")
        state = json.loads(raw)
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["content_blocks"][0]["content"], "部分内容")
        self.assertEqual(state["error"], "网络中断")

    async def test_get_stream_status_returns_none_when_not_exists(self):
        from app.services.stream_state_service import get_stream_status

        result = await get_stream_status("nonexistent")
        self.assertIsNone(result)

    async def test_get_stream_status_returns_state(self):
        from app.services.stream_state_service import set_stream_start, get_stream_status

        await set_stream_start("conv-123", "user-456", "gpt-4")
        result = await get_stream_status("conv-123")

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "streaming")

    async def test_acquire_stream_lock_later_overwrites(self):
        from app.services.stream_state_service import acquire_stream_lock, is_lock_owner

        first_id = await acquire_stream_lock("conv-123")
        second_id = await acquire_stream_lock("conv-123")

        self.assertFalse(await is_lock_owner("conv-123", first_id))
        self.assertTrue(await is_lock_owner("conv-123", second_id))

    async def test_redis_unavailable_degrades_gracefully(self):
        """Redis 不可用时所有操作静默降级"""
        self.patcher.stop()
        with patch(
            "app.services.stream_state_service.get_redis_pool",
            return_value=None,
        ):
            from app.services.stream_state_service import (
                acquire_stream_lock, is_lock_owner, set_stream_start,
                append_stream_chunk, set_stream_complete, get_stream_status,
            )

            # 所有操作不抛异常
            request_id = await acquire_stream_lock("conv-123")
            self.assertIsInstance(request_id, str)
            self.assertTrue(await is_lock_owner("conv-123", request_id))
            await set_stream_start("conv-123", "user", "model")
            await append_stream_chunk("conv-123", "answering", "test")
            await set_stream_complete("conv-123")
            result = await get_stream_status("conv-123")
            self.assertIsNone(result)
        self.patcher.start()


if __name__ == "__main__":
    unittest.main()
