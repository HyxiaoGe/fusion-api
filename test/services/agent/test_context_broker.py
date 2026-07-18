import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis


class AgentContextBrokerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        self.redis_patcher = patch(
            "app.services.agent.context_broker.get_redis_pool",
            return_value=self.redis,
        )
        self.redis_patcher.start()

    async def asyncTearDown(self):
        await self.redis.flushall()
        await self.redis.aclose()

    def tearDown(self):
        self.redis_patcher.stop()

    async def _create_pending(self, *, expires_at=200.0):
        from app.services.agent.context_broker import create_context_request

        return await create_context_request(
            request_id="ctx-1",
            context_type="geolocation",
            purpose="nearby_search",
            reason="搜索当前位置附近的地点",
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            expires_at=expires_at,
        )

    async def _init_stream_meta(self, *, task_id="task-1"):
        await self.redis.hset(
            "stream:meta:conv-1",
            mapping={
                "status": "streaming",
                "user_id": "user-1",
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "task_id": task_id,
            },
        )

    async def test_provided_result_wakes_same_run_waiter_and_does_not_echo_location_in_outcome(self):
        from app.services.agent.context_broker import submit_context_result, wait_for_context_result

        pending = await self._create_pending()
        await self._init_stream_meta()
        waiter = asyncio.create_task(
            wait_for_context_result(
                pending,
                clock=lambda: 100.0,
                ownership_check=AsyncMock(return_value=True),
                poll_interval_seconds=0.01,
            )
        )

        submission = await submit_context_result(
            request_id="ctx-1",
            user_id="user-1",
            conversation_id="conv-1",
            run_id="run-1",
            status="provided",
            location={
                "latitude": 22.616,
                "longitude": 114.031,
                "accuracy_m": 25.0,
                "acquired_at": 99.0,
            },
            reason=None,
            now=100.0,
        )
        resolved = await asyncio.wait_for(waiter, timeout=1)

        self.assertEqual(submission.outcome, "accepted")
        self.assertEqual(resolved.status, "provided")
        self.assertEqual(resolved.location.latitude, 22.616)
        self.assertEqual(resolved.location.longitude, 114.031)
        self.assertNotIn("latitude", submission.model_dump_json())
        self.assertNotIn("longitude", submission.model_dump_json())

    async def test_same_result_is_idempotent_but_conflicting_result_is_rejected(self):
        from app.services.agent.context_broker import submit_context_result

        await self._create_pending()
        await self._init_stream_meta()
        kwargs = dict(
            request_id="ctx-1",
            user_id="user-1",
            conversation_id="conv-1",
            run_id="run-1",
            status="denied",
            location=None,
            reason="permission_denied",
            now=100.0,
        )

        first = await submit_context_result(**kwargs)
        duplicate = await submit_context_result(**kwargs)
        conflict = await submit_context_result(**{**kwargs, "status": "unavailable", "reason": "not_supported"})

        self.assertEqual(first.outcome, "accepted")
        self.assertEqual(duplicate.outcome, "idempotent")
        self.assertEqual(conflict.outcome, "conflict")

    async def test_rejects_wrong_user_run_replaced_task_and_expired_request(self):
        from app.services.agent.context_broker import submit_context_result

        cases = (
            ({"user_id": "user-other"}, "forbidden"),
            ({"run_id": "run-other"}, "forbidden"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                await self.redis.flushall()
                await self._create_pending()
                await self._init_stream_meta()
                result = await submit_context_result(
                    request_id="ctx-1",
                    user_id=overrides.get("user_id", "user-1"),
                    conversation_id="conv-1",
                    run_id=overrides.get("run_id", "run-1"),
                    status="denied",
                    location=None,
                    reason="permission_denied",
                    now=100.0,
                )
                self.assertEqual(result.outcome, expected)

        await self.redis.flushall()
        await self._create_pending()
        await self._init_stream_meta(task_id="task-new")
        stale = await submit_context_result(
            request_id="ctx-1",
            user_id="user-1",
            conversation_id="conv-1",
            run_id="run-1",
            status="denied",
            location=None,
            reason="permission_denied",
            now=100.0,
        )
        self.assertEqual(stale.outcome, "stale")

        await self.redis.flushall()
        await self._create_pending(expires_at=99.0)
        await self._init_stream_meta()
        expired = await submit_context_result(
            request_id="ctx-1",
            user_id="user-1",
            conversation_id="conv-1",
            run_id="run-1",
            status="denied",
            location=None,
            reason="permission_denied",
            now=100.0,
        )
        self.assertEqual(expired.outcome, "expired")

    async def test_wait_timeout_returns_timeout_and_ownership_loss_returns_unavailable(self):
        from app.services.agent.context_broker import wait_for_context_result

        expired_pending = await self._create_pending(expires_at=99.0)
        timed_out = await wait_for_context_result(
            expired_pending,
            clock=lambda: 100.0,
            ownership_check=AsyncMock(return_value=True),
            poll_interval_seconds=0.01,
        )
        self.assertEqual(timed_out.status, "timeout")

        await self.redis.flushall()
        pending = await self._create_pending()
        ownership_lost = await wait_for_context_result(
            pending,
            clock=lambda: 100.0,
            ownership_check=AsyncMock(return_value=False),
            poll_interval_seconds=0.01,
        )
        self.assertEqual(ownership_lost.status, "unavailable")
        self.assertEqual(ownership_lost.reason, "stream_replaced")


if __name__ == "__main__":
    unittest.main()
