import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.responses import StreamingResponse

from app.api.chat import reconnect_stream
from app.schemas.response import ApiException


class ChatStreamReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_returns_recoverable_503_when_redis_pool_is_missing(self):
        with patch("app.api.chat.get_redis_pool", return_value=None):
            with self.assertRaises(ApiException) as raised:
                await reconnect_stream("conv-1", current_user=SimpleNamespace(id="user-1"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_RECONNECT_UNAVAILABLE")

    async def test_reconnect_returns_recoverable_503_when_redis_read_fails(self):
        redis = SimpleNamespace(
            ping=AsyncMock(return_value=True),
            hgetall=AsyncMock(side_effect=RuntimeError("temporary redis error")),
        )
        with patch("app.api.chat.get_redis_pool", return_value=redis):
            with self.assertRaises(ApiException) as raised:
                await reconnect_stream("conv-1", current_user=SimpleNamespace(id="user-1"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_RECONNECT_UNAVAILABLE")

    async def test_reconnect_returns_recoverable_503_when_redis_ping_fails(self):
        redis = SimpleNamespace(
            ping=AsyncMock(side_effect=RuntimeError("temporary redis error")),
            hgetall=AsyncMock(),
        )
        with patch("app.api.chat.get_redis_pool", return_value=redis):
            with self.assertRaises(ApiException) as raised:
                await reconnect_stream("conv-1", current_user=SimpleNamespace(id="user-1"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_RECONNECT_UNAVAILABLE")
        redis.hgetall.assert_not_awaited()

    async def test_reconnect_returns_404_only_when_meta_is_really_missing(self):
        redis = SimpleNamespace(
            ping=AsyncMock(return_value=True),
            hgetall=AsyncMock(return_value={}),
        )
        with patch("app.api.chat.get_redis_pool", return_value=redis):
            with self.assertRaises(ApiException) as raised:
                await reconnect_stream("conv-1", current_user=SimpleNamespace(id="user-1"))

        self.assertEqual(raised.exception.status_code, 404)
        redis.hgetall.assert_awaited_once_with("stream:meta:conv-1")

    async def test_reconnect_keeps_other_users_stream_hidden_as_404(self):
        redis = SimpleNamespace(
            ping=AsyncMock(return_value=True),
            hgetall=AsyncMock(return_value={"user_id": "user-2", "message_id": "msg-1"}),
        )
        with patch("app.api.chat.get_redis_pool", return_value=redis):
            with self.assertRaises(ApiException) as raised:
                await reconnect_stream("conv-1", current_user=SimpleNamespace(id="user-1"))

        self.assertEqual(raised.exception.status_code, 404)
        redis.hgetall.assert_awaited_once_with("stream:meta:conv-1")

    async def test_reconnect_returns_sse_for_owned_stream(self):
        redis = SimpleNamespace(
            ping=AsyncMock(return_value=True),
            hgetall=AsyncMock(
                return_value={
                    "user_id": "user-1",
                    "message_id": "msg-1",
                    "task_id": "task-1",
                    "status": "streaming",
                }
            ),
        )
        with (
            patch("app.api.chat.get_redis_pool", return_value=redis),
            patch("app.api.chat.stream_redis_as_sse") as stream_sse,
        ):
            response = await reconnect_stream(
                "conv-1",
                last_entry_id="10-0",
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertIsInstance(response, StreamingResponse)
        self.assertEqual(response.media_type, "text/event-stream")
        stream_sse.assert_called_once_with(
            conversation_id="conv-1",
            message_id="msg-1",
            task_id="task-1",
            last_entry_id="10-0",
        )


if __name__ == "__main__":
    unittest.main()
