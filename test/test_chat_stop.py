import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.chat import stop_stream
from app.schemas.chat import StopStreamRequest, TextBlock


class StopStreamSchemaTests(unittest.TestCase):
    def test_partial_content_parses_content_blocks_and_defaults_empty(self):
        request = StopStreamRequest(
            partial_content=[{"type": "text", "id": "answer-1", "text": "半截回答"}],
        )

        self.assertEqual(request.partial_content, [TextBlock(type="text", id="answer-1", text="半截回答")])
        self.assertEqual(StopStreamRequest().partial_content, [])


class StopStreamApiTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _request():
        return SimpleNamespace(state=SimpleNamespace(request_id="req-1"))

    async def test_empty_partial_body_uses_task_safe_cancel_without_db_persist(self):
        chat_service = MagicMock()
        meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }
        cancel_redis = AsyncMock(return_value=True)
        with (
            patch(
                "app.api.chat._read_stream_meta_strict",
                new=AsyncMock(return_value=(object(), meta)),
            ) as read_meta,
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)) as claim_stop,
            patch(
                "app.api.chat.release_stream_stop_guard",
                new=AsyncMock(return_value=True),
            ) as release_guard,
            patch("app.api.chat.cancel_task", return_value=True) as cancel_local,
            patch("app.api.chat.cancel_stream", new=cancel_redis),
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": True})
        read_meta.assert_awaited_once_with("conv-1")
        claim_stop.assert_awaited_once_with("conv-1", "msg-1", "task-1")
        release_guard.assert_awaited_once_with("conv-1", "task-1")
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        cancel_local.assert_called_once_with("conv-1", "task-1")
        cancel_redis.assert_awaited_once_with("conv-1", "msg-1", "task-1")

    async def test_none_body_keeps_legacy_cancel_flow(self):
        chat_service = MagicMock()
        cancel_redis = AsyncMock(return_value=False)
        with (
            patch("app.api.chat._read_stream_meta_strict", new=AsyncMock()) as read_meta,
            patch("app.api.chat.claim_stream_stop", new=AsyncMock()) as claim_stop,
            patch("app.api.chat.release_stream_stop_guard", new=AsyncMock()) as release_guard,
            patch("app.api.chat.cancel_task", return_value=True) as cancel_local,
            patch("app.api.chat.cancel_stream", new=cancel_redis),
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=None,
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": True})
        read_meta.assert_not_awaited()
        claim_stop.assert_not_awaited()
        release_guard.assert_not_awaited()
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        cancel_local.assert_called_once_with("conv-1")
        cancel_redis.assert_awaited_once_with("conv-1", "msg-1")

    async def test_partial_freezes_redis_and_local_task_before_persist_then_releases_guard(self):
        calls = []
        chat_service = MagicMock()

        def persist_partial(**_kwargs):
            calls.append("persist")
            return True

        chat_service.persist_stream_partial_before_stop.side_effect = persist_partial

        async def claim_stop(_conversation_id, _message_id, _task_id):
            calls.append("claim")
            return True

        async def release_guard(_conversation_id, _task_id):
            calls.append("release")
            return True

        def cancel_local(_conversation_id, _task_id):
            calls.append("local_cancel")
            return True

        async def cancel_redis(_conversation_id, _message_id, _task_id):
            calls.append("redis_cancel")
            return True

        meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }
        partial = [TextBlock(type="text", id="answer-1", text="半截回答")]
        with (
            patch("app.api.chat._read_stream_meta_strict", new=AsyncMock(return_value=(object(), meta))),
            patch("app.api.chat.claim_stream_stop", side_effect=claim_stop),
            patch("app.api.chat.release_stream_stop_guard", side_effect=release_guard),
            patch("app.api.chat.cancel_task", side_effect=cancel_local),
            patch("app.api.chat.cancel_stream", side_effect=cancel_redis),
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(partial_content=partial),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(calls, ["claim", "redis_cancel", "local_cancel", "persist", "release"])
        self.assertEqual(response.data, {"cancelled": True})
        chat_service.persist_stream_partial_before_stop.assert_called_once_with(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=partial,
            stream_meta=meta,
        )

    async def test_partial_old_task_claim_failure_does_not_write_db_or_cancel(self):
        chat_service = MagicMock()
        meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-old",
            "model": "gpt-4",
        }
        with (
            patch("app.api.chat._read_stream_meta_strict", new=AsyncMock(return_value=(object(), meta))),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=False)) as claim_stop,
            patch("app.api.chat.release_stream_stop_guard", new=AsyncMock()) as release_guard,
            patch("app.api.chat.cancel_task") as cancel_local,
            patch("app.api.chat.cancel_stream", new=AsyncMock()) as cancel_redis,
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(
                    partial_content=[TextBlock(type="text", id="answer-1", text="旧 partial")]
                ),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": False})
        claim_stop.assert_awaited_once_with("conv-1", "msg-1", "task-old")
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        cancel_redis.assert_not_awaited()
        cancel_local.assert_not_called()
        release_guard.assert_not_awaited()

    async def test_partial_db_failure_happens_after_freeze_and_still_releases_guard(self):
        calls = []
        chat_service = MagicMock()
        chat_service.persist_stream_partial_before_stop.side_effect = RuntimeError("db failed")
        meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }

        async def release_guard(_conversation_id, _task_id):
            calls.append("release")
            return True

        def cancel_local(_conversation_id, _task_id):
            calls.append("local_cancel")
            return True

        async def cancel_redis(_conversation_id, _message_id, _task_id):
            calls.append("redis_cancel")
            return True

        def persist_partial(**_kwargs):
            calls.append("persist")
            raise RuntimeError("db failed")

        chat_service.persist_stream_partial_before_stop.side_effect = persist_partial

        with (
            patch("app.api.chat._read_stream_meta_strict", new=AsyncMock(return_value=(object(), meta))),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)),
            patch("app.api.chat.release_stream_stop_guard", side_effect=release_guard),
            patch("app.api.chat.cancel_task", side_effect=cancel_local) as cancel_local_mock,
            patch("app.api.chat.cancel_stream", side_effect=cancel_redis) as cancel_redis_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "db failed"):
                await stop_stream(
                    "conv-1",
                    request=self._request(),
                    stop_request=StopStreamRequest(
                        partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")]
                    ),
                    message_id="msg-1",
                    chat_service=chat_service,
                    current_user=SimpleNamespace(id="user-1"),
                )

        self.assertEqual(calls, ["redis_cancel", "local_cancel", "persist", "release"])
        cancel_redis_mock.assert_awaited_once_with("conv-1", "msg-1", "task-1")
        cancel_local_mock.assert_called_once_with("conv-1", "task-1")

    async def test_partial_does_not_cancel_local_task_if_redis_cas_says_stream_was_replaced(self):
        chat_service = MagicMock()
        initial_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-old",
            "model": "gpt-4",
        }
        replaced_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-new",
            "model": "gpt-4",
        }
        read_meta = AsyncMock(side_effect=[(object(), initial_meta), (object(), replaced_meta)])
        with (
            patch("app.api.chat._read_stream_meta_strict", new=read_meta),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)),
            patch("app.api.chat.release_stream_stop_guard", new=AsyncMock(return_value=True)) as release_guard,
            patch("app.api.chat.cancel_task", return_value=True) as cancel_local,
            patch("app.api.chat.cancel_stream", new=AsyncMock(return_value=False)) as cancel_redis,
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(
                    partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")]
                ),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": False})
        cancel_redis.assert_awaited_once_with("conv-1", "msg-1", "task-old")
        cancel_local.assert_not_called()
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        release_guard.assert_awaited_once_with("conv-1", "task-old")
        self.assertEqual(read_meta.await_count, 2)

    async def test_partial_cancel_false_with_done_meta_does_not_persist(self):
        chat_service = MagicMock()
        initial_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }
        done_meta = {**initial_meta, "status": "done"}
        with (
            patch(
                "app.api.chat._read_stream_meta_strict",
                new=AsyncMock(side_effect=[(object(), initial_meta), (object(), done_meta)]),
            ),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)),
            patch("app.api.chat.release_stream_stop_guard", new=AsyncMock(return_value=True)) as release_guard,
            patch("app.api.chat.cancel_task") as cancel_local,
            patch("app.api.chat.cancel_stream", new=AsyncMock(return_value=False)) as cancel_redis,
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(
                    partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")]
                ),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": False})
        cancel_redis.assert_awaited_once_with("conv-1", "msg-1", "task-1")
        cancel_local.assert_not_called()
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        release_guard.assert_awaited_once_with("conv-1", "task-1")

    async def test_partial_cancel_ambiguous_but_cancelled_meta_continues_persist(self):
        calls = []
        chat_service = MagicMock()

        def persist_partial(**_kwargs):
            calls.append("persist")
            return True

        chat_service.persist_stream_partial_before_stop.side_effect = persist_partial
        initial_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }
        cancelled_meta = {**initial_meta, "status": "cancelled"}

        def cancel_local(_conversation_id, _task_id):
            calls.append("local_cancel")
            return True

        async def release_guard(_conversation_id, _task_id):
            calls.append("release")
            return True

        with (
            patch(
                "app.api.chat._read_stream_meta_strict",
                new=AsyncMock(side_effect=[(object(), initial_meta), (object(), cancelled_meta)]),
            ),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)),
            patch("app.api.chat.release_stream_stop_guard", side_effect=release_guard),
            patch("app.api.chat.cancel_task", side_effect=cancel_local),
            patch("app.api.chat.cancel_stream", new=AsyncMock(return_value=False)),
        ):
            response = await stop_stream(
                "conv-1",
                request=self._request(),
                stop_request=StopStreamRequest(
                    partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")]
                ),
                message_id="msg-1",
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(response.data, {"cancelled": True})
        self.assertEqual(calls, ["local_cancel", "persist", "release"])

    async def test_partial_cancel_ambiguous_and_meta_reread_failure_returns_retryable_error(self):
        from app.schemas.response import ApiException, ErrorCode

        chat_service = MagicMock()
        initial_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt-4",
        }
        reread_error = ApiException.service_unavailable(
            "流式连接暂时不可用，请稍后重试",
            code=ErrorCode.STREAM_RECONNECT_UNAVAILABLE,
        )
        with (
            patch(
                "app.api.chat._read_stream_meta_strict",
                new=AsyncMock(side_effect=[(object(), initial_meta), reread_error]),
            ),
            patch("app.api.chat.claim_stream_stop", new=AsyncMock(return_value=True)),
            patch("app.api.chat.release_stream_stop_guard", new=AsyncMock(return_value=True)) as release_guard,
            patch("app.api.chat.cancel_task") as cancel_local,
            patch("app.api.chat.cancel_stream", new=AsyncMock(return_value=False)),
        ):
            with self.assertRaises(ApiException) as raised:
                await stop_stream(
                    "conv-1",
                    request=self._request(),
                    stop_request=StopStreamRequest(
                        partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")]
                    ),
                    message_id="msg-1",
                    chat_service=chat_service,
                    current_user=SimpleNamespace(id="user-1"),
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_RECONNECT_UNAVAILABLE")
        cancel_local.assert_not_called()
        chat_service.persist_stream_partial_before_stop.assert_not_called()
        release_guard.assert_awaited_once_with("conv-1", "task-1")


if __name__ == "__main__":
    unittest.main()
