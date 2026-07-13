import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.response import ApiException
from app.services.chat_service import ChatService
from app.services.stream_state_service import StreamInitResult


class ChatContinueTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_agent_run_reuses_assistant_message_id(self):
        db = MagicMock()
        service = ChatService(db)
        conversation = SimpleNamespace(
            id="conv-1",
            user_id="user-1",
            model_id="deepseek-chat",
            messages=[],
        )
        service.conversation_service.get_conversation = MagicMock(return_value=conversation)
        continuation_context = SimpleNamespace(
            assistant_message=SimpleNamespace(sequence=42),
            initial_content_blocks=[],
            limits=SimpleNamespace(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        )
        service.stream_handler.generate_to_redis = AsyncMock()

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("deepseek/deepseek-chat", "deepseek", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True},
            ),
            patch("app.services.chat_service.build_continuation_context", return_value=continuation_context),
            patch("app.services.chat_service.get_stream_meta", new=AsyncMock(return_value=None)),
            patch(
                "app.services.chat_service.init_stream",
                new=AsyncMock(return_value=StreamInitResult(ok=True)),
            ) as init_stream_mock,
            patch("app.services.chat_service.register_task") as register_task_mock,
            patch("app.services.chat_service.asyncio.create_task") as create_task_mock,
        ):
            task = object()

            def close_created_coroutine(coro):
                coro.close()
                return task

            create_task_mock.side_effect = close_created_coroutine

            response = await service.continue_agent_run(
                conversation_id="conv-1",
                assistant_message_id="msg-1",
                user_id="user-1",
                previous_run_id="run-old",
                trace_id="trace-1",
            )

        init_stream_mock.assert_awaited_once()
        self.assertEqual(init_stream_mock.await_args.args[3], "msg-1")
        self.assertEqual(init_stream_mock.await_args.kwargs["stream_mode"], "continuation")
        self.assertEqual(init_stream_mock.await_args.kwargs["message_sequence"], 42)
        self.assertEqual(
            service.stream_handler.generate_to_redis.call_args.kwargs["assistant_message_sequence"],
            42,
        )
        register_task_mock.assert_called_once()
        self.assertIs(register_task_mock.call_args.args[1], task)
        self.assertEqual(response.media_type, "text/event-stream")

    async def test_continue_agent_run_rejects_missing_conversation(self):
        service = ChatService(MagicMock())
        service.conversation_service.get_conversation = MagicMock(return_value=None)

        with self.assertRaises(ApiException) as raised:
            await service.continue_agent_run(
                conversation_id="missing",
                assistant_message_id="msg-1",
                user_id="user-1",
                previous_run_id=None,
                trace_id="trace-1",
            )

        self.assertEqual(raised.exception.status_code, 404)

    async def test_continue_agent_run_rejects_active_stream(self):
        service = ChatService(MagicMock())
        service.conversation_service.get_conversation = MagicMock(
            return_value=SimpleNamespace(id="conv-1", user_id="user-1", model_id="deepseek-chat", messages=[])
        )

        with patch("app.services.chat_service.get_stream_meta", new=AsyncMock(return_value={"status": "streaming"})):
            with self.assertRaises(ApiException) as raised:
                await service.continue_agent_run(
                    conversation_id="conv-1",
                    assistant_message_id="msg-1",
                    user_id="user-1",
                    previous_run_id=None,
                    trace_id="trace-1",
                )

        self.assertEqual(raised.exception.status_code, 409)

    async def test_continue_agent_run_fails_before_starting_generation_when_stream_init_fails(self):
        service = ChatService(MagicMock())
        service.conversation_service.get_conversation = MagicMock(
            return_value=SimpleNamespace(
                id="conv-1",
                user_id="user-1",
                model_id="deepseek-chat",
                messages=[],
            )
        )
        continuation_context = SimpleNamespace(
            assistant_message=SimpleNamespace(sequence=42),
            initial_content_blocks=[],
            limits=SimpleNamespace(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("deepseek/deepseek-chat", "deepseek", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True},
            ),
            patch("app.services.chat_service.build_continuation_context", return_value=continuation_context),
            patch("app.services.chat_service.get_stream_meta", new=AsyncMock(return_value=None)),
            patch(
                "app.services.chat_service.init_stream",
                new=AsyncMock(
                    return_value=StreamInitResult(
                        ok=False,
                        error_code="stream_init_failed",
                        message="初始化失败",
                    )
                ),
            ),
            patch("app.services.chat_service.register_task") as register_task_mock,
            patch("app.services.chat_service.asyncio.create_task") as create_task_mock,
        ):
            with self.assertRaises(ApiException) as raised:
                await service.continue_agent_run(
                    conversation_id="conv-1",
                    assistant_message_id="msg-1",
                    user_id="user-1",
                    previous_run_id="run-old",
                    trace_id="trace-1",
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_UNAVAILABLE")
        create_task_mock.assert_not_called()
        register_task_mock.assert_not_called()

    async def test_continue_agent_run_route_requires_streaming(self):
        from app.api.chat import continue_agent_run
        from app.schemas.chat import ContinueAgentRunRequest

        request = SimpleNamespace(state=SimpleNamespace(request_id="trace-1"))
        chat_service = MagicMock()

        with self.assertRaises(ApiException) as raised:
            await continue_agent_run(
                conversation_id="conv-1",
                message_id="msg-1",
                continue_request=ContinueAgentRunRequest(stream=False),
                request=request,
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(raised.exception.status_code, 400)

    async def test_continue_agent_run_route_delegates_to_service(self):
        from app.api.chat import continue_agent_run
        from app.schemas.chat import ContinueAgentRunRequest

        response = object()
        chat_service = SimpleNamespace(continue_agent_run=AsyncMock(return_value=response))
        request = SimpleNamespace(state=SimpleNamespace(request_id="trace-1"))

        result = await continue_agent_run(
            conversation_id="conv-1",
            message_id="msg-1",
            continue_request=ContinueAgentRunRequest(previous_run_id="run-old"),
            request=request,
            chat_service=chat_service,
            current_user=SimpleNamespace(id="user-1"),
        )

        self.assertIs(result, response)
        chat_service.continue_agent_run.assert_awaited_once_with(
            conversation_id="conv-1",
            assistant_message_id="msg-1",
            user_id="user-1",
            previous_run_id="run-old",
            trace_id="trace-1",
        )


if __name__ == "__main__":
    unittest.main()
