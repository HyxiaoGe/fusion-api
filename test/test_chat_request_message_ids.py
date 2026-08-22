import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError
from starlette.responses import StreamingResponse

from app.api.chat import send_message
from app.schemas.chat import ChatRequest
from app.schemas.response import ApiException

USER_MESSAGE_ID = "11111111-1111-4111-8111-111111111111"
ASSISTANT_MESSAGE_ID = "22222222-2222-4222-8222-222222222222"
PREVIOUS_RUN_ID = "run-previous"


class ChatRequestMessageIdTests(unittest.TestCase):
    def test_accepts_distinct_uuid4_message_ids(self):
        request = ChatRequest(
            model_id="deepseek-chat",
            message="你好",
            user_message_id=USER_MESSAGE_ID,
            assistant_message_id=ASSISTANT_MESSAGE_ID,
        )

        self.assertEqual(request.user_message_id, USER_MESSAGE_ID)
        self.assertEqual(request.assistant_message_id, ASSISTANT_MESSAGE_ID)

    def test_old_client_can_omit_message_ids(self):
        request = ChatRequest(model_id="deepseek-chat", message="你好")

        self.assertIsNone(request.user_message_id)
        self.assertIsNone(request.assistant_message_id)
        self.assertIsNone(request.retry_user_message_id)
        self.assertIsNone(request.retry_assistant_message_id)
        self.assertIsNone(request.previous_run_id)

    def test_accepts_retry_ids_that_reuse_the_original_turn(self):
        request = ChatRequest(
            model_id="deepseek-chat",
            message="你好",
            conversation_id="conversation-1",
            user_message_id=USER_MESSAGE_ID,
            assistant_message_id=ASSISTANT_MESSAGE_ID,
            retry_user_message_id=USER_MESSAGE_ID,
            retry_assistant_message_id=ASSISTANT_MESSAGE_ID,
            previous_run_id=PREVIOUS_RUN_ID,
        )

        self.assertEqual(request.retry_user_message_id, USER_MESSAGE_ID)
        self.assertEqual(request.retry_assistant_message_id, ASSISTANT_MESSAGE_ID)
        self.assertEqual(request.previous_run_id, PREVIOUS_RUN_ID)

    def test_send_route_forwards_optional_previous_run_id(self):
        response = StreamingResponse(iter(()))
        chat_service = SimpleNamespace(process_message=AsyncMock(return_value=response))
        chat_request = ChatRequest(
            model_id="deepseek-chat",
            message="重试",
            conversation_id="conversation-1",
            retry_user_message_id=USER_MESSAGE_ID,
            previous_run_id=PREVIOUS_RUN_ID,
        )

        result = asyncio.run(
            send_message(
                chat_request=chat_request,
                request=SimpleNamespace(state=SimpleNamespace(request_id="run-new")),
                chat_service=chat_service,
                current_user=SimpleNamespace(id="user-1"),
            )
        )

        self.assertIs(result, response)
        self.assertEqual(chat_service.process_message.await_args.kwargs["previous_run_id"], PREVIOUS_RUN_ID)

    def test_send_route_preserves_stale_previous_run_conflict_contract(self):
        conflict = ApiException.conflict("所选 Agent 运行已不是最新执行，请刷新轨迹后重试")
        chat_service = SimpleNamespace(process_message=AsyncMock(side_effect=conflict))
        chat_request = ChatRequest(
            model_id="deepseek-chat",
            message="重试",
            conversation_id="conversation-1",
            retry_user_message_id=USER_MESSAGE_ID,
            previous_run_id=PREVIOUS_RUN_ID,
        )

        with self.assertRaises(ApiException) as raised:
            asyncio.run(
                send_message(
                    chat_request=chat_request,
                    request=SimpleNamespace(state=SimpleNamespace(request_id="run-new")),
                    chat_service=chat_service,
                    current_user=SimpleNamespace(id="user-1"),
                )
            )

        self.assertIs(raised.exception, conflict)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.code, "CONFLICT")

    def test_rejects_retry_without_conversation(self):
        with self.assertRaisesRegex(ValidationError, "conversation_id"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                retry_user_message_id=USER_MESSAGE_ID,
            )

    def test_rejects_retry_assistant_without_retry_user(self):
        with self.assertRaisesRegex(ValidationError, "retry_user_message_id"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                conversation_id="conversation-1",
                retry_assistant_message_id=ASSISTANT_MESSAGE_ID,
            )

    def test_rejects_previous_run_without_retry_turn(self):
        with self.assertRaisesRegex(ValidationError, "retry_user_message_id"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                conversation_id="conversation-1",
                previous_run_id=PREVIOUS_RUN_ID,
            )

    def test_rejects_retry_ids_that_do_not_match_request_message_ids(self):
        with self.assertRaisesRegex(ValidationError, "必须一致"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                conversation_id="conversation-1",
                user_message_id="33333333-3333-4333-8333-333333333333",
                retry_user_message_id=USER_MESSAGE_ID,
            )

    def test_rejects_non_uuid4_message_id(self):
        with self.assertRaises(ValidationError):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                user_message_id="11111111-1111-1111-8111-111111111111",
            )

    def test_rejects_same_user_and_assistant_message_id(self):
        with self.assertRaisesRegex(ValidationError, "必须不同"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                user_message_id=USER_MESSAGE_ID,
                assistant_message_id=USER_MESSAGE_ID,
            )

    def test_rejects_same_uuid_with_different_letter_case(self):
        lowercase_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        with self.assertRaisesRegex(ValidationError, "必须不同"):
            ChatRequest(
                model_id="deepseek-chat",
                message="你好",
                user_message_id=lowercase_id,
                assistant_message_id=lowercase_id.upper(),
            )


if __name__ == "__main__":
    unittest.main()
