import unittest

from pydantic import ValidationError

from app.schemas.chat import ChatRequest

USER_MESSAGE_ID = "11111111-1111-4111-8111-111111111111"
ASSISTANT_MESSAGE_ID = "22222222-2222-4222-8222-222222222222"


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
