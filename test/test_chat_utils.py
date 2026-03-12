from types import SimpleNamespace
import unittest

from app.constants import MessageRoles, MessageTexts
from app.services.chat.utils import ChatUtils


class ChatUtilsTests(unittest.TestCase):
    def test_extract_latest_user_content_prefers_last_user_message(self):
        messages = [
            {"role": MessageRoles.USER, "content": "first"},
            SimpleNamespace(type="human", content="second"),
        ]

        self.assertEqual(ChatUtils.extract_latest_user_content(messages), "second")

    def test_extract_latest_user_content_returns_default_when_missing(self):
        self.assertEqual(
            ChatUtils.extract_latest_user_content([], MessageTexts.USER_PREVIOUS_QUESTION),
            MessageTexts.USER_PREVIOUS_QUESTION,
        )

    def test_stringify_function_arguments_normalizes_invalid_json(self):
        self.assertEqual(ChatUtils.stringify_function_arguments("{not-json}"), "{}")

    def test_stringify_function_arguments_preserves_dict_payload(self):
        self.assertEqual(
            ChatUtils.stringify_function_arguments({"query": "fusion"}),
            '{"query": "fusion"}',
        )


if __name__ == "__main__":
    unittest.main()
