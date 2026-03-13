import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.constants import MessageRoles, MessageTexts
from app.services.chat.utils import ChatUtils


class ChatUtilsTests(unittest.TestCase):
    def test_get_response_text_prefers_content_attribute(self):
        self.assertEqual(
            ChatUtils.get_response_text(SimpleNamespace(content="hello")),
            "hello",
        )

    def test_clean_model_text_trims_whitespace_and_quotes(self):
        self.assertEqual(ChatUtils.clean_model_text('  "hello"  '), "hello")

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

    def test_generate_search_query_reuses_text_cleaning(self):
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='  "fusion ai"  ')))

        query = asyncio.run(ChatUtils.generate_search_query("fusion 是什么", llm))

        self.assertEqual(query, "fusion ai")

    def test_strip_question_prefix_removes_numbering(self):
        self.assertEqual(ChatUtils._strip_question_prefix("2. 第二个问题"), "第二个问题")

    def test_parse_questions_cleans_line_prefixes(self):
        questions = ChatUtils.parse_questions("1. 第一个问题\n2) 第二个问题\n3. 第三个问题")

        self.assertEqual(questions, ["第一个问题", "第二个问题", "第三个问题"])


if __name__ == "__main__":
    unittest.main()
