import unittest

from app.services.chat.utils import ChatUtils


class ChatUtilsTests(unittest.TestCase):
    def test_get_response_text_prefers_content_attribute(self):
        from types import SimpleNamespace
        self.assertEqual(
            ChatUtils.get_response_text(SimpleNamespace(content="hello")),
            "hello",
        )

    def test_clean_model_text_trims_whitespace_and_quotes(self):
        self.assertEqual(ChatUtils.clean_model_text('  "hello"  '), "hello")

    def test_stringify_function_arguments_normalizes_invalid_json(self):
        self.assertEqual(ChatUtils.stringify_function_arguments("{not-json}"), "{}")

    def test_stringify_function_arguments_preserves_dict_payload(self):
        self.assertEqual(
            ChatUtils.stringify_function_arguments({"query": "fusion"}),
            '{"query": "fusion"}',
        )

    def test_strip_question_prefix_removes_numbering(self):
        self.assertEqual(ChatUtils._strip_question_prefix("2. 第二个问题"), "第二个问题")

    def test_split_non_empty_lines_skips_blank_rows(self):
        self.assertEqual(
            ChatUtils._split_non_empty_lines("第一行\n\n 第二行 \n"),
            ["第一行", "第二行"],
        )

    def test_extract_numbered_questions_returns_clean_items(self):
        self.assertEqual(
            ChatUtils._extract_numbered_questions("1. 第一个问题\n2) 第二个问题\n3. 第三个问题"),
            ["第一个问题", "第二个问题", "第三个问题"],
        )

    def test_parse_questions_cleans_line_prefixes(self):
        questions = ChatUtils.parse_questions("1. 第一个问题\n2) 第二个问题\n3. 第三个问题")
        self.assertEqual(questions, ["第一个问题", "第二个问题", "第三个问题"])


if __name__ == "__main__":
    unittest.main()
