import unittest
from unittest.mock import MagicMock

from app.constants import MessageRoles
from app.services.chat.function_call_processor import FunctionCallProcessor


class FunctionCallProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = FunctionCallProcessor(MagicMock(), MagicMock())

    def test_prepare_function_call_messages_replaces_existing_system_prompt(self):
        messages = [
            {"role": MessageRoles.SYSTEM, "content": "old system"},
            {"role": MessageRoles.USER, "content": "hello"},
        ]

        prepared = self.processor._prepare_function_call_messages(messages, {"use_reasoning": False})

        self.assertEqual(prepared[0]["role"], MessageRoles.SYSTEM)
        self.assertEqual(prepared[1:], [{"role": MessageRoles.USER, "content": "hello"}])

    def test_resolve_tool_call_id_returns_fallback_when_missing(self):
        fallback = self.processor._resolve_tool_call_id(None)
        self.assertTrue(fallback.startswith("call_"))
        self.assertNotEqual(fallback, "call_")

    def test_resolve_tool_call_id_preserves_existing_value(self):
        self.assertEqual(self.processor._resolve_tool_call_id("tool-123"), "tool-123")


if __name__ == "__main__":
    unittest.main()
