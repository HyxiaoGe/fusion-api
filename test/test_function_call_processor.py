import asyncio
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from app.constants import FUNCTION_DESCRIPTIONS, MessageRoles, MessageTypes
from app.schemas.chat import Conversation, Message
from app.services.chat.function_call_processor import FunctionCallProcessor


class FunctionCallProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = FunctionCallProcessor(MagicMock(), MagicMock())

    def test_build_assistant_content_message_uses_assistant_content_type(self):
        message = self.processor._build_assistant_content_message("hello", "turn-1")

        self.assertEqual(message.role, MessageRoles.ASSISTANT)
        self.assertEqual(message.type, MessageTypes.ASSISTANT_CONTENT)
        self.assertEqual(message.content, "hello")
        self.assertEqual(message.turn_id, "turn-1")

    def test_build_function_call_history_messages_uses_fallback_description(self):
        messages = self.processor._build_function_call_history_messages(
            function_name="web_search",
            function_result={"status": "ok"},
            final_response="final answer",
            turn_id="turn-1",
            first_llm_thought=None,
        )

        self.assertEqual(
            [(msg.role, msg.type) for msg in messages],
            [
                (MessageRoles.ASSISTANT, MessageTypes.FUNCTION_CALL),
                (MessageRoles.SYSTEM, MessageTypes.FUNCTION_RESULT),
                (MessageRoles.ASSISTANT, MessageTypes.ASSISTANT_CONTENT),
            ],
        )
        self.assertEqual(messages[0].content, FUNCTION_DESCRIPTIONS["web_search"])

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

    def test_save_stream_response_persists_conversation_changes(self):
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            provider="qwen",
            model="qwen-max-latest",
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.processor.memory_service.get_conversation.return_value = conversation

        asyncio.run(
            self.processor._save_stream_response(
                conversation_id="conv-1",
                response_content="assistant reply",
                user_id="user-1",
            )
        )

        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, MessageRoles.ASSISTANT)
        self.assertEqual(conversation.messages[0].type, MessageTypes.ASSISTANT_CONTENT)
        self.assertEqual(conversation.messages[0].content, "assistant reply")
        self.processor.memory_service.save_conversation.assert_called_once_with(conversation)
        self.processor.db.commit.assert_called_once()

    def test_save_function_call_stream_response_persists_all_messages(self):
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            provider="qwen",
            model="qwen-max-latest",
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.processor.memory_service.get_conversation.return_value = conversation

        asyncio.run(
            self.processor._save_function_call_stream_response(
                conversation_id="conv-1",
                function_name="web_search",
                function_result={"status": "ok"},
                final_response="final answer",
                turn_id="turn-1",
                user_id="user-1",
                first_llm_thought="我需要查一下",
            )
        )

        self.assertEqual(
            [(msg.role, msg.type) for msg in conversation.messages],
            [
                (MessageRoles.ASSISTANT, MessageTypes.FUNCTION_CALL),
                (MessageRoles.SYSTEM, MessageTypes.FUNCTION_RESULT),
                (MessageRoles.ASSISTANT, MessageTypes.ASSISTANT_CONTENT),
            ],
        )
        self.processor.memory_service.save_conversation.assert_called_once_with(conversation)
        self.processor.db.commit.assert_called_once()

    def test_save_stream_response_skips_persistence_without_user_id(self):
        asyncio.run(
            self.processor._save_stream_response(
                conversation_id="conv-1",
                response_content="assistant reply",
                user_id=None,
            )
        )

        self.processor.memory_service.get_conversation.assert_not_called()
        self.processor.memory_service.save_conversation.assert_not_called()
        self.processor.db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
