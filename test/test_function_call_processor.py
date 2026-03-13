import asyncio
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.constants import FUNCTION_DESCRIPTIONS, USER_FRIENDLY_FUNCTION_DESCRIPTIONS, MessageRoles, MessageTypes
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

    def test_build_assistant_function_call_message_uses_tool_calls_for_supported_provider(self):
        message, tool_call_id = self.processor._build_assistant_function_call_message(
            "qwen",
            {"function": {"name": "web_search", "arguments": {"query": "fusion"}}},
            "thinking",
        )

        self.assertEqual(message["role"], MessageRoles.ASSISTANT)
        self.assertEqual(message["content"], "thinking")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"query": "fusion"}',
        )
        self.assertEqual(message["tool_calls"][0]["id"], tool_call_id)

    def test_build_assistant_function_call_message_falls_back_to_function_call_for_legacy_provider(self):
        message, tool_call_id = self.processor._build_assistant_function_call_message(
            "legacy",
            {"function": {"name": "web_search", "arguments": "{}"}},
        )

        self.assertEqual(message["role"], MessageRoles.ASSISTANT)
        self.assertEqual(message["content"], "")
        self.assertEqual(message["function_call"]["name"], "web_search")
        self.assertTrue(tool_call_id.startswith("call_"))

    def test_build_function_detected_event_data_uses_user_friendly_description(self):
        payload = self.processor._build_function_detected_event_data(
            {"function": {"name": "web_search"}}
        )

        self.assertEqual(payload["function_type"], "web_search")
        self.assertEqual(
            payload["description"],
            USER_FRIENDLY_FUNCTION_DESCRIPTIONS["web_search"],
        )

    def test_update_function_arguments_serializes_with_utf8(self):
        function_call_data = {
            "function": {
                "name": "web_search",
                "arguments": "{}",
            }
        }

        updated = self.processor._update_function_arguments(
            function_call_data,
            {"query": "融合聊天"},
        )

        self.assertEqual(
            updated["function"]["arguments"],
            '{"query": "融合聊天"}',
        )

    def test_build_function_result_event_data_preserves_result_payload(self):
        payload = self.processor._build_function_result_event_data(
            "web_search",
            {"status": "ok", "items": [1, 2]},
        )

        self.assertEqual(
            payload,
            {"function_type": "web_search", "result": {"status": "ok", "items": [1, 2]}},
        )

    def test_extract_final_stream_response_returns_last_non_event_string(self):
        final_response = self.processor._extract_final_stream_response(
            ["data: {\"type\": \"content\"}\n\n", "final answer"]
        )

        self.assertEqual(final_response, "final answer")

    def test_extract_final_stream_response_returns_empty_for_event_only_results(self):
        final_response = self.processor._extract_final_stream_response(
            ["data: {\"type\": \"done\"}\n\n"]
        )

        self.assertEqual(final_response, "")

    def test_finalize_first_pass_response_uses_friendly_description_when_content_missing(self):
        final_response = self.processor._finalize_first_pass_response(
            True,
            {"function": {"name": "web_search"}},
            "",
        )

        self.assertEqual(
            final_response,
            USER_FRIENDLY_FUNCTION_DESCRIPTIONS["web_search"],
        )

    def test_finalize_first_pass_response_preserves_existing_content(self):
        final_response = self.processor._finalize_first_pass_response(
            True,
            {"function": {"name": "web_search"}},
            "已有回答",
        )

        self.assertEqual(final_response, "已有回答")

    @patch("app.services.chat.function_call_processor.StreamProcessor.create_tool_synthesis_messages")
    def test_build_web_search_followup_messages_appends_tool_messages(self, mock_create_tool_synthesis_messages):
        mock_create_tool_synthesis_messages.return_value = [{"role": MessageRoles.SYSTEM, "content": "seed"}]

        messages = self.processor._build_web_search_followup_messages(
            messages=[{"role": MessageRoles.USER, "content": "帮我查 fusion"}],
            function_call_data={
                "tool_call_id": "tool-1",
                "function": {"name": "web_search", "arguments": {"query": "fusion"}},
                "first_llm_thought": "我先查一下",
            },
            function_result={"status": "ok"},
            provider="qwen",
        )

        self.assertEqual(messages[1]["tool_calls"][0]["id"], "tool-1")
        self.assertEqual(messages[1]["content"], "我先查一下")
        self.assertEqual(messages[2]["tool_call_id"], "tool-1")

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
