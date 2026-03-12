import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message
from app.services.chat_service import ChatService


class ChatServiceTests(unittest.TestCase):
    def test_handle_normal_response_commits_assistant_message(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            provider="qwen",
            model="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    type="user_query",
                    content="hello",
                    turn_id="user-msg-1",
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        response = SimpleNamespace(content="OK")

        with patch(
            "app.services.chat_service.ChatService._invoke_non_stream_model",
            new=AsyncMock(return_value=response),
        ):
            response = asyncio.run(
                service._handle_normal_response(
                    "qwen",
                    "qwen-max-latest",
                    messages=[{"role": "user", "content": "hello"}],
                    conversation_id="conv-1",
                    user_id="user-1",
                    options={},
                    turn_id="user-msg-1",
                )
            )

        self.assertEqual(response.message.content, "OK")
        self.assertEqual(response.reasoning, "")
        self.assertEqual(
            [(msg.role, msg.type, msg.content) for msg in conversation.messages],
            [
                ("user", "user_query", "hello"),
                ("assistant", "assistant_content", "OK"),
            ],
        )
        service.memory_service.save_conversation.assert_called_once_with(conversation)
        service.db.commit.assert_called_once()

    def test_update_message_commits_when_update_succeeds(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()

        updated_message = Message(
            id="assistant-msg-1",
            role="assistant",
            type="assistant_content",
            content="updated",
            turn_id="turn-1",
        )
        service.memory_service.update_message.return_value = updated_message

        result = service.update_message("assistant-msg-1", {"content": "updated"})

        self.assertIs(result, updated_message)
        service.memory_service.update_message.assert_called_once_with(
            "assistant-msg-1",
            {"content": "updated"},
        )
        service.db.commit.assert_called_once()

    def test_generate_suggested_questions_uses_prompt_manager_and_limits_output(self):
        service = object.__new__(ChatService)
        service.memory_service = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            provider="qwen",
            model="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    type="user_query",
                    content="Explain this answer",
                    turn_id="turn-1",
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="Here is the answer",
                    turn_id="turn-1",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(
            content="1. Follow-up A\n2. Follow-up B\n3. Follow-up C\n4. Follow-up D"
        )

        with patch(
            "app.services.chat_service.llm_manager.get_default_model",
            return_value=llm,
        ):
            questions = asyncio.run(
                service.generate_suggested_questions(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(
            questions,
            ["Follow-up A", "Follow-up B", "Follow-up C"],
        )
        llm.invoke.assert_called_once()

    def test_generate_title_persists_conversation_inside_service(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="old",
            provider="qwen",
            model="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    type="user_query",
                    content="Explain fusion",
                    turn_id="turn-1",
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="Fusion is a chat product.",
                    turn_id="turn-1",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="Fusion Chat")

        with patch(
            "app.services.chat_service.llm_manager.get_default_model",
            return_value=llm,
        ):
            title = asyncio.run(
                service.generate_title(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(title, "Fusion Chat")
        self.assertEqual(conversation.title, "Fusion Chat")
        service.memory_service.save_conversation.assert_called_once_with(conversation)
        service.db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
