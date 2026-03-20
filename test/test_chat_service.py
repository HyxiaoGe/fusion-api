import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message
from app.services.chat_service import ChatService


class ChatServiceTests(unittest.TestCase):
    def test_get_response_text_prefers_content_attribute(self):
        service = object.__new__(ChatService)

        text = service._get_response_text(SimpleNamespace(content="hello"))

        self.assertEqual(text, "hello")

    def test_get_response_text_falls_back_to_raw_response(self):
        service = object.__new__(ChatService)

        text = service._get_response_text("hello")

        self.assertEqual(text, "hello")

    def test_build_recent_dialog_content_prefers_latest_user_assistant_pair(self):
        service = object.__new__(ChatService)
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
                    content="old question",
                    turn_id="turn-1",
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="old answer",
                    turn_id="turn-1",
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    type="user_query",
                    content="latest question",
                    turn_id="turn-2",
                ),
                Message(
                    id="assistant-msg-2",
                    role="assistant",
                    type="assistant_content",
                    content="latest answer",
                    turn_id="turn-2",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        content = service._build_recent_dialog_content(conversation)

        self.assertEqual(content, "用户: latest question\n助手: latest answer")

    def test_build_recent_dialog_content_falls_back_to_user_messages(self):
        service = object.__new__(ChatService)
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
                    content="question one",
                    turn_id="turn-1",
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    type="user_query",
                    content="question two",
                    turn_id="turn-2",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        content = service._build_recent_dialog_content(conversation)

        self.assertEqual(content, "用户: question two")

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

    def test_get_conversation_attaches_reasoning_to_assistant_message(self):
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
                    content="question",
                    turn_id="turn-1",
                ),
                Message(
                    id="reasoning-msg-1",
                    role="assistant",
                    type="reasoning_content",
                    content="thought",
                    turn_id="turn-1",
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="answer",
                    turn_id="turn-1",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        result = service.get_conversation("conv-1", "user-1")

        self.assertEqual([msg.id for msg in result.messages], ["user-msg-1", "assistant-msg-1"])
        self.assertEqual(result.messages[1].reasoning, "thought")

    def test_get_conversation_concatenates_multiple_reasoning_messages_by_created_at(self):
        service = object.__new__(ChatService)
        service.memory_service = MagicMock()
        now = datetime.now()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            provider="qwen",
            model="qwen-max-latest",
            messages=[
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="answer",
                    turn_id="turn-1",
                    created_at=now,
                ),
                Message(
                    id="reasoning-msg-2",
                    role="assistant",
                    type="reasoning_content",
                    content="second",
                    turn_id="turn-1",
                    created_at=now.replace(microsecond=2),
                ),
                Message(
                    id="reasoning-msg-1",
                    role="assistant",
                    type="reasoning_content",
                    content="first",
                    turn_id="turn-1",
                    created_at=now.replace(microsecond=1),
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        service.memory_service.get_conversation.return_value = conversation

        result = service.get_conversation("conv-1", "user-1")

        self.assertEqual(result.messages[0].reasoning, "firstsecond")

    def test_get_conversation_returns_null_reasoning_when_absent(self):
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
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="answer",
                    turn_id="turn-1",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        result = service.get_conversation("conv-1", "user-1")

        self.assertIsNone(result.messages[0].reasoning)

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

        with patch(
            "app.services.chat_service.ChatService._invoke_non_stream_model",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    content="1. Follow-up A\n2. Follow-up B\n3. Follow-up C\n4. Follow-up D"
                )
            ),
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

        with patch(
            "app.services.chat_service.ChatService._invoke_non_stream_model",
            new=AsyncMock(return_value=SimpleNamespace(content="Fusion Chat")),
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

    def test_generate_title_prefers_latest_user_message_only(self):
        service = object.__new__(ChatService)
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
                    content="第一个问题",
                    turn_id="turn-1",
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    type="assistant_content",
                    content="第一个回答",
                    turn_id="turn-1",
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    type="user_query",
                    content="最后一个问题",
                    turn_id="turn-2",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.assertEqual(
            service._get_latest_user_message_content(conversation),
            "最后一个问题",
        )


if __name__ == "__main__":
    unittest.main()
