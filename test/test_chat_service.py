import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message, TextBlock, ThinkingBlock
from app.services.chat_service import ChatService


class ChatServiceTests(unittest.TestCase):
    def test_build_recent_dialog_content_prefers_latest_user_assistant_pair(self):
        service = object.__new__(ChatService)
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="old question")],
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    content=[TextBlock(type="text", text="old answer")],
                    model_id="qwen-max-latest",
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    content=[TextBlock(type="text", text="latest question")],
                ),
                Message(
                    id="assistant-msg-2",
                    role="assistant",
                    content=[TextBlock(type="text", text="latest answer")],
                    model_id="qwen-max-latest",
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
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="question one")],
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    content=[TextBlock(type="text", text="question two")],
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        content = service._build_recent_dialog_content(conversation)

        self.assertEqual(content, "用户: question two")

    def test_build_llm_messages_filters_thinking_blocks(self):
        """thinking block 不应传给 LLM"""
        messages = [
            Message(
                role="user",
                content=[TextBlock(type="text", text="hello")],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(type="thinking", thinking="let me think"),
                    TextBlock(type="text", text="answer"),
                ],
                model_id="qwen-max-latest",
            ),
        ]

        result = ChatService._build_llm_messages(messages)

        self.assertEqual(
            result,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_update_message_commits_when_update_succeeds(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()

        updated_message = Message(
            id="assistant-msg-1",
            role="assistant",
            content=[TextBlock(type="text", text="updated")],
            model_id="qwen-max-latest",
        )
        service.memory_service.update_message.return_value = updated_message

        result = service.update_message(
            "assistant-msg-1",
            {"content": [TextBlock(type="text", text="updated")]},
        )

        self.assertIs(result, updated_message)
        service.db.commit.assert_called_once()

    def test_generate_suggested_questions_limits_output_to_three(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()
        service.file_repo = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="Explain this answer")],
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    content=[TextBlock(type="text", text="Here is the answer")],
                    model_id="qwen-max-latest",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="1. Follow-up A\n2. Follow-up B\n3. Follow-up C\n4. Follow-up D")
                )
            ]
        )

        with (
            patch("app.services.chat_service.litellm") as mock_litellm,
            patch("app.services.chat_service.llm_manager") as mock_manager,
        ):
            mock_manager.resolve_model.return_value = ("openai/qwen-max-latest", "qwen", {})
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            questions = asyncio.run(
                service.generate_suggested_questions(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(len(questions), 3)
        self.assertEqual(questions, ["Follow-up A", "Follow-up B", "Follow-up C"])

    def test_generate_title_persists_title_to_database(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.memory_service = MagicMock()
        service.file_repo = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="old",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="Explain fusion")],
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.memory_service.get_conversation.return_value = conversation

        mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Fusion Chat"))])

        with (
            patch("app.services.chat_service.litellm") as mock_litellm,
            patch("app.services.chat_service.llm_manager") as mock_manager,
        ):
            mock_manager.resolve_model.return_value = ("openai/qwen-max-latest", "qwen", {})
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            title = asyncio.run(
                service.generate_title(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(title, "Fusion Chat")
        service.memory_service.repo.update_title.assert_called_once_with("conv-1", "Fusion Chat")
        service.db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
