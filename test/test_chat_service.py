import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message, TextBlock
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

    # _build_llm_messages 已重构为独立 async 函数 app/services/chat/message_builder.py::build_llm_messages
    # 该函数还注入了 system date prompt + user_system_prompt + 图片 base64 等副作用，
    # 不再适合作为 ChatService 的私有方法测。新测试应该写在
    # test/services/chat/test_message_builder.py（暂未补，等需要时再加）。

    def test_update_message_commits_when_update_succeeds(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()

        updated_message = Message(
            id="assistant-msg-1",
            role="assistant",
            content=[TextBlock(type="text", text="updated")],
            model_id="qwen-max-latest",
        )
        service.conversation_service.update_message.return_value = updated_message

        result = service.update_message(
            "assistant-msg-1",
            {"content": [TextBlock(type="text", text="updated")]},
        )

        self.assertIs(result, updated_message)
        service.db.commit.assert_called_once()

    def test_process_message_non_stream_injects_no_tool_network_boundary(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="OpenAI 最近发布了什么模型？",
                    model_id="qwen-vl-max",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                True,
            )
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="基于已有知识回答。"))],
            usage=None,
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-vl-max", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True, "agentTools": False},
            ),
            patch(
                "app.services.chat_service.build_llm_messages",
                new=AsyncMock(
                    return_value=[
                        {"role": "system", "content": "日期 system"},
                        {"role": "user", "content": "OpenAI 最近发布了什么模型？"},
                    ]
                ),
            ),
            patch("app.services.chat_service.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            asyncio.run(
                service.process_message(
                    model_id="qwen-vl-max",
                    message="OpenAI 最近发布了什么模型？",
                    user_id="user-1",
                    stream=False,
                )
            )

        sent_messages = mock_litellm.acompletion.await_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in sent_messages[:3]], ["system", "system", "user"])
        self.assertIn("日期 system", sent_messages[0]["content"])
        self.assertIn("【无联网工具边界规则】", sent_messages[1]["content"])
        self.assertIn("无法实时核验", sent_messages[1]["content"])
        self.assertEqual(sent_messages[2]["content"], "OpenAI 最近发布了什么模型？")

    def test_generate_suggested_questions_limits_output_to_three(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
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
        service.conversation_service.get_conversation.return_value = conversation

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
        self.assertEqual(mock_litellm.acompletion.await_args.kwargs["max_tokens"], 512)

    def test_generate_title_persists_title_to_database(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
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
        service.conversation_service.get_conversation.return_value = conversation

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
        self.assertEqual(mock_litellm.acompletion.await_args.kwargs["max_tokens"], 128)
        service.conversation_service.repo.update_title.assert_called_once_with("conv-1", "Fusion Chat")
        service.db.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
