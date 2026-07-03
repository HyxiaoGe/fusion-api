import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message, TextBlock
from app.schemas.response import ApiException
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
        self.assertIn("不要把已有知识包装成最新事实", sent_messages[1]["content"])
        self.assertIn("不要把缺少工具描述成系统故障", sent_messages[1]["content"])
        self.assertEqual(sent_messages[2]["content"], "OpenAI 最近发布了什么模型？")
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:chat_non_stream"]}},
        )

    def test_process_message_non_stream_injects_no_vision_boundary_for_image_on_text_model(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="image-1",
            user_id="user-1",
            original_filename="chart.png",
            mimetype="image/png",
            status="processed",
            storage_key="conv-1/image-1/chart.png",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="这张图里有什么？",
                    model_id="qwen-vl-max",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                True,
            )
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="当前模型无法查看图片。"))],
            usage=None,
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-vl-max", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True, "agentTools": False, "vision": False},
            ),
            patch(
                "app.services.chat_service.build_llm_messages",
                new=AsyncMock(
                    return_value=[
                        {"role": "system", "content": "日期 system"},
                        {"role": "user", "content": "这张图里有什么？"},
                    ]
                ),
            ),
            patch("app.services.chat_service.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            asyncio.run(
                service.process_message(
                    model_id="qwen-vl-max",
                    message="这张图里有什么？",
                    user_id="user-1",
                    stream=False,
                    file_ids=["image-1"],
                )
            )

        sent_messages = mock_litellm.acompletion.await_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in sent_messages[:4]], ["system", "system", "system", "user"])
        self.assertIn("日期 system", sent_messages[0]["content"])
        self.assertIn("【无图片理解能力边界规则】", sent_messages[1]["content"])
        self.assertIn("当前模型不能读取或理解图片附件", sent_messages[1]["content"])
        self.assertIn("【无联网工具边界规则】", sent_messages[2]["content"])
        self.assertEqual(sent_messages[3]["content"], "这张图里有什么？")

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
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:suggest_questions"]}},
        )

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
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:generate_title"]}},
        )
        service.conversation_service.repo.update_title.assert_called_once_with("conv-1", "Fusion Chat")
        service.db.commit.assert_called_once()

    def test_validate_message_files_accepts_processed_same_conversation_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        file_record = SimpleNamespace(
            id="file-1",
            user_id="user-1",
            original_filename="note.txt",
            mimetype="text/plain",
            status="processed",
            storage_key="conv-1/file-1/note.txt",
            thumbnail_key=None,
        )
        service.file_repo.get_file_by_id.return_value = file_record
        service.file_repo.is_file_linked_to_conversation.return_value = True

        result = service._validate_message_files(["file-1"], "user-1", "conv-1")

        self.assertEqual(result, [file_record])
        service.file_repo.get_file_by_id.assert_called_once_with("file-1", user_id="user-1")
        service.file_repo.is_file_linked_to_conversation.assert_called_once_with("conv-1", "file-1")

    def test_validate_message_files_rejects_other_user_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = None

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-1"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件不存在或无权访问")
        service.file_repo.is_file_linked_to_conversation.assert_not_called()

    def test_validate_message_files_rejects_same_user_file_from_other_conversation(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-2",
            user_id="user-1",
            original_filename="other.txt",
            mimetype="text/plain",
            status="processed",
            storage_key="conv-2/file-2/other.txt",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = False

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-2"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件不属于当前会话")

    def test_validate_message_files_rejects_unprocessed_non_image_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-3",
            user_id="user-1",
            original_filename="draft.pdf",
            mimetype="application/pdf",
            status="parsing",
            storage_key="conv-1/file-3/draft.pdf",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-3"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件仍在处理，请稍后再发送")

    def test_process_message_rejects_invalid_file_before_creating_message(self):
        db = MagicMock()
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = None
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="继续分析",
                    model_id="qwen-max-latest",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                False,
            )
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-max-latest", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": False, "agentTools": False, "vision": False},
            ),
        ):
            with self.assertRaises(ApiException):
                asyncio.run(
                    service.process_message(
                        model_id="qwen-max-latest",
                        message="继续分析",
                        user_id="user-1",
                        conversation_id="conv-1",
                        stream=True,
                        file_ids=["missing-file"],
                    )
                )

        service.conversation_service.create_message.assert_not_called()
        db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
