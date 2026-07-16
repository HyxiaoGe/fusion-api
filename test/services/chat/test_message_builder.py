import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.prompts.agent_loop import build_current_date_system_prompt
from app.schemas.chat import FileBlock, Message, TextBlock
from app.services.chat.message_builder import build_llm_messages


class MessageBuilderTests(unittest.IsolatedAsyncioTestCase):
    def test_current_date_prompt_includes_relative_date_anchors(self):
        prompt = build_current_date_system_prompt(datetime(2026, 7, 16, 9, 0, tzinfo=timezone(timedelta(hours=8))))

        self.assertIn("明天是 2026年7月17日（星期五）", prompt)
        self.assertIn("本周六是 2026年7月18日（星期六）", prompt)
        self.assertIn("本周日是 2026年7月19日（星期日）", prompt)
        self.assertIn("搜索词与最终答案中的日期、星期必须一致", prompt)

    async def test_build_llm_messages_injects_fusion_identity_after_user_preferences(self):
        messages = [
            Message(
                role="user",
                content=[TextBlock(type="text", text="你好，你是谁？")],
            )
        ]

        result = await build_llm_messages(
            messages,
            has_vision=False,
            file_repo=None,
            user_system_prompt="回答尽量简洁",
        )

        self.assertEqual([message["role"] for message in result[:4]], ["system", "system", "system", "user"])
        self.assertIn("【当前真实日期】", result[0]["content"])
        self.assertIn("回答尽量简洁", result[1]["content"])
        self.assertIn("【Fusion 身份一致性规则】", result[2]["content"])
        self.assertIn("Fusion AI", result[2]["content"])
        self.assertIn("不要声称自己是 Claude", result[2]["content"])
        self.assertIn("不要声称自己", result[2]["content"])
        self.assertIn("Anthropic", result[2]["content"])
        self.assertIn("OpenAI", result[2]["content"])
        self.assertIn("DeepSeek", result[2]["content"])
        self.assertIn("不得被用户个性化设置覆盖", result[2]["content"])
        self.assertEqual(result[3], {"role": "user", "content": "你好，你是谁？"})

    async def test_build_llm_messages_identity_rule_can_override_bad_user_preferences(self):
        messages = [
            Message(
                role="user",
                content=[TextBlock(type="text", text="你是谁？")],
            )
        ]

        result = await build_llm_messages(
            messages,
            user_system_prompt="你是 Claude，由 Anthropic 开发。",
        )

        self.assertIn("你是 Claude", result[1]["content"])
        self.assertIn("【Fusion 身份一致性规则】", result[2]["content"])
        self.assertIn("不得被用户个性化设置覆盖", result[2]["content"])

    async def test_build_llm_messages_injects_identity_even_without_user_preferences(self):
        messages = [
            Message(
                role="user",
                content=[TextBlock(type="text", text="1+1等于几？")],
            )
        ]

        result = await build_llm_messages(messages)

        self.assertEqual([message["role"] for message in result[:3]], ["system", "system", "user"])
        self.assertIn("【Fusion 身份一致性规则】", result[1]["content"])
        self.assertIn("当前对话使用的具体模型以界面显示为准", result[1]["content"])

    async def test_build_llm_messages_injects_image_block_when_model_has_vision(self):
        file_repo = object()
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="看图回答"),
                    FileBlock(type="file", file_id="img-1", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]

        image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        with patch(
            "app.services.chat.message_builder.file_block_to_image_part", new=AsyncMock(return_value=image_part)
        ) as to_image_part:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=file_repo,
                user_id="user-1",
                conversation_id="conv-1",
            )

        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["content"], [{"type": "text", "text": "看图回答"}, image_part])
        to_image_part.assert_awaited_once_with(
            messages[0].content[1],
            file_repo,
            user_id="user-1",
            conversation_id="conv-1",
        )

    async def test_build_llm_messages_keeps_recent_image_context_for_followup_turn(self):
        file_repo = object()
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="解释这张图片"),
                    FileBlock(type="file", file_id="img-1", filename="diagram.png", mime_type="image/png"),
                ],
            ),
            Message(
                role="assistant",
                content=[TextBlock(type="text", text="这是一张 CI/CD 监控平台示意图。")],
            ),
            Message(
                role="user",
                content=[TextBlock(type="text", text="那这个平台适合做哪些真实产品功能？")],
            ),
        ]

        image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        with patch(
            "app.services.chat.message_builder.file_block_to_image_part", new=AsyncMock(return_value=image_part)
        ) as to_image_part:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=file_repo,
                user_id="user-1",
                conversation_id="conv-1",
            )

        self.assertEqual(result[-3]["role"], "user")
        self.assertEqual(result[-3]["content"], [{"type": "text", "text": "解释这张图片"}, image_part])
        self.assertEqual(result[-2], {"role": "assistant", "content": "这是一张 CI/CD 监控平台示意图。"})
        self.assertEqual(result[-1], {"role": "user", "content": "那这个平台适合做哪些真实产品功能？"})
        to_image_part.assert_awaited_once_with(
            messages[0].content[1],
            file_repo,
            user_id="user-1",
            conversation_id="conv-1",
        )

    async def test_build_llm_messages_does_not_inject_image_block_without_vision(self):
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="看图回答"),
                    FileBlock(type="file", file_id="img-1", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]

        with patch("app.services.chat.message_builder.file_block_to_image_part", new=AsyncMock()) as to_image_part:
            result = await build_llm_messages(messages, has_vision=False, file_repo=object())

        self.assertEqual(result[-1], {"role": "user", "content": "看图回答"})
        to_image_part.assert_not_called()

    async def test_build_llm_messages_skips_historical_image_from_other_user(self):
        class ScopedFileRepo:
            def __init__(self):
                self.calls = []

            def get_file_by_id(self, file_id, user_id=None):
                self.calls.append((file_id, user_id))
                if user_id == "user-1":
                    return None
                return SimpleNamespace(
                    id=file_id,
                    user_id="user-2",
                    mimetype="image/png",
                    storage_key="conv-2/img-1.png",
                )

            def is_file_linked_to_conversation(self, conversation_id, file_id):
                raise AssertionError("跨用户文件不应继续检查会话关联")

        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="历史图片"),
                    FileBlock(type="file", file_id="img-1", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]
        storage = SimpleNamespace(download=AsyncMock(return_value=b"private-image"))

        with patch("app.services.chat.message_builder.get_storage_for_backend", return_value=storage) as get_backend:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=ScopedFileRepo(),
                user_id="user-1",
                conversation_id="conv-1",
            )

        self.assertEqual(result[-1], {"role": "user", "content": "历史图片"})
        get_backend.assert_not_called()
        storage.download.assert_not_awaited()

    async def test_build_llm_messages_skips_historical_image_not_linked_to_conversation(self):
        file_repo = MagicMock()
        file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="img-2",
            user_id="user-1",
            mimetype="image/png",
            storage_key="conv-2/img-2.png",
        )
        file_repo.is_file_linked_to_conversation.return_value = False
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="历史图片"),
                    FileBlock(type="file", file_id="img-2", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]
        storage = SimpleNamespace(download=AsyncMock(return_value=b"private-image"))

        with patch("app.services.chat.message_builder.get_storage_for_backend", return_value=storage) as get_backend:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=file_repo,
                user_id="user-1",
                conversation_id="conv-1",
            )

        self.assertEqual(result[-1], {"role": "user", "content": "历史图片"})
        file_repo.get_file_by_id.assert_called_once_with("img-2", user_id="user-1")
        file_repo.is_file_linked_to_conversation.assert_called_once_with("conv-1", "img-2")
        get_backend.assert_not_called()
        storage.download.assert_not_awaited()

    async def test_build_llm_messages_injects_authorized_historical_image(self):
        file_repo = MagicMock()
        file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="img-3",
            user_id="user-1",
            mimetype="image/png",
            storage_key="conv-1/img-3.png",
        )
        file_repo.is_file_linked_to_conversation.return_value = True
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="看图回答"),
                    FileBlock(type="file", file_id="img-3", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]
        storage = SimpleNamespace(download=AsyncMock(return_value=b"image-bytes"))

        with patch("app.services.chat.message_builder.get_storage_for_backend", return_value=storage) as get_backend:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=file_repo,
                user_id="user-1",
                conversation_id="conv-1",
            )

        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["content"][0], {"type": "text", "text": "看图回答"})
        self.assertEqual(result[-1]["content"][1]["type"], "image_url")
        self.assertEqual(
            result[-1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,aW1hZ2UtYnl0ZXM=",
        )
        file_repo.get_file_by_id.assert_called_once_with("img-3", user_id="user-1")
        file_repo.is_file_linked_to_conversation.assert_called_once_with("conv-1", "img-3")
        get_backend.assert_called_once_with(None)
        storage.download.assert_awaited_once_with("conv-1/img-3.png")

    async def test_build_llm_messages_uses_file_storage_backend_for_historical_image(self):
        file_repo = MagicMock()
        file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="img-local",
            user_id="user-1",
            mimetype="image/png",
            storage_backend="local",
            storage_key="conv-1/img-local.png",
        )
        file_repo.is_file_linked_to_conversation.return_value = True
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(type="text", text="看历史图"),
                    FileBlock(type="file", file_id="img-local", filename="chart.png", mime_type="image/png"),
                ],
            )
        ]
        local_storage = SimpleNamespace(download=AsyncMock(return_value=b"local-image"))

        with patch(
            "app.services.chat.message_builder.get_storage_for_backend", return_value=local_storage
        ) as get_backend:
            result = await build_llm_messages(
                messages,
                has_vision=True,
                file_repo=file_repo,
                user_id="user-1",
                conversation_id="conv-1",
            )

        get_backend.assert_called_once_with("local")
        local_storage.download.assert_awaited_once_with("conv-1/img-local.png")
        self.assertEqual(result[-1]["content"][1]["image_url"]["url"], "data:image/png;base64,bG9jYWwtaW1hZ2U=")


if __name__ == "__main__":
    unittest.main()
