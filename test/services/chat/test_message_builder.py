import unittest
from unittest.mock import AsyncMock, patch

from app.schemas.chat import FileBlock, Message, TextBlock
from app.services.chat.message_builder import build_llm_messages


class MessageBuilderTests(unittest.IsolatedAsyncioTestCase):
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
        ):
            result = await build_llm_messages(messages, has_vision=True, file_repo=object())

        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["content"], [{"type": "text", "text": "看图回答"}, image_part])

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


if __name__ == "__main__":
    unittest.main()
