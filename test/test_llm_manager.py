import unittest
from unittest.mock import patch

from app.ai.llm_manager import QwenFactory, get_model_display_name


class QwenFactoryTests(unittest.TestCase):
    @patch("langchain_community.chat_models.tongyi.ChatTongyi")
    def test_non_qwen3_model_uses_database_api_key(self, chat_tongyi_cls):
        factory = QwenFactory()

        factory.create_model(
            "qwen-max-latest",
            credentials={"api_key": "dashscope-test-key"},
        )

        chat_tongyi_cls.assert_called_once_with(
            model="qwen-max-latest",
            streaming=True,
            dashscope_api_key="dashscope-test-key",
        )

    def test_get_model_display_name_uses_wenxin_key(self):
        self.assertEqual(get_model_display_name("wenxin"), "文心一言")


if __name__ == "__main__":
    unittest.main()
