import unittest
from unittest.mock import MagicMock

from app.ai.llm_manager import get_model_display_name, LLMManager, PROVIDER_LITELLM_PREFIX


class LLMManagerTests(unittest.TestCase):
    def test_get_model_display_name_uses_wenxin_key(self):
        self.assertEqual(get_model_display_name("wenxin"), "文心一言")

    def test_get_model_display_name_returns_raw_for_unknown(self):
        self.assertEqual(get_model_display_name("unknown_provider"), "unknown_provider")

    def test_provider_prefix_mapping_covers_all_providers(self):
        expected_providers = {
            "openai", "anthropic", "deepseek", "google",
            "qwen", "volcengine", "wenxin", "hunyuan", "xai",
        }
        self.assertEqual(set(PROVIDER_LITELLM_PREFIX.keys()), expected_providers)

    def test_resolve_model_constructs_litellm_params(self):
        manager = LLMManager()
        db = MagicMock()

        # 模拟 ModelSource
        mock_source = MagicMock()
        mock_source.provider = "qwen"

        # 模拟 ModelCredential
        mock_credential = MagicMock()
        mock_credential.credentials = {"api_key": "test-key", "base_url": "https://api.example.com"}

        with unittest.mock.patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo, \
             unittest.mock.patch("app.ai.llm_manager.ModelCredentialRepository") as mock_cred_repo:
            mock_source_repo.return_value.get_by_id.return_value = mock_source
            mock_cred_repo.return_value.get_default.return_value = mock_credential

            litellm_model, provider, kwargs = manager.resolve_model("qwen-max-latest", db)

        self.assertEqual(litellm_model, "openai/qwen-max-latest")
        self.assertEqual(provider, "qwen")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["api_base"], "https://api.example.com")

    def test_resolve_model_raises_on_missing_source(self):
        manager = LLMManager()
        db = MagicMock()

        with unittest.mock.patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo:
            mock_source_repo.return_value.get_by_id.return_value = None

            with self.assertRaises(ValueError) as ctx:
                manager.resolve_model("nonexistent-model", db)

            self.assertIn("未找到模型配置", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
