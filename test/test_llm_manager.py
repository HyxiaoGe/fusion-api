import unittest
from unittest.mock import MagicMock, patch

from app.ai.llm_manager import LLMManager


class LLMManagerTests(unittest.TestCase):
    def test_resolve_model_constructs_litellm_params(self):
        manager = LLMManager()
        db = MagicMock()

        # 模拟 Provider
        mock_provider = MagicMock()
        mock_provider.litellm_prefix = "openai"
        mock_provider.custom_base_url = True

        # 模拟 ModelSource
        mock_source = MagicMock()
        mock_source.provider = "qwen"
        mock_source.provider_rel = mock_provider

        # 模拟 ModelCredential
        mock_credential = MagicMock()
        mock_credential.credentials = {"api_key": "test-key", "base_url": "https://api.example.com"}

        with (
            patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo,
            patch("app.ai.llm_manager.ModelCredentialRepository") as mock_cred_repo,
        ):
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

        with patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo:
            mock_source_repo.return_value.get_by_id.return_value = None

            with self.assertRaises(ValueError) as ctx:
                manager.resolve_model("nonexistent-model", db)

            self.assertIn("未找到模型配置", str(ctx.exception))

    def test_resolve_model_raises_on_missing_provider(self):
        manager = LLMManager()
        db = MagicMock()

        mock_source = MagicMock()
        mock_source.provider = "unknown"
        mock_source.provider_rel = None

        with patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo:
            mock_source_repo.return_value.get_by_id.return_value = mock_source

            with self.assertRaises(ValueError) as ctx:
                manager.resolve_model("some-model", db)

            self.assertIn("未配置", str(ctx.exception))

    def test_resolve_model_no_base_url_when_not_custom(self):
        manager = LLMManager()
        db = MagicMock()

        mock_provider = MagicMock()
        mock_provider.litellm_prefix = "openrouter/openai"
        mock_provider.custom_base_url = False

        mock_source = MagicMock()
        mock_source.provider = "openai"
        mock_source.provider_rel = mock_provider

        mock_credential = MagicMock()
        mock_credential.credentials = {"api_key": "test-key"}

        with (
            patch("app.ai.llm_manager.ModelSourceRepository") as mock_source_repo,
            patch("app.ai.llm_manager.ModelCredentialRepository") as mock_cred_repo,
        ):
            mock_source_repo.return_value.get_by_id.return_value = mock_source
            mock_cred_repo.return_value.get_default.return_value = mock_credential

            litellm_model, provider, kwargs = manager.resolve_model("gpt-5.4", db)

        self.assertEqual(litellm_model, "openrouter/openai/gpt-5.4")
        self.assertNotIn("api_base", kwargs)


if __name__ == "__main__":
    unittest.main()
