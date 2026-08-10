import unittest
from unittest.mock import patch

from app.ai.llm_manager import LLMManager


class LLMManagerCatalogFallbackTests(unittest.TestCase):
    def test_degraded_catalog_without_cache_keeps_proxy_alias_routable(self):
        manager = LLMManager()
        with (
            patch("app.ai.llm_manager.litellm_catalog.get_model_entry", return_value=None),
            patch(
                "app.ai.llm_manager.litellm_catalog.get_cache_status",
                return_value={"availability": "degraded", "has_cache": False},
            ),
            patch("app.ai.llm_manager.litellm_catalog.get_underlying_provider", return_value="litellm"),
        ):
            model, provider, kwargs = manager.resolve_model("fallback/model")

        self.assertEqual(model, "litellm_proxy/fallback/model")
        self.assertEqual(provider, "litellm")
        self.assertIn("api_base", kwargs)

    def test_available_or_cached_catalog_still_rejects_missing_alias(self):
        manager = LLMManager()
        for status in (
            {"availability": "available", "has_cache": False},
            {"availability": "degraded", "has_cache": True},
        ):
            with (
                self.subTest(status=status),
                patch("app.ai.llm_manager.litellm_catalog.get_model_entry", return_value=None),
                patch("app.ai.llm_manager.litellm_catalog.get_cache_status", return_value=status),
                self.assertRaisesRegex(ValueError, "不在 LiteLLM Proxy 注册表"),
            ):
                manager.resolve_model("missing/model")


if __name__ == "__main__":
    unittest.main()
