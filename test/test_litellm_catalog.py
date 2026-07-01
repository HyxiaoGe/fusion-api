import unittest

from app.ai import litellm_catalog


class LiteLLMCatalogTests(unittest.TestCase):
    def test_normalize_capabilities_adds_agent_tools_from_function_calling(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "deepseek-chat",
            {"functionCalling": True, "vision": False},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertTrue(capabilities["agentTools"])

    def test_normalize_capabilities_disables_known_non_agent_models_by_default(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "qwen-vl-max",
            {"functionCalling": True, "vision": True},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])

    def test_normalize_capabilities_respects_explicit_agent_tools_false(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "future-model",
            {"functionCalling": True, "agentTools": False},
        )

        self.assertFalse(capabilities["agentTools"])


if __name__ == "__main__":
    unittest.main()
