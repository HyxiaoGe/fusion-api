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
        self.assertTrue(capabilities["searchCapable"])
        self.assertTrue(capabilities["webSearch"])

    def test_normalize_capabilities_disables_known_non_agent_models_by_default(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "qwen-vl-max",
            {"functionCalling": True, "vision": True},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])
        self.assertFalse(capabilities["webSearch"])

    def test_normalize_capabilities_respects_explicit_agent_tools_false(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "future-model",
            {"functionCalling": True, "agentTools": False},
        )

        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])

    def test_normalize_capabilities_never_enables_agent_tools_without_function_calling(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "metadata-mismatch",
            {"functionCalling": False, "agentTools": True, "webSearch": True},
        )

        self.assertFalse(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])
        self.assertFalse(capabilities["webSearch"])

    def test_normalize_capabilities_preserves_vision_independent_of_search_tools(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "vision-only",
            {"functionCalling": False, "vision": True},
        )

        self.assertTrue(capabilities["vision"])
        self.assertFalse(capabilities["searchCapable"])

    def test_normalize_capabilities_uses_runtime_disabled_agent_tool_aliases(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "deepseek-chat",
            {"functionCalling": True},
            agent_tools_disabled_aliases={"deepseek-chat"},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])


if __name__ == "__main__":
    unittest.main()
