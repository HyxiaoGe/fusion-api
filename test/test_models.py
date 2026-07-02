import unittest

from app.api.models import _entry_to_card


class ModelsApiTests(unittest.TestCase):
    def test_entry_to_card_exposes_context_token_limits(self):
        card = _entry_to_card(
            "xiaomi/mimo-v2.5-pro",
            {
                "underlying": "xiaomi/mimo-v2.5-pro",
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 32_768,
                "metadata": {
                    "provider_key": "xiaomi",
                    "capabilities": {"functionCalling": True, "vision": False},
                },
            },
        )

        self.assertEqual(card["contextWindowTokens"], 1_000_000)
        self.assertEqual(card["maxOutputTokens"], 32_768)

    def test_entry_to_card_omits_invalid_context_token_limits(self):
        card = _entry_to_card(
            "unknown-model",
            {
                "underlying": "unknown/model",
                "max_input_tokens": "not-a-number",
                "max_output_tokens": 0,
                "metadata": {
                    "provider_key": "unknown",
                    "capabilities": {"functionCalling": False},
                },
            },
        )

        self.assertIsNone(card["contextWindowTokens"])
        self.assertIsNone(card["maxOutputTokens"])

    def test_entry_to_card_exposes_agent_tools_capability(self):
        card = _entry_to_card(
            "qwen-vl-max",
            {
                "underlying": "qwen/qwen-vl-max",
                "metadata": {
                    "provider_key": "qwen",
                    "capabilities": {"functionCalling": True, "vision": True},
                },
            },
        )

        self.assertTrue(card["capabilities"]["functionCalling"])
        self.assertFalse(card["capabilities"]["agentTools"])
        self.assertFalse(card["capabilities"]["searchCapable"])
        self.assertFalse(card["capabilities"]["webSearch"])
        self.assertTrue(card["capabilities"]["vision"])

    def test_entry_to_card_derives_search_capable_from_runtime_tool_contract(self):
        card = _entry_to_card(
            "deepseek-chat",
            {
                "underlying": "deepseek/deepseek-chat",
                "metadata": {
                    "provider_key": "deepseek",
                    "capabilities": {"functionCalling": True, "webSearch": False},
                },
            },
        )

        self.assertTrue(card["capabilities"]["functionCalling"])
        self.assertTrue(card["capabilities"]["agentTools"])
        self.assertTrue(card["capabilities"]["searchCapable"])
        self.assertTrue(card["capabilities"]["webSearch"])

    def test_entry_to_card_does_not_treat_legacy_web_search_as_runtime_capability(self):
        card = _entry_to_card(
            "legacy-model",
            {
                "underlying": "legacy/model",
                "metadata": {
                    "provider_key": "legacy",
                    "capabilities": {"functionCalling": False, "agentTools": True, "webSearch": True},
                },
            },
        )

        self.assertFalse(card["capabilities"]["functionCalling"])
        self.assertFalse(card["capabilities"]["agentTools"])
        self.assertFalse(card["capabilities"]["searchCapable"])
        self.assertFalse(card["capabilities"]["webSearch"])


if __name__ == "__main__":
    unittest.main()
