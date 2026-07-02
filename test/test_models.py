import unittest

from app.api.models import _entry_to_card


class ModelsApiTests(unittest.TestCase):
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
