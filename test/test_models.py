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


if __name__ == "__main__":
    unittest.main()
