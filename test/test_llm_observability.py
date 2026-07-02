import unittest

from app.ai.llm_observability import build_litellm_metadata, merge_openai_extra_body


class LLMObservabilityTests(unittest.TestCase):
    def test_build_litellm_metadata_uses_low_cardinality_tags(self):
        metadata = build_litellm_metadata("chat_stream")

        self.assertEqual(metadata, {"tags": ["app:fusion", "phase:chat_stream"]})

    def test_merge_openai_extra_body_preserves_existing_fields(self):
        extra_body = {"thinking": {"type": "disabled"}}

        merged = merge_openai_extra_body("search_summary", extra_body)

        self.assertEqual(
            merged,
            {
                "thinking": {"type": "disabled"},
                "metadata": {"tags": ["app:fusion", "phase:search_summary"]},
            },
        )
        self.assertEqual(extra_body, {"thinking": {"type": "disabled"}})

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            build_litellm_metadata("conversation-123")


if __name__ == "__main__":
    unittest.main()
