import unittest

from app.ai.llm_observability import build_litellm_metadata, merge_litellm_kwargs, merge_openai_extra_body


class LLMObservabilityTests(unittest.TestCase):
    def test_build_litellm_metadata_uses_low_cardinality_tags(self):
        metadata = build_litellm_metadata("chat_stream")

        self.assertEqual(metadata, {"tags": ["app:fusion", "phase:chat_stream"]})

    def test_merge_openai_extra_body_preserves_existing_fields(self):
        extra_body = {"thinking": {"type": "disabled"}}

        merged = merge_openai_extra_body("file_processing", extra_body)

        self.assertEqual(
            merged,
            {
                "thinking": {"type": "disabled"},
                "metadata": {"tags": ["app:fusion", "phase:file_processing"]},
            },
        )
        self.assertEqual(extra_body, {"thinking": {"type": "disabled"}})

    def test_merge_litellm_kwargs_sends_tags_through_extra_body(self):
        kwargs = {
            "api_key": "test-key",
            "extra_body": {
                "thinking": {"type": "disabled"},
                "metadata": {"existing": "keep"},
            },
        }

        merged = merge_litellm_kwargs("chat_stream", kwargs)

        self.assertEqual(merged["api_key"], "test-key")
        self.assertEqual(
            merged["extra_body"],
            {
                "thinking": {"type": "disabled"},
                "metadata": {
                    "existing": "keep",
                    "tags": ["app:fusion", "phase:chat_stream"],
                },
            },
        )
        self.assertNotIn("metadata", merged)
        self.assertEqual(
            kwargs,
            {
                "api_key": "test-key",
                "extra_body": {
                    "thinking": {"type": "disabled"},
                    "metadata": {"existing": "keep"},
                },
            },
        )

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            build_litellm_metadata("conversation-123")


if __name__ == "__main__":
    unittest.main()
