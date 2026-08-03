import unittest

from app.services.stream.reasoning_policy import configure_reasoning_call_kwargs


class ReasoningPolicyTests(unittest.TestCase):
    def test_deepseek_thinking_tools_remove_tool_choice_and_preserve_extra_body(self):
        original = {
            "tools": [{"function": {"name": "web_search"}}],
            "tool_choice": "auto",
            "extra_body": {"trace": "kept"},
        }

        configured = configure_reasoning_call_kwargs(
            original,
            provider="deepseek",
            should_use_reasoning=True,
        )

        self.assertNotIn("tool_choice", configured)
        self.assertEqual(
            configured["extra_body"],
            {"trace": "kept", "thinking": {"type": "enabled"}},
        )
        self.assertEqual(original["tool_choice"], "auto")

    def test_gemini_requests_reasoning_content_mapping(self):
        configured = configure_reasoning_call_kwargs(
            {"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto"},
            provider="gemini",
            should_use_reasoning=True,
        )

        self.assertEqual(configured["reasoning_effort"], "high")
        self.assertEqual(configured["tool_choice"], "auto")

    def test_volcengine_disables_thinking_only_while_tools_are_present(self):
        tool_round = configure_reasoning_call_kwargs(
            {
                "tools": [{"function": {"name": "web_search"}}],
                "extra_body": {"trace": "kept"},
            },
            provider="volcengine",
            should_use_reasoning=True,
        )
        synthesis_round = configure_reasoning_call_kwargs(
            {
                "extra_body": {"trace": "kept", "thinking": {"type": "disabled"}},
            },
            provider="volcengine",
            should_use_reasoning=True,
        )

        self.assertEqual(
            tool_round["extra_body"],
            {"trace": "kept", "thinking": {"type": "disabled"}},
        )
        self.assertEqual(synthesis_round["extra_body"], {"trace": "kept"})

    def test_reasoning_override_off_leaves_provider_parameters_untouched(self):
        original = {"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto"}

        configured = configure_reasoning_call_kwargs(
            original,
            provider="deepseek",
            should_use_reasoning=False,
        )

        self.assertEqual(configured, original)


if __name__ == "__main__":
    unittest.main()
