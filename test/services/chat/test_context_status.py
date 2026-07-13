import unittest

from app.schemas.chat import ContextUsage, Usage
from app.services.chat.context_manager import ContextPlan


class ContextStatusTests(unittest.TestCase):
    def test_context_plan_exposes_only_safe_status_fields(self):
        plan = ContextPlan(
            messages=[{"role": "system", "content": "private prompt"}],
            status="trimmed",
            context_window_tokens=100_000,
            context_window_source="private-registry-path",
            context_window_status="known",
            estimated_tokens_before=90_000,
            estimated_tokens_after=70_000,
            removed_turns=2,
            removed_messages=4,
            removed_tool_transactions=1,
        )

        context = plan.to_usage_context(actual_prompt_tokens=69_500)

        self.assertEqual(
            context.model_dump(),
            {
                "status": "trimmed",
                "round_index": None,
                "window_tokens": 100_000,
                "estimated_tokens_before": 90_000,
                "estimated_tokens_after": 70_000,
                "actual_prompt_tokens": 69_500,
                "removed_turns": 2,
                "removed_messages": 4,
                "removed_tool_transactions": 1,
            },
        )
        serialized = str(context.model_dump())
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("private-registry-path", serialized)

    def test_usage_context_is_optional_for_old_history(self):
        legacy = Usage.model_validate({"input_tokens": 12, "output_tokens": 8})
        current = Usage(
            input_tokens=12,
            output_tokens=8,
            context=ContextUsage(
                status="no_op_fast_path",
                window_tokens=128_000,
                actual_prompt_tokens=12,
            ),
        )

        self.assertIsNone(legacy.context)
        self.assertEqual(current.context.actual_prompt_tokens, 12)

    def test_invalid_context_is_discarded_without_losing_legacy_usage(self):
        invalid_samples = [
            {"status": "future-private-status", "window_tokens": 1000},
            {"status": "trimmed", "window_tokens": -1},
            {"status": "trimmed", "removed_messages": -2},
            {"status": "trimmed", "round_index": 0},
            "not-an-object",
        ]

        for invalid_context in invalid_samples:
            with self.subTest(invalid_context=invalid_context):
                usage = Usage.model_validate(
                    {
                        "input_tokens": 12,
                        "output_tokens": 8,
                        "context": invalid_context,
                    }
                )
                self.assertEqual(usage.input_tokens, 12)
                self.assertEqual(usage.output_tokens, 8)
                self.assertIsNone(usage.context)


if __name__ == "__main__":
    unittest.main()
