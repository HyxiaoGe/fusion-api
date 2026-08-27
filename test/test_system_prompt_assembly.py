"""系统提示词纯本地组装契约。"""

import unittest
from unittest.mock import patch


class SystemPromptAssemblyTests(unittest.TestCase):
    def test_sections_deduplicate_by_identity_not_user_text(self):
        from app.ai.prompts.system_prompt import SystemPromptSection, assemble_system_prompt

        result = assemble_system_prompt(
            user_system_prompt="请解释工具规则",
            sections=lambda: [SystemPromptSection("tool", "工具规则正文"), SystemPromptSection("tool", "重复规则")],
        )
        self.assertEqual(result.metadata["section_ids"], ["app_identity", "tool", "current_date", "user_preferences"])
        self.assertEqual(result.metadata["status"], "ready")
        self.assertEqual(result.metadata["source"], "code")
        self.assertEqual(len(result.metadata["fingerprint"]), 64)
        self.assertEqual(result.metadata["char_count"], sum(len(m["content"]) for m in result.messages))
        self.assertNotIn("请解释", str(result.metadata))
        self.assertNotIn("重复规则", str(result.messages))
        self.assertTrue(all(set(m) == {"role", "content"} for m in result.messages))

    def test_failure_has_safe_metadata_and_no_exception_text(self):
        from app.ai.prompts.system_prompt import SystemPromptAssemblyError, assemble_system_prompt

        with patch("app.ai.prompts.system_prompt.build_current_date_system_prompt", side_effect=ValueError("秘密偏好")):
            with self.assertRaises(SystemPromptAssemblyError) as raised:
                assemble_system_prompt()
        metadata = raised.exception.metadata
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["message"], "系统提示词组装失败")
        self.assertNotIn("秘密", str(metadata))
        self.assertIsNone(metadata["fingerprint"])
        self.assertGreaterEqual(metadata["duration_ms"], 0)

    def test_without_preferences_keeps_base_and_timer_is_local(self):
        from app.ai.prompts.system_prompt import assemble_system_prompt

        with patch("app.ai.prompts.system_prompt.perf_counter", side_effect=[10, 10.0255]):
            result = assemble_system_prompt()
        self.assertEqual(result.metadata["section_ids"], ["app_identity", "current_date"])
        self.assertIsInstance(result.metadata["duration_ms"], int)
        self.assertEqual(result.metadata["duration_ms"], 25)
        self.assertIn("Fusion AI", result.messages[0]["content"])
