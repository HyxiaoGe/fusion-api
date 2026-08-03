import unittest
from copy import deepcopy

from app.ai.prompts.agent_loop import VISIBLE_RESPONSE_LANGUAGE_PROMPT
from app.services.chat.model_call_language_policy import finalize_model_call_language_policy


class ModelCallLanguagePolicyTests(unittest.TestCase):
    def test_moves_single_contract_to_last_effective_system_instruction(self):
        messages = [
            {
                "role": "system",
                "content": f"身份规则\n\n{VISIBLE_RESPONSE_LANGUAGE_PROMPT}",
            },
            {"role": "system", "content": "深度研究阶段规则"},
            {"role": "user", "content": "具身智能产业的影响"},
        ]
        original = deepcopy(messages)

        finalized = finalize_model_call_language_policy(messages)

        self.assertEqual(messages, original)
        self.assertEqual(_contract_count(finalized), 1)
        self.assertEqual(finalized[-2]["role"], "system")
        self.assertTrue(finalized[-2]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertIn("深度研究阶段规则", finalized[-2]["content"])
        self.assertNotIn(VISIBLE_RESPONSE_LANGUAGE_PROMPT, finalized[0]["content"])

    def test_preserves_tool_transaction_and_raw_reasoning_when_repair_rule_is_appended(self):
        messages = [
            {"role": "system", "content": "身份规则"},
            {"role": "user", "content": "分析最新进展"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should inspect the source.",
                "tool_calls": [{"id": "call-1", "function": {"name": "web_search"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "搜索结果"},
            {"role": "system", "content": "修复计划状态后继续"},
        ]
        original = deepcopy(messages)

        finalized = finalize_model_call_language_policy(messages)

        self.assertEqual(messages, original)
        self.assertEqual([item["role"] for item in finalized], [item["role"] for item in original])
        self.assertEqual(finalized[2], original[2])
        self.assertEqual(finalized[3], original[3])
        self.assertTrue(finalized[-1]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertEqual(_contract_count(finalized), 1)

    def test_preserves_normal_tool_transaction_without_a_tail_repair_system(self):
        messages = [
            {"role": "system", "content": "身份规则"},
            {"role": "user", "content": "分析最新进展"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should inspect the source.",
                "tool_calls": [{"id": "call-1", "function": {"name": "web_search"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "搜索结果"},
        ]
        original = deepcopy(messages)

        finalized = finalize_model_call_language_policy(messages)

        self.assertEqual(messages, original)
        self.assertEqual([item["role"] for item in finalized], [item["role"] for item in original])
        self.assertEqual(finalized[1:], original[1:])
        self.assertTrue(finalized[0]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertEqual(_contract_count(finalized), 1)

    def test_is_idempotent_and_adds_a_system_instruction_when_missing(self):
        messages = [{"role": "user", "content": "Explain optimistic locking."}]

        once = finalize_model_call_language_policy(messages)
        twice = finalize_model_call_language_policy(once)

        self.assertEqual(once, twice)
        self.assertEqual(once[0], {"role": "system", "content": VISIBLE_RESPONSE_LANGUAGE_PROMPT})
        self.assertEqual(_contract_count(once), 1)

    def test_contract_follows_real_user_language_instead_of_forcing_chinese(self):
        self.assertIn("最后一条真实用户请求", VISIBLE_RESPONSE_LANGUAGE_PROMPT)
        self.assertIn("内部控制消息不得改变", VISIBLE_RESPONSE_LANGUAGE_PROMPT)
        self.assertIn("reasoning_content 输出的思考过程会原样实时展示", VISIBLE_RESPONSE_LANGUAGE_PROMPT)
        self.assertIn("第一个 reasoning_content token", VISIBLE_RESPONSE_LANGUAGE_PROMPT)
        self.assertNotIn("必须使用中文", VISIBLE_RESPONSE_LANGUAGE_PROMPT)


def _contract_count(messages: list[dict]) -> int:
    return sum(str(message.get("content") or "").count(VISIBLE_RESPONSE_LANGUAGE_PROMPT) for message in messages)
