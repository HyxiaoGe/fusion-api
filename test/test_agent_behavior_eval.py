import io
import json
import tempfile
import unittest
from pathlib import Path


class AgentBehaviorEvalTests(unittest.TestCase):
    def test_load_samples_rejects_duplicate_ids(self):
        from scripts.agent_behavior_eval import load_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "same",
                            "category": "direct_answer",
                            "question": "你好，你是谁？",
                            "expected_tool_policy": "no_search",
                            "expected_surface": "direct_answer",
                        },
                        {
                            "id": "same",
                            "category": "freshness",
                            "question": "微信A2A互通怎么用？",
                            "expected_tool_policy": "search",
                            "expected_surface": "evidence",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复样本 id"):
                load_samples(path)

    def test_load_samples_rejects_invalid_policy(self):
        from scripts.agent_behavior_eval import load_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "bad-policy",
                            "category": "direct_answer",
                            "question": "你好",
                            "expected_tool_policy": "maybe_search",
                            "expected_surface": "direct_answer",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected_tool_policy 非法"):
                load_samples(path)

    def test_default_sample_file_covers_core_agent_behavior_matrix(self):
        from scripts.agent_behavior_eval import DEFAULT_SAMPLE_PATH, load_samples

        samples = load_samples(DEFAULT_SAMPLE_PATH)

        ids = {sample["id"] for sample in samples}
        self.assertGreaterEqual(len(samples), 6)
        self.assertIn("identity-direct-answer", ids)
        self.assertIn("simple-math-direct-answer", ids)
        self.assertIn("realtime-product-feature-search", ids)
        self.assertIn("search-failure-degraded", ids)
        self.assertIn("url-read-failure-skipped", ids)
        self.assertIn("refresh-recovery-preserves-surface", ids)
        self.assertIn("console-error-regression", ids)

    def test_identity_sample_blocks_upstream_identity_variants(self):
        from scripts.agent_behavior_eval import DEFAULT_SAMPLE_PATH, load_samples

        samples = load_samples(DEFAULT_SAMPLE_PATH)
        identity_sample = next(sample for sample in samples if sample["id"] == "identity-direct-answer")

        forbidden_terms = set(identity_sample["forbidden_answer_terms"])
        self.assertGreaterEqual(
            forbidden_terms,
            {"Claude", "Anthropic", "ChatGPT", "OpenAI", "Gemini", "Google", "DeepSeek"},
        )

    def test_score_observation_passes_direct_answer_without_tools_or_surfaces(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "identity-direct-answer",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "forbidden_answer_terms": ["Claude", "Anthropic"],
            "forbidden_internal_terms": ["url_read", "reader-service"],
        }
        observation = {
            "answer_text": "我是 Fusion AI 中的 AI 助手。",
            "tool_calls": [],
            "surfaces": [],
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertTrue(score["passed"])
        self.assertEqual(score["issues"], [])

    def test_score_observation_fails_direct_answer_with_search_surface_or_wrong_identity(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "identity-direct-answer",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "forbidden_answer_terms": ["Claude", "Anthropic"],
            "forbidden_internal_terms": ["url_read", "reader-service"],
        }
        observation = {
            "answer_text": "我是 Claude，由 Anthropic 开发。",
            "tool_calls": ["web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "console_errors": ["React #185"],
        }

        score = score_observation(sample, observation)

        self.assertFalse(score["passed"])
        self.assertIn("no_search 场景不应调用 web_search", score["issues"])
        self.assertIn("direct_answer 场景不应展示 execution_process", score["issues"])
        self.assertIn("direct_answer 场景不应展示 answer_evidence", score["issues"])
        self.assertIn("回答包含禁止身份词: Claude", score["issues"])
        self.assertIn("存在 console error: React #185", score["issues"])

    def test_score_observation_flags_upstream_identity_variants(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "identity-direct-answer",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "forbidden_answer_terms": ["Claude", "Anthropic", "ChatGPT", "OpenAI", "Gemini", "Google", "DeepSeek"],
        }

        for term in sample["forbidden_answer_terms"]:
            with self.subTest(term=term):
                score = score_observation(
                    sample,
                    {
                        "answer_text": f"我是 {term}。",
                        "tool_calls": [],
                        "surfaces": [],
                        "console_errors": [],
                    },
                )

                self.assertFalse(score["passed"])
                self.assertIn(f"回答包含禁止身份词: {term}", score["issues"])

    def test_score_observation_requires_search_keywords_and_sources_for_search_case(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "realtime-product-feature-search",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "min_sources": 3,
            "requires_search_keywords": True,
            "forbidden_internal_terms": ["url_read", "reader-service"],
        }
        observation = {
            "answer_text": "微信A2A互通需要查看最新资料。",
            "tool_calls": ["web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "search_keywords": ["微信A2A互通 使用方法 2026"],
            "source_count": 3,
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertTrue(score["passed"])
        self.assertEqual(score["issues"], [])

    def test_score_observation_flags_internal_leaks_and_missing_search_context(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "realtime-product-feature-search",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "min_sources": 3,
            "requires_search_keywords": True,
            "forbidden_internal_terms": ["url_read", "reader-service"],
        }
        observation = {
            "answer_text": "工具 url_read 调用了 reader-service。",
            "tool_calls": [],
            "surfaces": ["answer_evidence"],
            "search_keywords": [],
            "source_count": 1,
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertFalse(score["passed"])
        self.assertIn("search 场景必须调用 web_search", score["issues"])
        self.assertIn("evidence 场景应展示 execution_process", score["issues"])
        self.assertIn("搜索场景应展示搜索关键词", score["issues"])
        self.assertIn("来源数量不足: actual=1 min=3", score["issues"])
        self.assertIn("输出包含内部实现词: url_read", score["issues"])
        self.assertIn("输出包含内部实现词: reader-service", score["issues"])

    def test_write_dry_run_outputs_jsonl(self):
        from scripts.agent_behavior_eval import write_dry_run

        samples = [
            {
                "id": "identity-direct-answer",
                "category": "direct_answer",
                "question": "你好，你是谁？",
                "expected_tool_policy": "no_search",
                "expected_surface": "direct_answer",
            }
        ]
        output = io.StringIO()

        write_dry_run(samples, output)

        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["sample_id"], "identity-direct-answer")
        self.assertEqual(lines[0]["expected_tool_policy"], "no_search")
        self.assertEqual(lines[0]["expected_surface"], "direct_answer")
        self.assertFalse(lines[0]["passed"])


if __name__ == "__main__":
    unittest.main()
