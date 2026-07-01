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

    def test_load_samples_rejects_invalid_planner_limits(self):
        from scripts.agent_behavior_eval import load_samples

        base_sample = {
            "id": "planner-limits",
            "category": "search_read_planner",
            "question": "OpenAI 最近发布了哪些产品更新？",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
        }

        cases = [
            ("max_duplicate_search_keywords", -1),
            ("max_duplicate_search_keywords", True),
            ("max_recommended_reads", -1),
            ("max_recommended_reads", "2"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "samples.json"
                    sample = dict(base_sample)
                    sample["id"] = f"planner-limits-{field}"
                    sample[field] = value
                    path.write_text(json.dumps([sample], ensure_ascii=False), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, f"{field} 必须是非负整数"):
                        load_samples(path)

    def test_load_samples_rejects_invalid_v1_2_planner_fields(self):
        from scripts.agent_behavior_eval import load_samples

        base_sample = {
            "id": "planner-v1-2",
            "category": "search_read_planner",
            "question": "OpenAI 最近发布了哪些产品更新？",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
        }

        cases = [
            ("max_search_calls", -1, "必须是非负整数"),
            ("max_provider_search_calls", True, "必须是非负整数"),
            ("expected_search_budgets", ["freshness", 1], "必须是字符串数组"),
            ("forbidden_read_domains", "youtube.com", "必须是字符串数组"),
            ("required_decision_reason_codes", ["official_original", None], "必须是字符串数组"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "samples.json"
                    sample = dict(base_sample)
                    sample["id"] = f"planner-v1-2-{field}"
                    sample[field] = value
                    path.write_text(json.dumps([sample], ensure_ascii=False), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, f"{field} {message}"):
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
        self.assertIn("search-read-planner-dedup-and-read-limit", ids)

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

    def test_score_observation_flags_duplicate_search_keywords_and_excess_recommended_reads(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "search-read-planner-dedup-and-read-limit",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "requires_search_keywords": True,
            "max_duplicate_search_keywords": 0,
            "max_recommended_reads": 2,
        }
        observation = {
            "answer_text": "基于搜索结果回答。",
            "tool_calls": ["web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "search_keywords": [
                "OpenAI 最新公告 2026年6月 新闻",
                "OpenAI 最新公告 2026年6月 新闻",
            ],
            "recommended_read_count": 3,
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertFalse(score["passed"])
        self.assertIn("搜索关键词重复次数过多: duplicate_count=1 max=0", score["issues"])
        self.assertIn("推荐深读数量过多: actual=3 max=2", score["issues"])

    def test_score_observation_flags_v1_2_planner_decision_regressions(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "search-read-planner-v1-2-decision-ledger",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "max_search_calls": 2,
            "max_provider_search_calls": 2,
            "expected_search_budgets": ["freshness", "freshness_followup"],
            "forbidden_read_domains": ["youtube.com"],
            "required_decision_reason_codes": ["official_original"],
        }
        observation = {
            "answer_text": "基于搜索结果回答。",
            "tool_calls": ["web_search", "web_search", "web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "search_call_count": 3,
            "provider_search_call_count": 3,
            "search_budgets": ["freshness", "standard", "standard"],
            "read_domains": ["youtube.com"],
            "decision_reason_codes": [],
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertFalse(score["passed"])
        joined_issues = "\n".join(score["issues"])
        self.assertIn("搜索调用次数过多", joined_issues)
        self.assertIn("provider 搜索次数过多", joined_issues)
        self.assertIn("搜索预算不符合预期", joined_issues)
        self.assertIn("读取了禁止深读的域名", joined_issues)
        self.assertIn("缺少必需决策原因", joined_issues)

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
