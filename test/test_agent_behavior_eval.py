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

    def test_load_samples_rejects_invalid_v1_3_recovery_fields(self):
        from scripts.agent_behavior_eval import load_samples

        base_sample = {
            "id": "planner-v1-3",
            "category": "search_failure_recovery",
            "question": "OpenAI 最近发布了哪些产品更新？",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
        }

        cases = [
            ("expected_search_actions", ["execute", 1], "必须是字符串数组"),
            ("required_search_actions", "repair_search", "必须是字符串数组"),
            ("forbidden_search_actions", ["redirect_to_read_alternative", None], "必须是字符串数组"),
            ("max_repair_search_calls", True, "必须是非负整数"),
            ("max_repair_search_calls", -1, "必须是非负整数"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "samples.json"
                    sample = dict(base_sample)
                    sample["id"] = f"planner-v1-3-{field}"
                    sample[field] = value
                    path.write_text(json.dumps([sample], ensure_ascii=False), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, f"{field} {message}"):
                        load_samples(path)

    def test_load_samples_validates_optional_run_capability_fields(self):
        from scripts.agent_behavior_eval import load_samples

        base_sample = {
            "id": "run-capability-weather",
            "category": "run_capability",
            "question": "明天上海天气怎样？",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "expected_package_id": "weather",
            "expected_announced_tools": ["weather_forecast"],
            "expected_called_tools": ["weather_forecast"],
            "expected_prompt_section_ids": ["app_identity", "current_date"],
            "expected_resolution_mode": "routed",
            "required_capability_reason_codes": ["explicit_weather_request"],
            "expected_effective_plan_mode": "off",
            "expected_network_boundary_required": False,
        }
        invalid_cases = [
            ("expected_package_id", "", "必须是非空字符串"),
            ("expected_package_id", "future_package", "非法"),
            ("expected_announced_tools", ["weather_forecast", 1], "必须是字符串数组"),
            ("expected_called_tools", "weather_forecast", "必须是字符串数组"),
            ("expected_prompt_section_ids", ["app_identity", None], "必须是字符串数组"),
            ("expected_resolution_mode", "future", "非法"),
            ("required_capability_reason_codes", [], "必须是非空字符串数组"),
            ("expected_effective_plan_mode", "sometimes", "非法"),
            ("expected_network_boundary_required", 0, "必须是布尔值"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(json.dumps([base_sample], ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_samples(path), [base_sample])

        for field, value, message in invalid_cases:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "samples.json"
                sample = {**base_sample, field: value}
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
        self.assertIn("search-failure-recovery-v1-3", ids)

    def test_default_sample_file_has_24_main_routes_and_four_adversarial_variants(self):
        from scripts.agent_behavior_eval import DEFAULT_SAMPLE_PATH, load_samples

        samples = load_samples(DEFAULT_SAMPLE_PATH)
        route_samples = [sample for sample in samples if sample["id"].startswith("run-route-")]
        main_samples = [sample for sample in route_samples if sample.get("capability_eval_group") == "main"]
        adversarial_variants = {
            sample.get("adversarial_variant_id")
            for sample in route_samples
            if sample.get("capability_eval_group") == "adversarial"
        }
        required_fields = {
            "expected_package_id",
            "expected_announced_tools",
            "expected_prompt_section_ids",
            "expected_resolution_mode",
            "required_capability_reason_codes",
            "expected_effective_plan_mode",
            "expected_network_boundary_required",
        }

        self.assertGreaterEqual(len(main_samples), 24)
        self.assertGreaterEqual(len(adversarial_variants - {None}), 4)
        for sample in route_samples:
            with self.subTest(sample_id=sample["id"]):
                self.assertEqual(required_fields - set(sample), set())

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

    def test_score_observation_compares_run_capability_observations_exactly(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "run-capability-weather",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "expected_package_id": "weather",
            "expected_announced_tools": ["weather_forecast"],
            "expected_called_tools": ["weather_forecast"],
            "expected_prompt_section_ids": ["app_identity", "current_date"],
            "expected_resolution_mode": "routed",
            "required_capability_reason_codes": ["explicit_weather_request"],
            "expected_effective_plan_mode": "off",
            "expected_network_boundary_required": False,
        }
        observation = {
            "tool_calls": [],
            "surfaces": [],
            "console_errors": [],
            "package_id": "weather",
            "announced_tools": ["weather_forecast"],
            "called_tools": ["weather_forecast"],
            "prompt_section_ids": ["app_identity", "current_date"],
            "resolution_mode": "routed",
            "capability_reason_codes": ["explicit_weather_request", "extra_safe_reason"],
            "effective_plan_mode": "off",
            "network_boundary_required": False,
        }

        score = score_observation(sample, observation)

        self.assertTrue(score["passed"])
        self.assertEqual(score["issues"], [])

    def test_score_observation_fails_when_required_run_capability_observations_are_missing(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "run-capability-direct",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "expected_package_id": "direct",
            "expected_announced_tools": [],
            "expected_called_tools": [],
            "expected_prompt_section_ids": ["app_identity"],
            "expected_resolution_mode": "routed",
            "required_capability_reason_codes": ["direct_greeting"],
            "expected_effective_plan_mode": "off",
            "expected_network_boundary_required": False,
        }

        score = score_observation(
            sample,
            {"tool_calls": [], "surfaces": [], "console_errors": []},
        )

        self.assertFalse(score["passed"])
        for field in (
            "package_id",
            "announced_tools",
            "called_tools",
            "prompt_section_ids",
            "resolution_mode",
            "capability_reason_codes",
            "effective_plan_mode",
            "network_boundary_required",
        ):
            with self.subTest(field=field):
                self.assertIn(f"缺少必需观测字段: {field}", score["issues"])

    def test_score_observation_reports_run_capability_mismatches(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "run-capability-weather",
            "expected_tool_policy": "no_search",
            "expected_surface": "direct_answer",
            "expected_package_id": "weather",
            "expected_announced_tools": ["weather_forecast"],
            "expected_called_tools": ["weather_forecast"],
            "expected_prompt_section_ids": ["app_identity", "current_date"],
            "expected_resolution_mode": "routed",
            "required_capability_reason_codes": ["explicit_weather_request"],
            "expected_effective_plan_mode": "off",
            "expected_network_boundary_required": False,
        }
        observation = {
            "tool_calls": [],
            "surfaces": [],
            "console_errors": [],
            "package_id": "direct",
            "announced_tools": [],
            "called_tools": [],
            "prompt_section_ids": ["app_identity"],
            "resolution_mode": "clarification",
            "capability_reason_codes": ["insufficient_capability_signal"],
            "effective_plan_mode": "auto",
            "network_boundary_required": True,
        }

        score = score_observation(sample, observation)
        issues = "\n".join(score["issues"])

        self.assertFalse(score["passed"])
        self.assertIn("能力包不符合预期", issues)
        self.assertIn("公告工具不符合预期", issues)
        self.assertIn("实际调用工具不符合预期", issues)
        self.assertIn("Prompt section IDs不符合预期", issues)
        self.assertIn("resolution mode不符合预期", issues)
        self.assertIn("缺少必需能力原因码", issues)
        self.assertIn("effective plan mode不符合预期", issues)
        self.assertIn("network boundary 标记不符合预期", issues)

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

    def test_score_observation_flags_v1_3_recovery_action_regressions(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "search-failure-recovery-v1-3",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "expected_search_actions": ["execute", "repair_search"],
            "required_search_actions": ["repair_search"],
            "forbidden_search_actions": ["redirect_to_read_alternative"],
            "max_repair_search_calls": 1,
        }
        observation = {
            "answer_text": "基于搜索结果回答。",
            "tool_calls": ["web_search", "web_search", "web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "search_actions": ["execute", "repair_search", "repair_search", "redirect_to_read_alternative"],
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertFalse(score["passed"])
        joined_issues = "\n".join(score["issues"])
        self.assertIn("搜索动作不符合预期", joined_issues)
        self.assertIn("包含禁止搜索动作", joined_issues)
        self.assertIn("repair 搜索次数过多", joined_issues)

    def test_score_observation_passes_v1_3_recovery_action_sample(self):
        from scripts.agent_behavior_eval import score_observation

        sample = {
            "id": "search-failure-recovery-v1-3",
            "expected_tool_policy": "search",
            "expected_surface": "evidence",
            "expected_search_actions": ["execute", "repair_search"],
            "required_search_actions": ["repair_search"],
            "max_repair_search_calls": 1,
        }
        observation = {
            "answer_text": "基于搜索结果回答。",
            "tool_calls": ["web_search", "web_search"],
            "surfaces": ["execution_process", "answer_evidence"],
            "search_actions": ["execute", "repair_search"],
            "console_errors": [],
        }

        score = score_observation(sample, observation)

        self.assertTrue(score["passed"])
        self.assertEqual(score["issues"], [])

    def test_write_dry_run_outputs_jsonl(self):
        from scripts.agent_behavior_eval import write_dry_run

        samples = [
            {
                "id": "identity-direct-answer",
                "category": "direct_answer",
                "question": "你好，你是谁？",
                "expected_tool_policy": "no_search",
                "expected_surface": "direct_answer",
                "expected_package_id": "direct",
                "expected_announced_tools": [],
                "expected_prompt_section_ids": ["app_identity"],
            }
        ]
        output = io.StringIO()

        write_dry_run(samples, output)

        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["sample_id"], "identity-direct-answer")
        self.assertEqual(lines[0]["expected_tool_policy"], "no_search")
        self.assertEqual(lines[0]["expected_surface"], "direct_answer")
        self.assertEqual(lines[0]["expected_package_id"], "direct")
        self.assertEqual(lines[0]["expected_announced_tools"], [])
        self.assertEqual(lines[0]["expected_prompt_section_ids"], ["app_identity"])
        self.assertFalse(lines[0]["passed"])


class RunCapabilityBehaviorEvalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_fixture_runs_through_real_router_and_prompt_assembly(self):
        from app.services.mcp.amap_product_tools import AMAP_PRODUCT_DEFINITIONS
        from app.services.mcp.flyai_travel_tools import FLYAI_TRAVEL_DEFINITIONS
        from app.services.stream.agent_loop_request_prep import (
            build_agent_loop_call_config,
            prepare_agent_loop_messages,
        )
        from scripts.agent_behavior_eval import DEFAULT_SAMPLE_PATH, load_samples

        samples = [sample for sample in load_samples(DEFAULT_SAMPLE_PATH) if sample["id"].startswith("run-route-")]
        self.assertGreaterEqual(len(samples), 29)
        dynamic_tools = [
            *AMAP_PRODUCT_DEFINITIONS,
            *FLYAI_TRAVEL_DEFINITIONS,
            {
                "type": "function",
                "function": {
                    "name": "mcp_unrelated_tool",
                    "description": "行为评估专用 MCP 工具。",
                    "parameters": {"type": "object", "additionalProperties": False},
                },
            },
        ]
        dynamic_tool_names = [tool["function"]["name"] for tool in dynamic_tools]
        handlers = {name: object() for name in dynamic_tool_names}
        bindings = [
            {"alias": name, "server_id": f"eval-server-{index}"} for index, name in enumerate(dynamic_tool_names)
        ]

        async def build_messages(*_args, **_kwargs):
            return []

        for sample in samples:
            with self.subTest(sample_id=sample["id"]):
                options = sample.get("options", {})
                capabilities = sample.get(
                    "capabilities",
                    {"functionCalling": True, "searchCapable": True, "agentTools": True},
                )
                task_context_messages = sample.get(
                    "task_context_messages",
                    [{"role": "user", "content": sample["question"]}],
                )
                call_config = build_agent_loop_call_config(
                    provider="openai",
                    options=options,
                    capabilities=capabilities,
                    additional_tools=dynamic_tools,
                    dynamic_tool_handlers=handlers,
                    tool_bindings=bindings,
                    original_message=sample["question"],
                    task_context_messages=task_context_messages,
                )
                prepared = await prepare_agent_loop_messages(
                    db=object(),
                    user_id="eval-user",
                    raw_messages=task_context_messages,
                    has_vision=False,
                    file_ids=None,
                    original_message=sample["question"],
                    call_config=call_config,
                    file_repo_factory=lambda _db: object(),
                    load_user_system_prompt_fn=lambda _db, _user_id: sample.get("user_system_prompt"),
                    build_llm_messages_fn=build_messages,
                    preprocess_user_input=False,
                )
                resolution = call_config.capability_resolution
                announced_definition_names = [
                    tool["function"]["name"]
                    for tool in call_config.call_kwargs.get("tools", [])
                    if tool["function"]["name"] != "update_plan"
                ]
                expected_dynamic_tools = [name for name in sample["expected_announced_tools"] if name in handlers]

                self.assertEqual(resolution.package_id, sample["expected_package_id"])
                self.assertEqual(call_config.announced_tools, sample["expected_announced_tools"])
                self.assertEqual(list(resolution.external_tool_names), sample["expected_announced_tools"])
                self.assertEqual(announced_definition_names, sample["expected_announced_tools"])
                self.assertEqual(
                    list(call_config.dynamic_tool_handlers),
                    expected_dynamic_tools,
                )
                self.assertEqual(
                    [binding["alias"] for binding in call_config.tool_bindings],
                    expected_dynamic_tools,
                )
                self.assertEqual(prepared.final_tool_names, sample["expected_announced_tools"])
                self.assertEqual(
                    prepared.prompt_assembly["section_ids"],
                    sample["expected_prompt_section_ids"],
                )
                self.assertEqual(resolution.resolution_mode, sample["expected_resolution_mode"])
                self.assertTrue(set(sample["required_capability_reason_codes"]).issubset(resolution.reason_codes))
                self.assertEqual(
                    resolution.effective_plan_mode,
                    sample["expected_effective_plan_mode"],
                )
                self.assertEqual(
                    resolution.include_current_date,
                    "current_date" in sample["expected_prompt_section_ids"],
                )
                self.assertEqual(
                    resolution.network_boundary_required,
                    sample["expected_network_boundary_required"],
                )


if __name__ == "__main__":
    unittest.main()
