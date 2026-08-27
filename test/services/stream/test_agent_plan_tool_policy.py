import unittest
from types import SimpleNamespace

from app.services.stream.agent_plan_tool_policy import (
    resolve_agent_plan_tool_policy,
    resolve_product_capability_signals,
)


class AgentPlanToolPolicyTests(unittest.TestCase):
    def test_product_signal_resolver_is_reusable_by_run_router(self):
        signals = resolve_product_capability_signals(
            original_message="我住在南景新村，公司在双子塔，请比较地铁和驾车通勤路线。",
            task_context_messages=None,
        )

        self.assertTrue(signals.explicit_route)
        self.assertFalse(signals.adjacent_route_followup)
        self.assertFalse(signals.weather)

    def test_product_signal_resolver_recognizes_bare_intercity_choice(self):
        signals = resolve_product_capability_signals(
            original_message="北京到上海哪种方式好？",
            task_context_messages=None,
        )

        self.assertTrue(signals.endpoint_relation)
        self.assertTrue(signals.intercity_mobility)

    def test_product_signal_resolver_rejects_non_travel_city_relation(self):
        signals = resolve_product_capability_signals(
            original_message="从北京大学到上海交通大学申请哪个更适合我？",
            task_context_messages=None,
        )

        self.assertTrue(signals.endpoint_relation)
        self.assertFalse(signals.intercity_mobility)

    def test_structured_endpoint_relation_is_a_mobility_signal_without_extra_keywords(self):
        for message in (
            "从北京到上海",
            "从北京去上海",
            "住在北京，公司在上海",
        ):
            with self.subTest(message=message):
                signals = resolve_product_capability_signals(
                    original_message=message,
                    task_context_messages=None,
                )

                self.assertTrue(signals.endpoint_relation)
                self.assertTrue(signals.intercity_mobility)

    def test_bare_business_route_is_not_an_explicit_or_intercity_mobility_signal(self):
        signals = resolve_product_capability_signals(
            original_message="比较北京到上海两家公司的发展路线",
            task_context_messages=None,
        )

        self.assertTrue(signals.endpoint_relation)
        self.assertFalse(signals.explicit_route)
        self.assertFalse(signals.intercity_mobility)

    def test_verified_research_requires_search_and_two_independent_reads(self):
        messages = [
            "帮我调研一下韩国股市近几年的起伏，重点说说主要原因和争议。",
            "请围绕韩国股市进行联网调研，给出关键结论。",
            "请全面调研并对比两家公司的技术路线。",
            "请联网调研官方公告并整理时间线。",
            "请分析这个政策，并提供可靠来源和可核验原文。",
            "请核验各方争议说法，给出权威证据。",
        ]

        for message in messages:
            with self.subTest(message=message):
                policy = resolve_agent_plan_tool_policy(
                    original_message=message,
                    announced_tool_names=["web_search", "url_read", "route_compare"],
                )

                self.assertEqual(
                    policy.required_initial_tool_counts,
                    {"web_search": 1, "url_read": 2},
                )
                self.assertIsNone(policy.allowed_tool_names)
                self.assertEqual(policy.reason, "verified_research_request")

    def test_single_fact_lookup_does_not_force_verified_research_plan(self):
        messages = [
            "KOSPI 今天多少点？",
            "帮我查一下 OpenAI 官网地址。",
            "Python 3.14 是什么？",
            "不要深入调研，简单回答即可。",
            "不用核验争议说法，概括已有信息即可。",
        ]

        for message in messages:
            with self.subTest(message=message):
                policy = resolve_agent_plan_tool_policy(
                    original_message=message,
                    announced_tool_names=["web_search", "url_read"],
                )

                self.assertEqual(policy.required_initial_tool_counts, {})
                self.assertIsNone(policy.allowed_tool_names)

        reliable_source_policy = resolve_agent_plan_tool_policy(
            original_message="不要使用不可靠来源，请给出权威原文。",
            announced_tool_names=["web_search", "url_read"],
        )
        self.assertEqual(
            reliable_source_policy.required_initial_tool_counts,
            {"web_search": 1, "url_read": 2},
        )

    def test_verified_route_research_merges_route_and_evidence_requirements(self):
        policy = resolve_agent_plan_tool_policy(
            original_message=(
                "请联网调研从南景新村到双子塔的驾车和地铁路线，"
                "核验争议说法并给出权威来源。"
            ),
            announced_tool_names=["web_search", "url_read", "route_compare"],
        )

        self.assertEqual(
            policy.required_initial_tool_counts,
            {"web_search": 1, "url_read": 2, "route_compare": 1},
        )
        self.assertIsNone(policy.allowed_tool_names)
        self.assertEqual(
            policy.reason,
            "verified_research_request+explicit_route_task",
        )

    def test_explicit_commute_comparison_requires_route_tool_and_blocks_generic_substitutes(self):
        policy = resolve_agent_plan_tool_policy(
            original_message=("我住在南景新村，公司在双子塔，请帮我比较驾车、公交和地铁的通勤路线，并给出推荐选择。"),
            announced_tool_names=[
                "web_search",
                "local_place_search",
                "route_compare",
                "weather_forecast",
            ],
        )

        self.assertEqual(policy.required_initial_tool_counts, {"route_compare": 1})
        self.assertEqual(policy.allowed_tool_names, frozenset({"route_compare"}))
        self.assertEqual(policy.reason, "explicit_route_task")

    def test_route_news_question_is_not_misclassified_as_route_planning(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="深圳公交路线最近有什么调整？请联网查下新闻。",
            announced_tool_names=["web_search", "route_compare"],
        )

        self.assertEqual(policy.required_initial_tool_counts, {})
        self.assertIsNone(policy.allowed_tool_names)

    def test_policy_or_environmental_research_is_not_misclassified_as_route_planning(self):
        messages = [
            "请联网研究深圳绿色通勤政策，包括公交、步行和骑行路线的环保影响。",
            "从环保角度分析公交路线、步行和驾车的碳排放差异，不需要规划具体行程。",
        ]

        for message in messages:
            with self.subTest(message=message):
                policy = resolve_agent_plan_tool_policy(
                    original_message=message,
                    announced_tool_names=["web_search", "route_compare"],
                )

                self.assertEqual(policy.required_initial_tool_counts, {})
                self.assertIsNone(policy.allowed_tool_names)

    def test_current_message_explicit_route_ignores_generic_network_word(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="请联网查询从南景新村到双子塔怎么走，比较驾车和地铁路线。",
            announced_tool_names=["web_search", "route_compare"],
        )

        self.assertEqual(policy.required_initial_tool_counts, {"route_compare": 1})
        self.assertEqual(policy.allowed_tool_names, frozenset({"route_compare"}))
        self.assertEqual(policy.reason, "explicit_route_task")

    def test_followup_route_mode_uses_adjacent_structured_route_result(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="那公交和驾车哪个更合适？",
            announced_tool_names=["web_search", "route_compare"],
            task_context_messages=[
                {"role": "user", "content": "我想从南景新村到双子塔"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "route_results", "schema_version": 1},
                        {"type": "text", "text": "已为你完成路线比较。"},
                    ],
                },
                {"role": "user", "content": "那公交和驾车哪个更合适？"},
            ],
        )

        self.assertEqual(policy.required_initial_tool_counts, {})
        self.assertEqual(policy.allowed_tool_names, frozenset({"route_compare"}))
        self.assertEqual(policy.reason, "adjacent_route_followup")

    def test_elliptical_followup_reads_real_content_block_message_shape(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="那你更推荐哪个？",
            announced_tool_names=["web_search", "route_compare"],
            task_context_messages=[
                SimpleNamespace(
                    role="user",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text="我住在南景新村，公司在双子塔，请比较驾车、公交和地铁通勤方案。",
                        )
                    ],
                ),
                SimpleNamespace(
                    role="assistant",
                    content=[
                        SimpleNamespace(type="route_results", schema_version=1),
                        SimpleNamespace(type="text", text="已为你完成路线比较。"),
                    ],
                ),
                SimpleNamespace(
                    role="user",
                    content=[SimpleNamespace(type="text", text="那你更推荐哪个？")],
                ),
            ],
        )

        self.assertEqual(policy.required_initial_tool_counts, {})
        self.assertEqual(policy.allowed_tool_names, frozenset({"route_compare"}))
        self.assertEqual(policy.reason, "adjacent_route_followup")

    def test_old_route_result_does_not_leak_across_a_completed_topic_change(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="那地铁和开车哪个更环保？",
            announced_tool_names=["web_search", "route_compare"],
            task_context_messages=[
                {"role": "user", "content": "从南景新村到双子塔怎么走？"},
                {"role": "assistant", "content": [{"type": "route_results", "schema_version": 1}]},
                {"role": "user", "content": "讲讲 Python 3.14"},
                {"role": "assistant", "content": [{"type": "text", "text": "Python 3.14 的主要变化如下。"}]},
                {"role": "user", "content": "那地铁和开车哪个更环保？"},
            ],
        )

        self.assertEqual(policy.required_initial_tool_counts, {})
        self.assertIsNone(policy.allowed_tool_names)

    def test_route_and_weather_request_keeps_both_structured_tools(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="从南景新村到双子塔怎么走？比较地铁和驾车，也看看今天的天气。",
            announced_tool_names=["web_search", "route_compare", "weather_forecast"],
        )

        self.assertEqual(policy.required_initial_tool_counts, {"route_compare": 1})
        self.assertEqual(
            policy.allowed_tool_names,
            frozenset({"route_compare", "weather_forecast"}),
        )

    def test_ambiguous_itinerary_request_is_left_to_model(self):
        policy = resolve_agent_plan_tool_policy(
            original_message="帮我规划广州一日游。",
            announced_tool_names=["web_search", "local_place_search", "route_compare"],
        )

        self.assertEqual(policy.required_initial_tool_counts, {})
        self.assertIsNone(policy.allowed_tool_names)
