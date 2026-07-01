import unittest

from app.services.stream.network_budget import NetworkToolBudget


class NetworkToolBudgetTests(unittest.TestCase):
    def test_web_search_uses_standard_budget_when_intent_missing(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "redis"})

        self.assertIsNone(degraded)
        self.assertEqual(args["count"], 5)
        self.assertEqual(args["context_source_limit"], 5)
        self.assertEqual(args["search_budget"], "standard")

    def test_web_search_infers_official_source_budget_from_query(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "OpenAI GPT-5.6 Sol 2026年6月 官方公告"})

        self.assertIsNone(degraded)
        self.assertEqual(args["intent"], "official_source")
        self.assertEqual(args["count"], 5)
        self.assertEqual(args["context_source_limit"], 4)
        self.assertEqual(args["search_budget"], "official_source")

    def test_initial_search_records_budget_decision(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})

        self.assertIsNone(degraded)
        self.assertEqual(args["budget_decision"]["query"], "OpenAI 最新公告 2026年7月")
        self.assertEqual(args["budget_decision"]["action"], "execute")
        self.assertEqual(args["budget_decision"]["reason_code"], "initial_search")
        self.assertEqual(args["budget_decision"]["budget_name"], args["search_budget"])
        self.assertEqual(args["budget_decision"]["requested_count"], args["count"])
        self.assertEqual(args["budget_decision"]["context_source_limit"], args["context_source_limit"])
        self.assertEqual(args["budget_decision"]["previous_query_count"], 0)
        self.assertEqual(args["budget_decision"]["planned_search_limit"], 2)

    def test_similar_followup_records_narrow_followup_decision(self):
        budget = NetworkToolBudget()

        _first_args, first_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol official announcement June 2026"}
        )
        second_args, second_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol 2026年6月 官方公告"}
        )

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertEqual(second_args["search_budget"], "official_source_followup")
        self.assertEqual(second_args["budget_decision"]["action"], "narrow_followup")
        self.assertEqual(second_args["budget_decision"]["reason_code"], "similar_followup")
        self.assertEqual(second_args["budget_decision"]["budget_name"], "official_source_followup")
        self.assertEqual(second_args["budget_decision"]["previous_query_count"], 1)
        self.assertEqual(second_args["count"], 3)

    def test_duplicate_search_records_skip_duplicate_decision(self):
        budget = NetworkToolBudget()

        _first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})

        self.assertIsNone(first_degraded)
        self.assertIsNotNone(second_degraded)
        self.assertEqual(second_args["budget_decision"]["action"], "skip_duplicate")
        self.assertEqual(second_args["budget_decision"]["reason_code"], "duplicate_query")
        self.assertEqual(second_degraded.data["budget_decision"]["action"], "skip_duplicate")
        self.assertEqual(second_degraded.data["budget_decision"]["reason_code"], "duplicate_query")
        self.assertEqual(second_args["count"], 0)
        self.assertEqual(budget.web_search_calls, 1)

    def test_planner_limited_search_records_limit_decision(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年产品更新 最新发布"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年最新新闻 媒体报道"})
        third_args, third_degraded = budget.prepare_web_search_args({"query": "OpenAI GPT-5.6 Sol 预览 2026年7月"})

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertIsNotNone(third_degraded)
        self.assertEqual(third_args["budget_decision"]["action"], "limit_planner")
        self.assertEqual(third_args["budget_decision"]["reason_code"], "planned_search_limit_reached")
        self.assertEqual(third_args["budget_decision"]["previous_query_count"], 2)
        self.assertEqual(third_degraded.data["budget_decision"]["action"], "limit_planner")
        self.assertEqual(third_degraded.data["budget_decision"]["reason_code"], "planned_search_limit_reached")
        self.assertEqual(third_args["count"], 0)
        self.assertEqual(budget.web_search_calls, 2)

    def test_chinese_year_query_infers_freshness_intent(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "SpaceX 估值 上市 2026年"})

        self.assertIsNone(degraded)
        self.assertEqual(args["intent"], "freshness")
        self.assertEqual(args["count"], 5)
        self.assertEqual(args["search_budget"], "freshness")

    def test_second_similar_chinese_year_query_uses_followup_budget(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args({"query": "SpaceX 估值 上市 2026年"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "SpaceX IPO 估值 2026 最新"})

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertEqual(first_args["search_budget"], "freshness")
        self.assertEqual(second_args["search_budget"], "freshness_followup")
        self.assertEqual(second_args["count"], 3)
        self.assertEqual(second_args["context_source_limit"], 3)

    def test_duplicate_web_search_returns_degraded_without_consuming_provider_budget(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年6月 新闻"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年6月 新闻"})

        self.assertIsNone(first_degraded)
        self.assertEqual(first_args["search_budget"], "official_source")
        self.assertIsNotNone(second_degraded)
        self.assertEqual(second_degraded.status, "degraded")
        self.assertTrue(second_degraded.data["duplicate_search_skipped"])
        self.assertEqual(second_args["search_budget"], "duplicate_skipped")
        self.assertEqual(second_args["count"], 0)
        self.assertEqual(budget.web_search_calls, 1)

    def test_web_search_narrows_similar_followup_query(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol official announcement June 2026"}
        )
        second_args, second_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol 2026年6月 官方公告"}
        )

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertEqual(first_args["search_budget"], "official_source")
        self.assertEqual(first_args["count"], 5)
        self.assertEqual(first_args["context_source_limit"], 4)
        self.assertEqual(second_args["intent"], "official_source")
        self.assertEqual(second_args["search_budget"], "official_source_followup")
        self.assertEqual(second_args["count"], 3)
        self.assertEqual(second_args["context_source_limit"], 3)

    def test_web_search_keeps_complementary_media_followup_broad(self):
        budget = NetworkToolBudget()

        official_args, official_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol official announcement June 2026"}
        )
        media_args, media_degraded = budget.prepare_web_search_args(
            {"query": "OpenAI GPT-5.6 Sol TechCrunch Reuters 权威媒体报道"}
        )

        self.assertIsNone(official_degraded)
        self.assertIsNone(media_degraded)
        self.assertEqual(official_args["search_budget"], "official_source")
        self.assertEqual(media_args["intent"], "comparison")
        self.assertEqual(media_args["search_budget"], "comparison")
        self.assertEqual(media_args["count"], 8)
        self.assertEqual(media_args["context_source_limit"], 6)

    def test_third_non_deep_search_returns_plan_limited_without_provider_call(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年产品更新 最新发布"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年最新新闻 媒体报道"})
        third_args, third_degraded = budget.prepare_web_search_args({"query": "OpenAI GPT-5.6 Sol 预览 2026年6月"})

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertEqual(first_args["search_budget"], "official_source")
        self.assertEqual(second_args["search_budget"], "comparison")
        self.assertIsNotNone(third_degraded)
        self.assertEqual(third_degraded.status, "degraded")
        self.assertTrue(third_degraded.data["search_plan_limited"])
        self.assertFalse(third_degraded.data["budget_limited"])
        self.assertEqual(third_args["search_budget"], "planner_limited")
        self.assertEqual(third_args["count"], 0)
        self.assertEqual(third_args["context_source_limit"], 0)
        self.assertEqual(budget.web_search_calls, 2)
        self.assertEqual(
            budget.web_search_queries,
            ["OpenAI 2026年产品更新 最新发布", "OpenAI 2026年最新新闻 媒体报道"],
        )

    def test_deep_research_allows_third_effective_search(self):
        budget = NetworkToolBudget()

        first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年 深入调研 技术报告"})
        second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年 官方公告"})
        third_args, third_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026年 权威媒体报道"})
        fourth_args, fourth_degraded = budget.prepare_web_search_args({"query": "OpenAI GPT-5.6 Sol 预览 2026年6月"})

        self.assertIsNone(first_degraded)
        self.assertIsNone(second_degraded)
        self.assertIsNone(third_degraded)
        self.assertEqual(first_args["search_budget"], "deep_research")
        self.assertEqual(third_args["search_budget"], "comparison")
        self.assertIsNotNone(fourth_degraded)
        self.assertTrue(fourth_degraded.data["search_plan_limited"])
        self.assertEqual(fourth_args["search_budget"], "planner_limited")
        self.assertEqual(budget.web_search_calls, 3)

    def test_web_search_ignores_model_supplied_count(self):
        budget = NetworkToolBudget()

        low_args, low_degraded = budget.prepare_web_search_args({"query": "redis", "count": 1})
        high_args, high_degraded = budget.prepare_web_search_args({"query": "postgres", "count": 99})

        self.assertIsNone(low_degraded)
        self.assertIsNone(high_degraded)
        self.assertEqual(low_args["count"], 5)
        self.assertEqual(high_args["count"], 5)
        self.assertEqual(low_args["search_budget"], "standard")
        self.assertEqual(high_args["search_budget"], "standard")

    def test_web_search_maps_supported_intents_to_search_budgets(self):
        expected = {
            "quick_fact": ("quick_fact", 3, 3),
            "freshness": ("freshness", 5, 5),
            "comparison": ("comparison", 8, 6),
            "deep_research": ("deep_research", 10, 8),
            "official_source": ("official_source", 5, 4),
        }

        for intent, (budget_name, requested_count, context_limit) in expected.items():
            with self.subTest(intent=intent):
                budget = NetworkToolBudget()

                args, degraded = budget.prepare_web_search_args({"query": "redis", "intent": intent, "count": 99})

                self.assertIsNone(degraded)
                self.assertEqual(args["intent"], intent)
                self.assertEqual(args["search_budget"], budget_name)
                self.assertEqual(args["count"], requested_count)
                self.assertEqual(args["context_source_limit"], context_limit)

    def test_web_search_drops_unsupported_intent(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "redis", "intent": "ignore-system", "count": 10})

        self.assertIsNone(degraded)
        self.assertNotIn("intent", args)
        self.assertEqual(args["count"], 5)
        self.assertEqual(args["search_budget"], "standard")

    def test_web_search_keeps_at_most_five_plain_domains(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args(
            {
                "query": "redis",
                "domains": [
                    "https://Redis.io/docs",
                    "docs.python.org",
                    "bad domain",
                    "openai.com/path?q=1",
                    "api.example.com:443",
                    "*.example.com",
                    "example",
                    "sub.example.org",
                    "localhost",
                    "ietf.org",
                    "github.com",
                    "www.python.org",
                    "mozilla.org",
                ],
            }
        )

        self.assertIsNone(degraded)
        self.assertEqual(
            args["domains"],
            ["docs.python.org", "sub.example.org", "ietf.org", "github.com", "python.org"],
        )

    def test_web_search_clamps_recency_days(self):
        budget = NetworkToolBudget()

        low_args, low_degraded = budget.prepare_web_search_args({"query": "redis", "recency_days": 0})
        high_args, high_degraded = budget.prepare_web_search_args({"query": "postgres", "recency_days": 999})

        self.assertIsNone(low_degraded)
        self.assertIsNone(high_degraded)
        self.assertEqual(low_args["recency_days"], 1)
        self.assertEqual(high_args["recency_days"], 365)

    def test_web_search_hard_cap_returns_degraded_without_consuming_handler(self):
        budget = NetworkToolBudget(web_search_calls=4)

        args, degraded = budget.prepare_web_search_args({"query": "q4", "count": 8})

        self.assertEqual(args["query"], "q4")
        self.assertIsNotNone(degraded)
        self.assertEqual(degraded.status, "degraded")
        self.assertTrue(degraded.data["budget_limited"])

    def test_sixth_url_read_returns_degraded(self):
        budget = NetworkToolBudget()

        for i in range(5):
            _args, degraded = budget.prepare_url_read_args({"url": f"https://example.com/{i}"})
            self.assertIsNone(degraded)

        args, degraded = budget.prepare_url_read_args({"url": "https://example.com/5"})

        self.assertEqual(args["url"], "https://example.com/5")
        self.assertIsNotNone(degraded)
        self.assertEqual(degraded.status, "degraded")
        self.assertTrue(degraded.data["budget_limited"])


if __name__ == "__main__":
    unittest.main()
