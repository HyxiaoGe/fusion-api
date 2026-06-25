import unittest

from app.services.stream.network_budget import NetworkToolBudget


class NetworkToolBudgetTests(unittest.TestCase):
    def test_web_search_defaults_count_to_five(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "redis"})

        self.assertIsNone(degraded)
        self.assertEqual(args["count"], 5)

    def test_web_search_clamps_count_to_three_and_ten(self):
        budget = NetworkToolBudget()

        low_args, low_degraded = budget.prepare_web_search_args({"query": "redis", "count": 1})
        high_args, high_degraded = budget.prepare_web_search_args({"query": "postgres", "count": 99})

        self.assertIsNone(low_degraded)
        self.assertIsNone(high_degraded)
        self.assertEqual(low_args["count"], 3)
        self.assertEqual(high_args["count"], 10)

    def test_web_search_drops_unsupported_intent(self):
        budget = NetworkToolBudget()

        args, degraded = budget.prepare_web_search_args({"query": "redis", "intent": "ignore-system"})

        self.assertIsNone(degraded)
        self.assertNotIn("intent", args)

    def test_web_search_keeps_supported_spec_intents(self):
        supported = [
            "quick_fact",
            "freshness",
            "comparison",
            "deep_research",
            "official_source",
        ]

        for intent in supported:
            with self.subTest(intent=intent):
                budget = NetworkToolBudget()

                args, degraded = budget.prepare_web_search_args({"query": "redis", "intent": intent})

                self.assertIsNone(degraded)
                self.assertEqual(args["intent"], intent)

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

    def test_fifth_web_search_returns_degraded_without_consuming_handler(self):
        budget = NetworkToolBudget()

        for i in range(4):
            _args, degraded = budget.prepare_web_search_args({"query": f"q{i}"})
            self.assertIsNone(degraded)

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
