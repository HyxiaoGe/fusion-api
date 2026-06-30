import unittest


class AiToolSchemaTests(unittest.TestCase):
    def test_web_search_schema_exposes_decision_options_but_not_count(self):
        from app.ai.tools import build_web_search_tool

        tool = build_web_search_tool()
        properties = tool["function"]["parameters"]["properties"]

        self.assertIn("query", properties)
        self.assertNotIn("count", properties)
        self.assertIn("intent", properties)
        self.assertIn("domains", properties)
        self.assertIn("recency_days", properties)
        self.assertEqual(
            properties["intent"]["enum"],
            [
                "quick_fact",
                "freshness",
                "comparison",
                "deep_research",
                "official_source",
            ],
        )
        self.assertEqual(tool["function"]["parameters"]["required"], ["query"])

    def test_url_read_schema_exposes_optional_reason(self):
        from app.ai.tools import URL_READ_TOOL

        properties = URL_READ_TOOL["function"]["parameters"]["properties"]

        self.assertIn("url", properties)
        self.assertIn("reason", properties)
        self.assertEqual(URL_READ_TOOL["function"]["parameters"]["required"], ["url"])

    def test_url_read_schema_prioritizes_high_value_search_results(self):
        from app.ai.tools import URL_READ_TOOL

        description = URL_READ_TOOL["function"]["description"]

        self.assertIn("官方来源", description)
        self.assertIn("原文公告", description)
        self.assertIn("高相关结果", description)
        self.assertIn("视频", description)
        self.assertIn("论坛", description)
        self.assertIn("低相关结果", description)
        self.assertIn("降权", description)


if __name__ == "__main__":
    unittest.main()
