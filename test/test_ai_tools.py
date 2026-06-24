import unittest


class AiToolSchemaTests(unittest.TestCase):
    def test_web_search_schema_exposes_dynamic_network_options(self):
        from app.ai.tools import build_web_search_tool

        tool = build_web_search_tool()
        properties = tool["function"]["parameters"]["properties"]

        self.assertIn("query", properties)
        self.assertIn("count", properties)
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


if __name__ == "__main__":
    unittest.main()
