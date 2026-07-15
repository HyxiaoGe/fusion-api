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

    def test_web_search_tool_description_avoids_duplicate_queries(self):
        from app.ai.tools import build_web_search_tool

        tool = build_web_search_tool()

        description = tool["function"]["description"]
        query_description = tool["function"]["parameters"]["properties"]["query"]["description"]

        self.assertIn("默认只发起 1 次搜索", description)
        self.assertIn("第二个互补搜索", description)
        self.assertIn("官方来源、权威媒体、地区、时间范围", description)
        self.assertIn("第三次搜索只适用于 deep_research", description)
        self.assertIn("同义改写重复搜索", description)
        self.assertIn("不要用中英文翻译或同义改写重复搜索同一意图", query_description)

    def test_web_search_tool_description_guides_autonomous_natural_questions(self):
        from app.ai.tools import build_web_search_tool

        tool = build_web_search_tool()
        description = tool["function"]["description"]

        self.assertIn("即使用户没有说", description)
        self.assertIn("使用方法", description)
        self.assertIn("接入", description)
        self.assertIn("互通", description)
        self.assertIn("微信A2A互通怎么用？", description)
        self.assertIn("纯闲聊", description)
        self.assertIn("1+1", description)

    def test_web_search_tool_description_keeps_stable_product_facts_offline(self):
        from app.ai.tools import build_web_search_tool

        tool = build_web_search_tool()
        description = tool["function"]["description"]

        self.assertIn("稳定背景", description)
        self.assertIn("历史原因", description)
        self.assertIn("iPhone 从 Lightning 换成 USB-C 的核心原因", description)
        self.assertIn("价值、风险和落地建议", description)

    def test_url_read_schema_exposes_optional_reason(self):
        from app.ai.tools import URL_READ_TOOL

        parameters = URL_READ_TOOL["function"]["parameters"]
        properties = parameters["properties"]

        self.assertIn("url", properties)
        self.assertIn("reason", properties)
        self.assertEqual(parameters["required"], ["url"])
        self.assertFalse(parameters["additionalProperties"])

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
