import unittest

from app.schemas.chat import SearchSource
from app.services.source_candidate_ranker import (
    SearchResultForRanking,
    format_source_selection_guidance,
    rank_search_sources,
)


class SourceCandidateRankerTests(unittest.TestCase):
    def test_rank_search_sources_deduplicates_and_recommends_high_value_sources(self):
        search_results = [
            SearchResultForRanking(
                tool_call_id="search-1",
                query="OpenAI GPT-5.6 Sol 发布 2026年6月 新闻",
                sources=[
                    SearchSource(
                        title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                        url="https://openai.com/index/previewing-gpt-5-6-sol?utm_source=feed",
                        description="OpenAI official announcement for GPT-5.6 Sol.",
                    ),
                    SearchSource(
                        title="OpenAI 新闻中心| 最新动态",
                        url="https://openai.com/zh-Hans-CN/news/company-announcements",
                        description="OpenAI company announcements.",
                    ),
                    SearchSource(
                        title="不過，這次GPT-5.6 並未全面開放。應美國政府要求，OpenAI 先以 ...",
                        url="https://threads.com/@kufutw/post/DaJ2_DLD8cS",
                        description="Social repost about GPT-5.6.",
                    ),
                    SearchSource(
                        title="GPT-5.6 Sol 解读视频",
                        url="https://youtube.com/watch?v=abc",
                        description="A video commentary.",
                    ),
                    SearchSource(
                        title="OpenAI 的閃電戰：GPT-5.6 亮相與全棧帝國的終局構想 - Yahoo 財經",
                        url="https://hk.finance.yahoo.com/news/openai-gpt-5-6.html",
                        description="Syndicated finance article.",
                    ),
                ],
            ),
            SearchResultForRanking(
                tool_call_id="search-2",
                query="OpenAI GPT-5.6 Sol 2026年6月 官方公告",
                sources=[
                    SearchSource(
                        title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                        url="https://openai.com/index/previewing-gpt-5-6-sol",
                        description="Duplicate official announcement.",
                    ),
                    SearchSource(
                        title="OpenAI releases powerful new GPT-5.6 model - Axios",
                        url="https://axios.com/2026/06/26/openai-gpt-sol-terra-luna-trump",
                        description="Axios reports on OpenAI GPT-5.6 release restrictions.",
                    ),
                    SearchSource(
                        title="[PDF] GPT-5.6 Preview System Card - Deployment Safety Hub",
                        url="https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf",
                        description="Official system card PDF.",
                    ),
                    SearchSource(
                        title="ChatGPT 5.6 Sol: Release Date, Price & Review | Coursiv Blog",
                        url="https://coursiv.io/blog/chatgpt-5-6-sol",
                        description="SEO blog summary.",
                    ),
                    SearchSource(
                        title="OpenAI limits new AI models to trusted partners request US government",
                        url="https://cnbc.com/2026/06/26/openai-limits-new-ai-models-to-trusted-partners-request-us-government.html",
                        description="Media report about OpenAI restrictions.",
                    ),
                ],
            ),
        ]

        plan = rank_search_sources(search_results, max_recommended=3)

        self.assertEqual(plan.total_source_count, 10)
        self.assertEqual(plan.unique_source_count, 9)
        self.assertEqual(
            [candidate.url for candidate in plan.recommended],
            [
                "https://openai.com/index/previewing-gpt-5-6-sol",
                "https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf",
                "https://axios.com/2026/06/26/openai-gpt-sol-terra-luna-trump",
            ],
        )
        self.assertIn("官方来源", plan.recommended[0].reasons)
        self.assertIn("原文公告", plan.recommended[0].reasons)
        self.assertIn("官方 PDF/技术报告", plan.recommended[1].reasons)
        self.assertIn("权威媒体", plan.recommended[2].reasons)

        by_domain = {candidate.domain: candidate for candidate in plan.candidates}
        self.assertEqual(by_domain["threads.com"].priority, "low")
        self.assertIn("社交/论坛来源默认降权", by_domain["threads.com"].reasons)
        self.assertEqual(by_domain["youtube.com"].priority, "low")
        self.assertIn("视频来源默认降权", by_domain["youtube.com"].reasons)

    def test_format_source_selection_guidance_explains_recommended_reads(self):
        search_results = [
            SearchResultForRanking(
                tool_call_id="search-1",
                query="OpenAI GPT-5.6 Sol 发布 2026年6月 新闻",
                sources=[
                    SearchSource(
                        title="Previewing GPT-5.6 Sol: a next-generation model | OpenAI",
                        url="https://openai.com/index/previewing-gpt-5-6-sol",
                        description="OpenAI official announcement.",
                    ),
                    SearchSource(
                        title="不過，這次GPT-5.6 並未全面開放。應美國政府要求，OpenAI 先以 ...",
                        url="https://threads.com/@kufutw/post/DaJ2_DLD8cS",
                        description="Social repost.",
                    ),
                ],
            )
        ]
        plan = rank_search_sources(search_results, max_recommended=1)

        guidance = format_source_selection_guidance(plan)

        self.assertIn("结构化来源选择建议", guidance)
        self.assertIn("合并候选 2 条，去重后 2 条", guidance)
        self.assertIn("建议优先深读", guidance)
        self.assertIn("Previewing GPT-5.6 Sol", guidance)
        self.assertIn("官方来源", guidance)
        self.assertIn("低优先级候选", guidance)
        self.assertIn("threads.com", guidance)
        self.assertIn("不要为了形式读满所有搜索结果", guidance)
