import unittest

from app.schemas.chat import SearchBlock, SearchSourceSummary, SourceReference, TextBlock, UrlBlock
from app.services.source_evidence_ledger import stable_web_evidence_id


def search_ref(title: str, url: str) -> SourceReference:
    return SourceReference(kind="search", title=title, url=url)


def read_ref(title: str, url: str) -> SourceReference:
    return SourceReference(kind="url_read", title=title, url=url)


def search_block(refs: list[SourceReference]) -> SearchBlock:
    return SearchBlock(
        type="search",
        id="blk-search",
        query="OpenAI 产品更新",
        sources=[SearchSourceSummary(title=ref.title, url=ref.url) for ref in refs],
        source_refs=refs,
        source_count=len(refs),
    )


class FinalAnswerEvidenceTests(unittest.TestCase):
    def test_marks_numbered_markdown_citation_as_used(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告", "https://openai.com/news/product"),
            search_ref("媒体报道", "https://example.com/media"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs), TextBlock(type="text", id="text-1", text="答案引用官方公告。[1]")],
            answer_text="答案引用官方公告。[1]",
        )

        self.assertEqual(
            [item["id"] for item in used],
            [stable_web_evidence_id("https://openai.com/news/product", fallback="unused")],
        )
        self.assertEqual(used[0]["status"], "used")
        self.assertTrue(used[0]["used_by_final_answer"])

    def test_marks_citation_placeholder_as_used(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告", "https://openai.com/news/product"),
            search_ref("媒体报道", "https://example.com/media"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs)],
            answer_text="媒体报道提到监管要求。⟦2⟧",
        )

        self.assertEqual([item["url"] for item in used], ["https://example.com/media"])

    def test_maps_run_level_citation_to_later_search_block(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        first_refs = [search_ref(f"第一轮来源 {index}", f"https://first.example.com/{index}") for index in range(1, 6)]
        second_refs = [
            search_ref(f"第二轮来源 {index}", f"https://second.example.com/{index}") for index in range(1, 6)
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(first_refs), search_block(second_refs)],
            answer_text="北京周末天气以第二轮第 4 条为准。[9]",
        )

        self.assertEqual([item["url"] for item in used], ["https://second.example.com/4"])

    def test_marks_exact_url_mention_as_used(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告", "https://openai.com/news/product?utm_source=x"),
            search_ref("媒体报道", "https://example.com/media"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs)],
            answer_text="详见 https://openai.com/news/product 的官方公告。",
        )

        self.assertEqual([item["url"] for item in used], ["https://openai.com/news/product"])

    def test_marks_unique_domain_mention_as_used(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告", "https://openai.com/news/product"),
            search_ref("媒体报道", "https://example.com/media"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs)],
            answer_text="这条结论以 openai.com 官方页面为准。",
        )

        self.assertEqual([item["url"] for item in used], ["https://openai.com/news/product"])

    def test_does_not_mark_ambiguous_domain_mention(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告 A", "https://openai.com/news/a"),
            search_ref("官方公告 B", "https://openai.com/news/b"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs)],
            answer_text="OpenAI 官方页面 openai.com 给出了更新。",
        )

        self.assertEqual(used, [])

    def test_single_successful_url_read_falls_back_to_used(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        block = UrlBlock(
            type="url_read",
            id="url-1",
            url="https://openai.com/news/product",
            title="官方公告",
            source_refs=[read_ref("官方公告", "https://openai.com/news/product")],
        )

        used = build_used_final_answer_evidence(
            content_blocks=[block],
            answer_text="OpenAI 近期发布了产品更新。",
        )

        self.assertEqual([item["url"] for item in used], ["https://openai.com/news/product"])

    def test_no_match_with_multiple_sources_returns_empty(self):
        from app.services.final_answer_evidence import build_used_final_answer_evidence

        refs = [
            search_ref("官方公告", "https://openai.com/news/product"),
            search_ref("媒体报道", "https://example.com/media"),
        ]

        used = build_used_final_answer_evidence(
            content_blocks=[search_block(refs)],
            answer_text="OpenAI 近期发布了产品更新。",
        )

        self.assertEqual(used, [])


if __name__ == "__main__":
    unittest.main()
