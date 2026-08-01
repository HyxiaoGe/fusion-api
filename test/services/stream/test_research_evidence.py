import unittest

from app.schemas.chat import SearchBlock, SearchSourceSummary, SourceReference, UrlBlock
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.stream.research_evidence import (
    ResearchEvidenceWorkset,
    assign_missing_source_reference_metadata,
    build_research_untrusted_context_messages,
    build_research_workset_prompt,
    resolve_deep_research_stage,
    validate_research_completion,
)


def _search_block(*, citation_index: int = 1) -> SearchBlock:
    return SearchBlock(
        type="search",
        query="官方资料",
        sources=[SearchSourceSummary(title="官方资料", url="https://example.com/report")],
        source_refs=[
            SourceReference(
                kind="search",
                title="官方资料",
                url="https://example.com/report",
                evidence_id="ev-web-report",
                citation_index=citation_index,
            )
        ],
        source_count=1,
    )


def _read_block(url: str, *, evidence_id: str, citation_index: int) -> UrlBlock:
    return UrlBlock(
        type="url_read",
        url=url,
        title=f"来源 {citation_index}",
        status="success",
        source_count=1,
        source_refs=[
            SourceReference(
                kind="url_read",
                title=f"来源 {citation_index}",
                url=url,
                evidence_id=evidence_id,
                citation_index=citation_index,
            )
        ],
    )


class ResearchEvidenceTests(unittest.TestCase):
    def test_preprocessed_url_block_receives_source_metadata_for_workset_restore(self):
        block = UrlBlock(
            type="url_read",
            url="https://example.com/preprocessed",
            title="预读取来源",
        )

        assign_missing_source_reference_metadata([block])
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks([block])

        self.assertEqual(block.source_count, 1)
        self.assertEqual(len(block.source_refs), 1)
        self.assertEqual(block.source_refs[0]["citation_index"], 1)
        self.assertEqual(workset.successful_read_urls, {"https://example.com/preprocessed"})

    def test_restored_source_without_summary_exposes_url_only_in_untrusted_context(self):
        block = _read_block(
            "https://example.com/retry-source",
            evidence_id="ev-retry",
            citation_index=19,
        )
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks([block], allow_read_success=False)

        control = build_research_workset_prompt(workset)
        contexts = build_research_untrusted_context_messages(workset)

        self.assertIn("[19] evidence_id=ev-retry status=candidate", control)
        self.assertNotIn("example.com", control)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["role"], "user")
        self.assertIn('source_url="https://example.com/retry-source"', contexts[0]["content"])
        self.assertIn("仅恢复来源身份，正文未持久化，引用前必须重新读取", contexts[0]["content"])
        self.assertEqual(workset.attempted_read_urls, set())
        self.assertEqual(workset.unread_candidate_urls, {"https://example.com/retry-source"})

    def test_unsuccessful_read_exhausts_candidate_and_enters_search_repair(self):
        for status in ("failed", "degraded"):
            with self.subTest(status=status):
                workset = ResearchEvidenceWorkset()
                workset.record_content_blocks(
                    [
                        _search_block(),
                        UrlBlock(
                            type="url_read",
                            url="https://example.com/report",
                            title="读取未成功",
                            status=status,
                            source_count=0,
                            source_refs=[],
                            error_message="无法读取",
                        ),
                    ]
                )

                self.assertEqual(workset.successful_read_urls, set())
                self.assertEqual(workset.attempted_read_urls, {"https://example.com/report"})
                self.assertEqual(workset.unread_candidate_urls, set())
                control = build_research_workset_prompt(workset)
                self.assertIn(
                    "[1] evidence_id=ev-web-report status=read_failed retry=forbidden",
                    control,
                )
                self.assertNotIn("example.com", control)
                self.assertEqual(build_research_untrusted_context_messages(workset), [])
                self.assertEqual(
                    resolve_deep_research_stage(workset, has_valid_plan=True),
                    "search_repair",
                )

    def test_source_reference_new_fields_are_optional_for_legacy_blocks(self):
        legacy = SourceReference(kind="search", title="旧来源", url="https://example.com/legacy")

        self.assertIsNone(legacy.evidence_id)
        self.assertIsNone(legacy.citation_index)

    def test_workset_keeps_bounded_safe_projection_and_deduplicates_urls(self):
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                _search_block(),
                _read_block("https://example.com/a?utm_source=test", evidence_id="ev-a", citation_index=2),
                _read_block("https://example.com/a", evidence_id="ev-a", citation_index=2),
            ],
            summaries={
                "ev-web-report": ("官方摘要", ["关键发现一"]),
                "ev-a": ("原文摘要", ["关键发现二"]),
            },
        )

        self.assertEqual(workset.successful_searches, 1)
        self.assertEqual(workset.successful_read_urls, {"https://example.com/a"})
        self.assertEqual(len(workset.sources), 2)
        prompt = build_research_workset_prompt(workset)
        self.assertIn("[1] evidence_id=ev-web-report status=candidate", prompt)
        self.assertIn("[2] evidence_id=ev-a status=read_success", prompt)
        self.assertNotIn("官方资料", prompt)
        self.assertNotIn("example.com", prompt)
        self.assertNotIn("关键发现一", prompt)
        summary_prompt = build_research_workset_prompt(workset, include_candidates=False)
        self.assertNotIn("[1]", summary_prompt)
        self.assertIn("[2] evidence_id=ev-a status=read_success", summary_prompt)

        context_messages = build_research_untrusted_context_messages(workset)
        self.assertTrue(all(message["role"] == "user" for message in context_messages))
        self.assertIn("官方摘要", context_messages[0]["content"])
        self.assertIn("关键发现一", context_messages[0]["content"])
        summary_context_messages = build_research_untrusted_context_messages(
            workset,
            include_candidates=False,
        )
        self.assertEqual(len(summary_context_messages), 1)
        self.assertNotIn("官方摘要", summary_context_messages[0]["content"])

    def test_late_high_index_read_source_is_retained_ahead_of_early_candidates(self):
        candidate_refs = [
            SourceReference(
                kind="search",
                title=f"候选 {index}",
                url=f"https://example.com/{index}",
                evidence_id=f"ev-{index}",
                citation_index=index,
            )
            for index in range(1, 14)
        ]
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                SearchBlock(
                    type="search",
                    query="候选",
                    sources=[SearchSourceSummary(title=ref.title, url=ref.url) for ref in candidate_refs],
                    source_refs=candidate_refs,
                    source_count=len(candidate_refs),
                ),
                _read_block("https://example.com/13", evidence_id="ev-13", citation_index=13),
                _read_block("https://example.com/12", evidence_id="ev-12", citation_index=12),
            ]
        )

        self.assertEqual(len(workset.sources), 12)
        self.assertIn("ev-13", workset.sources)
        self.assertIn(13, workset.valid_citation_indexes)
        self.assertTrue(validate_research_completion(workset, "高编号已读来源仍可引用。[13]").is_valid)

    def test_new_unattempted_candidate_is_retained_ahead_of_twelve_failed_sources(self):
        failed_refs = [
            SourceReference(
                kind="search",
                title=f"失败候选 {index}",
                url=f"https://example.com/failed/{index}",
                evidence_id=f"ev-failed-{index}",
                citation_index=index,
            )
            for index in range(1, 13)
        ]
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                SearchBlock(
                    type="search",
                    query="旧候选",
                    sources=[SearchSourceSummary(title=ref.title, url=ref.url) for ref in failed_refs],
                    source_refs=failed_refs,
                    source_count=len(failed_refs),
                ),
                *[
                    UrlBlock(
                        type="url_read",
                        url=ref.url,
                        title=ref.title,
                        status="failed",
                        source_count=0,
                        source_refs=[],
                        error_message="无法读取",
                    )
                    for ref in failed_refs
                ],
                SearchBlock(
                    type="search",
                    query="新候选",
                    sources=[
                        SearchSourceSummary(
                            title="新候选",
                            url="https://example.com/new",
                        )
                    ],
                    source_refs=[
                        SourceReference(
                            kind="search",
                            title="新候选",
                            url="https://example.com/new",
                            evidence_id="ev-new",
                            citation_index=13,
                        )
                    ],
                    source_count=1,
                ),
            ]
        )

        self.assertEqual(len(workset.sources), 12)
        self.assertIn("ev-new", workset.sources)
        self.assertNotIn("ev-failed-12", workset.sources)
        self.assertEqual(workset.unread_candidate_urls, {"https://example.com/new"})

    def test_completion_requires_search_two_distinct_reads_and_mapped_citation(self):
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                _search_block(),
                _read_block("https://example.com/a", evidence_id="ev-a", citation_index=2),
                _read_block("https://example.com/b", evidence_id="ev-b", citation_index=3),
            ]
        )

        self.assertTrue(validate_research_completion(workset, "结论来自已深读来源。[2]").is_valid)

        unread_search = validate_research_completion(workset, "只引用搜索候选。[1]")
        self.assertFalse(unread_search.is_valid)
        self.assertEqual(unread_search.reason, "invalid_citation")

        invalid = validate_research_completion(workset, "引用了不存在的来源。[9]")
        self.assertFalse(invalid.is_valid)
        self.assertEqual(invalid.reason, "invalid_citation")

    def test_fallback_plan_keeps_full_research_evidence_gate(self):
        coordinator = PlanCoordinator(run_id="run-fallback", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 2,
            }
        )
        self.assertTrue(coordinator.adopt_research_fallback().accepted)
        workset = ResearchEvidenceWorkset()

        self.assertEqual(
            resolve_deep_research_stage(workset, has_valid_plan=coordinator.has_valid_model_plan),
            "search",
        )
        self.assertEqual(validate_research_completion(workset, "无依据结论").reason, "missing_search")

        workset.record_content_blocks([_search_block()])
        self.assertEqual(
            resolve_deep_research_stage(workset, has_valid_plan=coordinator.has_valid_model_plan),
            "read",
        )
        workset.record_content_blocks([_read_block("https://example.com/a", evidence_id="ev-a", citation_index=2)])
        self.assertEqual(validate_research_completion(workset, "只有一条来源。[2]").reason, "insufficient_reads")

        workset.record_content_blocks([_read_block("https://example.com/b", evidence_id="ev-b", citation_index=3)])
        self.assertEqual(
            resolve_deep_research_stage(workset, has_valid_plan=coordinator.has_valid_model_plan),
            "synthesis",
        )
        self.assertTrue(validate_research_completion(workset, "已核验结论。[2]").is_valid)
        self.assertEqual(
            validate_research_completion(workset, "引用不存在来源。[9]").reason,
            "invalid_citation",
        )

    def test_completion_rejects_insufficient_reads(self):
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                _search_block(),
                _read_block("https://example.com/a", evidence_id="ev-a", citation_index=2),
            ]
        )

        result = validate_research_completion(workset, "初步结论。[1]")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "insufficient_reads")

    def test_only_blocks_recorded_by_current_run_count_toward_gate(self):
        workset = ResearchEvidenceWorkset()

        result = validate_research_completion(workset, "沿用旧回答。[1]")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.reason, "missing_search")

    def test_research_stage_does_not_synthesize_while_planned_tool_item_is_unexecuted(self):
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                _search_block(),
                _read_block("https://example.com/a", evidence_id="ev-a", citation_index=2),
                _read_block("https://example.com/b", evidence_id="ev-b", citation_index=3),
            ]
        )

        self.assertEqual(
            resolve_deep_research_stage(
                workset,
                has_valid_plan=True,
                unexecuted_plan_tool_names={"web_search"},
            ),
            "search",
        )
        self.assertEqual(
            resolve_deep_research_stage(
                workset,
                has_valid_plan=True,
                unexecuted_plan_tool_names=set(),
            ),
            "synthesis",
        )

    def test_rejected_non_stage_tool_plan_cannot_enter_synthesis(self):
        workset = ResearchEvidenceWorkset()
        workset.record_content_blocks(
            [
                _search_block(),
                _read_block("https://example.com/a", evidence_id="ev-a", citation_index=2),
                _read_block("https://example.com/b", evidence_id="ev-b", citation_index=3),
            ]
        )

        self.assertEqual(
            resolve_deep_research_stage(
                workset,
                has_valid_plan=False,
                unexecuted_plan_tool_names={"route_compare"},
            ),
            "planning",
        )


if __name__ == "__main__":
    unittest.main()
