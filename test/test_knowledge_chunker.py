import unittest
from unittest.mock import patch

from app.services.knowledge.chunker import (
    DeterministicKnowledgeChunker,
    KnowledgeChunkLimitExceeded,
)
from app.services.knowledge.parser import ParsedSection


class DeterministicKnowledgeChunkerTests(unittest.TestCase):
    def test_chunks_and_ids_are_deterministic_with_source_location(self):
        chunker = DeterministicKnowledgeChunker(chunk_size=200, overlap=20)
        sections = [ParsedSection("第一段。" * 80, page=2, section="page")]

        first = chunker.chunk(sections, document_id="doc-1", index_version="version-1")
        second = chunker.chunk(sections, document_id="doc-1", index_version="version-1")

        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertEqual(first[0].page, 2)
        self.assertEqual(first[0].section, "page")
        self.assertTrue(all(len(chunk.chunk_id) == 64 for chunk in first))
        self.assertEqual([chunk.ordinal for chunk in first], list(range(len(first))))

    def test_invalid_overlap_is_rejected(self):
        for chunk_size, overlap in ((100, 100), (1000, 501), (150, 60)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap), self.assertRaises(ValueError):
                DeterministicKnowledgeChunker(chunk_size=chunk_size, overlap=overlap)

    def test_pathological_boundary_keeps_safe_progress_for_near_two_million_characters(self):
        chunker = DeterministicKnowledgeChunker(chunk_size=1200, overlap=500)
        text = "x" * 1_999_999

        with patch.object(
            chunker,
            "_prefer_boundary",
            side_effect=lambda value, start, hard_end: (
                len(value) if hard_end >= len(value) else start + chunker.overlap + 1
            ),
        ):
            chunks = chunker.chunk(
                [ParsedSection(text, page=None, section="large")],
                document_id="doc-large",
                index_version="version-large",
                max_chunks=10_000,
            )

        starts = [chunk.char_start for chunk in chunks]
        self.assertTrue(
            all(current - previous >= chunker.minimum_start_advance for previous, current in zip(starts, starts[1:]))
        )
        self.assertLess(len(chunks), 10_000)

    def test_chunk_limit_stops_before_unbounded_manifest_allocation(self):
        chunker = DeterministicKnowledgeChunker(chunk_size=100, overlap=0)

        with self.assertRaises(KnowledgeChunkLimitExceeded) as raised:
            chunker.chunk(
                [ParsedSection("x" * 501, page=None, section=None)],
                document_id="doc-limit",
                index_version="version-limit",
                max_chunks=5,
            )

        self.assertEqual(raised.exception.max_chunks, 5)


if __name__ == "__main__":
    unittest.main()
