import unittest

from app.services.knowledge.chunker import DeterministicKnowledgeChunker
from app.services.knowledge.parser import ParsedSection


class DeterministicKnowledgeChunkerTests(unittest.TestCase):
    def test_chunks_and_ids_are_deterministic_with_source_location(self):
        chunker = DeterministicKnowledgeChunker(chunk_size=100, overlap=20)
        sections = [ParsedSection("第一段。" * 30, page=2, section="page")]

        first = chunker.chunk(sections, document_id="doc-1", index_version="version-1")
        second = chunker.chunk(sections, document_id="doc-1", index_version="version-1")

        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertEqual(first[0].page, 2)
        self.assertEqual(first[0].section, "page")
        self.assertTrue(all(len(chunk.chunk_id) == 64 for chunk in first))
        self.assertEqual([chunk.ordinal for chunk in first], list(range(len(first))))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            DeterministicKnowledgeChunker(chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
