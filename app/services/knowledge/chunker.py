from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.services.knowledge.parser import ParsedSection


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page: int | None
    section: str | None


class DeterministicKnowledgeChunker:
    """固定字符窗切片；相同文档、版本与参数总是产生相同 ID。"""

    VERSION = "chunker-v1"

    def __init__(self, *, chunk_size: int, overlap: int):
        if chunk_size < 100 or overlap < 0 or overlap * 2 > chunk_size or chunk_size - overlap < 100:
            raise ValueError("知识库切片参数无效")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        sections: list[ParsedSection],
        *,
        document_id: str,
        index_version: str,
    ) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        global_offset = 0
        for section in sections:
            start = 0
            while start < len(section.text):
                hard_end = min(start + self.chunk_size, len(section.text))
                end = self._prefer_boundary(section.text, start, hard_end)
                text = section.text[start:end].strip()
                if text:
                    ordinal = len(chunks)
                    chunk_id = self._chunk_id(document_id, index_version, ordinal, text)
                    left_trim = len(section.text[start:end]) - len(section.text[start:end].lstrip())
                    right_trimmed_length = len(section.text[start:end].rstrip())
                    char_start = global_offset + start + left_trim
                    char_end = global_offset + start + right_trimmed_length
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=chunk_id,
                            ordinal=ordinal,
                            text=text,
                            char_start=char_start,
                            char_end=char_end,
                            page=section.page,
                            section=section.section,
                        )
                    )
                if end >= len(section.text):
                    break
                next_start = end - self.overlap
                start = max(start + 1, next_start)
            global_offset += len(section.text) + 2
        if not chunks:
            raise ValueError("文档未产生可索引切片")
        return chunks

    @staticmethod
    def _prefer_boundary(text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)
        minimum = start + max(1, (hard_end - start) // 2)
        for marker in ("\n\n", "\n", "。", ". ", "；", "; ", " "):
            position = text.rfind(marker, minimum, hard_end)
            if position >= minimum:
                return position + len(marker)
        return hard_end

    @staticmethod
    def _chunk_id(document_id: str, index_version: str, ordinal: int, text: str) -> str:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = f"{document_id}:{index_version}:{ordinal}:{content_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
