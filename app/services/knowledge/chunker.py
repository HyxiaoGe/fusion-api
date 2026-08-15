from __future__ import annotations

import hashlib
from collections.abc import Iterator
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


class KnowledgeChunkLimitExceeded(ValueError):
    """文档切片数量超过 Worker 的有界处理上限。"""

    def __init__(self, max_chunks: int):
        self.max_chunks = max_chunks
        super().__init__(f"文档切片数量超过上限 {max_chunks}")


class DeterministicKnowledgeChunker:
    """固定字符窗切片；相同文档、版本与参数总是产生相同 ID。"""

    VERSION = "chunker-v2"

    def __init__(self, *, chunk_size: int, overlap: int):
        if chunk_size < 100 or overlap < 0 or overlap * 2 > chunk_size or chunk_size - overlap < 100:
            raise ValueError("知识库切片参数无效")
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 边界偏好只能在安全窗口内回退，避免病态标点令 start 每轮仅推进一个字符。
        self.minimum_start_advance = max(100, (chunk_size - overlap) // 2)

    def chunk(
        self,
        sections: list[ParsedSection],
        *,
        document_id: str,
        index_version: str,
        max_chunks: int | None = None,
    ) -> list[KnowledgeChunk]:
        return list(
            self.iter_chunks(
                sections,
                document_id=document_id,
                index_version=index_version,
                max_chunks=max_chunks,
            )
        )

    def iter_chunks(
        self,
        sections: list[ParsedSection],
        *,
        document_id: str,
        index_version: str,
        max_chunks: int | None = None,
    ) -> Iterator[KnowledgeChunk]:
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("文档切片数量上限必须大于零")
        ordinal = 0
        global_offset = 0
        for section in sections:
            start = 0
            while start < len(section.text):
                hard_end = min(start + self.chunk_size, len(section.text))
                end = self._prefer_boundary(section.text, start, hard_end)
                if hard_end < len(section.text):
                    minimum_end = start + self.overlap + self.minimum_start_advance
                    end = max(end, min(minimum_end, hard_end))
                text = section.text[start:end].strip()
                if text:
                    if max_chunks is not None and ordinal >= max_chunks:
                        raise KnowledgeChunkLimitExceeded(max_chunks)
                    chunk_id = self._chunk_id(document_id, index_version, ordinal, text)
                    left_trim = len(section.text[start:end]) - len(section.text[start:end].lstrip())
                    right_trimmed_length = len(section.text[start:end].rstrip())
                    char_start = global_offset + start + left_trim
                    char_end = global_offset + start + right_trimmed_length
                    yield KnowledgeChunk(
                        chunk_id=chunk_id,
                        ordinal=ordinal,
                        text=text,
                        char_start=char_start,
                        char_end=char_end,
                        page=section.page,
                        section=section.section,
                    )
                    ordinal += 1
                if end >= len(section.text):
                    break
                next_start = end - self.overlap
                start = max(start + 1, next_start)
            global_offset += len(section.text) + 2
        if ordinal == 0:
            raise ValueError("文档未产生可索引切片")

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


class LegacyKnowledgeChunkerV1(DeterministicKnowledgeChunker):
    """升级期间继续处理已持久化的 v1 在途索引版本。"""

    VERSION = "chunker-v1"

    def __init__(self, *, chunk_size: int, overlap: int):
        if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("知识库 v1 切片参数无效")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        sections: list[ParsedSection],
        *,
        document_id: str,
        index_version: str,
        max_chunks: int | None = None,
    ) -> list[KnowledgeChunk]:
        return list(
            self.iter_chunks(
                sections,
                document_id=document_id,
                index_version=index_version,
                max_chunks=max_chunks,
            )
        )

    def iter_chunks(
        self,
        sections: list[ParsedSection],
        *,
        document_id: str,
        index_version: str,
        max_chunks: int | None = None,
    ) -> Iterator[KnowledgeChunk]:
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("文档切片数量上限必须大于零")
        ordinal = 0
        global_offset = 0
        for section in sections:
            start = 0
            while start < len(section.text):
                hard_end = min(start + self.chunk_size, len(section.text))
                end = self._prefer_boundary(section.text, start, hard_end)
                text = section.text[start:end].strip()
                if text:
                    if max_chunks is not None and ordinal >= max_chunks:
                        raise KnowledgeChunkLimitExceeded(max_chunks)
                    chunk_id = self._chunk_id(document_id, index_version, ordinal, text)
                    left_trim = len(section.text[start:end]) - len(section.text[start:end].lstrip())
                    right_trimmed_length = len(section.text[start:end].rstrip())
                    yield KnowledgeChunk(
                        chunk_id=chunk_id,
                        ordinal=ordinal,
                        text=text,
                        char_start=global_offset + start + left_trim,
                        char_end=global_offset + start + right_trimmed_length,
                        page=section.page,
                        section=section.section,
                    )
                    ordinal += 1
                if end >= len(section.text):
                    break
                start = max(start + 1, end - self.overlap)
            global_offset += len(section.text) + 2
        if ordinal == 0:
            raise ValueError("文档未产生可索引切片")


def knowledge_chunker_for_version(version: str, *, chunk_size: int, overlap: int):
    chunker_types = {
        LegacyKnowledgeChunkerV1.VERSION: LegacyKnowledgeChunkerV1,
        DeterministicKnowledgeChunker.VERSION: DeterministicKnowledgeChunker,
    }
    chunker_type = chunker_types.get(version)
    if chunker_type is None:
        raise ValueError("Worker 不支持该切片版本")
    return chunker_type(chunk_size=chunk_size, overlap=overlap)
