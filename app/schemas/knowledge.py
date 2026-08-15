import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

KnowledgeBaseStatus = Literal["active", "deleting", "deleted"]
KnowledgeDocumentStatus = Literal[
    "queued",
    "parsing",
    "chunking",
    "embedding",
    "writing",
    "ready",
    "failed",
    "deleting",
    "deleted",
]
KNOWLEDGE_BASE_NORMALIZED_NAME_MAX_LENGTH = 200
KNOWLEDGE_SEARCH_MAX_BASES_PER_REQUEST = 5


def normalize_knowledge_base_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    return " ".join(normalized.split()).casefold()


def validate_normalized_name_length(name: str) -> str:
    if len(normalize_knowledge_base_name(name)) > KNOWLEDGE_BASE_NORMALIZED_NAME_MAX_LENGTH:
        raise ValueError("知识库名称规范化后不能超过 200 个字符")
    return name


def reject_nul_character(value: str) -> str:
    if isinstance(value, str) and "\x00" in value:
        raise ValueError("字段不能包含 NUL 字符")
    return value


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    business_type: str = Field(default="general", min_length=1, max_length=60, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("name", "description", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = reject_nul_character(value)
        return value.strip() if isinstance(value, str) else value

    @field_validator("business_type", mode="before")
    @classmethod
    def reject_nul_business_type(cls, value: str) -> str:
        return reject_nul_character(value)

    _validate_normalized_name_length = field_validator("name")(validate_normalized_name_length)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    business_type: str | None = Field(default=None, min_length=1, max_length=60, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("name", "description", mode="before")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            raise ValueError("字段不能为 null")
        value = reject_nul_character(value)
        return value.strip() if isinstance(value, str) else value

    @field_validator("business_type", mode="before")
    @classmethod
    def reject_null_business_type(cls, value: str | None) -> str | None:
        if value is None:
            raise ValueError("字段不能为 null")
        return reject_nul_character(value)

    _validate_normalized_name_length = field_validator("name")(validate_normalized_name_length)


class KnowledgeDocumentStats(BaseModel):
    total: int
    ready: int
    processing: int
    failed: int


class KnowledgeBaseView(BaseModel):
    id: str
    name: str
    description: str
    business_type: str
    status: KnowledgeBaseStatus
    document_stats: KnowledgeDocumentStats
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    distance_metric: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class KnowledgeBasePage(BaseModel):
    items: list[KnowledgeBaseView]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class KnowledgeDocumentView(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    mimetype: str
    size: int
    checksum_sha256: str
    status: KnowledgeDocumentStatus
    parser_version: str
    chunker_version: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    distance_metric: str
    desired_index_version: str
    active_index_version: str | None
    chunk_count: int
    error_code: str | None
    error_summary: str | None
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    ready_at: datetime | None
    deleted_at: datetime | None


class KnowledgeDocumentPage(BaseModel):
    items: list[KnowledgeDocumentView]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class KnowledgeChunkView(BaseModel):
    chunk_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page: int | None
    section: str | None


class KnowledgeChunkPage(BaseModel):
    document_id: str
    active_index_version: str
    chunker_version: str
    chunk_size: int
    chunk_overlap: int
    items: list[KnowledgeChunkView]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class KnowledgeTaskView(BaseModel):
    id: str
    task_type: str
    status: str
    phase: str
    attempt_count: int
    max_attempts: int
    error_code: str | None
    error_summary: str | None
    created_at: datetime
    updated_at: datetime


class KnowledgeDocumentUploadResult(BaseModel):
    document: KnowledgeDocumentView
    task: KnowledgeTaskView


class KnowledgeBaseEnvelope(BaseModel):
    knowledge_base: KnowledgeBaseView


class KnowledgeDocumentEnvelope(BaseModel):
    document: KnowledgeDocumentView


class KnowledgeTaskEnvelope(BaseModel):
    task: KnowledgeTaskView


class KnowledgeRetrievalRequest(BaseModel):
    knowledge_base_ids: list[str] = Field(
        min_length=1,
        max_length=KNOWLEDGE_SEARCH_MAX_BASES_PER_REQUEST,
    )
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = reject_nul_character(value)
        return value.strip() if isinstance(value, str) else value

    @field_validator("knowledge_base_ids")
    @classmethod
    def unique_knowledge_base_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            reject_nul_character(value)
        if len(set(values)) != len(values):
            raise ValueError("knowledge_base_ids 不能重复")
        return values


class KnowledgeRetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    knowledge_base_id: str
    knowledge_base_name: str
    index_version: str
    ordinal: int
    text: str
    similarity: float
    filename: str
    source: dict[str, int | str | None]


class KnowledgeRetrievalResult(BaseModel):
    hits: list[KnowledgeRetrievalHit]
    query: str
    top_k: int
