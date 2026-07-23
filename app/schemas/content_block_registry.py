"""消息内容块的统一注册与兼容解析入口。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from app.schemas.chat import (
    ContentBlock,
    FileBlock,
    FlightResultsBlock,
    PlaceResultsBlock,
    RouteResultsBlock,
    SearchBlock,
    TextBlock,
    ThinkingBlock,
    TrainResultsBlock,
    UnsupportedContentBlock,
    UrlBlock,
    WeatherResultsBlock,
)

logger = logging.getLogger(__name__)

ContentBlockPayloadNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ContentBlockRegistration:
    """单个内容块解码契约；富结果以 schema_version 区分注册项。"""

    model: type[BaseModel]
    schema_version: int | None = None
    normalize_payload: ContentBlockPayloadNormalizer | None = None


def _normalize_legacy_search(payload: dict[str, Any]) -> dict[str, Any]:
    """补齐最早期 SearchBlock 尚未持久化的必填展示字段。"""
    normalized = dict(payload)
    normalized.setdefault("query", "")
    normalized.setdefault("sources", [])
    return normalized


CONTENT_BLOCK_REGISTRY: dict[tuple[str, int | None], ContentBlockRegistration] = {
    ("text", None): ContentBlockRegistration(TextBlock),
    ("thinking", None): ContentBlockRegistration(ThinkingBlock),
    ("file", None): ContentBlockRegistration(FileBlock),
    ("search", None): ContentBlockRegistration(SearchBlock, normalize_payload=_normalize_legacy_search),
    ("url_read", None): ContentBlockRegistration(UrlBlock),
    ("unsupported_result", None): ContentBlockRegistration(UnsupportedContentBlock),
    ("place_results", 1): ContentBlockRegistration(PlaceResultsBlock, schema_version=1),
    ("route_results", 1): ContentBlockRegistration(RouteResultsBlock, schema_version=1),
    ("weather_results", 1): ContentBlockRegistration(WeatherResultsBlock, schema_version=1),
    ("flight_results", 1): ContentBlockRegistration(FlightResultsBlock, schema_version=1),
    ("train_results", 1): ContentBlockRegistration(TrainResultsBlock, schema_version=1),
}

_SAFE_BLOCK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_SAFE_SOURCE_TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")


def _is_registered_rich_content_type(block_type: str) -> bool:
    return any(
        registered_type == block_type and schema_version is not None
        for registered_type, schema_version in CONTENT_BLOCK_REGISTRY
    )


def _source_schema_version(payload: Mapping[str, Any]) -> int | None:
    candidate = payload.get("schema_version")
    return candidate if type(candidate) is int else None


def _safe_source_type(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("type")
    if isinstance(candidate, str) and _SAFE_SOURCE_TYPE_PATTERN.fullmatch(candidate):
        return candidate
    return "unknown"


def _stable_unsupported_id(raw_block: object, *, position: int, collision_index: int = 0) -> str:
    payload = raw_block.model_dump(mode="json") if isinstance(raw_block, BaseModel) else raw_block
    try:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = repr(type(payload))
    digest = hashlib.sha256(f"{position}:{collision_index}:{canonical}".encode()).hexdigest()[:16]
    return f"blk_unsupported_{digest}"


def _unsupported_content_block(
    raw_block: object,
    *,
    reason: str,
    position: int,
    force_generated_id: bool = False,
    collision_index: int = 0,
) -> UnsupportedContentBlock:
    payload = (
        raw_block.model_dump(mode="python")
        if isinstance(raw_block, BaseModel)
        else dict(raw_block)
        if isinstance(raw_block, Mapping)
        else {}
    )
    candidate_id = payload.get("id")
    block_id = (
        candidate_id
        if not force_generated_id and isinstance(candidate_id, str) and _SAFE_BLOCK_ID_PATTERN.fullmatch(candidate_id)
        else _stable_unsupported_id(raw_block, position=position, collision_index=collision_index)
    )
    return UnsupportedContentBlock(
        type="unsupported_result",
        id=block_id,
        source_type=_safe_source_type(payload),
        source_schema_version=_source_schema_version(payload),
        reason=reason,
    )


def deserialize_content_block(raw_block: object, *, position: int = 0) -> ContentBlock:
    """解析单个内容块；未知类型、未知版本和损坏块降级为安全占位。"""
    if isinstance(raw_block, BaseModel):
        payload = raw_block.model_dump(mode="python")
    elif isinstance(raw_block, Mapping):
        payload = dict(raw_block)
    else:
        return _unsupported_content_block(raw_block, reason="invalid_payload", position=position)

    block_type = payload.get("type")
    if not isinstance(block_type, str):
        return _unsupported_content_block(raw_block, reason="invalid_payload", position=position)

    schema_version: int | None = None
    if _is_registered_rich_content_type(block_type):
        candidate_version = payload.get("schema_version")
        if type(candidate_version) is not int:
            return _unsupported_content_block(raw_block, reason="invalid_payload", position=position)
        schema_version = candidate_version

    registration = CONTENT_BLOCK_REGISTRY.get((block_type, schema_version))
    if registration is None:
        reason = "unsupported_version" if _is_registered_rich_content_type(block_type) else "unsupported_type"
        return _unsupported_content_block(raw_block, reason=reason, position=position)

    normalized = registration.normalize_payload(payload) if registration.normalize_payload else payload
    try:
        return cast(ContentBlock, registration.model.model_validate(normalized))
    except (ValidationError, TypeError, ValueError):
        logger.warning(
            "跳过无法解析的消息内容块: type=%s schema_version=%s",
            block_type,
            schema_version,
        )
        return _unsupported_content_block(raw_block, reason="invalid_payload", position=position)


def is_registered_rich_content_block(block: object) -> bool:
    """判断 block 是否匹配一个已注册的富结果版本。"""
    if not isinstance(block, BaseModel):
        return False
    payload = block.model_dump(mode="python")
    block_type = payload.get("type")
    schema_version = payload.get("schema_version")
    if not isinstance(block_type, str) or type(schema_version) is not int:
        return False
    registration = CONTENT_BLOCK_REGISTRY.get((block_type, schema_version))
    return (
        registration is not None and registration.schema_version is not None and isinstance(block, registration.model)
    )


def deserialize_content_blocks(raw_blocks: object) -> list[ContentBlock]:
    """逐块恢复内容，单个坏块不能拖垮整条消息或 continuation。"""
    if not isinstance(raw_blocks, list):
        return []

    content_blocks: list[ContentBlock] = []
    used_ids: set[str] = set()
    for position, raw_block in enumerate(raw_blocks):
        block = deserialize_content_block(raw_block, position=position)
        collision_index = 0
        while isinstance(block, UnsupportedContentBlock) and block.id in used_ids:
            collision_index += 1
            block = _unsupported_content_block(
                raw_block,
                reason=block.reason,
                position=position,
                force_generated_id=True,
                collision_index=collision_index,
            )
        used_ids.add(block.id)
        content_blocks.append(block)
    return content_blocks
