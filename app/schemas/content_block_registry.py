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
    ItineraryResultsBlock,
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
    ("itinerary_results", 1): ContentBlockRegistration(ItineraryResultsBlock, schema_version=1),
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
    return _reject_invalid_itinerary_references(content_blocks, raw_blocks)


def validate_itinerary_references(
    itinerary: ItineraryResultsBlock,
    content_blocks: list[ContentBlock],
) -> bool:
    """验证跨块引用；任一引用损坏时必须保留源块并拒绝整个 itinerary。"""

    blocks_by_id: dict[str, list[ContentBlock]] = {}
    for block in content_blocks:
        blocks_by_id.setdefault(block.id, []).append(block)

    for plan in itinerary.plans:
        if not _validate_itinerary_plan(itinerary, plan, blocks_by_id):
            return False
        for section in plan.sections:
            ref_keys = [(ref.block_id, tuple(ref.item_ids)) for ref in section.result_refs]
            if len(ref_keys) != len(set(ref_keys)):
                return False
            expected_types = _expected_itinerary_source_types(section.kind)
            for result_ref in section.result_refs:
                candidates = blocks_by_id.get(result_ref.block_id, [])
                if len(candidates) != 1:
                    return False
                source = candidates[0]
                if source.type not in expected_types:
                    return False
                if not _validate_itinerary_item_ids(source, result_ref.item_ids):
                    return False
            if section.kind == "destination_weather" and not _validate_weather_section(
                itinerary,
                section,
                blocks_by_id,
            ):
                return False
            if section.kind == "local_route" and not _validate_local_route_section(
                itinerary,
                plan,
                section,
                blocks_by_id,
            ):
                return False
    return _validate_itinerary_availability(itinerary, blocks_by_id) and _validate_itinerary_status(
        itinerary,
        blocks_by_id,
    )


def _validate_itinerary_plan(
    itinerary: ItineraryResultsBlock,
    plan: Any,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> bool:
    transport_items: list[Any] = []
    transport_kinds: set[str] = set()
    for section in plan.sections:
        if section.kind not in {"outbound_transport", "return_transport"}:
            continue
        transport_kinds.add(section.kind)
        resolved = _resolve_transport_items(itinerary, section, blocks_by_id)
        if resolved is None:
            return False
        transport_items.extend(resolved)

    expected_kinds = {"outbound_transport"}
    if itinerary.trip_type == "round_trip":
        expected_kinds.add("return_transport")
    if plan.status == "complete" and transport_kinds != expected_kinds:
        return False
    complete = plan.status == "complete" and transport_kinds == expected_kinds
    return _validate_plan_totals(plan, transport_items, complete=complete)


def _resolve_transport_items(
    itinerary: ItineraryResultsBlock,
    section: Any,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> list[Any] | None:
    expected_origin, expected_destination, expected_date = _transport_expectation(itinerary, section.kind)
    if expected_date is None:
        return None
    resolved_items: list[Any] = []
    for result_ref in section.result_refs:
        sources = blocks_by_id.get(result_ref.block_id, [])
        if len(sources) != 1 or not isinstance(sources[0], (FlightResultsBlock, TrainResultsBlock)):
            return None
        source = sources[0]
        if not (
            _locations_compatible(source.origin, expected_origin)
            and _locations_compatible(source.destination, expected_destination)
            and source.departure_date == expected_date.isoformat()
        ):
            return None
        items = source.flights if isinstance(source, FlightResultsBlock) else source.trains
        option_ids = [item.option_id for item in items]
        if len(option_ids) != len(set(option_ids)):
            return None
        items_by_id = {item.option_id: item for item in items}
        for item_id in result_ref.item_ids:
            item = items_by_id.get(item_id)
            if item is None or not _validate_transport_item(item, expected_origin, expected_destination, expected_date):
                return None
            resolved_items.append(item)
    return resolved_items or None


def _transport_expectation(
    itinerary: ItineraryResultsBlock,
    section_kind: str,
) -> tuple[str, str, Any]:
    if section_kind == "outbound_transport":
        return itinerary.origin, itinerary.destination, itinerary.start_date
    if section_kind == "return_transport":
        return itinerary.destination, itinerary.origin, itinerary.end_date
    return "", "", None


def _validate_transport_item(
    item: Any,
    expected_origin: str,
    expected_destination: str,
    expected_date: Any,
) -> bool:
    return bool(
        _locations_compatible(item.departure.city, expected_origin)
        and _locations_compatible(item.arrival.city, expected_destination)
        and item.departure.scheduled_at.date() == expected_date
    )


def _validate_plan_totals(
    plan: Any,
    transport_items: list[Any],
    *,
    complete: bool,
) -> bool:
    if not complete:
        return plan.known_cost is None and plan.known_duration_s is None
    expected_duration = sum(item.duration_s for item in transport_items)
    if plan.known_duration_s != expected_duration:
        return False
    prices = [item.price.amount_minor if item.price is not None else None for item in transport_items]
    if any(price is None for price in prices):
        return plan.known_cost is None
    return bool(
        plan.known_cost is not None
        and plan.known_cost.currency == "CNY"
        and plan.known_cost.amount_minor == sum(price for price in prices if price is not None)
    )


def _validate_weather_section(
    itinerary: ItineraryResultsBlock,
    section: Any,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> bool:
    if len(section.result_refs) != 1:
        return False
    sources = blocks_by_id.get(section.result_refs[0].block_id, [])
    if len(sources) != 1 or not isinstance(sources[0], WeatherResultsBlock):
        return False
    source = sources[0]
    if not _locations_compatible(source.resolved_location, itinerary.destination):
        return False
    end_date = itinerary.end_date or itinerary.start_date
    expected_days = (end_date - itinerary.start_date).days + 1
    covered_days = sum(itinerary.start_date <= item.date <= end_date for item in source.forecast_days)
    expected_coverage = "full" if covered_days == expected_days else "partial" if covered_days else "outside_range"
    return section.coverage == expected_coverage


def _validate_local_route_section(
    itinerary: ItineraryResultsBlock,
    plan: Any,
    section: Any,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> bool:
    station_names: set[str] = set()
    for transport_section in plan.sections:
        if transport_section.kind not in {"outbound_transport", "return_transport"}:
            continue
        items = _resolve_transport_items(itinerary, transport_section, blocks_by_id)
        if items is None:
            return False
        for item in items:
            if _locations_compatible(item.arrival.city, itinerary.destination):
                station_names.add(item.arrival.station_name)
            if _locations_compatible(item.departure.city, itinerary.destination):
                station_names.add(item.departure.station_name)
    if not station_names:
        return False

    for result_ref in section.result_refs:
        sources = blocks_by_id.get(result_ref.block_id, [])
        if len(sources) != 1 or not isinstance(sources[0], RouteResultsBlock):
            return False
        source = sources[0]
        endpoint_cities = [endpoint.city for endpoint in (source.origin, source.destination) if endpoint.city]
        if endpoint_cities and any(not _locations_compatible(city, itinerary.destination) for city in endpoint_cities):
            return False
        endpoint_labels = (source.origin.label, source.destination.label)
        if not any(_locations_compatible(label, station) for label in endpoint_labels for station in station_names):
            return False
    return True


def _validate_itinerary_availability(
    itinerary: ItineraryResultsBlock,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> bool:
    availability = {(item.journey, item.mode): item.status for item in itinerary.availability}
    selected: set[tuple[str, str]] = set()
    for plan in itinerary.plans:
        for section in plan.sections:
            journey = {
                "outbound_transport": "outbound",
                "return_transport": "return",
                "destination_weather": "destination_weather",
                "local_route": "local_route",
            }[section.kind]
            for result_ref in section.result_refs:
                sources = blocks_by_id.get(result_ref.block_id, [])
                if len(sources) != 1:
                    return False
                source = sources[0]
                mode = {
                    "flight_results": "flight",
                    "train_results": "train",
                    "weather_results": "weather",
                    "route_results": "route",
                }.get(source.type)
                if mode is None:
                    return False
                selected.add((journey, mode))
    return all(availability.get(key) == "available" for key in selected)


def _validate_itinerary_status(
    itinerary: ItineraryResultsBlock,
    blocks_by_id: dict[str, list[ContentBlock]],
) -> bool:
    degraded = any(
        plan.status == "partial" or any(section.status != "complete" for section in plan.sections)
        for plan in itinerary.plans
    )
    degraded = degraded or any(item.status == "unavailable" for item in itinerary.availability)
    degraded = degraded or any(
        getattr(source, "status", "success") != "success"
        for plan in itinerary.plans
        for section in plan.sections
        for result_ref in section.result_refs
        for source in blocks_by_id.get(result_ref.block_id, [])
    )
    return itinerary.status == ("degraded" if degraded else "success")


def _expected_itinerary_source_types(section_kind: str) -> frozenset[str]:
    return {
        "outbound_transport": frozenset({"flight_results", "train_results"}),
        "return_transport": frozenset({"flight_results", "train_results"}),
        "destination_weather": frozenset({"weather_results"}),
        "local_route": frozenset({"route_results"}),
    }.get(section_kind, frozenset())


def _validate_itinerary_item_ids(source: ContentBlock, item_ids: list[str]) -> bool:
    if source.type == "flight_results":
        option_ids = [option.option_id for option in source.flights]
        if len(option_ids) != len(set(option_ids)):
            return False
        available_ids = set(option_ids)
        return bool(item_ids) and set(item_ids) <= available_ids
    if source.type == "train_results":
        option_ids = [option.option_id for option in source.trains]
        if len(option_ids) != len(set(option_ids)):
            return False
        available_ids = set(option_ids)
        return bool(item_ids) and set(item_ids) <= available_ids
    return not item_ids


def _locations_compatible(left: str, right: str) -> bool:
    normalized_left = _normalize_location(left)
    normalized_right = _normalize_location(right)
    return bool(
        normalized_left
        and normalized_right
        and (normalized_left in normalized_right or normalized_right in normalized_left)
    )


def _normalize_location(value: str) -> str:
    compact = re.sub(r"[\s·•（）()\-—_]+", "", value).lower()
    return re.sub(r"(?:省|市|区|县|自治州|特别行政区)$", "", compact)


def _reject_invalid_itinerary_references(
    content_blocks: list[ContentBlock],
    raw_blocks: list[object],
) -> list[ContentBlock]:
    validated = list(content_blocks)
    for position, block in enumerate(content_blocks):
        if not isinstance(block, ItineraryResultsBlock):
            continue
        if validate_itinerary_references(block, content_blocks):
            continue
        validated[position] = _unsupported_content_block(
            raw_blocks[position],
            reason="invalid_payload",
            position=position,
        )
    return validated
