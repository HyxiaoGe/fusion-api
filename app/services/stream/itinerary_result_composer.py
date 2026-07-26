"""把既有产品结果组合成引用型行程结果，不调用模型或网络。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from app.schemas.chat import (
    FlightResultsBlock,
    ItineraryAvailability,
    ItineraryPlan,
    ItineraryResultRef,
    ItineraryResultsBlock,
    ItinerarySection,
    RouteResultsBlock,
    TrainResultsBlock,
    TravelMoney,
    WeatherResultsBlock,
)

TravelKind = Literal["flight", "train"]
JourneyKind = Literal["outbound", "return"]

_TRAVEL_LIMITATION = "班次时长不包含前后接驳、值机、安检或候车时间"
_COST_LIMITATION = "参考票价之和不等于完整旅行总预算"


@dataclass(frozen=True)
class _TravelCandidate:
    kind: TravelKind
    block_id: str
    option_id: str
    number: str
    origin: str
    destination: str
    departure_date: date
    departure_station: str
    arrival_station: str
    duration_s: int
    price_minor: int | None
    source_status: str
    position: int

    @property
    def stable_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.kind,
            self.number.upper(),
            self.departure_date.isoformat(),
            _normalize_location(self.departure_station),
            _normalize_location(self.arrival_station),
        )


@dataclass
class _TravelGroup:
    origin: str
    destination: str
    departure_date: date
    position: int
    candidates: dict[tuple[str, str, str, str, str], _TravelCandidate]


@dataclass(frozen=True)
class _Journey:
    kind: JourneyKind
    group: _TravelGroup


def compose_itinerary_result(
    content_blocks: list[Any],
    *,
    product_outcomes: list[Any] | None = None,
) -> ItineraryResultsBlock | None:
    """从同一消息中的规范化结果构造一个可持久化的 itinerary。"""

    groups = _collect_travel_groups(content_blocks)
    outcomes = product_outcomes or []
    journeys = _select_journeys(groups, outcomes)
    if not journeys:
        return None

    existing = _find_existing_itinerary(content_blocks, journeys[0])
    weather = _select_weather(content_blocks, journeys)
    plans = _build_plans(journeys, weather, content_blocks)
    if not plans:
        return None

    availability = _build_availability(journeys, weather, plans, outcomes, existing)
    limitations = [_TRAVEL_LIMITATION, _COST_LIMITATION]
    if weather is not None and weather[1] != "full":
        limitations.append(
            "天气预报未完整覆盖行程日期" if weather[1] == "partial" else "当前天气预报窗口未覆盖行程日期"
        )
    degraded = any(plan.status == "partial" for plan in plans) or any(
        section.status != "complete" for plan in plans for section in plan.sections
    )
    degraded = degraded or any(item.status == "unavailable" for item in availability)
    degraded = degraded or _has_degraded_source(journeys, weather, plans, content_blocks)
    outbound = journeys[0].group
    end_date = journeys[-1].group.departure_date if len(journeys) == 2 else None
    return ItineraryResultsBlock(
        type="itinerary_results",
        id=existing.id if existing is not None else _stable_itinerary_id(outbound),
        schema_version=1,
        provider="fusion",
        status="degraded" if degraded else "success",
        trip_type="round_trip" if len(journeys) == 2 else "one_way",
        origin=outbound.origin,
        destination=outbound.destination,
        start_date=outbound.departure_date,
        end_date=end_date,
        recommended_plan_id=None,
        plans=plans,
        availability=availability,
        limitations=limitations,
    )


def _collect_travel_groups(content_blocks: list[Any]) -> list[_TravelGroup]:
    groups: dict[tuple[str, str, date], _TravelGroup] = {}
    for position, block in enumerate(content_blocks):
        if not isinstance(block, (FlightResultsBlock, TrainResultsBlock)):
            continue
        key = (
            _normalize_location(block.origin),
            _normalize_location(block.destination),
            date.fromisoformat(block.departure_date),
        )
        group = groups.setdefault(
            key,
            _TravelGroup(
                origin=block.origin,
                destination=block.destination,
                departure_date=key[2],
                position=position,
                candidates={},
            ),
        )
        group.origin = block.origin
        group.destination = block.destination
        for candidate in _travel_candidates(block, position):
            group.candidates[candidate.stable_key] = candidate
    return [group for group in groups.values() if group.candidates]


def _travel_candidates(
    block: FlightResultsBlock | TrainResultsBlock,
    position: int,
) -> list[_TravelCandidate]:
    if isinstance(block, FlightResultsBlock):
        options = block.flights
        kind: TravelKind = "flight"
        number_key = "flight_no"
    else:
        options = block.trains
        kind = "train"
        number_key = "train_no"
    candidates: list[_TravelCandidate] = []
    for option in options:
        price = option.price.amount_minor if option.price is not None else None
        candidates.append(
            _TravelCandidate(
                kind=kind,
                block_id=block.id,
                option_id=option.option_id,
                number=str(getattr(option, number_key)),
                origin=block.origin,
                destination=block.destination,
                departure_date=date.fromisoformat(block.departure_date),
                departure_station=option.departure.station_name,
                arrival_station=option.arrival.station_name,
                duration_s=option.duration_s,
                price_minor=price,
                source_status=block.status,
                position=position,
            )
        )
    return candidates


def _select_journeys(
    groups: list[_TravelGroup],
    product_outcomes: list[Any],
) -> list[_Journey]:
    if not groups:
        return []
    ordered = sorted(groups, key=lambda item: (item.departure_date, item.position))
    outbound = ordered[0]
    if _has_earlier_failed_reciprocal(outbound, product_outcomes):
        # 已知更早的反向去程失败时，当前唯一成功组其实是返程。
        # v1 不能在没有任何去程引用的情况下构造合法 partial plan，
        # 因此保留成功源卡，等待后续去程结果，而不是伪装成完整单程。
        return []
    reciprocal = [
        group
        for group in ordered[1:]
        if group.departure_date > outbound.departure_date
        and _normalize_location(group.origin) == _normalize_location(outbound.destination)
        and _normalize_location(group.destination) == _normalize_location(outbound.origin)
    ]
    journeys = [_Journey(kind="outbound", group=outbound)]
    if reciprocal:
        journeys.append(_Journey(kind="return", group=reciprocal[0]))
    elif failed_return := _failed_return_group(outbound, product_outcomes):
        journeys.append(_Journey(kind="return", group=failed_return))
    return journeys


def _has_earlier_failed_reciprocal(
    successful_group: _TravelGroup,
    product_outcomes: list[Any],
) -> bool:
    return any(
        getattr(outcome, "mode", None) in {"flight", "train"}
        and getattr(outcome, "status", None) == "unavailable"
        and (failed_date := _safe_date(getattr(outcome, "departure_date", None))) is not None
        and failed_date < successful_group.departure_date
        and _locations_compatible(getattr(outcome, "origin", ""), successful_group.destination)
        and _locations_compatible(getattr(outcome, "destination", ""), successful_group.origin)
        for outcome in product_outcomes
    )


def _failed_return_group(
    outbound: _TravelGroup,
    product_outcomes: list[Any],
) -> _TravelGroup | None:
    candidates: list[_TravelGroup] = []
    for position, outcome in enumerate(product_outcomes):
        if getattr(outcome, "mode", None) not in {"flight", "train"}:
            continue
        if getattr(outcome, "status", None) != "unavailable":
            continue
        departure_date = _safe_date(getattr(outcome, "departure_date", None))
        if departure_date is None or departure_date <= outbound.departure_date:
            continue
        origin = getattr(outcome, "origin", None)
        destination = getattr(outcome, "destination", None)
        if not (
            isinstance(origin, str)
            and isinstance(destination, str)
            and _locations_compatible(origin, outbound.destination)
            and _locations_compatible(destination, outbound.origin)
        ):
            continue
        candidates.append(
            _TravelGroup(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                position=position,
                candidates={},
            )
        )
    return min(candidates, key=lambda item: (item.departure_date, item.position)) if candidates else None


def _build_plans(
    journeys: list[_Journey],
    weather: tuple[WeatherResultsBlock, str] | None,
    content_blocks: list[Any],
) -> list[ItineraryPlan]:
    plans: list[ItineraryPlan] = []
    for strategy in ("lowest_reference_price", "shortest_scheduled_duration"):
        selections = _select_candidates(journeys, strategy)
        if not selections:
            continue
        plan = _build_plan(strategy, journeys, selections, weather, content_blocks)
        if any(_plan_signature(existing) == _plan_signature(plan) for existing in plans):
            continue
        plans.append(plan)
    return plans


def _select_candidates(
    journeys: list[_Journey],
    strategy: str,
) -> dict[JourneyKind, _TravelCandidate]:
    selected: dict[JourneyKind, _TravelCandidate] = {}
    for journey in journeys:
        candidates = list(journey.group.candidates.values())
        if strategy == "lowest_reference_price":
            candidates = [candidate for candidate in candidates if candidate.price_minor is not None]
            sort_key = _lowest_price_sort_key
        else:
            sort_key = _shortest_duration_sort_key
        if candidates:
            selected[journey.kind] = min(candidates, key=sort_key)
    return selected


def _lowest_price_sort_key(candidate: _TravelCandidate) -> tuple[int, int, int]:
    return (candidate.price_minor or 0, candidate.duration_s, -candidate.position)


def _shortest_duration_sort_key(candidate: _TravelCandidate) -> tuple[int, bool, int]:
    return (candidate.duration_s, candidate.price_minor is None, candidate.price_minor or 0)


def _build_plan(
    strategy: str,
    journeys: list[_Journey],
    selections: dict[JourneyKind, _TravelCandidate],
    weather: tuple[WeatherResultsBlock, str] | None,
    content_blocks: list[Any],
) -> ItineraryPlan:
    sections = [
        _transport_section(journey, selections[journey.kind]) for journey in journeys if journey.kind in selections
    ]
    if weather is not None:
        sections.append(_weather_section(weather))
    route = _select_route(content_blocks, selections)
    if route is not None:
        sections.append(_route_section(route))
    complete = len(selections) == len(journeys)
    known_cost = _known_cost(selections, complete=complete)
    known_duration = sum(candidate.duration_s for candidate in selections.values()) if complete else None
    return ItineraryPlan(
        id="lowest-price" if strategy == "lowest_reference_price" else "shortest-duration",
        title="最低参考价组合" if strategy == "lowest_reference_price" else "班次计划时长最短组合",
        status="complete" if complete else "partial",
        strategy=strategy,
        tags=[strategy],
        known_cost=known_cost,
        known_duration_s=known_duration,
        sections=sections,
    )


def _transport_section(journey: _Journey, candidate: _TravelCandidate) -> ItinerarySection:
    return ItinerarySection(
        id=journey.kind,
        kind="outbound_transport" if journey.kind == "outbound" else "return_transport",
        status="complete",
        title="去程" if journey.kind == "outbound" else "返程",
        coverage=None,
        result_refs=[ItineraryResultRef(block_id=candidate.block_id, item_ids=[candidate.option_id])],
    )


def _weather_section(weather: tuple[WeatherResultsBlock, str]) -> ItinerarySection:
    block, coverage = weather
    return ItinerarySection(
        id="destination-weather",
        kind="destination_weather",
        status="complete" if coverage == "full" else "partial",
        title="目的地天气",
        coverage=coverage,
        result_refs=[ItineraryResultRef(block_id=block.id)],
    )


def _route_section(route: RouteResultsBlock) -> ItinerarySection:
    return ItinerarySection(
        id="local-route",
        kind="local_route",
        status="complete" if route.status == "success" else "partial",
        title="当地路线",
        coverage=None,
        result_refs=[ItineraryResultRef(block_id=route.id)],
    )


def _known_cost(
    selections: dict[JourneyKind, _TravelCandidate],
    *,
    complete: bool,
) -> TravelMoney | None:
    prices = [candidate.price_minor for candidate in selections.values()]
    if not complete or not prices or any(price is None for price in prices):
        return None
    return TravelMoney(currency="CNY", amount_minor=sum(price for price in prices if price is not None))


def _select_weather(
    content_blocks: list[Any],
    journeys: list[_Journey],
) -> tuple[WeatherResultsBlock, str] | None:
    destination = journeys[0].group.destination
    start_date = journeys[0].group.departure_date
    end_date = journeys[-1].group.departure_date
    candidates: list[tuple[int, int, int, WeatherResultsBlock, str]] = []
    for position, block in enumerate(content_blocks):
        if not isinstance(block, WeatherResultsBlock):
            continue
        if not _locations_compatible(destination, block.resolved_location):
            continue
        coverage, covered_days = _weather_coverage(block, start_date, end_date)
        rank = {"outside_range": 0, "partial": 1, "full": 2}[coverage]
        candidates.append((rank, covered_days, position, block, coverage))
    if not candidates:
        return None
    _, _, _, block, coverage = max(candidates, key=lambda item: item[:3])
    return block, coverage


def _weather_coverage(
    block: WeatherResultsBlock,
    start_date: date,
    end_date: date,
) -> tuple[str, int]:
    expected_days = (end_date - start_date).days + 1
    covered_days = sum(start_date <= item.date <= end_date for item in block.forecast_days)
    if covered_days == expected_days:
        return "full", covered_days
    if covered_days > 0:
        return "partial", covered_days
    return "outside_range", 0


def _select_route(
    content_blocks: list[Any],
    selections: dict[JourneyKind, _TravelCandidate],
) -> RouteResultsBlock | None:
    destination = selections.get("outbound").destination if selections.get("outbound") is not None else ""
    station_names: set[str] = set()
    for candidate in selections.values():
        if _locations_compatible(destination, candidate.destination):
            station_names.add(candidate.arrival_station)
        if _locations_compatible(destination, candidate.origin):
            station_names.add(candidate.departure_station)
    matches = [
        block
        for block in content_blocks
        if isinstance(block, RouteResultsBlock) and _route_matches_destination(block, destination, station_names)
    ]
    return matches[-1] if matches else None


def _route_matches_destination(
    block: RouteResultsBlock,
    destination: str,
    station_names: set[str],
) -> bool:
    endpoint_cities = [endpoint.city for endpoint in (block.origin, block.destination) if endpoint.city]
    if endpoint_cities and any(not _locations_compatible(destination, city) for city in endpoint_cities):
        return False
    endpoint_labels = (block.origin.label, block.destination.label)
    return any(_labels_compatible(label, station) for label in endpoint_labels for station in station_names)


def _build_availability(
    journeys: list[_Journey],
    weather: tuple[WeatherResultsBlock, str] | None,
    plans: list[ItineraryPlan],
    product_outcomes: list[Any],
    existing: ItineraryResultsBlock | None,
) -> list[ItineraryAvailability]:
    statuses: dict[tuple[str, str], str] = (
        {(item.journey, item.mode): item.status for item in existing.availability} if existing is not None else {}
    )
    for journey in journeys:
        modes = {candidate.kind for candidate in journey.group.candidates.values()}
        for mode in ("flight", "train"):
            if mode in modes:
                statuses[(journey.kind, mode)] = "available"
    if weather is not None:
        statuses[("destination_weather", "weather")] = "available"
    if any(section.kind == "local_route" for plan in plans for section in plan.sections):
        statuses[("local_route", "route")] = "available"
    _merge_outcome_availability(statuses, journeys, product_outcomes)
    ordering = {
        ("outbound", "flight"): 0,
        ("outbound", "train"): 1,
        ("return", "flight"): 2,
        ("return", "train"): 3,
        ("destination_weather", "weather"): 4,
        ("local_route", "route"): 5,
    }
    return [
        ItineraryAvailability(journey=journey, mode=mode, status=status)
        for (journey, mode), status in sorted(statuses.items(), key=lambda item: ordering[item[0]])
    ]


def _merge_outcome_availability(
    statuses: dict[tuple[str, str], str],
    journeys: list[_Journey],
    product_outcomes: list[Any],
) -> None:
    for outcome in product_outcomes:
        mode = getattr(outcome, "mode", None)
        status = getattr(outcome, "status", None)
        journey = _outcome_journey(outcome, journeys)
        if journey is None or mode not in {"flight", "train", "weather", "route"}:
            continue
        key = (journey, mode)
        if statuses.get(key) == "available":
            continue
        if status in {"available", "unavailable"}:
            statuses[key] = status


def _outcome_journey(outcome: Any, journeys: list[_Journey]) -> str | None:
    mode = getattr(outcome, "mode", None)
    if mode in {"flight", "train"}:
        outcome_date = _safe_date(getattr(outcome, "departure_date", None))
        for journey in journeys:
            if (
                outcome_date == journey.group.departure_date
                and _locations_compatible(getattr(outcome, "origin", ""), journey.group.origin)
                and _locations_compatible(getattr(outcome, "destination", ""), journey.group.destination)
            ):
                return journey.kind
        return None
    destination = journeys[0].group.destination
    if mode == "weather":
        location = getattr(outcome, "location", None)
        return (
            "destination_weather"
            if isinstance(location, str) and _locations_compatible(location, destination)
            else None
        )
    if mode == "route":
        endpoints = (getattr(outcome, "origin", None), getattr(outcome, "destination", None))
        return (
            "local_route"
            if any(isinstance(endpoint, str) and _locations_compatible(endpoint, destination) for endpoint in endpoints)
            else None
        )
    return None


def _find_existing_itinerary(
    content_blocks: list[Any],
    outbound: _Journey,
) -> ItineraryResultsBlock | None:
    for block in reversed(content_blocks):
        if not isinstance(block, ItineraryResultsBlock):
            continue
        if (
            block.start_date == outbound.group.departure_date
            and _locations_compatible(block.origin, outbound.group.origin)
            and _locations_compatible(block.destination, outbound.group.destination)
        ):
            return block
    return None


def _has_degraded_source(
    journeys: list[_Journey],
    weather: tuple[WeatherResultsBlock, str] | None,
    plans: list[ItineraryPlan],
    content_blocks: list[Any],
) -> bool:
    if any(
        candidate.source_status != "success" for journey in journeys for candidate in journey.group.candidates.values()
    ):
        return True
    if weather is not None and weather[0].status != "success":
        return True
    route_ids = {
        ref.block_id
        for plan in plans
        for section in plan.sections
        if section.kind == "local_route"
        for ref in section.result_refs
    }
    return any(
        isinstance(block, RouteResultsBlock) and block.id in route_ids and block.status != "success"
        for block in content_blocks
    )


def _plan_signature(plan: ItineraryPlan) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple(
        (section.kind, ref.block_id, tuple(ref.item_ids))
        for section in plan.sections
        if section.kind in {"outbound_transport", "return_transport"}
        for ref in section.result_refs
    )


def _stable_itinerary_id(outbound: _TravelGroup) -> str:
    canonical = "|".join(
        (
            _normalize_location(outbound.origin),
            _normalize_location(outbound.destination),
            outbound.departure_date.isoformat(),
        )
    )
    return f"blk_itinerary_{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _locations_compatible(left: str, right: str) -> bool:
    normalized_left = _normalize_location(left)
    normalized_right = _normalize_location(right)
    return bool(
        normalized_left
        and normalized_right
        and (normalized_left in normalized_right or normalized_right in normalized_left)
    )


def _labels_compatible(left: str, right: str) -> bool:
    normalized_left = _normalize_location(left)
    normalized_right = _normalize_location(right)
    return bool(
        normalized_left
        and normalized_right
        and (
            normalized_left == normalized_right
            or normalized_left in normalized_right
            or normalized_right in normalized_left
        )
    )


def _normalize_location(value: str) -> str:
    compact = re.sub(r"[\s·•（）()\-—_]+", "", value).lower()
    return re.sub(r"(?:省|市|区|县|自治州|特别行政区)$", "", compact)


def _safe_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
