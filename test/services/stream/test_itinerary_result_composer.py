from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.schemas.chat import (
    FlightOption,
    FlightResultsBlock,
    ItineraryAvailability,
    ItineraryResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    TrainOption,
    TrainResultsBlock,
    TravelEndpoint,
    TravelMoney,
    WeatherForecastDay,
    WeatherResultsBlock,
)
from app.services.stream.agent_loop_state import ProductToolOutcome
from app.services.stream.itinerary_result_composer import compose_itinerary_result

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _endpoint(city: str, station: str, scheduled_at: str) -> TravelEndpoint:
    return TravelEndpoint(
        city=city,
        station_name=station,
        scheduled_at=datetime.fromisoformat(scheduled_at),
    )


def _flight_block(
    *,
    block_id: str,
    origin: str,
    destination: str,
    departure_date: str,
    option_id: str,
    flight_no: str,
    price_minor: int | None,
    duration_s: int,
    departure_at: str,
    arrival_at: str,
    status: str = "success",
) -> FlightResultsBlock:
    option = FlightOption(
        option_id=option_id,
        airline_name="测试航空",
        flight_no=flight_no,
        departure=_endpoint(origin, f"{origin}机场", departure_at),
        arrival=_endpoint(destination, f"{destination}机场", arrival_at),
        duration_s=duration_s,
        stops=0,
        price=TravelMoney(currency="CNY", amount_minor=price_minor) if price_minor is not None else None,
    )
    return FlightResultsBlock(
        type="flight_results",
        id=block_id,
        schema_version=1,
        provider="flyai",
        status=status,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        observed_at=datetime(2026, 7, 26, 8, tzinfo=SHANGHAI_TZ),
        result_count=1,
        flights=[option],
    )


def _train_block(
    *,
    block_id: str,
    origin: str,
    destination: str,
    departure_date: str,
    option_id: str,
    train_no: str,
    price_minor: int | None,
    duration_s: int,
    departure_at: str,
    arrival_at: str,
    status: str = "success",
) -> TrainResultsBlock:
    option = TrainOption(
        option_id=option_id,
        train_no=train_no,
        train_type="高铁",
        departure=_endpoint(origin, f"{origin}站", departure_at),
        arrival=_endpoint(destination, f"{destination}站", arrival_at),
        duration_s=duration_s,
        stops=0,
        price=TravelMoney(currency="CNY", amount_minor=price_minor) if price_minor is not None else None,
    )
    return TrainResultsBlock(
        type="train_results",
        id=block_id,
        schema_version=1,
        provider="flyai",
        status=status,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        observed_at=datetime(2026, 7, 26, 8, tzinfo=SHANGHAI_TZ),
        result_count=1,
        trains=[option],
    )


def _weather_block(
    *,
    block_id: str,
    location: str,
    dates: list[str],
) -> WeatherResultsBlock:
    days = [
        WeatherForecastDay(
            date=date.fromisoformat(value),
            weekday=date.fromisoformat(value).isoweekday(),
            day_weather="晴",
            night_weather="多云",
            high_c=31,
            low_c=25,
        )
        for value in dates
    ]
    return WeatherResultsBlock(
        type="weather_results",
        id=block_id,
        schema_version=1,
        provider="amap",
        status="success" if len(days) == 4 else "degraded",
        query=location,
        resolved_location=location,
        day_count=len(days),
        forecast_days=days,
        fetched_at=datetime(2026, 7, 26, 8, tzinfo=SHANGHAI_TZ),
    )


def _route_block(*, block_id: str, city: str, station: str) -> RouteResultsBlock:
    return RouteResultsBlock(
        type="route_results",
        id=block_id,
        schema_version=1,
        provider="amap",
        status="success",
        origin=RouteEndpoint(label=station, city=city),
        destination=RouteEndpoint(label=f"{city}人民广场", city=city),
        routes=[RouteOption(mode="transit", duration_s=1800, distance_m=20_000)],
    )


def _round_trip_blocks() -> list:
    return [
        _flight_block(
            block_id="flight-out",
            origin="深圳",
            destination="上海",
            departure_date="2026-08-01",
            option_id="flight-out-1",
            flight_no="ZH9501",
            price_minor=68000,
            duration_s=8400,
            departure_at="2026-08-01T08:00:00+08:00",
            arrival_at="2026-08-01T10:20:00+08:00",
        ),
        _train_block(
            block_id="train-out",
            origin="深圳",
            destination="上海",
            departure_date="2026-08-01",
            option_id="train-out-1",
            train_no="G100",
            price_minor=56000,
            duration_s=28800,
            departure_at="2026-08-01T07:00:00+08:00",
            arrival_at="2026-08-01T15:00:00+08:00",
        ),
        _flight_block(
            block_id="flight-return",
            origin="上海",
            destination="深圳",
            departure_date="2026-08-03",
            option_id="flight-return-1",
            flight_no="ZH9502",
            price_minor=72000,
            duration_s=9000,
            departure_at="2026-08-03T18:00:00+08:00",
            arrival_at="2026-08-03T20:30:00+08:00",
        ),
        _train_block(
            block_id="train-return",
            origin="上海",
            destination="深圳",
            departure_date="2026-08-03",
            option_id="train-return-1",
            train_no="G101",
            price_minor=57000,
            duration_s=29400,
            departure_at="2026-08-03T08:00:00+08:00",
            arrival_at="2026-08-03T16:10:00+08:00",
        ),
    ]


def test_compose_round_trip_builds_lowest_price_and_shortest_duration_plans():
    itinerary = compose_itinerary_result(_round_trip_blocks())

    assert isinstance(itinerary, ItineraryResultsBlock)
    assert itinerary.trip_type == "round_trip"
    assert itinerary.origin == "深圳"
    assert itinerary.destination == "上海"
    assert itinerary.start_date.isoformat() == "2026-08-01"
    assert itinerary.end_date.isoformat() == "2026-08-03"
    assert [plan.strategy for plan in itinerary.plans] == [
        "lowest_reference_price",
        "shortest_scheduled_duration",
    ]
    lowest, shortest = itinerary.plans
    assert lowest.known_cost is not None
    assert lowest.known_cost.amount_minor == 113000
    assert lowest.known_duration_s == 58200
    assert [ref.block_id for section in lowest.sections[:2] for ref in section.result_refs] == [
        "train-out",
        "train-return",
    ]
    assert shortest.known_cost is not None
    assert shortest.known_cost.amount_minor == 140000
    assert shortest.known_duration_s == 17400
    assert [ref.block_id for section in shortest.sections[:2] for ref in section.result_refs] == [
        "flight-out",
        "flight-return",
    ]


def test_compose_reuses_stable_id_and_updates_existing_itinerary():
    outbound = _round_trip_blocks()[:2]
    first = compose_itinerary_result(outbound)
    assert first is not None

    updated = compose_itinerary_result([*outbound, first, *_round_trip_blocks()[2:]])

    assert updated is not None
    assert updated.id == first.id
    assert updated.trip_type == "round_trip"


def test_compose_continuation_preserves_unavailable_until_same_mode_succeeds():
    train_outbound = _round_trip_blocks()[1]
    unavailable_flight = ProductToolOutcome(
        tool_name="search_flights",
        mode="flight",
        status="unavailable",
        origin="深圳",
        destination="上海",
        departure_date="2026-08-01",
    )
    first = compose_itinerary_result(
        [train_outbound],
        product_outcomes=[unavailable_flight],
    )
    assert first is not None
    assert (
        ItineraryAvailability(
            journey="outbound",
            mode="flight",
            status="unavailable",
        )
        in first.availability
    )

    continued = compose_itinerary_result(
        [
            train_outbound,
            first,
            _weather_block(
                block_id="weather-continuation",
                location="上海市",
                dates=["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
            ),
        ]
    )
    assert continued is not None
    assert (
        ItineraryAvailability(
            journey="outbound",
            mode="flight",
            status="unavailable",
        )
        in continued.availability
    )

    recovered = compose_itinerary_result([train_outbound, first, _round_trip_blocks()[0]])
    assert recovered is not None
    assert (
        ItineraryAvailability(
            journey="outbound",
            mode="flight",
            status="available",
        )
        in recovered.availability
    )


def test_compose_does_not_misclassify_return_as_successful_one_way_when_outbound_failed():
    successful_return = _round_trip_blocks()[2]
    failed_outbound = ProductToolOutcome(
        tool_name="search_flights",
        mode="flight",
        status="unavailable",
        origin="深圳",
        destination="上海",
        departure_date="2026-08-01",
    )

    itinerary = compose_itinerary_result(
        [successful_return],
        product_outcomes=[failed_outbound],
    )

    assert itinerary is None


def test_compose_deduplicates_candidates_and_weather_refs():
    blocks = _round_trip_blocks()
    duplicate_train = _train_block(
        block_id="train-out-new",
        origin="深圳",
        destination="上海",
        departure_date="2026-08-01",
        option_id="train-out-1",
        train_no="G100",
        price_minor=55000,
        duration_s=28800,
        departure_at="2026-08-01T07:00:00+08:00",
        arrival_at="2026-08-01T15:00:00+08:00",
    )
    blocks.extend(
        [
            duplicate_train,
            _weather_block(
                block_id="weather-old",
                location="上海市",
                dates=["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
            ),
            _weather_block(
                block_id="weather-new",
                location="上海市",
                dates=["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
            ),
        ]
    )

    itinerary = compose_itinerary_result(blocks)

    assert itinerary is not None
    lowest = itinerary.plans[0]
    outbound = next(section for section in lowest.sections if section.kind == "outbound_transport")
    weather = next(section for section in lowest.sections if section.kind == "destination_weather")
    assert outbound.result_refs[0].block_id == "train-out-new"
    assert weather.coverage == "full"
    assert [ref.block_id for ref in weather.result_refs] == ["weather-new"]


def test_compose_marks_weather_outside_range_and_excludes_wrong_city():
    blocks = [
        *_round_trip_blocks(),
        _weather_block(
            block_id="weather-wrong-city",
            location="深圳市",
            dates=["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
        ),
        _weather_block(
            block_id="weather-outside",
            location="上海市",
            dates=["2026-07-26", "2026-07-27", "2026-07-28", "2026-07-29"],
        ),
    ]

    itinerary = compose_itinerary_result(blocks)

    assert itinerary is not None
    assert itinerary.status == "degraded"
    weather_sections = [
        section for plan in itinerary.plans for section in plan.sections if section.kind == "destination_weather"
    ]
    assert weather_sections
    assert all(section.coverage == "outside_range" for section in weather_sections)
    assert all(section.result_refs[0].block_id == "weather-outside" for section in weather_sections)


def test_compose_only_includes_route_with_strict_destination_station_match():
    blocks = [
        *_round_trip_blocks(),
        _route_block(block_id="route-valid", city="上海", station="上海机场"),
        _route_block(block_id="route-wrong", city="深圳", station="深圳机场"),
    ]

    itinerary = compose_itinerary_result(blocks)

    assert itinerary is not None
    route_sections = [section for plan in itinerary.plans for section in plan.sections if section.kind == "local_route"]
    assert route_sections
    assert all(section.result_refs[0].block_id == "route-valid" for section in route_sections)


def test_compose_returns_none_without_travel_results():
    itinerary = compose_itinerary_result(
        [
            _weather_block(
                block_id="weather-only",
                location="上海市",
                dates=["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"],
            )
        ]
    )

    assert itinerary is None
