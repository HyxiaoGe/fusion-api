import unittest
from datetime import datetime, timezone

from app.schemas.chat import (
    FlightResultsBlock,
    ItineraryResultsBlock,
    PlaceResultsBlock,
    SearchBlock,
    TextBlock,
    UnsupportedContentBlock,
    WeatherResultsBlock,
)
from app.schemas.content_block_registry import deserialize_content_blocks


class ContentBlockRegistryTests(unittest.TestCase):
    def test_deserialize_blocks_isolates_unknown_corrupt_and_future_blocks(self):
        raw_blocks = [
            {"type": "text", "id": "text-1", "text": "保留的回答"},
            {"type": "future_widget", "id": "future-1", "payload": {"value": 1}},
            {
                "type": "place_results",
                "id": "place-future",
                "schema_version": 2,
                "provider": "future-provider",
            },
            {"type": "route_results", "id": "route-corrupt", "schema_version": 1},
            {"type": "text", "id": "text-2", "text": "后续回答"},
        ]

        blocks = deserialize_content_blocks(raw_blocks)

        self.assertEqual(
            [block.type for block in blocks],
            [
                "text",
                "unsupported_result",
                "unsupported_result",
                "unsupported_result",
                "text",
            ],
        )
        self.assertEqual(blocks[0], TextBlock(type="text", id="text-1", text="保留的回答"))
        self.assertEqual(blocks[1].source_type, "future_widget")
        self.assertEqual(blocks[1].reason, "unsupported_type")
        self.assertEqual(blocks[2].source_type, "place_results")
        self.assertEqual(blocks[2].source_schema_version, 2)
        self.assertEqual(blocks[2].reason, "unsupported_version")
        self.assertEqual(blocks[3].source_type, "route_results")
        self.assertEqual(blocks[3].source_schema_version, 1)
        self.assertEqual(blocks[3].reason, "invalid_payload")
        self.assertEqual(blocks[4], TextBlock(type="text", id="text-2", text="后续回答"))
        self.assertEqual(raw_blocks[1]["type"], "future_widget")

    def test_unsupported_fallback_is_safe_stable_and_unique(self):
        raw_blocks = [
            {"type": "future_widget", "payload": {"provider": "PRIVATE_PROVIDER"}},
            {"type": "future_widget", "payload": {"provider": "PRIVATE_PROVIDER"}},
            {"type": "future_widget", "id": "duplicate-id", "tool_name": "PRIVATE_TOOL"},
            {"type": "future_widget", "id": "duplicate-id", "tool_name": "PRIVATE_TOOL"},
        ]

        first = deserialize_content_blocks(raw_blocks)
        second = deserialize_content_blocks(raw_blocks)

        self.assertTrue(all(isinstance(block, UnsupportedContentBlock) for block in first))
        self.assertEqual([block.id for block in first], [block.id for block in second])
        self.assertEqual(len({block.id for block in first}), len(first))
        serialized = [block.model_dump(mode="json") for block in first]
        self.assertNotIn("PRIVATE_PROVIDER", str(serialized))
        self.assertNotIn("PRIVATE_TOOL", str(serialized))
        self.assertTrue(
            all(
                set(block)
                <= {
                    "type",
                    "id",
                    "source_type",
                    "source_schema_version",
                    "reason",
                }
                for block in serialized
            )
        )

    def test_deserialize_blocks_keeps_legacy_search_defaults(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "search",
                    "tool_call_log_id": "log-legacy",
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], SearchBlock)
        self.assertEqual(blocks[0].query, "")
        self.assertEqual(blocks[0].sources, [])
        self.assertEqual(blocks[0].tool_call_log_id, "log-legacy")

    def test_deserialize_blocks_accepts_current_rich_schema_version(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "place_results",
                    "id": "place-1",
                    "schema_version": 1,
                    "provider": "amap",
                    "query": "咖啡",
                    "status": "success",
                    "result_count": 0,
                    "places": [],
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], PlaceResultsBlock)
        self.assertEqual(blocks[0].schema_version, 1)

    def test_deserialize_blocks_accepts_weather_schema_v1(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "weather_results",
                    "id": "weather-1",
                    "schema_version": 1,
                    "provider": "amap",
                    "status": "degraded",
                    "query": "深圳市",
                    "resolved_location": "深圳市",
                    "day_count": 1,
                    "forecast_days": [
                        {
                            "date": "2026-07-23",
                            "weekday": 4,
                            "day_weather": "多云",
                            "night_weather": "阵雨",
                            "high_c": 32,
                            "low_c": 27,
                        }
                    ],
                    "fetched_at": datetime(2026, 7, 23, 8, tzinfo=timezone.utc).isoformat(),
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], WeatherResultsBlock)
        self.assertEqual(blocks[0].forecast_days[0].weekday, 4)

    def test_deserialize_blocks_treats_non_list_payload_as_empty(self):
        self.assertEqual(deserialize_content_blocks(None), [])
        self.assertEqual(deserialize_content_blocks({"type": "text"}), [])
        self.assertEqual(deserialize_content_blocks("invalid"), [])

    def test_malformed_block_becomes_fallback_instead_of_empty_message(self):
        blocks = deserialize_content_blocks([{"type": "place_results", "schema_version": 9}])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].type, "unsupported_result")
        self.assertEqual(blocks[0].source_type, "place_results")
        self.assertEqual(blocks[0].source_schema_version, 9)
        self.assertEqual(blocks[0].reason, "unsupported_version")

    def test_deserialize_blocks_accepts_itinerary_with_valid_source_reference(self):
        raw_blocks = [
            _flight_block_payload(),
            _itinerary_payload(),
        ]

        blocks = deserialize_content_blocks(raw_blocks)

        self.assertIsInstance(blocks[0], FlightResultsBlock)
        self.assertIsInstance(blocks[1], ItineraryResultsBlock)
        self.assertEqual(blocks[1].plans[0].sections[0].result_refs[0].item_ids, ["flight-1"])

    def test_deserialize_blocks_rejects_itinerary_with_missing_source_without_hiding_sources(self):
        raw_blocks = [
            _flight_block_payload(),
            _itinerary_payload(source_block_id="missing-flight"),
        ]

        blocks = deserialize_content_blocks(raw_blocks)

        self.assertIsInstance(blocks[0], FlightResultsBlock)
        self.assertIsInstance(blocks[1], UnsupportedContentBlock)
        self.assertEqual(blocks[1].source_type, "itinerary_results")
        self.assertEqual(blocks[1].reason, "invalid_payload")

    def test_deserialize_blocks_rejects_itinerary_with_unknown_item_reference(self):
        raw_blocks = [
            _flight_block_payload(),
            _itinerary_payload(item_id="unknown-flight"),
        ]

        blocks = deserialize_content_blocks(raw_blocks)

        self.assertIsInstance(blocks[0], FlightResultsBlock)
        self.assertIsInstance(blocks[1], UnsupportedContentBlock)
        self.assertEqual(blocks[1].reason, "invalid_payload")

    def test_deserialize_blocks_treats_future_itinerary_version_as_unsupported(self):
        payload = _itinerary_payload()
        payload["schema_version"] = 2

        blocks = deserialize_content_blocks([payload])

        self.assertIsInstance(blocks[0], UnsupportedContentBlock)
        self.assertEqual(blocks[0].source_type, "itinerary_results")
        self.assertEqual(blocks[0].source_schema_version, 2)
        self.assertEqual(blocks[0].reason, "unsupported_version")

    def test_deserialize_blocks_rejects_round_trip_without_end_date(self):
        payload = _itinerary_payload()
        payload["trip_type"] = "round_trip"
        payload["end_date"] = None

        blocks = deserialize_content_blocks([_flight_block_payload(), payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)
        self.assertEqual(blocks[1].reason, "invalid_payload")

    def test_deserialize_blocks_rejects_one_way_with_end_date(self):
        payload = _itinerary_payload()
        payload["end_date"] = "2026-08-02"

        blocks = deserialize_content_blocks([_flight_block_payload(), payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)
        self.assertEqual(blocks[1].reason, "invalid_payload")

    def test_deserialize_blocks_rejects_transport_reference_with_wrong_direction_or_date(self):
        wrong_direction = _flight_block_payload(
            block_id="flight-wrong-direction",
            option_id="flight-wrong-direction-1",
            origin="上海",
            destination="深圳",
        )
        wrong_date = _flight_block_payload(
            block_id="flight-wrong-date",
            option_id="flight-wrong-date-1",
            departure_date="2026-08-02",
        )

        direction_blocks = deserialize_content_blocks(
            [
                wrong_direction,
                _itinerary_payload(
                    source_block_id="flight-wrong-direction",
                    item_id="flight-wrong-direction-1",
                ),
            ]
        )
        date_blocks = deserialize_content_blocks(
            [
                wrong_date,
                _itinerary_payload(
                    source_block_id="flight-wrong-date",
                    item_id="flight-wrong-date-1",
                ),
            ]
        )

        self.assertIsInstance(direction_blocks[1], UnsupportedContentBlock)
        self.assertIsInstance(date_blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_forged_plan_cost_or_duration(self):
        forged_cost = _itinerary_payload()
        forged_cost["plans"][0]["known_cost"]["amount_minor"] = 1
        forged_duration = _itinerary_payload()
        forged_duration["plans"][0]["known_duration_s"] = 1

        cost_blocks = deserialize_content_blocks([_flight_block_payload(), forged_cost])
        duration_blocks = deserialize_content_blocks([_flight_block_payload(), forged_duration])

        self.assertIsInstance(cost_blocks[1], UnsupportedContentBlock)
        self.assertIsInstance(duration_blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_requires_empty_known_cost_when_selected_price_is_missing(self):
        source = _flight_block_payload(price_minor=None)
        payload = _itinerary_payload()

        blocks = deserialize_content_blocks([source, payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_recomputes_weather_coverage_and_destination(self):
        wrong_coverage = _itinerary_payload()
        wrong_coverage["plans"][0]["sections"].append(_weather_section_payload(coverage="outside_range"))
        wrong_location = _itinerary_payload()
        wrong_location["plans"][0]["sections"].append(_weather_section_payload())

        coverage_blocks = deserialize_content_blocks(
            [
                _flight_block_payload(),
                _weather_block_payload(),
                wrong_coverage,
            ]
        )
        location_blocks = deserialize_content_blocks(
            [
                _flight_block_payload(),
                _weather_block_payload(location="深圳市"),
                wrong_location,
            ]
        )

        self.assertIsInstance(coverage_blocks[2], UnsupportedContentBlock)
        self.assertIsInstance(location_blocks[2], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_inconsistent_root_status(self):
        partial = _itinerary_payload()
        partial["plans"][0]["status"] = "partial"
        unavailable = _itinerary_payload()
        unavailable["availability"][0]["status"] = "unavailable"

        partial_blocks = deserialize_content_blocks([_flight_block_payload(), partial])
        unavailable_blocks = deserialize_content_blocks([_flight_block_payload(), unavailable])

        self.assertIsInstance(partial_blocks[1], UnsupportedContentBlock)
        self.assertIsInstance(unavailable_blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_complete_round_trip_plan_without_return_section(self):
        payload = _itinerary_payload()
        payload["trip_type"] = "round_trip"
        payload["end_date"] = "2026-08-03"

        blocks = deserialize_content_blocks([_flight_block_payload(), payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_duplicate_result_ref_in_same_section(self):
        payload = _itinerary_payload()
        result_ref = dict(payload["plans"][0]["sections"][0]["result_refs"][0])
        payload["plans"][0]["sections"][0]["result_refs"].append(result_ref)

        blocks = deserialize_content_blocks([_flight_block_payload(), payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_duplicate_source_option_ids(self):
        source = _flight_block_payload()
        source["flights"].append(dict(source["flights"][0]))
        source["result_count"] = 2

        blocks = deserialize_content_blocks([source, _itinerary_payload()])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_selected_source_marked_unavailable(self):
        payload = _itinerary_payload()
        payload["status"] = "degraded"
        payload["availability"][0]["status"] = "unavailable"

        blocks = deserialize_content_blocks([_flight_block_payload(), payload])

        self.assertIsInstance(blocks[1], UnsupportedContentBlock)

    def test_deserialize_blocks_rejects_unrelated_local_route(self):
        payload = _itinerary_payload()
        payload["plans"][0]["sections"].append(
            {
                "id": "local-route",
                "kind": "local_route",
                "status": "complete",
                "title": "当地路线",
                "coverage": None,
                "result_refs": [{"block_id": "route-unrelated", "item_ids": []}],
            }
        )
        payload["availability"].append(
            {
                "journey": "local_route",
                "mode": "route",
                "status": "available",
            }
        )

        blocks = deserialize_content_blocks(
            [
                _flight_block_payload(),
                _route_block_payload(
                    block_id="route-unrelated",
                    origin_label="广州南站",
                    destination_label="广州塔",
                    city="广州市",
                ),
                payload,
            ]
        )

        self.assertIsInstance(blocks[2], UnsupportedContentBlock)


def _flight_block_payload(
    *,
    block_id: str = "flight-outbound",
    option_id: str = "flight-1",
    origin: str = "深圳",
    destination: str = "上海",
    departure_date: str = "2026-08-01",
    price_minor: int | None = 68000,
) -> dict:
    price = {"currency": "CNY", "amount_minor": price_minor} if price_minor is not None else None
    return {
        "type": "flight_results",
        "id": block_id,
        "schema_version": 1,
        "provider": "flyai",
        "status": "success",
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "observed_at": "2026-07-26T08:00:00+08:00",
        "result_count": 1,
        "flights": [
            {
                "option_id": option_id,
                "airline_name": "测试航空",
                "flight_no": "ZH9501",
                "departure": {
                    "city": origin,
                    "station_name": f"{origin}机场",
                    "scheduled_at": f"{departure_date}T08:00:00+08:00",
                },
                "arrival": {
                    "city": destination,
                    "station_name": f"{destination}机场",
                    "scheduled_at": f"{departure_date}T10:20:00+08:00",
                },
                "duration_s": 8400,
                "stops": 0,
                "price": price,
            }
        ],
    }


def _route_block_payload(
    *,
    block_id: str,
    origin_label: str,
    destination_label: str,
    city: str,
) -> dict:
    return {
        "type": "route_results",
        "id": block_id,
        "schema_version": 1,
        "provider": "amap",
        "status": "success",
        "origin": {"label": origin_label, "city": city},
        "destination": {"label": destination_label, "city": city},
        "routes": [{"mode": "transit", "duration_s": 1800}],
        "unavailable_modes": [],
        "limitations": [],
    }


def _itinerary_payload(
    *,
    source_block_id: str = "flight-outbound",
    item_id: str = "flight-1",
) -> dict:
    return {
        "type": "itinerary_results",
        "id": "blk-itinerary",
        "schema_version": 1,
        "provider": "fusion",
        "status": "success",
        "trip_type": "one_way",
        "origin": "深圳",
        "destination": "上海",
        "start_date": "2026-08-01",
        "end_date": None,
        "recommended_plan_id": None,
        "plans": [
            {
                "id": "lowest-price",
                "title": "最低参考价方案",
                "status": "complete",
                "strategy": "lowest_reference_price",
                "tags": ["lowest_reference_price"],
                "known_cost": {"currency": "CNY", "amount_minor": 68000},
                "known_duration_s": 8400,
                "sections": [
                    {
                        "id": "outbound",
                        "kind": "outbound_transport",
                        "status": "complete",
                        "title": "去程",
                        "coverage": None,
                        "result_refs": [
                            {
                                "block_id": source_block_id,
                                "item_ids": [item_id],
                            }
                        ],
                    }
                ],
            }
        ],
        "availability": [
            {
                "journey": "outbound",
                "mode": "flight",
                "status": "available",
            }
        ],
        "limitations": ["参考票价不等于完整旅行总预算"],
    }


def _weather_block_payload(*, location: str = "上海市") -> dict:
    return {
        "type": "weather_results",
        "id": "weather-destination",
        "schema_version": 1,
        "provider": "amap",
        "status": "degraded",
        "query": location,
        "resolved_location": location,
        "day_count": 1,
        "forecast_days": [
            {
                "date": "2026-08-01",
                "weekday": 6,
                "day_weather": "晴",
                "night_weather": "多云",
                "high_c": 31,
                "low_c": 25,
            }
        ],
        "fetched_at": "2026-07-26T08:00:00+08:00",
    }


def _weather_section_payload(*, coverage: str = "full") -> dict:
    return {
        "id": "destination-weather",
        "kind": "destination_weather",
        "status": "complete" if coverage == "full" else "partial",
        "title": "目的地天气",
        "coverage": coverage,
        "result_refs": [{"block_id": "weather-destination", "item_ids": []}],
    }


if __name__ == "__main__":
    unittest.main()
