import unittest
from datetime import date, datetime, timezone

from app.schemas.chat import (
    PlaceResult,
    PlaceResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    TransitLeg,
    WeatherForecastDay,
    WeatherResultsBlock,
)
from app.services.stream.product_answer_validator import validate_product_answer
from app.services.stream.product_result_answer import (
    build_grounded_mixed_travel_answer,
    build_grounded_product_answer,
    build_product_tool_failure_answer,
    build_tool_repair_clarification,
    has_product_result_blocks,
    neutralize_product_provider_mentions,
)


class ProductResultAnswerTests(unittest.TestCase):
    def test_mixed_travel_deterministic_answer_requires_same_route_and_date(self):
        flight = _travel_block(
            block_type="flight_results",
            block_id="flight-out",
            origin="北京",
            destination="上海",
            departure_date="2026-08-29",
            option_id="flight-out-1",
            number="MU5101",
            duration_s=8100,
            price_minor=76000,
        )
        same_day_train = _travel_block(
            block_type="train_results",
            block_id="train-out",
            origin="北京",
            destination="上海",
            departure_date="2026-08-29",
            option_id="train-out-1",
            number="G1",
            duration_s=17640,
            price_minor=66100,
        )
        other_day_train = {
            **same_day_train,
            "departure_date": "2026-08-30",
        }

        answer = build_grounded_mixed_travel_answer([flight, same_day_train])

        self.assertIn("同时返回北京到上海", answer)
        self.assertIn("MU5101", answer)
        self.assertIn("G1", answer)
        self.assertEqual(build_grounded_mixed_travel_answer([flight, other_day_train]), "")

    def test_fallback_keeps_both_directions_when_more_than_four_product_blocks_exist(self):
        blocks = [
            _travel_block(
                block_type="flight_results",
                block_id="flight-out",
                origin="深圳",
                destination="上海",
                departure_date="2026-08-01",
                option_id="flight-out-1",
                number="ZH1001",
                duration_s=8400,
                price_minor=68000,
            ),
            _travel_block(
                block_type="train_results",
                block_id="train-out",
                origin="深圳",
                destination="上海",
                departure_date="2026-08-01",
                option_id="train-out-1",
                number="G100",
                duration_s=28800,
                price_minor=56000,
            ),
            _weather_result("weather-old"),
            _travel_block(
                block_type="flight_results",
                block_id="flight-return",
                origin="上海",
                destination="深圳",
                departure_date="2026-08-03",
                option_id="flight-return-1",
                number="ZH1002",
                duration_s=9000,
                price_minor=72000,
            ),
            _travel_block(
                block_type="train_results",
                block_id="train-return",
                origin="上海",
                destination="深圳",
                departure_date="2026-08-03",
                option_id="train-return-1",
                number="G101",
                duration_s=29400,
                price_minor=57000,
            ),
            _weather_result("weather-new"),
        ]

        answer = build_grounded_product_answer(blocks)

        self.assertIn("深圳到上海", answer)
        self.assertIn("上海到深圳", answer)

    def test_itinerary_fallback_uses_referenced_plan_and_weather_coverage(self):
        source_blocks = [
            _travel_block(
                block_type="train_results",
                block_id="train-out",
                origin="深圳",
                destination="上海",
                departure_date="2026-08-01",
                option_id="train-out-1",
                number="G100",
                duration_s=28800,
                price_minor=56000,
            ),
            _travel_block(
                block_type="train_results",
                block_id="train-return",
                origin="上海",
                destination="深圳",
                departure_date="2026-08-03",
                option_id="train-return-1",
                number="G101",
                duration_s=29400,
                price_minor=57000,
            ),
            _weather_result("weather-destination"),
        ]
        itinerary = _itinerary_result()

        answer = build_grounded_product_answer([*source_blocks, itinerary])

        self.assertIn("最低参考价组合", answer)
        self.assertIn("去程G100", answer)
        self.assertIn("返程G101", answer)
        self.assertIn("参考票价合计1130元", answer)
        self.assertIn("天气预报已完整覆盖行程日期", answer)
        self.assertIn("上海市天气预报", answer)
        self.assertNotIn("7月31日", answer)
        validation = validate_product_answer(answer, [*source_blocks, itinerary])
        self.assertTrue(validation.is_valid, validation.reason_code)
        invalid_travel_weekday = validate_product_answer(
            f"{answer}\n\n返程周日出发。",
            [*source_blocks, itinerary],
        )
        self.assertFalse(invalid_travel_weekday.is_valid)
        self.assertEqual(invalid_travel_weekday.reason_code, "unknown_travel_date")

    def test_itinerary_fallback_keeps_referenced_local_route(self):
        source_blocks = [
            _travel_block(
                block_type="train_results",
                block_id="train-out",
                origin="深圳",
                destination="上海",
                departure_date="2026-08-01",
                option_id="train-out-1",
                number="G100",
                duration_s=28800,
                price_minor=56000,
            ),
            _travel_block(
                block_type="train_results",
                block_id="train-return",
                origin="上海",
                destination="深圳",
                departure_date="2026-08-03",
                option_id="train-return-1",
                number="G101",
                duration_s=29400,
                price_minor=57000,
            ),
            _weather_result("weather-destination"),
            {
                "type": "route_results",
                "id": "route-local",
                "status": "success",
                "origin": {"label": "上海站"},
                "destination": {"label": "外滩"},
                "routes": [
                    {
                        "mode": "transit",
                        "transit_type": "subway",
                        "duration_s": 2100,
                        "transfers": 1,
                        "legs": [{"kind": "subway", "line_name": "地铁1号线"}],
                    }
                ],
                "limitations": ["路线时间和距离仅代表本次查询结果"],
            },
        ]
        itinerary = _itinerary_result()
        itinerary["plans"][0]["sections"].append(
            {
                "kind": "local_route",
                "coverage": None,
                "result_refs": [{"block_id": "route-local", "item_ids": []}],
            }
        )

        answer = build_grounded_product_answer([*source_blocks, itinerary])

        self.assertIn("上海站到外滩", answer)
        self.assertIn("地铁约 35 分钟", answer)
        self.assertIn("地铁1号线", answer)
        self.assertTrue(validate_product_answer(answer, [*source_blocks, itinerary]).is_valid)

    def test_itinerary_fallback_keeps_unreferenced_route_returned_in_same_round(self):
        source_blocks = [
            _travel_block(
                block_type="train_results",
                block_id="train-out",
                origin="深圳",
                destination="上海",
                departure_date="2026-08-01",
                option_id="train-out-1",
                number="G100",
                duration_s=28800,
                price_minor=56000,
            ),
            _travel_block(
                block_type="train_results",
                block_id="train-return",
                origin="上海",
                destination="深圳",
                departure_date="2026-08-03",
                option_id="train-return-1",
                number="G101",
                duration_s=29400,
                price_minor=57000,
            ),
            _weather_result("weather-destination"),
            {
                "type": "route_results",
                "id": "route-unreferenced",
                "status": "success",
                "origin": {"label": "上海虹桥站"},
                "destination": {"label": "外滩"},
                "routes": [{"mode": "driving", "duration_s": 2400, "distance_m": 21000}],
                "limitations": ["路线时间和距离仅代表本次查询结果"],
            },
        ]

        answer = build_grounded_product_answer([*source_blocks, _itinerary_result()])

        self.assertIn("上海虹桥站到外滩", answer)
        self.assertIn("驾车约 40 分钟、21 公里", answer)
        self.assertTrue(validate_product_answer(answer, [*source_blocks, _itinerary_result()]).is_valid)

    def test_itinerary_fallback_keeps_unreferenced_travel_type_returned_in_same_round(self):
        flight = _travel_block(
            block_type="flight_results",
            block_id="flight-out",
            origin="北京",
            destination="上海",
            departure_date="2026-08-29",
            option_id="flight-out-1",
            number="MU5101",
            duration_s=8100,
            price_minor=76000,
        )
        train = _travel_block(
            block_type="train_results",
            block_id="train-out",
            origin="北京",
            destination="上海",
            departure_date="2026-08-29",
            option_id="train-out-1",
            number="G1",
            duration_s=16200,
            price_minor=55300,
        )
        itinerary = {
            "type": "itinerary_results",
            "id": "itinerary-one-way",
            "status": "success",
            "trip_type": "one_way",
            "origin": "北京",
            "destination": "上海",
            "start_date": "2026-08-29",
            "end_date": None,
            "plans": [
                {
                    "id": "fastest",
                    "title": "最快方案",
                    "status": "complete",
                    "strategy": "fastest_known_duration",
                    "known_cost": {"currency": "CNY", "amount_minor": 76000},
                    "known_duration_s": 8100,
                    "sections": [
                        {
                            "kind": "outbound_transport",
                            "coverage": None,
                            "result_refs": [{"block_id": "flight-out", "item_ids": ["flight-out-1"]}],
                        }
                    ],
                }
            ],
            "limitations": [],
        }
        blocks = [flight, train, itinerary]

        answer = build_grounded_product_answer(blocks)

        self.assertIn("同时返回北京到上海", answer)
        self.assertIn("MU5101", answer)
        self.assertIn("G1", answer)
        validation = validate_product_answer(answer, blocks)
        self.assertTrue(validation.is_valid, validation.reason_code)

    def test_weather_fallback_uses_only_forecast_fields_and_safe_advice(self):
        block = WeatherResultsBlock(
            type="weather_results",
            schema_version=1,
            provider="amap",
            status="degraded",
            query="南山区",
            resolved_location="南山区",
            day_count=2,
            forecast_days=[
                WeatherForecastDay(
                    date=date(2026, 7, 23),
                    weekday=4,
                    day_weather="多云",
                    night_weather="阵雨",
                    high_c=32,
                    low_c=27,
                ),
                WeatherForecastDay(
                    date=date(2026, 7, 24),
                    weekday=5,
                    day_weather="雷阵雨",
                    night_weather="多云",
                    high_c=31,
                    low_c=26,
                    day_wind_direction="南",
                    day_wind_power="≤3",
                ),
            ],
            fetched_at=datetime(2026, 7, 23, 8, tzinfo=timezone.utc),
            limitations=["天气预报按行政区提供，不代表具体建筑物", "仅返回 2 天有效预报"],
        )

        answer = build_grounded_product_answer([block])

        self.assertTrue(has_product_result_blocks([block]))
        self.assertIn("南山区天气预报", answer)
        self.assertIn("7月23日（周四）白天多云、夜间阵雨，27–32℃", answer)
        self.assertIn("7月24日（周五）白天雷阵雨、夜间多云，26–31℃", answer)
        self.assertIn("建议携带雨具", answer)
        self.assertTrue(validate_product_answer(answer, [block]).is_valid)
        self.assertNotIn("高德", answer)
        for unsupported in ("湿度", "空气质量", "降雨概率", "预警"):
            self.assertNotIn(unsupported, answer)

    def test_weather_fallback_answers_requested_activity_condition_without_inference(self):
        block = _weather_result("weather-activity")
        messages = [
            {
                "role": "user",
                "content": "2026年8月2日上海天气怎么样，适合上午骑行吗？",
            }
        ]

        answer = build_grounded_product_answer([block], messages=messages)

        self.assertIn("8月2日（周日）白天阵雨", answer)
        self.assertNotIn("8月1日", answer)
        self.assertNotIn("8月3日", answer)
        self.assertIn("只有白天和夜间粒度，无法确认上午这一细分时段", answer)
        self.assertIn("如果你的条件是上午骑行时避开降水", answer)
        self.assertIn("不满足这一条件", answer)
        self.assertNotIn("建议", answer)
        self.assertNotIn("适合骑行", answer)
        validation = validate_product_answer(answer, [block], messages=messages)
        self.assertTrue(validation.is_valid, validation.reason_code)

    def test_geolocation_failure_answer_is_product_neutral(self):
        answer = build_product_tool_failure_answer(
            [
                {
                    "role": "tool",
                    "content": (
                        '{"error_code":"context_required_not_provided","context_type":"geolocation",'
                        '"context_status":"denied"}'
                    ),
                }
            ]
        )

        self.assertIn("依赖当前位置的查询尚未执行", answer)
        self.assertNotIn("路线查询尚未执行", answer)
        self.assertNotIn("起点", answer)

    def test_neutralize_product_provider_mentions_keeps_sentences_natural(self):
        answer = neutralize_product_provider_mentions(
            "根据高德返回的路线结果，驾车更快。"
            "路线时间和距离仅代表高德本次返回结果。"
            "未返回的费用信息，本次高德结果无法确认。"
        )

        self.assertNotIn("高德", answer)
        self.assertIn("根据本次查询返回的路线结果", answer)
        self.assertIn("路线时间和距离仅代表本次查询结果", answer)
        self.assertIn("未返回的费用信息，本次查询结果无法确认", answer)

        entity_block = PlaceResultsBlock(
            type="place_results",
            schema_version=1,
            provider="amap",
            query="商场",
            status="success",
            result_count=1,
            places=[PlaceResult(name="高德置地广场")],
        )
        entity_answer = neutralize_product_provider_mentions(
            "高德置地广场是候选地点。高德返回了路线结果。",
            [entity_block],
        )
        self.assertIn("高德置地广场", entity_answer)
        self.assertIn("本次查询返回了路线结果", entity_answer)

    def test_neutralize_product_provider_mentions_handles_recent_generated_variants(self):
        answer = neutralize_product_provider_mentions(
            "高德结果无法确认排队情况。"
            "高德当前未能返回公共交通方案。"
            "高德预估行驶时间约15分钟。"
            "根据高德在民治附近查到的结果，可以考虑这些地点。"
        )

        self.assertNotIn("高德", answer)
        self.assertIn("本次查询结果无法确认排队情况", answer)
        self.assertIn("本次查询未能返回公共交通方案", answer)
        self.assertIn("本次查询预估行驶时间约15分钟", answer)
        self.assertIn("根据地图服务在民治附近查到的结果", answer)

    def test_product_tool_failure_answer_does_not_expose_provider_name(self):
        answer = build_product_tool_failure_answer()

        self.assertNotIn("高德", answer)
        self.assertIn("本次未取得可用的地点或路线数据", answer)

    def test_argument_repair_clarification_is_specific_and_provider_neutral(self):
        answer = build_tool_repair_clarification(
            {
                "weather_forecast": {
                    "required_fields": ["location"],
                    "requires_user_input": True,
                }
            }
        )

        self.assertIn("请补充包含城市的完整地点", answer)
        self.assertIn("不会猜测", answer)
        self.assertNotIn("高德", answer)
        self.assertNotIn("深圳", answer)
        self.assertNotIn("鹤岗", answer)

    def test_transit_answer_uses_transit_type_and_compact_line_names(self):
        block = RouteResultsBlock(
            type="route_results",
            schema_version=1,
            provider="amap",
            status="success",
            origin=RouteEndpoint(label="民治站"),
            destination=RouteEndpoint(label="市民中心"),
            routes=[
                RouteOption(
                    mode="transit",
                    transit_type="subway",
                    duration_s=1800,
                    transfers=1,
                    legs=[
                        TransitLeg(kind="walking"),
                        TransitLeg(kind="subway", line_name="地铁5号线"),
                        TransitLeg(kind="subway", line_name="地铁2号线"),
                    ],
                )
            ],
        )

        answer = build_grounded_product_answer([block])

        self.assertIn("地铁约 30 分钟", answer)
        self.assertIn("地铁5号线→地铁2号线", answer)
        self.assertNotIn("步行→", answer)
        self.assertNotIn("公里", answer)
        self.assertNotIn(" 米", answer)

    def test_place_answer_only_uses_structured_fields_and_limitations(self):
        block = PlaceResultsBlock(
            type="place_results",
            schema_version=1,
            provider="amap",
            query="烤肉|火锅",
            near="深圳民治",
            status="success",
            result_count=2,
            places=[
                PlaceResult(name="炭火一号", rating=4.7, reference_cost_yuan=88),
                PlaceResult(name="沸腾火锅", address="民治大道 1 号"),
            ],
            limitations=["不包含实时排队或空位信息", "参考消费不代表人均或实时价格"],
        )

        answer = build_grounded_product_answer([block])

        self.assertTrue(has_product_result_blocks([block]))
        self.assertIn("本次查询返回 2 个", answer)
        self.assertNotIn("高德", answer)
        self.assertIn("炭火一号、沸腾火锅", answer)
        self.assertIn("不包含实时排队或空位信息", answer)
        self.assertNotIn("三人", answer)
        self.assertNotIn("预算", answer)
        self.assertNotIn("适合", answer)

    def test_route_answer_formats_only_returned_route_facts(self):
        block = RouteResultsBlock(
            type="route_results",
            schema_version=1,
            provider="amap",
            status="success",
            origin=RouteEndpoint(label="深圳民治"),
            destination=RouteEndpoint(label="深圳湾公园"),
            routes=[
                RouteOption(mode="driving", distance_m=9645, duration_s=1642),
                RouteOption(mode="transit", distance_m=8887, duration_s=2324, transfers=1),
                RouteOption(mode="walking", distance_m=15589, duration_s=12471),
            ],
            limitations=["路线时间和距离仅代表本次查询结果"],
        )

        answer = build_grounded_product_answer([block])

        self.assertIn("深圳民治到深圳湾公园", answer)
        self.assertIn("驾车约 27 分钟、9.6 公里", answer)
        self.assertIn("公交约 39 分钟、换乘 1 次", answer)
        self.assertNotIn("8.9 公里", answer)
        self.assertIn("步行约 208 分钟、16 公里", answer)
        self.assertIn("如果优先考虑本次返回的用时，建议选择驾车", answer)
        self.assertIn("如果更倾向公共交通，可选择公交方案", answer)
        self.assertNotIn("高德", answer)
        for unsupported in ("停车", "路况", "候车", "费用", "拥堵"):
            self.assertNotIn(unsupported, answer)


def _travel_block(
    *,
    block_type: str,
    block_id: str,
    origin: str,
    destination: str,
    departure_date: str,
    option_id: str,
    number: str,
    duration_s: int,
    price_minor: int,
) -> dict:
    collection = "flights" if block_type == "flight_results" else "trains"
    number_key = "flight_no" if block_type == "flight_results" else "train_no"
    return {
        "type": block_type,
        "id": block_id,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        collection: [
            {
                "option_id": option_id,
                number_key: number,
                "departure": {
                    "city": origin,
                    "station_name": f"{origin}站",
                    "scheduled_at": f"{departure_date}T08:00:00+08:00",
                },
                "arrival": {
                    "city": destination,
                    "station_name": f"{destination}站",
                    "scheduled_at": f"{departure_date}T16:00:00+08:00",
                },
                "duration_s": duration_s,
                "price": {"currency": "CNY", "amount_minor": price_minor},
            }
        ],
        "limitations": [],
    }


def _weather_result(block_id: str) -> dict:
    return {
        "type": "weather_results",
        "id": block_id,
        "resolved_location": "上海市",
        "forecast_days": [
            {
                "date": "2026-08-01",
                "weekday": 6,
                "day_weather": "多云",
                "night_weather": "多云",
                "low_c": 27,
                "high_c": 34,
            },
            {
                "date": "2026-08-02",
                "weekday": 7,
                "day_weather": "阵雨",
                "night_weather": "多云",
                "low_c": 26,
                "high_c": 32,
            },
            {
                "date": "2026-08-03",
                "weekday": 1,
                "day_weather": "多云",
                "night_weather": "多云",
                "low_c": 27,
                "high_c": 33,
            },
            {
                "date": "2026-07-31",
                "weekday": 5,
                "day_weather": "晴",
                "night_weather": "晴",
                "low_c": 28,
                "high_c": 35,
            },
        ],
        "limitations": [],
    }


def _itinerary_result() -> dict:
    return {
        "type": "itinerary_results",
        "id": "itinerary-1",
        "status": "success",
        "trip_type": "round_trip",
        "origin": "深圳",
        "destination": "上海",
        "start_date": "2026-08-01",
        "end_date": "2026-08-03",
        "plans": [
            {
                "id": "lowest-price",
                "title": "最低参考价组合",
                "status": "complete",
                "strategy": "lowest_reference_price",
                "known_cost": {"currency": "CNY", "amount_minor": 113000},
                "known_duration_s": 58200,
                "sections": [
                    {
                        "kind": "outbound_transport",
                        "coverage": None,
                        "result_refs": [{"block_id": "train-out", "item_ids": ["train-out-1"]}],
                    },
                    {
                        "kind": "return_transport",
                        "coverage": None,
                        "result_refs": [{"block_id": "train-return", "item_ids": ["train-return-1"]}],
                    },
                    {
                        "kind": "destination_weather",
                        "coverage": "full",
                        "result_refs": [{"block_id": "weather-destination", "item_ids": []}],
                    },
                ],
            }
        ],
        "limitations": ["班次时长不包含前后接驳、值机、安检或候车时间"],
    }


if __name__ == "__main__":
    unittest.main()
