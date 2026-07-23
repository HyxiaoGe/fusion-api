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
    build_grounded_product_answer,
    build_product_tool_failure_answer,
    has_product_result_blocks,
    neutralize_product_provider_mentions,
)


class ProductResultAnswerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
