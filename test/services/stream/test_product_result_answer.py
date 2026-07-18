import unittest

from app.schemas.chat import (
    PlaceResult,
    PlaceResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    TransitLeg,
)
from app.services.stream.product_result_answer import (
    build_grounded_product_answer,
    has_product_result_blocks,
)


class ProductResultAnswerTests(unittest.TestCase):
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
        self.assertIn("高德返回 2 个", answer)
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
            limitations=["路线时间和距离仅代表高德本次返回结果"],
        )

        answer = build_grounded_product_answer([block])

        self.assertIn("深圳民治到深圳湾公园", answer)
        self.assertIn("驾车约 27 分钟、9.6 公里", answer)
        self.assertIn("公交约 39 分钟、换乘 1 次", answer)
        self.assertNotIn("8.9 公里", answer)
        self.assertIn("步行约 208 分钟、16 公里", answer)
        self.assertIn("如果优先考虑本次返回的用时，建议选择驾车", answer)
        self.assertIn("如果更倾向公共交通，可选择公交方案", answer)
        for unsupported in ("停车", "路况", "候车", "费用", "拥堵"):
            self.assertNotIn(unsupported, answer)


if __name__ == "__main__":
    unittest.main()
