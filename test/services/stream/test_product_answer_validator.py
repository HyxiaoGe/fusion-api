import unittest

from app.schemas.chat import (
    PlaceResult,
    PlaceResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    TransitLeg,
)
from app.services.stream.product_answer_validator import (
    repair_unsupported_product_answer,
    validate_product_answer,
)


def _place_block():
    return PlaceResultsBlock(
        type="place_results",
        schema_version=1,
        provider="amap",
        query="烤肉",
        near="深圳民治",
        status="success",
        result_count=1,
        places=[PlaceResult(name="炭火一号", rating=4.7, reference_cost_yuan=88)],
        limitations=["不包含实时排队或空位信息", "参考消费不代表人均或实时价格"],
    )


def _two_place_block():
    block = _place_block()
    block.result_count = 2
    block.places.append(PlaceResult(name="金杆桌球", rating=4.1))
    return block


def _route_block():
    return RouteResultsBlock(
        type="route_results",
        schema_version=1,
        provider="amap",
        status="success",
        origin=RouteEndpoint(label="民治站"),
        destination=RouteEndpoint(label="雅宝站"),
        routes=[
            RouteOption(mode="driving", duration_s=840, distance_m=6200, toll_yuan=5),
            RouteOption(
                mode="transit",
                transit_type="subway",
                duration_s=1920,
                walking_distance_m=420,
                transfers=1,
                legs=[
                    TransitLeg(
                        kind="subway",
                        line_name="地铁5号线",
                        departure_stop="民治站",
                        arrival_stop="五和站",
                        entrance="A口",
                    ),
                    TransitLeg(
                        kind="subway",
                        line_name="地铁10号线",
                        departure_stop="五和站",
                        arrival_stop="雅宝站",
                        exit="D口",
                    ),
                ],
            ),
        ],
        limitations=["路线时间和距离仅代表高德本次返回结果"],
    )


class ProductAnswerValidatorTests(unittest.TestCase):
    def test_valid_llm_prose_with_derived_difference_and_explicit_limits_is_retained(self):
        answer = (
            "结论：如果更看重用时，可以优先驾车。高德本次返回驾车约14分钟、公交约32分钟，"
            "两者相差18分钟；公交可乘地铁5号线换乘地铁10号线，共换乘1次。"
            "实时拥堵和公交票价本次未提供，建议出发前核实。"
        )

        validation = validate_product_answer(answer, [_route_block()])

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.reason_code, "ok")

    def test_derived_difference_cannot_be_reused_as_direct_leg_duration(self):
        route = _route_block()
        route.routes[1].alternatives = [RouteOption(mode="transit", duration_s=2280)]

        validation = validate_product_answer("地铁5号线到五和站约6分钟。", [route])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "numeric_mismatch")

    def test_route_difference_is_bound_to_the_compared_modes(self):
        route = _route_block()
        route.routes[1].alternatives = [RouteOption(mode="transit", duration_s=2280)]
        cases = (
            ("驾车比公交快18分钟。", True),
            ("驾车比公交快6分钟。", False),
            ("公交两个方案相差6分钟。", True),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(validate_product_answer(answer, [route]).is_valid, expected)

    def test_route_comparison_direction_must_match_returned_values(self):
        cases = (
            ("驾车比公交快18分钟。", True),
            ("公交比驾车快18分钟。", False),
            ("驾车比公交更快。", True),
            ("公交比驾车更快。", False),
            ("本次返回中驾车用时最短。", True),
            ("本次返回中公交用时最短。", False),
            ("本次返回中公交用时最长。", True),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(validate_product_answer(answer, [_route_block()]).is_valid, expected)

    def test_equal_route_values_cannot_be_described_as_faster(self):
        route = _route_block()
        route.routes[1].duration_s = route.routes[0].duration_s

        validation = validate_product_answer("驾车比公交更快。", [route])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "numeric_mismatch")

    def test_route_stop_and_access_names_must_come_from_structured_result(self):
        cases = (
            ("从民治站乘地铁5号线到五和站，再换乘地铁10号线到雅宝站。", True),
            ("从A口进站，最后从D口出站。", True),
            ("公交在深圳北站换乘地铁5号线。", False),
            ("深圳北站换乘地铁5号线。", False),
            ("从C口进站乘地铁5号线。", False),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(validate_product_answer(answer, [_route_block()]).is_valid, expected)

    def test_transit_walking_distance_cannot_be_compared_as_route_total_distance(self):
        validation = validate_product_answer("公交比驾车少5.78公里。", [_route_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "numeric_mismatch")

    def test_transit_walking_distance_requires_explicit_walking_semantics(self):
        cases = (
            ("公交方案步行约420米。", True),
            ("公交方案全程约420米。", False),
            ("公交方案约420米。", False),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(validate_product_answer(answer, [_route_block()]).is_valid, expected)

    def test_markdown_table_is_removed_while_grounded_prose_is_repaired(self):
        answer = (
            "结论：驾车约14分钟，公交约32分钟。\n\n| 方案 | 耗时 |\n| --- | --- |\n| 驾车 | 14分钟 |\n| 地铁 | 32分钟 |"
        )

        validation = validate_product_answer(answer, [_route_block()])
        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_format")
        self.assertEqual(reason_code, "ok")
        self.assertEqual(repaired, "结论：驾车约14分钟，公交约32分钟。")
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_markdown_table_only_still_requires_deterministic_fallback(self):
        answer = "| 方案 | 耗时 |\n| --- | --- |\n| 驾车 | 14分钟 |\n| 地铁 | 32分钟 |"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertIsNone(repaired)
        self.assertEqual(reason_code, "unsupported_format")

    def test_markdown_cleanup_cannot_turn_empty_headings_into_punctuation_answer(self):
        answer = "## 上午高铁推荐\n### 车次列表\n| 车次 | 耗时 |\n| --- | --- |\n| G426 | 2小时 |"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])
        punctuation_validation = validate_product_answer("。", [_route_block()])

        self.assertIsNone(repaired)
        self.assertEqual(reason_code, "not_repairable")
        self.assertFalse(punctuation_validation.is_valid)
        self.assertEqual(punctuation_validation.reason_code, "empty_answer")

    def test_markdown_repair_drops_unsafe_whole_sentence_instead_of_leaving_fragment(self):
        answer = (
            "## 方案概览\n"
            "### 路线明细\n"
            "| 方案 | 耗时 |\n"
            "| --- | --- |\n"
            "| 驾车 | 14分钟 |\n"
            "| 公交 | 32分钟 |\n\n"
            "驾车约14分钟，公交约32分钟。\n"
            "最终推荐：若重视用时，驾车更舒适。"
        )

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertEqual(reason_code, "ok")
        self.assertNotIn("路线明细", repaired)
        self.assertNotIn("最终推荐", repaired)
        self.assertNotIn("更舒适", repaired)
        self.assertIn("驾车约14分钟，公交约32分钟", repaired)

    def test_high_confidence_unsupported_claim_falls_back(self):
        validation = validate_product_answer(
            "炭火一号停车肯定方便，也不用排队，现场一定有空位。",
            [_place_block()],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_limitation_cue_does_not_mask_unsupported_claim_in_another_clause(self):
        validation = validate_product_answer(
            "实时排队无法确认，但停车肯定方便。",
            [_place_block()],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_weak_uncertainty_language_does_not_turn_realtime_claim_into_fact(self):
        validation = validate_product_answer(
            "炭火一号通常不用排队，现场一般有空位。",
            [_place_block()],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_unreturned_route_quality_claims_are_not_kept(self):
        answer = "地铁准点率高，早高峰最稳定，骑行更省钱也更安全，开车更舒适。"

        validation = validate_product_answer(answer, [_route_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_unreturned_waiting_and_flexibility_claims_are_not_kept(self):
        answer = "骑行比地铁少进出站和换乘等待，时间更灵活，也不用掐点赶车。"

        validation = validate_product_answer(answer, [_route_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_route_result_cannot_be_described_as_realtime_data(self):
        validation = validate_product_answer("高德返回了实时路线数据。", [_route_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_claim")

    def test_requested_departure_time_non_realtime_boundary_is_retained(self):
        answer = "用户指定的出发时间为“工作日早上 8:30”，本次结果未按该时刻的实时路况或班次计算。"

        validation = validate_product_answer(answer, [_route_block()])

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.reason_code, "ok")

    def test_repair_rewrites_realtime_route_data_label_and_keeps_grounded_comparison(self):
        answer = "根据高德返回的实时路线数据，驾车约14分钟，公共交通约32分钟。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("本次返回的路线数据", repaired)
        self.assertNotIn("高德", repaired)
        self.assertIn("驾车约14分钟", repaired)
        self.assertIn("公共交通约32分钟", repaired)
        self.assertNotIn("实时路线数据", repaired)
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_repair_salvages_explicitly_scoped_facts_from_mixed_unsafe_sentence(self):
        answer = "根据高德返回的实时路线数据，驾车约14分钟，公共交通约32分钟，实时拥堵较轻。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("驾车约14分钟", repaired)
        self.assertIn("公共交通约32分钟", repaired)
        self.assertNotIn("实时拥堵较轻", repaired)
        self.assertNotRegex(repaired, r"14分钟[^。\n]{0,30}32分钟")
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_repair_drops_source_only_and_dangling_predicate_clauses(self):
        route = _route_block()
        route.routes.append(RouteOption(mode="bicycling", duration_s=1200, distance_m=5000))
        answer = (
            "根据本次查询结果，驾车约14分钟，骑行约20分钟、5公里，是非常适合骑行的通勤距离，停车方便，地铁约32分钟。"
        )

        repaired, reason_code = repair_unsupported_product_answer(answer, [route])

        self.assertEqual(reason_code, "ok")
        self.assertIn("驾车约14分钟", repaired)
        self.assertIn("骑行约20分钟、5公里", repaired)
        self.assertIn("地铁约32分钟", repaired)
        self.assertNotIn("根据本次查询结果。", repaired)
        self.assertNotIn("是非常适合", repaired)
        self.assertNotIn("停车方便", repaired)
        self.assertTrue(validate_product_answer(repaired, [route]).is_valid)

    def test_repairs_only_unsupported_clause_and_keeps_grounded_model_prose(self):
        answer = "结论：驾车约14分钟，是本次用时最短的方案。高峰期可能拥堵。地铁约32分钟，适合能接受1次换乘的情况。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertIsNotNone(repaired)
        self.assertEqual(reason_code, "ok")
        self.assertIn("驾车约14分钟", repaired)
        self.assertIn("地铁约32分钟", repaired)
        self.assertNotIn("高峰期可能拥堵", repaired)
        self.assertIn("本次查询结果无法确认", repaired)
        self.assertNotIn("高德", repaired)
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_hard_fact_error_is_not_repaired(self):
        repaired, reason_code = repair_unsupported_product_answer(
            "结论：驾车约20分钟。高峰期可能拥堵。",
            [_route_block()],
        )

        self.assertIsNone(repaired)
        self.assertEqual(reason_code, "not_repairable")

    def test_removes_unsupported_numeric_clause_and_keeps_other_grounded_prose(self):
        repaired, reason_code = repair_unsupported_product_answer(
            "结论：驾车约14分钟。建议提前10分钟出门。地铁约32分钟。",
            [_route_block()],
        )

        self.assertEqual(reason_code, "ok")
        self.assertIn("驾车约14分钟", repaired)
        self.assertIn("地铁约32分钟", repaired)
        self.assertNotIn("提前10分钟", repaired)
        self.assertIn("请以卡片数值为准", repaired)

    def test_repair_drops_whole_unsafe_list_item_without_splicing_its_facts(self):
        answer = (
            "从民治站到雅宝站可按需求选择：\n"
            "1. 赶时间优先自驾，全程约6.2公里，耗时约14分钟。\n"
            "2. 骑行耗时约20分钟，通常更省钱且更安全。\n"
            "3. 公共交通约32分钟，需要换乘1次。"
        )

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("自驾", repaired)
        self.assertIn("公共交通约32分钟", repaired)
        self.assertNotIn("骑行耗时约20分钟", repaired)
        self.assertNotRegex(repaired, r"14分钟[^。\n]{0,30}20分钟")
        self.assertNotIn("3. 公共交通", repaired)
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_repair_salvages_second_route_fact_before_coverage_check(self):
        answer = "自驾全程约6.2公里，耗时约14分钟。公共交通最优选择。地铁约32分钟，准点稳定又省钱。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_route_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("自驾全程约6.2公里", repaired)
        self.assertIn("地铁约32分钟", repaired)
        self.assertNotIn("准点稳定又省钱", repaired)
        self.assertTrue(validate_product_answer(repaired, [_route_block()]).is_valid)

    def test_repair_falls_back_when_multiple_routes_shrink_to_one_grounded_mode(self):
        repaired, reason_code = repair_unsupported_product_answer(
            "自驾约14分钟。地铁准点稳定又省钱。",
            [_route_block()],
        )

        self.assertIsNone(repaired)
        self.assertEqual(reason_code, "insufficient_coverage")

    def test_only_returned_toll_amount_can_support_cost_claim(self):
        cases = (
            ("驾车方案有5元过路费。", True),
            ("公交票价5元。", False),
            ("驾车费用更低。", False),
            ("公交有5元过路费。", False),
            ("过路费88元。", False),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                validation = validate_product_answer(answer, [_route_block()])
                self.assertEqual(validation.is_valid, expected)

    def test_numeric_mismatch_falls_back(self):
        validation = validate_product_answer(
            "高德显示驾车约20分钟，建议优先驾车。",
            [_route_block()],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "numeric_mismatch")

    def test_numbers_are_bound_to_route_mode(self):
        cases = (
            "公交约14分钟。",
            "驾车约32分钟。",
            "公交全程约6.2公里。",
        )

        for answer in cases:
            with self.subTest(answer=answer):
                validation = validate_product_answer(answer, [_route_block()])
                self.assertFalse(validation.is_valid)

    def test_transit_leg_duration_cannot_be_used_as_total_route_duration(self):
        route = _route_block()
        route.routes[1].legs[0].duration_s = 3600

        validation = validate_product_answer("纯公交方案全程耗时超1小时。", [route])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "numeric_mismatch")

    def test_hour_and_minute_shorthand_is_supported(self):
        route = _route_block()
        route.routes[1].duration_s = 5400

        validation = validate_product_answer("公交约1小时30分。", [route])

        self.assertTrue(validation.is_valid)

    def test_user_budget_can_be_repeated_without_becoming_product_fact(self):
        validation = validate_product_answer(
            "你的总预算是500元；炭火一号的高德参考消费为88元。",
            [_place_block()],
            messages=[{"role": "user", "content": "三个人总预算500元"}],
        )

        self.assertTrue(validation.is_valid)

    def test_unreturned_relation_between_place_results_falls_back(self):
        validation = validate_product_answer(
            "模型自由文本：两家店步行五分钟。",
            [_place_block()],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "unsupported_place_relation")

    def test_address_similarity_cannot_be_inferred_as_walkable_proximity(self):
        cases = (
            "炭火一号和金杆桌球地址临近，吃完走几步就到。",
            "炭火一号到金杆桌球步行即达。",
            "两家距离也很近，溜达过去很方便。",
            "两家都在东边老村附近，地址非常接近。",
            "先吃完再到隔壁片区打桌球。",
            "这是就近组合，两个点最近，地址相近。",
            "两家地址的区域重叠度高。",
        )

        for answer in cases:
            with self.subTest(answer=answer):
                validation = validate_product_answer(answer, [_two_place_block()])
                self.assertFalse(validation.is_valid)
                self.assertEqual(validation.reason_code, "unsupported_place_relation")

    def test_repair_removes_unreturned_place_proximity_and_keeps_place_facts(self):
        answer = "炭火一号评分4.7分。两家地址临近，吃完走几步就到。金杆桌球评分4.1分。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_two_place_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("炭火一号评分4.7分", repaired)
        self.assertIn("金杆桌球评分4.1分", repaired)
        self.assertNotIn("地址临近", repaired)
        self.assertNotIn("走几步", repaired)
        self.assertIn("地点之间的距离和步行时间", repaired)
        self.assertTrue(validate_product_answer(repaired, [_two_place_block()]).is_valid)

    def test_place_experience_and_name_inference_are_not_treated_as_returned_facts(self):
        cases = (
            "吃完转场很方便。",
            "吃完随时去打桌球。",
            "这两个地点很顺路，也很好找好走。",
            "串说烧烤靠近商业街。",
            "眼镜哥鲜货烧烤适合爱吃鲜肉的朋友。",
            "烤肉配桌球，节奏比较自由。",
        )

        for answer in cases:
            with self.subTest(answer=answer):
                self.assertFalse(validate_product_answer(answer, [_two_place_block()]).is_valid)

    def test_place_repair_keeps_grounded_entities_and_drops_unscoped_experience_prose(self):
        answer = "炭火一号评分4.7分。吃完转场很方便。金杆桌球评分4.1分。适合爱吃鲜肉的朋友。"

        repaired, reason_code = repair_unsupported_product_answer(answer, [_two_place_block()])

        self.assertEqual(reason_code, "ok")
        self.assertIn("炭火一号评分4.7分", repaired)
        self.assertIn("金杆桌球评分4.1分", repaired)
        self.assertNotIn("转场很方便", repaired)
        self.assertNotIn("爱吃鲜肉", repaired)
        self.assertTrue(validate_product_answer(repaired, [_two_place_block()]).is_valid)

    def test_place_relation_request_requires_explicit_missing_route_caveat(self):
        messages = [
            {
                "role": "user",
                "content": "想吃烤肉，吃完去桌球厅，不想走太远，请给组合建议",
            }
        ]
        answer = "高德返回了炭火一号和金杆桌球，可作为候选。"

        validation = validate_product_answer(answer, [_two_place_block()], messages=messages)
        repaired, reason_code = repair_unsupported_product_answer(
            answer,
            [_two_place_block()],
            messages=messages,
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "missing_place_relation_caveat")
        self.assertEqual(reason_code, "ok")
        self.assertIn("地点之间的距离和步行时间", repaired)
        self.assertIn("另行查询路线", repaired)
        self.assertTrue(validate_product_answer(repaired, [_two_place_block()], messages=messages).is_valid)

    def test_place_superlative_requires_scope_to_returned_candidates(self):
        cases = (
            ("炭火一号评分4.7分，评分最高。", False),
            ("炭火一号评分4.7分，为本次返回候选中的最高评分。", True),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(validate_product_answer(answer, [_two_place_block()]).is_valid, expected)

    def test_unrelated_route_block_does_not_legalize_place_relation(self):
        validation = validate_product_answer(
            "炭火一号和海底捞民治店距离很近。",
            [_place_block(), _route_block()],
        )

        self.assertFalse(validation.is_valid)

    def test_unknown_place_and_line_fall_back(self):
        cases = (
            ("首选海底捞民治店。", [_place_block()], "unknown_place"),
            ("公交可以乘地铁11号线。", [_route_block()], "unknown_line"),
            ("公交可以乘999路。", [_route_block()], "unknown_line"),
            ("公交可以乘高峰专线91号。", [_route_block()], "unknown_line"),
            ("公交可以乘B917线。", [_route_block()], "unknown_line"),
            ("公交可以乘地铁十一号线。", [_route_block()], "unknown_line"),
            ("海底捞民治店评分4.7分。", [_place_block()], "unknown_place"),
        )

        for answer, blocks, reason_code in cases:
            with self.subTest(answer=answer):
                validation = validate_product_answer(answer, blocks)
                self.assertFalse(validation.is_valid)
                self.assertEqual(validation.reason_code, reason_code)

    def test_line_matching_is_exact_but_accepts_chinese_number_variant(self):
        route = _route_block()
        route.routes[1].legs[1].line_name = "地铁11号线"
        cases = (
            ("建议选择地铁十一号线。", True),
            ("建议选择地铁1号线。", False),
        )

        for answer, expected in cases:
            with self.subTest(answer=answer):
                validation = validate_product_answer(answer, [route])
                self.assertEqual(validation.is_valid, expected)

    def test_route_mode_recommendation_is_not_mistaken_for_unknown_place(self):
        validation = validate_product_answer(
            "建议选择耗时更短的驾车方案。",
            [_route_block()],
        )

        self.assertTrue(validation.is_valid)

    def test_empty_answer_falls_back(self):
        validation = validate_product_answer("   ", [_place_block()])

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.reason_code, "empty_answer")


if __name__ == "__main__":
    unittest.main()
