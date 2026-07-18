import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.db.models import Message as MessageModel
from app.db.repositories import ConversationRepository
from app.schemas.chat import (
    PlacePhoto,
    PlaceResult,
    PlaceResultsBlock,
    RouteEndpoint,
    RouteOption,
    RouteResultsBlock,
    TransitAlternative,
    TransitLeg,
)
from app.services.admin_audit_service import AdminAuditService
from app.services.agent.continuation import deserialize_content_blocks
from app.services.agent.emitter import AgentEventEmitter
from app.services.mcp.amap_product_tools import AmapProductToolHandler, build_amap_product_binding
from app.services.stream.persistence import merge_partial_content_blocks
from app.services.tool_handlers.base import ToolResult


def build_handler(product_name: str) -> AmapProductToolHandler:
    dependency_hashes = {
        "maps_geo": "hash-geo",
        "maps_text_search": "hash-text",
        "maps_around_search": "hash-around",
        "maps_search_detail": "hash-detail",
        "maps_direction_driving": "hash-driving",
        "maps_direction_transit_integrated": "hash-transit",
        "maps_direction_walking": "hash-walking",
        "maps_direction_bicycling": "hash-bicycling",
    }
    binding = build_amap_product_binding(
        row=SimpleNamespace(id="amap-1", provider="amap", config_version=3),
        product_name=product_name,
        dependency_hashes=dependency_hashes,
    )
    return AmapProductToolHandler(
        binding=binding,
        remote_executor=AsyncMock(),
        dependency_hashes=dependency_hashes,
    )


def place_block(*, block_id: str = "blk-place", result_count: int = 1) -> PlaceResultsBlock:
    places = [
        PlaceResult(
            provider_place_id="B0FFTEST",
            name="民治烤肉店",
            address="深圳市龙华区民治街道",
            district="龙华区",
            category="餐饮服务",
            distance_m=320,
            photos=[PlacePhoto(url="https://store.is.autonavi.com/photo.jpg", title="门店")],
            rating=4.6,
            reference_cost_yuan=98,
            business_area="民治",
            open_hours="周一至周日 11:00-23:00",
            detail_status="enriched",
            platform_url="https://uri.amap.com/marker?poiid=B0FFTEST&src=fusion&callnative=0",
        )
    ][:result_count]
    return PlaceResultsBlock(
        type="place_results",
        id=block_id,
        schema_version=1,
        provider="amap",
        query="烤肉",
        near="民治地铁站",
        status="success",
        result_count=len(places),
        places=places,
        limitations=["不包含实时排队或空位信息", "参考消费不代表人均或实时价格"],
        tool_call_log_id="log-place",
    )


def route_block(*, block_id: str = "blk-route") -> RouteResultsBlock:
    return RouteResultsBlock(
        type="route_results",
        id=block_id,
        schema_version=1,
        provider="amap",
        status="degraded",
        origin=RouteEndpoint(label="民治地铁站", city="深圳市"),
        destination=RouteEndpoint(label="深圳市民中心", city="深圳市"),
        routes=[
            RouteOption(
                mode="driving",
                distance_m=16_000,
                duration_s=1_500,
                summary="推荐路线",
                toll_yuan=0,
            )
        ],
        unavailable_modes=["transit"],
        limitations=["路线时间和距离仅代表高德本次返回结果"],
        tool_call_log_id="log-route",
    )


class ProductResultSchemaTests(unittest.TestCase):
    def test_route_transit_extension_keeps_schema_v1_and_old_payload_compatibility(self):
        old_payload = route_block().model_dump(mode="json")
        for key in ("transit_type", "walking_distance_m", "legs", "alternatives"):
            old_payload["routes"][0].pop(key, None)
        old = RouteResultsBlock.model_validate(old_payload)
        self.assertEqual(old.schema_version, 1)
        self.assertIsNone(old.routes[0].transit_type)
        self.assertEqual(old.routes[0].legs, [])
        self.assertEqual(old.routes[0].alternatives, [])
        restored = deserialize_content_blocks([old_payload])
        self.assertIsInstance(restored[0], RouteResultsBlock)
        self.assertEqual(restored[0].routes[0].mode, "driving")

        transit = RouteOption(
            mode="transit",
            transit_type="subway",
            walking_distance_m=320,
            legs=[TransitLeg(kind="walking"), TransitLeg(kind="subway", line_name="地铁5号线")],
            alternatives=[TransitAlternative(transit_type="bus", legs=[TransitLeg(kind="bus", line_name="M201路")])],
        )
        self.assertEqual(transit.model_dump(mode="json")["transit_type"], "subway")

        with self.assertRaises(ValidationError):
            RouteOption(mode="transit", legs=[TransitLeg(kind="walking") for _ in range(9)])
        with self.assertRaises(ValidationError):
            RouteOption(
                mode="transit",
                alternatives=[TransitAlternative() for _ in range(3)],
            )

    def test_provider_neutral_blocks_serialize_with_schema_version(self):
        place = place_block()
        route = route_block()

        self.assertEqual(place.model_dump(mode="json")["schema_version"], 1)
        self.assertEqual(place.model_dump(mode="json")["places"][0]["provider_place_id"], "B0FFTEST")
        self.assertEqual(route.model_dump(mode="json")["routes"][0]["mode"], "driving")
        self.assertNotIn("location", route.model_dump(mode="json")["origin"])

    def test_blocks_reject_unknown_fields_oversized_payloads_and_untrusted_urls(self):
        with self.assertRaises(ValidationError):
            PlaceResultsBlock.model_validate(
                {
                    **place_block().model_dump(mode="json"),
                    "provider_response": {"credential": "secret"},
                }
            )
        with self.assertRaises(ValidationError):
            place_block().model_copy(
                update={
                    "places": [
                        PlaceResult(provider_place_id=f"poi-{index}", name=f"地点-{index}") for index in range(6)
                    ]
                }
            ).__class__.model_validate(
                {
                    **place_block().model_dump(mode="json"),
                    "places": [{"provider_place_id": f"poi-{index}", "name": f"地点-{index}"} for index in range(6)],
                    "result_count": 6,
                }
            )
        with self.assertRaises(ValidationError):
            PlaceResult(
                provider_place_id="poi-1",
                name="地点",
                photos=[
                    PlacePhoto(url="https://store.is.autonavi.com/photo-1.jpg"),
                    PlacePhoto(url="https://store.is.autonavi.com/photo-2.jpg"),
                ],
            )
        with self.assertRaises(ValidationError):
            PlaceResult(
                provider_place_id="poi-1",
                name="地点",
                platform_url="https://evil.example/redirect?target=amap",
            )
        with self.assertRaises(ValidationError):
            PlacePhoto(url="http://store.is.autonavi.com/photo.jpg")
        with self.assertRaises(ValidationError):
            PlaceResult.model_validate(
                {
                    "provider_place_id": "poi-1",
                    "name": "地点",
                    "telephone": "0755-12345678",
                }
            )
        with self.assertRaises(ValidationError):
            PlaceResultsBlock.model_validate(
                {
                    **place_block().model_dump(mode="json"),
                    "limitations": ["x" * 100_000],
                }
            )


class AmapProductContentBlockTests(unittest.TestCase):
    def test_place_success_builds_bounded_redacted_block_and_ignores_upstream_url(self):
        handler = build_handler("local_place_search")
        result = ToolResult(
            status="success",
            data={
                "result": {
                    "query": "烤肉",
                    "near": "民治地铁站",
                    "result_count": 1,
                    "places": [
                        {
                            "poi_id": "B0FFTEST",
                            "name": "民治 api_key=NAME_SENTINEL 烤肉店",
                            "address": "authorization=Bearer ADDRESS_SENTINEL",
                            "district": "龙华区",
                            "type": "餐饮服务",
                            "distance_m": 320,
                            "rating": 4.6,
                            "reference_cost_yuan": 98,
                            "business_area": "民治",
                            "open_hours": "周一至周日 11:00-23:00",
                            "detail_status": "enriched",
                            "platform_url": "https://evil.example/redirect",
                            "photos": [
                                {"url": "https://store.is.autonavi.com/photo.jpg", "title": "门店"},
                                {"url": "http://evil.example/photo.jpg", "title": "不安全"},
                            ],
                        }
                    ],
                    "limitations": ["不包含实时排队或空位信息", "参考消费不代表人均或实时价格"],
                }
            },
        )

        block = handler.build_content_block(result, "blk-place", "log-place")

        self.assertIsInstance(block, PlaceResultsBlock)
        self.assertEqual(block.status, "success")
        self.assertEqual(block.result_count, 1)
        self.assertEqual(block.places[0].provider_place_id, "B0FFTEST")
        self.assertEqual(block.places[0].category, "餐饮服务")
        self.assertEqual(len(block.places[0].photos), 1)
        self.assertEqual(
            block.places[0].platform_url,
            "https://uri.amap.com/marker?poiid=B0FFTEST&src=fusion&callnative=0",
        )
        self.assertEqual(block.places[0].reference_cost_yuan, 98)
        self.assertEqual(block.places[0].business_area, "民治")
        serialized = json.dumps(block.model_dump(mode="json"), ensure_ascii=False)
        self.assertNotIn("NAME_SENTINEL", serialized)
        self.assertNotIn("ADDRESS_SENTINEL", serialized)
        self.assertNotIn("evil.example", serialized)
        self.assertNotIn("telephone", serialized)
        self.assertEqual(
            handler._build_result_summary(result),
            {
                "kind": "external_tool",
                "title": "高德地点搜索",
                "provider": "amap",
                "truncated": False,
                "result_count": 1,
            },
        )

    def test_route_degraded_builds_route_block_but_failed_result_builds_none(self):
        handler = build_handler("route_compare")
        degraded = ToolResult(
            status="degraded",
            data={
                "result": {
                    "origin": {"label": "民治地铁站", "city": "深圳市", "location": "114.0,22.5"},
                    "destination": {"label": "深圳市民中心", "city": "深圳市", "location": "114.1,22.6"},
                    "routes": [
                        {
                            "mode": "driving",
                            "distance_m": 16_000,
                            "duration_s": 1_500,
                            "summary": "推荐路线",
                            "toll_yuan": 0,
                        }
                    ],
                    "unavailable_modes": ["transit"],
                    "limitations": ["路线时间和距离仅代表高德本次返回结果"],
                }
            },
        )

        block = handler.build_content_block(degraded, "blk-route", "log-route")

        self.assertIsInstance(block, RouteResultsBlock)
        self.assertEqual(block.status, "degraded")
        self.assertEqual(block.origin.label, "民治地铁站")
        self.assertEqual(block.routes[0].duration_s, 1_500)
        self.assertNotIn("location", block.model_dump(mode="json")["origin"])
        self.assertEqual(handler._build_result_summary(degraded)["mode_count"], 1)
        self.assertIsNone(
            handler.build_content_block(
                ToolResult(status="failed", data={"error_code": "server_run_budget_exhausted"}),
                "blk-failed",
                "log-failed",
            )
        )


class ProductResultRecoveryTests(unittest.TestCase):
    def test_repository_hydration_and_continuation_restore_both_blocks(self):
        raw_blocks = [place_block().model_dump(mode="json"), route_block().model_dump(mode="json")]
        db_message = MessageModel(
            id="msg-results",
            conversation_id="conv-1",
            role="assistant",
            content=raw_blocks,
            created_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        )

        message = ConversationRepository(None)._convert_message_to_schema(db_message)
        continuation = deserialize_content_blocks(raw_blocks)

        self.assertIsInstance(message.content[0], PlaceResultsBlock)
        self.assertIsInstance(message.content[1], RouteResultsBlock)
        self.assertEqual(continuation, message.content)

    def test_partial_merge_upserts_same_product_block_and_keeps_other_blocks(self):
        existing = [place_block(result_count=0).model_dump(mode="json"), route_block().model_dump(mode="json")]
        incoming_place = place_block().model_dump(mode="json")

        merged = merge_partial_content_blocks(existing, [incoming_place])

        self.assertEqual([block["type"] for block in merged], ["place_results", "route_results"])
        self.assertEqual(merged[0], incoming_place)

    def test_admin_projection_keeps_only_product_contract_fields(self):
        raw = place_block().model_dump(mode="json")
        raw.update(
            provider_response={"credential": "PRIVATE_SENTINEL"},
            raw_payload="PRIVATE_PAYLOAD",
        )

        projection = AdminAuditService._content_block_projection(raw)
        serialized = json.dumps(projection, ensure_ascii=False)

        self.assertEqual(projection["type"], "place_results")
        self.assertEqual(projection["schema_version"], 1)
        self.assertEqual(projection["places"][0]["provider_place_id"], "B0FFTEST")
        self.assertNotIn("provider_response", projection)
        self.assertNotIn("raw_payload", projection)
        self.assertNotIn("PRIVATE_SENTINEL", serialized)

        raw_route = route_block().model_dump(mode="json")
        raw_route["origin"]["location"] = "114.0,22.5"
        raw_route["routes"][0]["provider_response"] = {"credential": "ROUTE_PRIVATE_SENTINEL"}

        route_projection = AdminAuditService._content_block_projection(raw_route)
        route_serialized = json.dumps(route_projection, ensure_ascii=False)

        self.assertEqual(route_projection["type"], "route_results")
        self.assertEqual(route_projection["status"], "degraded")
        self.assertNotIn("location", route_projection["origin"])
        self.assertNotIn("provider_response", route_serialized)
        self.assertNotIn("ROUTE_PRIVATE_SENTINEL", route_serialized)

    def test_admin_projection_preserves_bounded_transit_structure(self):
        raw = route_block().model_dump(mode="json")
        raw["routes"] = [
            {
                "mode": "transit",
                "distance_m": 12_000,
                "duration_s": 2_400,
                "transit_type": "mixed",
                "walking_distance_m": 420,
                "transfers": 1,
                "legs": [
                    {"kind": "subway", "line_name": "地铁5号线", "departure_stop": "民治站"},
                    {"kind": "bus", "line_name": "M201路", "arrival_stop": "市民中心站"},
                ],
                "alternatives": [
                    {
                        "transit_type": "bus",
                        "distance_m": 13_000,
                        "duration_s": 2_700,
                        "walking_distance_m": 300,
                        "transfers": 0,
                        "summary": "公交备选",
                        "legs": [{"kind": "bus", "line_name": "M202路"}],
                    }
                ],
                "unsafe": "DROP_ME",
            }
        ]

        projection = AdminAuditService._content_block_projection(raw)

        route = projection["routes"][0]
        self.assertEqual(route["transit_type"], "mixed")
        self.assertEqual(route["legs"][0]["line_name"], "地铁5号线")
        self.assertEqual(route["alternatives"][0]["transit_type"], "bus")
        self.assertNotIn("distance_m", route["alternatives"][0])
        self.assertNotIn("unsafe", json.dumps(projection, ensure_ascii=False))


class ProductResultEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_block_upsert_event_uses_strict_whitelist_and_full_block(self):
        writer = AsyncMock()
        emitter = AgentEventEmitter(
            run_id="run-1",
            trace_id="trace-1",
            conversation_id="conv-1",
            task_id="task-1",
            redis_writer=writer,
        )
        block = place_block()

        await emitter.content_block_upserted(tool_call_id="tc-place", content_block=block)

        payload = writer.append_chunk.await_args.args[3]
        self.assertEqual(payload["type"], "content_block_upserted")
        self.assertEqual(payload["protocol_version"], 2)
        self.assertEqual(payload["tool_call_id"], "tc-place")
        self.assertEqual(payload["content_block"], block.model_dump(mode="json"))
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False).encode()), 65_536)

        with self.assertRaises(ValidationError):
            await emitter.content_block_upserted(
                tool_call_id="tc-place",
                content_block={
                    **block.model_dump(mode="json"),
                    "provider_response": {"secret": "LEAK_SENTINEL"},
                },
            )
