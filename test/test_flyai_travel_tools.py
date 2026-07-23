import asyncio
import hashlib
import hmac
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from app.schemas.chat import FlightResultsBlock, TrainResultsBlock
from app.schemas.content_block_registry import deserialize_content_blocks
from app.services.agent.events import ContentBlockUpserted
from app.services.mcp.agent_tools import load_mcp_agent_tools
from app.services.mcp.flyai_travel_tools import (
    FLYAI_SEARCH_FLIGHTS,
    FLYAI_SEARCH_TRAINS,
    FLYAI_TRAVEL_DEFINITIONS,
    FlyAiTravelAdapterClient,
    FlyAiTravelRunControls,
    FlyAiTravelToolHandler,
    build_flyai_travel_binding,
    build_flyai_user_scope,
)
from app.services.stream.agent_loop_request_prep import inject_flyai_travel_fact_boundary
from app.services.stream.agent_loop_wiring import _load_dynamic_tools
from app.services.stream.product_answer_validator import (
    repair_unsupported_product_answer,
    validate_product_answer,
)
from app.services.stream.product_result_answer import (
    build_grounded_product_answer,
    build_product_tool_failure_answer,
    neutralize_product_provider_mentions,
)
from app.services.stream.tool_executor import ToolExecutionBatchRequest, execute_tool_handler


def _adapter_payload(
    *,
    transport_no: str = "CZ1234",
    booking_url: str | None = None,
    request: dict | None = None,
) -> dict:
    item = {
        "transport_no": transport_no,
        "operator_name": "南方航空" if transport_no.startswith("CZ") else "复兴号",
        "departure": {
            "city": "深圳",
            "station_name": "深圳宝安国际机场" if transport_no.startswith("CZ") else "深圳北站",
            "station_code": "SZX" if transport_no.startswith("CZ") else "IOQ",
            "terminal": "T3" if transport_no.startswith("CZ") else None,
            "scheduled_at": "2026-08-01T08:30:00+08:00",
        },
        "arrival": {
            "city": "上海",
            "station_name": "上海虹桥国际机场" if transport_no.startswith("CZ") else "上海虹桥站",
            "station_code": "SHA" if transport_no.startswith("CZ") else "AOH",
            "terminal": "T2" if transport_no.startswith("CZ") else None,
            "scheduled_at": "2026-08-01T10:45:00+08:00",
        },
        "duration_minutes": 135,
        "travel_class": "经济舱" if transport_no.startswith("CZ") else "二等座",
        "journey_type": "direct",
        "price": {"currency": "CNY", "amount_minor": 88000},
    }
    if booking_url is not None:
        item["booking_url"] = booking_url
    return {
        "observed_at": "2026-07-22T15:00:00+08:00",
        "request": request
        or {
            "origin": "深圳",
            "destination": "上海",
            "departure_date": "2026-08-01",
            "sort_by": "recommended",
            "limit": 5,
        },
        "items": [item],
    }


def _handler(
    tool_name: str,
    transport: httpx.AsyncBaseTransport,
    *,
    controls: FlyAiTravelRunControls | None = None,
    timeout_seconds: float = 1,
) -> FlyAiTravelToolHandler:
    client = FlyAiTravelAdapterClient(
        base_url="http://flyai-adapter:8080",
        token="adapter-secret",
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    return FlyAiTravelToolHandler(
        binding=build_flyai_travel_binding(tool_name),
        client=client,
        controls=controls or FlyAiTravelRunControls(max_calls=4, concurrency=2),
        user_scope="scope-digest",
    )


class FlyAiTravelToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_definitions_are_closed_bounded_and_use_confirmed_names(self):
        by_name = {item["function"]["name"]: item["function"]["parameters"] for item in FLYAI_TRAVEL_DEFINITIONS}

        self.assertEqual(set(by_name), {FLYAI_SEARCH_FLIGHTS, FLYAI_SEARCH_TRAINS})
        for parameters in by_name.values():
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(parameters["required"], ["origin", "destination", "departure_date"])
            self.assertEqual(parameters["properties"]["limit"]["maximum"], 5)
            self.assertEqual(
                parameters["properties"]["sort_by"]["enum"],
                list(("recommended", "price_asc", "duration_asc", "departure_asc")),
            )
        self.assertIn("cabin_class", by_name[FLYAI_SEARCH_FLIGHTS]["properties"])
        self.assertNotIn("seat_class", by_name[FLYAI_SEARCH_FLIGHTS]["properties"])
        self.assertIn("seat_class", by_name[FLYAI_SEARCH_TRAINS]["properties"])
        self.assertNotIn("cabin_class", by_name[FLYAI_SEARCH_TRAINS]["properties"])

    async def test_flight_success_projects_safe_fields_and_trusted_booking_action(self):
        captured: dict = {}

        async def respond(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            normalized_request = json.loads(request.content)
            return httpx.Response(
                200,
                json=_adapter_payload(
                    booking_url="https://a.feizhu.com/flight/detail?id=public",
                    request=normalized_request,
                ),
            )

        handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(respond))
        result = await handler.execute(
            {
                "origin": "深圳",
                "destination": "上海",
                "departure_date": "2026-08-01",
                "cabin_class": "经济舱",
                "limit": 5,
            }
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(captured["request"].url.path, "/v1/search/flights")
        self.assertEqual(captured["request"].headers["authorization"], "Bearer adapter-secret")
        self.assertEqual(captured["request"].headers["x-fusion-user-scope"], "scope-digest")
        serialized_result = json.dumps(result.data, ensure_ascii=False)
        self.assertNotIn("adapter-secret", serialized_result)
        self.assertNotIn("raw", serialized_result)

        block = handler.build_content_block(result, "blk-flight", "log-flight")
        self.assertIsInstance(block, FlightResultsBlock)
        self.assertEqual(block.flights[0].flight_no, "CZ1234")
        self.assertEqual(block.flights[0].price.amount_minor, 88000)
        self.assertEqual(block.flights[0].actions[0].url, "https://a.feizhu.com/flight/detail?id=public")
        self.assertNotIn("flight_number", block.model_dump(mode="json")["flights"][0])
        self.assertNotIn("reference_price", block.model_dump(mode="json")["flights"][0])

    async def test_train_projection_drops_untrusted_booking_url(self):
        async def respond(_request: httpx.Request) -> httpx.Response:
            normalized_request = json.loads(_request.content)
            return httpx.Response(
                200,
                json=_adapter_payload(
                    transport_no="G100",
                    booking_url="https://evil.example/redirect?token=secret",
                    request=normalized_request,
                ),
            )

        handler = _handler(FLYAI_SEARCH_TRAINS, httpx.MockTransport(respond))
        result = await handler.execute(
            {
                "origin": "深圳",
                "destination": "上海",
                "departure_date": "2026-08-01",
                "seat_class": "二等座",
            }
        )
        block = handler.build_content_block(result, "blk-train", "log-train")

        self.assertIsInstance(block, TrainResultsBlock)
        self.assertEqual(block.trains[0].train_no, "G100")
        self.assertEqual(block.trains[0].actions, [])
        self.assertNotIn("evil.example", handler.format_llm_context(result))

    async def test_invalid_arguments_do_not_consume_budget_or_send_request(self):
        calls = 0

        async def respond(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=_adapter_payload(request=json.loads(_request.content)))

        controls = FlyAiTravelRunControls(max_calls=4, concurrency=2)
        handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(respond), controls=controls)
        result = await handler.execute(
            {
                "origin": "深圳",
                "destination": "上海",
                "departure_date": "08/01/2026",
                "limit": 6,
                "unknown": True,
            }
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error_code"], "invalid_arguments")
        self.assertEqual(
            result.data["validation_errors"],
            [
                "departure_date:string_pattern_mismatch",
                "limit:less_than_equal",
                "unknown_field:extra_forbidden",
            ],
        )
        self.assertEqual(
            handler.sanitize_output_data_for_log(result)["validation_errors"],
            result.data["validation_errors"],
        )
        self.assertEqual(calls, 0)
        self.assertEqual(await controls.remaining(), 4)

    async def test_sent_failures_consume_shared_budget_and_never_retry(self):
        calls = 0

        async def respond(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": "secret upstream message"})

        controls = FlyAiTravelRunControls(max_calls=4, concurrency=2)
        flight = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(respond), controls=controls)
        train = _handler(FLYAI_SEARCH_TRAINS, httpx.MockTransport(respond), controls=controls)
        args = {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}

        results = [
            await flight.execute(args),
            await train.execute(args),
            await flight.execute(args),
            await train.execute(args),
        ]
        exhausted = await flight.execute(args)

        self.assertTrue(all(item.data["error_code"] == "unauthorized" for item in results))
        self.assertEqual(exhausted.data["error_code"], "travel_run_budget_exhausted")
        self.assertEqual(calls, 4)
        self.assertEqual(await controls.remaining(), 0)

    async def test_shared_run_controls_serialize_flight_and_train_for_same_user_scope(self):
        active = 0
        max_active = 0

        async def respond(request: httpx.Request) -> httpx.Response:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            transport_no = "CZ1234" if request.url.path.endswith("/flights") else "G100"
            return httpx.Response(
                200,
                json=_adapter_payload(
                    transport_no=transport_no,
                    request=json.loads(request.content),
                ),
            )

        controls = FlyAiTravelRunControls(max_calls=4, concurrency=1)
        transport = httpx.MockTransport(respond)
        flight = _handler(FLYAI_SEARCH_FLIGHTS, transport, controls=controls)
        train = _handler(FLYAI_SEARCH_TRAINS, transport, controls=controls)
        args = {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}

        flight_result, train_result = await asyncio.gather(flight.execute(args), train.execute(args))

        self.assertEqual(flight_result.status, "success")
        self.assertEqual(train_result.status, "success")
        self.assertEqual(max_active, 1)
        self.assertEqual(await controls.remaining(), 2)

    async def test_response_byte_limit_and_timeout_fail_closed(self):
        oversized = b"{" + (b"x" * 262_144) + b"}"

        async def oversized_response(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=oversized, headers={"content-type": "application/json"})

        oversized_handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(oversized_response))
        oversized_result = await oversized_handler.execute(
            {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}
        )
        self.assertEqual(oversized_result.data["error_code"], "response_too_large")

        async def slow_response(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.1)
            return httpx.Response(200, json=_adapter_payload(request=json.loads(_request.content)))

        timeout_handler = _handler(
            FLYAI_SEARCH_FLIGHTS,
            httpx.MockTransport(slow_response),
            timeout_seconds=0.01,
        )
        timeout_result = await timeout_handler.execute(
            {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}
        )
        self.assertEqual(timeout_result.data["error_code"], "call_timeout")

    async def test_audit_and_event_arguments_never_include_query_or_credentials(self):
        handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(lambda _request: httpx.Response(500)))
        args = {
            "origin": "深圳",
            "destination": "上海",
            "departure_date": "2026-08-01",
        }
        safe_event = handler.sanitize_input_params_for_event(args)
        safe_log = handler.sanitize_input_params_for_log(args)

        self.assertEqual(safe_event, {"argument_count": 3})
        self.assertEqual(safe_log["argument_count"], 3)
        serialized = json.dumps({"event": safe_event, "log": safe_log}, ensure_ascii=False)
        for forbidden in ("深圳", "上海", "2026-08-01", "adapter-secret"):
            self.assertNotIn(forbidden, serialized)

    async def test_started_and_completed_events_only_emit_safe_summaries(self):
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_adapter_payload(request=json.loads(request.content)))

        handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(respond))
        emitter = AsyncMock()
        args = {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}
        request = ToolExecutionBatchRequest(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="model-1",
            provider="provider-1",
            emitter=emitter,
        )

        result = await execute_tool_handler(
            request=request,
            tool_call={"id": "call-flight", "name": FLYAI_SEARCH_FLIGHTS},
            handler=handler,
            args=args,
        )

        self.assertEqual(result.status, "success")
        started_arguments = emitter.tool_call_started.await_args.kwargs["arguments"]
        completed_summary = emitter.tool_call_completed.await_args.kwargs["result_summary"]
        self.assertEqual(started_arguments, {"argument_count": 3})
        self.assertEqual(completed_summary["result_count"], 1)
        serialized = json.dumps({"started": started_arguments, "completed": completed_summary}, ensure_ascii=False)
        for forbidden in ("深圳", "上海", "2026-08-01", "adapter-secret"):
            self.assertNotIn(forbidden, serialized)

    async def test_strict_json_content_type_and_full_echo_request_are_enforced(self):
        cases = (
            (
                b'{"observed_at":"2026-07-22T15:00:00+08:00","observed_at":"2026-07-22T15:01:00+08:00"}',
                "application/json",
            ),
            (b'{"observed_at":NaN}', "application/json"),
            (json.dumps(_adapter_payload()).encode(), "text/plain"),
            (
                json.dumps(
                    _adapter_payload(
                        request={
                            "origin": "广州",
                            "destination": "上海",
                            "departure_date": "2026-08-01",
                            "sort_by": "recommended",
                            "limit": 5,
                        }
                    )
                ).encode(),
                "application/json",
            ),
        )
        args = {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}
        for body, content_type in cases:
            with self.subTest(body=body[:32], content_type=content_type):
                handler = _handler(
                    FLYAI_SEARCH_FLIGHTS,
                    httpx.MockTransport(
                        lambda _request, body=body, content_type=content_type: httpx.Response(
                            200,
                            content=body,
                            headers={"content-type": content_type},
                        )
                    ),
                )
                result = await handler.execute(args)
                self.assertEqual(result.data["error_code"], "invalid_response")

    async def test_registration_is_explicit_and_user_scope_is_irreversible(self):
        client = FlyAiTravelAdapterClient(
            base_url="http://flyai-adapter:8080",
            token="adapter-secret",
            timeout_seconds=1,
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        )

        def empty_repository(_db):
            return SimpleNamespace(list_enabled=lambda: [])

        with (
            patch("app.services.mcp.agent_tools.settings.ENABLE_FLYAI_TRAVEL_TOOLS", True),
            patch("app.services.mcp.agent_tools.settings.FLYAI_ADAPTER_BASE_URL", "http://flyai-adapter:8080"),
            patch("app.services.mcp.agent_tools.settings.FLYAI_ADAPTER_TOKEN", "adapter-secret"),
        ):
            disabled_without_user = load_mcp_agent_tools(
                object(),
                repository_factory=empty_repository,
                flyai_client=client,
            )
            enabled = load_mcp_agent_tools(
                object(),
                user_id="user-123",
                repository_factory=empty_repository,
                flyai_client=client,
            )

        self.assertEqual(disabled_without_user.definitions, [])
        self.assertEqual(
            [definition["function"]["name"] for definition in enabled.definitions],
            [FLYAI_SEARCH_FLIGHTS, FLYAI_SEARCH_TRAINS],
        )
        self.assertEqual(set(enabled.handlers), {FLYAI_SEARCH_FLIGHTS, FLYAI_SEARCH_TRAINS})
        flight_controls = enabled.handlers[FLYAI_SEARCH_FLIGHTS].controls
        train_controls = enabled.handlers[FLYAI_SEARCH_TRAINS].controls
        self.assertIs(flight_controls, train_controls)
        self.assertEqual(flight_controls.concurrency_limit, 1)
        expected_scope = hmac.new(b"adapter-secret", b"user-123", hashlib.sha256).hexdigest()
        self.assertEqual(build_flyai_user_scope("user-123", "adapter-secret"), expected_scope)
        self.assertNotIn("user-123", expected_scope)
        self.assertNotIn("adapter-secret", json.dumps(enabled.audit_bindings))

    async def test_content_block_registry_grounded_answer_and_validator_cover_travel(self):
        async def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_adapter_payload(
                    booking_url="https://a.feizhu.com/flight/detail?id=public",
                    request=json.loads(request.content),
                ),
            )

        handler = _handler(FLYAI_SEARCH_FLIGHTS, httpx.MockTransport(respond))
        result = await handler.execute({"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"})
        block = handler.build_content_block(result, "blk-flight", "log-flight")
        restored = deserialize_content_blocks([block.model_dump(mode="json")])
        answer = build_grounded_product_answer(restored)

        self.assertIsInstance(restored[0], FlightResultsBlock)
        event = ContentBlockUpserted(
            type="content_block_upserted",
            protocol_version=2,
            run_id="run-1",
            sequence=1,
            trace_id="trace-1",
            ts=1.0,
            content_block=restored[0],
        )
        self.assertIsInstance(event.content_block, FlightResultsBlock)
        self.assertIn("CZ1234", answer)
        self.assertIn("深圳宝安国际机场", answer)
        self.assertIn("08:30", answer)
        self.assertIn("880元", answer)
        self.assertTrue(validate_product_answer(answer, restored).is_valid)
        self.assertTrue(
            validate_product_answer(
                "本次返回中，CZ1234 的票价为 880 元，从深圳宝安国际机场 T3 出发，是其中较便宜的选择。",
                restored,
            ).is_valid
        )
        self.assertTrue(validate_product_answer("2026年8月1日（周六）可以考虑CZ1234。", restored).is_valid)
        table_answer = (
            "本次返回中，CZ1234 在08:30从深圳宝安国际机场T3出发，参考价880元，可以优先考虑。\n\n"
            "| 航班 | 出发时间 | 参考价 |\n"
            "| --- | --- | --- |\n"
            "| CZ1234 | 08:30 | 880元 |\n\n"
            "---\n"
            "CZ1234 是夜间航班，可以省住宿费。"
        )
        repaired, reason_code = repair_unsupported_product_answer(table_answer, restored)
        self.assertEqual(reason_code, "ok")
        self.assertNotIn("|", repaired)
        self.assertNotIn("---", repaired)
        self.assertNotIn("省住宿费", repaired)
        self.assertNotIn("实时排队", repaired)
        self.assertIn("CZ1234", repaired)
        self.assertTrue(validate_product_answer(repaired, restored).is_valid)

        async def respond_train(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_adapter_payload(transport_no="G100", request=json.loads(request.content)),
            )

        train_handler = _handler(FLYAI_SEARCH_TRAINS, httpx.MockTransport(respond_train))
        train_result = await train_handler.execute(
            {"origin": "深圳", "destination": "上海", "departure_date": "2026-08-01"}
        )
        train_block = train_handler.build_content_block(train_result, "blk-train", "log-train")
        mixed_blocks = [restored[0], train_block]
        comparison = build_grounded_product_answer(mixed_blocks)
        self.assertIn("同时返回深圳到上海", comparison)
        self.assertIn("本次返回航班中参考价最低的是CZ1234", comparison)
        self.assertIn("本次返回火车中用时最短的是G100", comparison)
        comparison_validation = validate_product_answer(comparison, mixed_blocks)
        self.assertTrue(
            comparison_validation.is_valid,
            f"{comparison_validation.reason_code}: {comparison}",
        )

        flight_template = restored[0].flights[0]
        slow_cheap_flight = flight_template.model_copy(
            update={
                "option_id": "opt-slow-cheap-flight",
                "flight_no": "CZ1001",
                "duration_s": 3 * 60 * 60,
                "price": flight_template.price.model_copy(update={"amount_minor": 30_000}),
            }
        )
        fast_expensive_flight = flight_template.model_copy(
            update={
                "option_id": "opt-fast-expensive-flight",
                "flight_no": "CZ2002",
                "duration_s": 60 * 60,
                "price": flight_template.price.model_copy(update={"amount_minor": 100_000}),
            }
        )
        multi_flight_block = restored[0].model_copy(
            update={
                "result_count": 2,
                "flights": [slow_cheap_flight, fast_expensive_flight],
            }
        )
        train_template = train_block.trains[0]
        cheaper_train_block = train_block.model_copy(
            update={
                "trains": [
                    train_template.model_copy(
                        update={
                            "duration_s": 2 * 60 * 60,
                            "price": train_template.price.model_copy(update={"amount_minor": 20_000}),
                        }
                    )
                ]
            }
        )
        mapped_comparison = build_grounded_product_answer([multi_flight_block, cheaper_train_block])
        self.assertIn("如果优先考虑本次返回的计划行程时长，可优先考虑航班CZ2002", mapped_comparison)
        self.assertNotIn("计划行程时长，可优先考虑航班CZ1001", mapped_comparison)
        self.assertIn("如果预算优先，可考虑高铁G100", mapped_comparison)
        self.assertTrue(
            validate_product_answer(mapped_comparison, [multi_flight_block, cheaper_train_block]).is_valid
        )

        faster_train_block = cheaper_train_block.model_copy(
            update={
                "trains": [
                    cheaper_train_block.trains[0].model_copy(
                        update={"duration_s": 45 * 60}
                    )
                ]
            }
        )
        cross_mode_comparison = build_grounded_product_answer([multi_flight_block, faster_train_block])
        self.assertIn("如果优先考虑本次返回的计划行程时长，可优先考虑高铁G100", cross_mode_comparison)
        self.assertTrue(
            validate_product_answer(cross_mode_comparison, [multi_flight_block, faster_train_block]).is_valid
        )
        mixed_repair, mixed_reason = repair_unsupported_product_answer(table_answer, mixed_blocks)
        self.assertIsNone(mixed_repair)
        self.assertEqual(mixed_reason, "unsupported_format")
        for invalid_answer, reason in (
            ("CA9999 在 08:30 起飞。", "unknown_travel_number"),
            ("CZ1234 在 09:30 起飞。", "unknown_travel_time"),
            ("CZ1234 从广州白云国际机场起飞。", "unknown_travel_entity"),
            ("CZ1234 参考价 999 元。", "numeric_mismatch"),
            ("CZ1234 票价 999 元。", "unsupported_claim"),
            ("本次返回中，CZ1234 的实时票价为 880 元。", "unsupported_claim"),
            ("本次返回中，CZ1234 从深圳宝安国际机场 T4 出发。", "unknown_travel_number"),
            ("CZ1234 所属航司班次更多，机场接机也方便。", "unsupported_claim"),
            ("CZ1234 是夜间航班，可以省住宿费。", "unsupported_claim"),
            ("2026年8月1日（周日）可以考虑CZ1234。", "unknown_travel_date"),
            ("CZ1234 的耗时约为另一个选项的4.7倍。", "numeric_mismatch"),
            ("CZ1234 余票充足且准点率很高。", "unsupported_claim"),
        ):
            validation = validate_product_answer(invalid_answer, restored)
            self.assertFalse(validation.is_valid)
            self.assertEqual(validation.reason_code, reason)

        neutralized = neutralize_product_provider_mentions(
            "根据 FlyAI 和飞猪旅行返回的结果，search_flights 返回 CZ1234。"
        )
        self.assertNotIn("FlyAI", neutralized)
        self.assertNotIn("飞猪", neutralized)
        self.assertNotIn("search_flights", neutralized)
        self.assertIn("航班查询", neutralized)
        failure = build_product_tool_failure_answer()
        self.assertIn("航班或高铁", failure)

    async def test_agent_request_injects_travel_boundary_and_loader_receives_user_id(self):
        call_kwargs = {"tools": [FLYAI_TRAVEL_DEFINITIONS[0]]}
        messages = inject_flyai_travel_fact_boundary([{"role": "user", "content": "查航班"}], call_kwargs)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("【航班与高铁事实边界规则】", messages[0]["content"])
        self.assertIn("正文不使用表格重复卡片", messages[0]["content"])
        self.assertNotIn("FlyAI", messages[0]["content"])
        self.assertNotIn("飞猪", messages[0]["content"])

        captured: dict = {}

        def new_loader(db, *, user_id):
            captured.update(db=db, user_id=user_id)
            return "loaded"

        self.assertEqual(_load_dynamic_tools(new_loader, db="db", user_id="user-1"), "loaded")
        self.assertEqual(captured, {"db": "db", "user_id": "user-1"})
        self.assertEqual(_load_dynamic_tools(lambda db: db, db="legacy-db", user_id="user-1"), "legacy-db")


if __name__ == "__main__":
    unittest.main()
