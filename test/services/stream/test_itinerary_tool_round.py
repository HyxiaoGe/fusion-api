from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

from app.schemas.chat import (
    ItineraryResultsBlock,
    TrainOption,
    TrainResultsBlock,
    TravelEndpoint,
    TravelMoney,
)
from app.services.stream import tool_round as tool_round_module
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.itinerary_result_composer import compose_itinerary_result
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_context import ToolContextResolution
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_round import handle_tool_calls_round
from app.services.tool_handlers.base import ToolResult

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _train_block() -> TrainResultsBlock:
    return TrainResultsBlock(
        type="train_results",
        id="blk-train-outbound",
        schema_version=1,
        provider="flyai",
        status="success",
        origin="深圳",
        destination="上海",
        departure_date="2026-08-01",
        observed_at=datetime(2026, 7, 26, 8, tzinfo=SHANGHAI_TZ),
        result_count=1,
        trains=[
            TrainOption(
                option_id="train-outbound-1",
                train_no="G100",
                train_type="高铁",
                departure=TravelEndpoint(
                    city="深圳",
                    station_name="深圳北站",
                    scheduled_at=datetime(2026, 8, 1, 8, tzinfo=SHANGHAI_TZ),
                ),
                arrival=TravelEndpoint(
                    city="上海",
                    station_name="上海虹桥站",
                    scheduled_at=datetime(2026, 8, 1, 16, tzinfo=SHANGHAI_TZ),
                ),
                duration_s=28_800,
                stops=0,
                price=TravelMoney(currency="CNY", amount_minor=56000),
            )
        ],
    )


def _record(
    *,
    tool_call: dict,
    status: str,
    block=None,
) -> ToolExecutionRecord:
    handler = Mock()
    handler.format_llm_context.return_value = "安全工具上下文"
    handler.build_content_block.return_value = block
    return ToolExecutionRecord(
        tool_call=tool_call,
        result=ToolResult(status=status),
        handler=handler,
        block_id=block.id if block is not None else f"blk-{tool_call['id']}",
        log_id=f"log-{tool_call['id']}",
    )


def _request(
    records: list[ToolExecutionRecord],
    *,
    state: AgentLoopState | None = None,
):
    state = state or AgentLoopState()
    tool_calls = [record.tool_call for record in records]
    order: list[tuple] = []
    emitter = AsyncMock()

    async def emit_content_block(*, tool_call_id, content_block):
        order.append(("emit", tool_call_id, content_block))

    emitter.content_block_upserted.side_effect = emit_content_block

    async def discard_content_block(*, block_id):
        order.append(("discard", block_id))

    emitter.content_block_discarded.side_effect = discard_content_block

    def persist_message_fn(db, msg_id, conv_id, model_id, blocks, usage_data=None, partial=False):
        order.append(("persist", partial, list(blocks)))

    async def execute_tools_fn(*args, **kwargs):
        order.append(("execute",))
        return records

    async def complete_step_fn(**kwargs):
        order.append(("complete",))

    request = tool_round_module.ToolRoundRequest(
        db="db",
        assistant_message_id="msg-itinerary",
        conversation_id="conv-itinerary",
        user_id="user-1",
        model_id="model-1",
        provider="provider-1",
        content_blocks=state.content_blocks,
        messages=[{"role": "user", "content": "8月1日从深圳去上海，比较飞机和高铁"}],
        tool_calls=tool_calls,
        reasoning_buf="",
        should_use_reasoning=False,
        step_context=AgentStepContext(
            step_id="step-1",
            run_id="run-1",
            step_number=1,
            started_at=1.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        ),
        step_number=1,
        run_id="run-1",
        emitter=emitter,
        session_cache=object(),
        network_budget=Mock(max_tool_calls=20, completed_tool_calls=0),
        call_kwargs={},
        persist_message_fn=persist_message_fn,
        execute_tools_fn=execute_tools_fn,
        complete_step_fn=complete_step_fn,
        agent_state=state,
        resolve_tool_context_fn=AsyncMock(return_value=ToolContextResolution(executable_calls=tool_calls)),
    )
    return request, state, order


async def _check_tool_round_checkpoints_source_and_itinerary_before_both_events():
    train_call = {
        "id": "tc-train",
        "name": "search_trains",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    request, state, order = _request([_record(tool_call=train_call, status="success", block=_train_block())])

    outcome = await handle_tool_calls_round(request=request)

    itinerary = next(block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock))
    final_checkpoint_index = max(index for index, item in enumerate(order) if item[0] == "persist")
    emit_indexes = [index for index, item in enumerate(order) if item[0] == "emit"]
    assert all(final_checkpoint_index < index for index in emit_indexes)
    assert [item[2].type for item in order if item[0] == "emit"] == [
        "train_results",
        "itinerary_results",
    ]
    assert itinerary.id.startswith("blk_itinerary_")
    assert outcome.itinerary_result_count == 1
    assert outcome.product_outcomes[0].status == "available"
    assert outcome.product_outcomes[0].block_id == "blk-train-outbound"


async def _check_tool_round_records_safe_failure_and_builds_degraded_itinerary():
    flight_call = {
        "id": "tc-flight",
        "name": "search_flights",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    train_call = {
        "id": "tc-train",
        "name": "search_trains",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    records = [
        _record(tool_call=flight_call, status="failed"),
        _record(tool_call=train_call, status="success", block=_train_block()),
    ]
    request, state, _ = _request(records)

    outcome = await handle_tool_calls_round(request=request)

    itinerary = next(block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock))
    availability = {(item.journey, item.mode): item.status for item in itinerary.availability}
    assert itinerary.status == "degraded"
    assert availability[("outbound", "flight")] == "unavailable"
    assert availability[("outbound", "train")] == "available"
    assert {item.status for item in outcome.product_outcomes} == {"available", "unavailable"}
    assert all(not hasattr(item, "error_code") for item in outcome.product_outcomes)


async def _check_tool_round_updates_existing_itinerary_in_place_with_stable_id():
    train_call = {
        "id": "tc-train",
        "name": "search_trains",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    first_request, state, _ = _request([_record(tool_call=train_call, status="success", block=_train_block())])
    await handle_tool_calls_round(request=first_request)
    first = next(block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock))

    weather_call = {
        "id": "tc-weather",
        "name": "weather_forecast",
        "arguments": '{"location":"上海","location_source":"named"}',
    }
    second_request, _, order = _request(
        [_record(tool_call=weather_call, status="failed")],
        state=state,
    )

    await handle_tool_calls_round(request=second_request)

    itineraries = [block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock)]
    assert len(itineraries) == 1
    assert itineraries[0].id == first.id
    assert itineraries[0].status == "degraded"
    assert any(
        item.journey == "destination_weather" and item.status == "unavailable" for item in itineraries[0].availability
    )
    assert [item[2].type for item in order if item[0] == "emit"] == ["itinerary_results"]


async def _check_tool_round_does_not_create_empty_itinerary_when_all_travel_tools_fail():
    flight_call = {
        "id": "tc-flight",
        "name": "search_flights",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    request, state, _ = _request([_record(tool_call=flight_call, status="failed")])

    outcome = await handle_tool_calls_round(request=request)

    assert not any(isinstance(block, ItineraryResultsBlock) for block in state.content_blocks)
    assert outcome.itinerary_result_count == 0


async def _check_tool_round_does_not_publish_return_as_one_way_after_outbound_failed():
    failed_outbound_call = {
        "id": "tc-outbound",
        "name": "search_flights",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    first_request, state, _ = _request([_record(tool_call=failed_outbound_call, status="failed")])
    await handle_tool_calls_round(request=first_request)

    successful_return = _train_block().model_copy(
        update={
            "id": "blk-train-return",
            "origin": "上海",
            "destination": "深圳",
            "departure_date": "2026-08-03",
            "trains": [
                _train_block()
                .trains[0]
                .model_copy(
                    update={
                        "option_id": "train-return-1",
                        "train_no": "G101",
                        "departure": TravelEndpoint(
                            city="上海",
                            station_name="上海虹桥站",
                            scheduled_at=datetime(2026, 8, 3, 8, tzinfo=SHANGHAI_TZ),
                        ),
                        "arrival": TravelEndpoint(
                            city="深圳",
                            station_name="深圳北站",
                            scheduled_at=datetime(2026, 8, 3, 16, tzinfo=SHANGHAI_TZ),
                        ),
                    }
                )
            ],
        }
    )
    return_call = {
        "id": "tc-return",
        "name": "search_trains",
        "arguments": '{"origin":"上海","destination":"深圳","departure_date":"2026-08-03"}',
    }
    second_request, _, order = _request(
        [_record(tool_call=return_call, status="success", block=successful_return)],
        state=state,
    )

    outcome = await handle_tool_calls_round(request=second_request)

    assert not any(isinstance(block, ItineraryResultsBlock) for block in state.content_blocks)
    assert outcome.itinerary_result_count == 0
    assert [item[2].type for item in order if item[0] == "emit"] == ["train_results"]


async def _check_tool_round_discards_existing_return_itinerary_when_outbound_later_fails():
    successful_return = _train_block().model_copy(
        update={
            "id": "blk-train-return",
            "origin": "上海",
            "destination": "深圳",
            "departure_date": "2026-08-03",
            "trains": [
                _train_block()
                .trains[0]
                .model_copy(
                    update={
                        "option_id": "train-return-1",
                        "train_no": "G101",
                        "departure": TravelEndpoint(
                            city="上海",
                            station_name="上海虹桥站",
                            scheduled_at=datetime(2026, 8, 3, 8, tzinfo=SHANGHAI_TZ),
                        ),
                        "arrival": TravelEndpoint(
                            city="深圳",
                            station_name="深圳北站",
                            scheduled_at=datetime(2026, 8, 3, 16, tzinfo=SHANGHAI_TZ),
                        ),
                    }
                )
            ],
        }
    )
    return_call = {
        "id": "tc-return-first",
        "name": "search_trains",
        "arguments": '{"origin":"上海","destination":"深圳","departure_date":"2026-08-03"}',
    }
    first_request, state, _ = _request([_record(tool_call=return_call, status="success", block=successful_return)])
    await handle_tool_calls_round(request=first_request)
    old_itinerary = next(block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock))

    failed_outbound_call = {
        "id": "tc-outbound-later",
        "name": "search_flights",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    second_request, _, order = _request(
        [_record(tool_call=failed_outbound_call, status="failed")],
        state=state,
    )

    outcome = await handle_tool_calls_round(request=second_request)

    assert not any(isinstance(block, ItineraryResultsBlock) for block in state.content_blocks)
    assert outcome.itinerary_result_count == 0
    assert [item for item in order if item[0] in {"emit", "discard"}] == [
        ("discard", old_itinerary.id),
    ]


async def _check_tool_round_discards_old_stream_itinerary_when_identity_changes():
    old_source = _train_block().model_copy(
        update={
            "id": "blk-old-trip",
            "origin": "上海",
            "destination": "深圳",
            "departure_date": "2026-08-03",
            "trains": [
                _train_block()
                .trains[0]
                .model_copy(
                    update={
                        "option_id": "old-trip-1",
                        "departure": TravelEndpoint(
                            city="上海",
                            station_name="上海虹桥站",
                            scheduled_at=datetime(2026, 8, 3, 8, tzinfo=SHANGHAI_TZ),
                        ),
                        "arrival": TravelEndpoint(
                            city="深圳",
                            station_name="深圳北站",
                            scheduled_at=datetime(2026, 8, 3, 16, tzinfo=SHANGHAI_TZ),
                        ),
                    }
                )
            ],
        }
    )
    old_itinerary = compose_itinerary_result([old_source])
    assert old_itinerary is not None
    state = AgentLoopState(content_blocks=[old_itinerary])
    new_call = {
        "id": "tc-new-trip",
        "name": "search_trains",
        "arguments": '{"origin":"深圳","destination":"上海","departure_date":"2026-08-01"}',
    }
    request, _, order = _request(
        [_record(tool_call=new_call, status="success", block=_train_block())],
        state=state,
    )

    await handle_tool_calls_round(request=request)

    new_itinerary = next(block for block in state.content_blocks if isinstance(block, ItineraryResultsBlock))
    assert new_itinerary.id != old_itinerary.id
    stream_events = [item for item in order if item[0] in {"emit", "discard"}]
    assert [(item[0], item[1]) for item in stream_events] == [
        ("emit", "tc-new-trip"),
        ("discard", old_itinerary.id),
        ("emit", f"itinerary:{new_itinerary.id}"),
    ]


def test_tool_round_checkpoints_source_and_itinerary_before_both_events():
    asyncio.run(_check_tool_round_checkpoints_source_and_itinerary_before_both_events())


def test_tool_round_records_safe_failure_and_builds_degraded_itinerary():
    asyncio.run(_check_tool_round_records_safe_failure_and_builds_degraded_itinerary())


def test_tool_round_updates_existing_itinerary_in_place_with_stable_id():
    asyncio.run(_check_tool_round_updates_existing_itinerary_in_place_with_stable_id())


def test_tool_round_does_not_create_empty_itinerary_when_all_travel_tools_fail():
    asyncio.run(_check_tool_round_does_not_create_empty_itinerary_when_all_travel_tools_fail())


def test_tool_round_does_not_publish_return_as_one_way_after_outbound_failed():
    asyncio.run(_check_tool_round_does_not_publish_return_as_one_way_after_outbound_failed())


def test_tool_round_discards_existing_return_itinerary_when_outbound_later_fails():
    asyncio.run(_check_tool_round_discards_existing_return_itinerary_when_outbound_later_fails())


def test_tool_round_discards_old_stream_itinerary_when_identity_changes():
    asyncio.run(_check_tool_round_discards_old_stream_itinerary_when_identity_changes())
