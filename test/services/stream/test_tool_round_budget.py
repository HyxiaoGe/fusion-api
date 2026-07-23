import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.step_lifecycle import AgentStepContext
from app.services.stream.tool_context import BlockedToolContext, ToolContextResolution, ToolRuntimeContext
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_round import ToolRoundOutcome, ToolRoundRequest, handle_tool_calls_round
from app.services.stream_state_service import StreamWriteTerminalError
from app.services.tool_handlers.base import ToolResult


def _tool_calls(count: int) -> list[dict]:
    return [
        {
            "id": f"tc-{index}",
            "name": "web_search",
            "arguments": json.dumps({"query": f"query-{index}"}),
        }
        for index in range(1, count + 1)
    ]


def _record(
    tool_call: dict,
    *,
    status: str = "success",
    search_budget: str | None = None,
) -> ToolExecutionRecord:
    handler = Mock()
    handler.format_llm_context.return_value = f"已执行 {tool_call['id']}"
    handler.build_content_block.return_value = None
    data = {"search_budget": search_budget} if search_budget is not None else {}
    return ToolExecutionRecord(
        tool_call=tool_call,
        result=ToolResult(status=status, data=data),
        handler=handler,
        block_id=f"blk-{tool_call['id']}",
        log_id=f"log-{tool_call['id']}",
    )


def _request(
    *,
    tool_calls: list[dict],
    completed_tool_calls: int | None,
    max_tool_calls: int | None,
    execute_tools_fn,
    on_tools_executed=None,
    complete_step_fn=None,
    network_budget=None,
    agent_state=None,
    resolve_tool_context_fn=None,
    tool_handlers=None,
    emitter=None,
) -> ToolRoundRequest:
    kwargs = dict(
        db="db",
        assistant_message_id="msg-1",
        conversation_id="conv-1",
        user_id="user-1",
        model_id="gpt-4",
        provider="openai",
        content_blocks=[],
        messages=[{"role": "user", "content": "请搜索"}],
        tool_calls=tool_calls,
        reasoning_buf="",
        should_use_reasoning=True,
        step_context=AgentStepContext(
            step_id="step-1",
            run_id="run-1",
            step_number=1,
            started_at=10.0,
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        ),
        step_number=1,
        run_id="run-1",
        emitter=SimpleNamespace() if emitter is None else (None if emitter is False else emitter),
        session_cache=SimpleNamespace(),
        network_budget=network_budget or SimpleNamespace(),
        call_kwargs={},
        persist_message_fn=Mock(),
        execute_tools_fn=execute_tools_fn,
        complete_step_fn=complete_step_fn or AsyncMock(),
        on_tools_executed=on_tools_executed,
        completed_tool_calls=completed_tool_calls,
        max_tool_calls=max_tool_calls,
        clock=Mock(return_value=10.5),
        tool_handlers=tool_handlers,
    )
    if agent_state is not None:
        kwargs["agent_state"] = agent_state
    if resolve_tool_context_fn is not None:
        kwargs["resolve_tool_context_fn"] = resolve_tool_context_fn
    return ToolRoundRequest(**kwargs)


class ToolRoundBudgetTests(IsolatedAsyncioTestCase):
    async def test_second_identical_successful_travel_round_reuses_result_without_consuming_quota(self):
        from app.services.mcp.flyai_travel_tools import (
            FLYAI_SEARCH_TRAINS,
            FlyAiTravelToolHandler,
            build_flyai_travel_binding,
        )
        from app.services.stream.tool_executor import execute_tools_parallel

        handler = FlyAiTravelToolHandler(
            binding=build_flyai_travel_binding(FLYAI_SEARCH_TRAINS),
            client=Mock(),
            controls=Mock(),
            user_scope="scope",
        )
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result_count": 1}))
        handler.log = AsyncMock()
        state = AgentLoopState()
        on_tools_executed = Mock(wraps=state.record_executed_tool_calls)
        first_call = {
            "id": "tc-train-1",
            "name": FLYAI_SEARCH_TRAINS,
            "arguments": {
                "origin": "深圳",
                "destination": "广州",
                "departure_date": "2026-07-25",
            },
        }
        repeated_call = {
            "id": "tc-train-2",
            "name": FLYAI_SEARCH_TRAINS,
            "arguments": json.dumps(
                {
                    "limit": 5,
                    "sort_by": "recommended",
                    "departure_date": "2026-07-25",
                    "destination": "广州",
                    "origin": " 深圳 ",
                }
            ),
        }

        first_request = _request(
            tool_calls=[first_call],
            completed_tool_calls=0,
            max_tool_calls=1,
            execute_tools_fn=execute_tools_parallel,
            on_tools_executed=on_tools_executed,
            agent_state=state,
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            emitter=False,
            resolve_tool_context_fn=AsyncMock(return_value=ToolContextResolution(executable_calls=[first_call])),
        )
        second_complete_step = AsyncMock()
        second_request = _request(
            tool_calls=[repeated_call],
            completed_tool_calls=1,
            max_tool_calls=1,
            execute_tools_fn=execute_tools_parallel,
            on_tools_executed=on_tools_executed,
            complete_step_fn=second_complete_step,
            agent_state=state,
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            emitter=False,
            resolve_tool_context_fn=AsyncMock(return_value=ToolContextResolution(executable_calls=[repeated_call])),
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()) as mark_started:
            first_outcome = await handle_tool_calls_round(request=first_request)
            second_outcome = await handle_tool_calls_round(request=second_request)

        self.assertEqual(first_outcome.tool_call_count, 1)
        self.assertEqual(second_outcome.tool_call_count, 0)
        self.assertEqual(state.total_tool_calls, 1)
        self.assertEqual(handler.execute.await_count, 1)
        self.assertEqual(handler.log.await_count, 1)
        self.assertEqual(mark_started.await_count, 1)
        self.assertEqual(second_complete_step.await_args.kwargs["tool_call_count"], 0)
        self.assertEqual(second_complete_step.await_args.kwargs["completed_tool_calls"], 1)
        self.assertEqual(
            [message["role"] for message in second_request.messages],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(second_request.messages[-1]["tool_call_id"], "tc-train-2")
        self.assertIn("复用上一条成功结果", second_request.messages[-1]["content"])
        self.assertEqual(second_request.content_blocks, [])

    async def test_qwen_four_train_calls_with_identical_last_two_count_three_actual_executions(self):
        from app.services.mcp.flyai_travel_tools import (
            FLYAI_SEARCH_TRAINS,
            FlyAiTravelToolHandler,
            build_flyai_travel_binding,
        )
        from app.services.stream.tool_executor import execute_tools_parallel

        handler = FlyAiTravelToolHandler(
            binding=build_flyai_travel_binding(FLYAI_SEARCH_TRAINS),
            client=Mock(),
            controls=Mock(),
            user_scope="scope",
        )
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result_count": 1}))
        handler.log = AsyncMock()
        state = AgentLoopState()
        calls = [
            {
                "id": f"tc-train-{index}",
                "name": FLYAI_SEARCH_TRAINS,
                "arguments": {
                    "origin": "深圳",
                    "destination": "广州",
                    "departure_date": "2026-07-25",
                    "departure_hour_start": hour,
                },
            }
            for index, hour in enumerate((6, 8, 10, 10), 1)
        ]
        request = _request(
            tool_calls=calls,
            completed_tool_calls=0,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_parallel,
            on_tools_executed=state.record_executed_tool_calls,
            agent_state=state,
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            emitter=False,
            resolve_tool_context_fn=AsyncMock(return_value=ToolContextResolution(executable_calls=calls)),
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(outcome.tool_call_count, 3)
        self.assertEqual(state.total_tool_calls, 3)
        self.assertEqual(handler.execute.await_count, 3)
        self.assertEqual(handler.log.await_count, 3)
        self.assertEqual(
            [message["role"] for message in request.messages],
            ["user", "assistant", "tool", "tool", "tool", "tool"],
        )
        self.assertIn("复用上一条成功结果", request.messages[-1]["content"])
        self.assertEqual(request.content_blocks, [])

    async def test_denied_context_executes_no_tool_consumes_no_quota_and_appends_synthetic_result(self):
        tool_call = {
            "id": "tc-current",
            "name": "local_place_search",
            "arguments": json.dumps({"query": "咖啡", "anchor_source": "current_location"}),
        }
        state = AgentLoopState()
        on_tools_executed = Mock(wraps=state.record_executed_tool_calls)
        execute_tools_fn = AsyncMock(return_value=[])
        resolver = AsyncMock(
            return_value=ToolContextResolution(
                executable_calls=[],
                blocked_calls={"tc-current": BlockedToolContext(status="denied", reason="permission_denied")},
            )
        )
        request = _request(
            tool_calls=[tool_call],
            completed_tool_calls=0,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
            agent_state=state,
            resolve_tool_context_fn=resolver,
        )

        outcome = await handle_tool_calls_round(request=request)

        execute_tools_fn.assert_not_awaited()
        self.assertEqual(outcome.tool_call_count, 0)
        self.assertEqual(state.total_tool_calls, 0)
        self.assertEqual(request.messages[-1]["role"], "tool")
        payload = json.loads(request.messages[-1]["content"])
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["context_status"], "denied")

    async def test_provided_context_is_forwarded_internally_and_actual_tool_counts_once(self):
        tool_call = {
            "id": "tc-current",
            "name": "local_place_search",
            "arguments": json.dumps({"query": "咖啡", "anchor_source": "current_location"}),
        }
        record = _record(tool_call)
        state = AgentLoopState()
        on_tools_executed = Mock(wraps=state.record_executed_tool_calls)
        execute_tools_fn = AsyncMock(return_value=[record])
        runtime_context = ToolRuntimeContext()
        resolver = AsyncMock(
            return_value=ToolContextResolution(
                executable_calls=[tool_call],
                runtime_context=runtime_context,
            )
        )
        request = _request(
            tool_calls=[tool_call],
            completed_tool_calls=0,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
            agent_state=state,
            resolve_tool_context_fn=resolver,
        )

        outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(outcome.tool_call_count, 1)
        self.assertEqual(state.total_tool_calls, 1)
        self.assertIs(execute_tools_fn.await_args.kwargs["runtime_context"], runtime_context)

    async def test_same_batch_marks_two_budget_stopped_search_results_as_no_progress(self):
        tool_calls = _tool_calls(2)
        execute_tools_fn = AsyncMock(
            return_value=[
                _record(tool_calls[0], status="degraded", search_budget="planner_limited"),
                _record(tool_calls[1], status="degraded", search_budget="duplicate_skipped"),
            ]
        )
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=0,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        state = AgentLoopState()
        state.record_no_progress_search_results(outcome.no_progress_search_results)
        self.assertEqual(outcome.no_progress_search_results, (True, True))
        self.assertTrue(state.should_summarize_no_progress_search())

    async def test_successful_search_resets_no_progress_results_in_same_batch(self):
        tool_calls = _tool_calls(3)
        execute_tools_fn = AsyncMock(
            return_value=[
                _record(tool_calls[0], status="degraded", search_budget="planner_limited"),
                _record(tool_calls[1], status="success", search_budget="normal"),
                _record(tool_calls[2], status="degraded", search_budget="duplicate_skipped"),
            ]
        )
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=0,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        state = AgentLoopState()
        state.record_no_progress_search_results(outcome.no_progress_search_results)
        self.assertEqual(outcome.no_progress_search_results, (True, False, True))
        self.assertEqual(state.consecutive_no_progress_search_results, 1)
        self.assertFalse(state.should_summarize_no_progress_search())

    async def test_excess_batch_executes_only_remaining_capacity_and_completes_protocol(self):
        tool_calls = _tool_calls(5)
        executed_record = _record(tool_calls[0])
        execute_tools_fn = AsyncMock(return_value=[executed_record])
        state = AgentLoopState(total_tool_calls=19)
        on_tools_executed = Mock(wraps=state.record_executed_tool_calls)
        complete_step_fn = AsyncMock()
        network_budget = SimpleNamespace(record_tool_results=Mock())
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=19,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
            complete_step_fn=complete_step_fn,
            network_budget=network_budget,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()) as mark_started:
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(
            outcome,
            ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["web_search"],
                no_progress_search_results=(False,),
            ),
        )
        self.assertEqual(execute_tools_fn.await_args.args[0], [tool_calls[0]])
        on_tools_executed.assert_called_once_with(1)
        self.assertEqual(state.total_tool_calls, 20)
        self.assertEqual(state.run_stats("run-1").total_tool_calls, 20)
        self.assertEqual(mark_started.await_args.kwargs["tool_call_count"], 1)
        self.assertEqual(mark_started.await_args.kwargs["tool_names"], ["web_search"])
        complete_step_fn.assert_awaited_once()
        self.assertEqual(complete_step_fn.await_args.kwargs["tool_call_count"], 1)
        self.assertEqual(complete_step_fn.await_args.kwargs["completed_tool_calls"], 20)
        network_budget.record_tool_results.assert_called_once()
        self.assertEqual(network_budget.record_tool_results.call_args.kwargs["results"], [executed_record])

        self.assertEqual(
            [call["id"] for call in request.messages[1]["tool_calls"]], [call["id"] for call in tool_calls]
        )
        tool_messages = request.messages[2:]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], [call["id"] for call in tool_calls])
        self.assertEqual(tool_messages[0]["content"], "已执行 tc-1")
        for message in tool_messages[1:]:
            result = json.loads(message["content"])
            self.assertEqual(result["status"], "not_executed")
            self.assertEqual(result["reason"], "limit_reached")
            self.assertEqual(result["limit_reason"], "max_tool_calls")

    async def test_zero_remaining_capacity_does_not_invoke_executor_or_count_calls(self):
        tool_calls = _tool_calls(2)
        execute_tools_fn = AsyncMock()
        on_tools_executed = Mock()
        complete_step_fn = AsyncMock()
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=20,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
            complete_step_fn=complete_step_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()) as mark_started:
            outcome = await handle_tool_calls_round(request=request)

        execute_tools_fn.assert_not_awaited()
        on_tools_executed.assert_called_once_with(0)
        self.assertEqual(outcome.tool_call_count, 0)
        self.assertEqual(outcome.tool_names, [])
        mark_started.assert_not_awaited()
        self.assertEqual(complete_step_fn.await_args.kwargs["completed_tool_calls"], 20)
        self.assertTrue(
            all(json.loads(message["content"])["status"] == "not_executed" for message in request.messages[2:])
        )

    async def test_batch_equal_to_remaining_capacity_executes_all_calls(self):
        tool_calls = _tool_calls(3)
        execute_tools_fn = AsyncMock(return_value=[_record(tool_call) for tool_call in tool_calls])
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=17,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(execute_tools_fn.await_args.args[0], tool_calls)
        self.assertEqual(outcome.tool_call_count, 3)
        self.assertEqual(len(request.messages), 5)
        self.assertTrue(all(not message["content"].startswith("{") for message in request.messages[2:]))

    async def test_missing_limit_keeps_legacy_execute_all_behavior(self):
        tool_calls = _tool_calls(3)
        execute_tools_fn = AsyncMock(return_value=[_record(tool_call) for tool_call in tool_calls])
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=19,
            max_tool_calls=None,
            execute_tools_fn=execute_tools_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(execute_tools_fn.await_args.args[0], tool_calls)
        self.assertEqual(outcome.tool_call_count, 3)

    async def test_missing_completed_count_keeps_legacy_execute_all_behavior(self):
        tool_calls = _tool_calls(3)
        execute_tools_fn = AsyncMock(return_value=[_record(tool_call) for tool_call in tool_calls])
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=None,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(execute_tools_fn.await_args.args[0], tool_calls)
        self.assertEqual(outcome.tool_call_count, 3)

    async def test_executor_terminal_error_still_counts_submitted_calls_once(self):
        tool_calls = _tool_calls(5)
        state = AgentLoopState(total_tool_calls=19)
        on_tools_executed = Mock(wraps=state.record_executed_tool_calls)
        execute_tools_fn = AsyncMock(side_effect=StreamWriteTerminalError("SSE 写入失败"))
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=19,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
        )

        with (
            patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()),
            self.assertRaises(StreamWriteTerminalError),
        ):
            await handle_tool_calls_round(request=request)

        self.assertEqual(execute_tools_fn.await_args.args[0], [tool_calls[0]])
        on_tools_executed.assert_called_once_with(1)
        self.assertEqual(state.total_tool_calls, 20)

    async def test_missing_execution_records_get_safe_tool_responses(self):
        tool_calls = _tool_calls(3)
        first_record = _record(tool_calls[0])
        execute_tools_fn = AsyncMock(return_value=[first_record])
        on_tools_executed = Mock()
        network_budget = SimpleNamespace(record_tool_results=Mock())
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=17,
            max_tool_calls=20,
            execute_tools_fn=execute_tools_fn,
            on_tools_executed=on_tools_executed,
            network_budget=network_budget,
        )

        with patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(outcome.tool_call_count, 3)
        on_tools_executed.assert_called_once_with(3)
        self.assertEqual(network_budget.record_tool_results.call_args.kwargs["results"], [first_record])
        tool_messages = request.messages[2:]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["tc-1", "tc-2", "tc-3"])
        for message in tool_messages[1:]:
            payload = json.loads(message["content"])
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["reason"], "execution_result_missing")
            self.assertEqual(payload["message"], "工具执行未返回可用记录，本次结果不能作为事实依据。")
            self.assertNotIn("SSE", message["content"])

    async def test_duplicate_execution_records_are_ignored(self):
        tool_calls = _tool_calls(2)
        first_record = _record(tool_calls[0])
        duplicate_record = _record(tool_calls[0])
        second_record = _record(tool_calls[1])
        network_budget = SimpleNamespace(record_tool_results=Mock())
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=18,
            max_tool_calls=20,
            execute_tools_fn=AsyncMock(return_value=[first_record, duplicate_record, second_record]),
            network_budget=network_budget,
        )

        with (
            patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()),
            patch("app.services.stream.tool_round.emit_selected_source_evidence", new=AsyncMock()) as emit_evidence,
        ):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(outcome.tool_call_count, 2)
        self.assertEqual(network_budget.record_tool_results.call_args.kwargs["results"], [first_record, second_record])
        self.assertEqual(emit_evidence.await_args.args[1], [first_record, second_record])
        self.assertEqual([message["tool_call_id"] for message in request.messages[2:]], ["tc-1", "tc-2"])

    async def test_extra_execution_records_do_not_enter_messages_or_feedback(self):
        tool_calls = _tool_calls(3)
        selected_record = _record(tool_calls[0])
        excluded_record = _record(tool_calls[1])
        unknown_record = _record(_tool_calls(999)[-1])
        network_budget = SimpleNamespace(record_tool_results=Mock())
        request = _request(
            tool_calls=tool_calls,
            completed_tool_calls=19,
            max_tool_calls=20,
            execute_tools_fn=AsyncMock(return_value=[selected_record, excluded_record, unknown_record]),
            network_budget=network_budget,
        )

        with (
            patch("app.services.stream.tool_round.mark_tool_round_started", new=AsyncMock()),
            patch("app.services.stream.tool_round.emit_selected_source_evidence", new=AsyncMock()) as emit_evidence,
        ):
            outcome = await handle_tool_calls_round(request=request)

        self.assertEqual(
            outcome,
            ToolRoundOutcome(
                tool_call_count=1,
                tool_names=["web_search"],
                no_progress_search_results=(False,),
            ),
        )
        self.assertEqual(network_budget.record_tool_results.call_args.kwargs["results"], [selected_record])
        self.assertEqual(emit_evidence.await_args.args[1], [selected_record])
        tool_messages = request.messages[2:]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["tc-1", "tc-2", "tc-3"])
        self.assertTrue(all(message["tool_call_id"] != "tc-999" for message in tool_messages))
        for message in tool_messages[1:]:
            self.assertEqual(json.loads(message["content"])["status"], "not_executed")
