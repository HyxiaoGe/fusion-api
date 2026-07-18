import unittest
from unittest.mock import AsyncMock

from app.services.agent.context_broker import Geolocation, ResolvedContext
from app.services.stream.agent_loop_state import AgentLoopState


class ToolContextResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_broker_failure_blocks_dependent_call_without_crashing_or_reprompting(self):
        from app.services.stream.tool_context import resolve_tool_context

        state = AgentLoopState()
        emitter = AsyncMock()
        create_request = AsyncMock(side_effect=RuntimeError("redis down"))
        current = {
            "id": "local-current",
            "name": "local_place_search",
            "arguments": '{"query":"咖啡","anchor_source":"current_location"}',
        }

        first = await resolve_tool_context(
            tool_calls=[current],
            state=state,
            emitter=emitter,
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: 100.0,
            create_request_fn=create_request,
            wait_result_fn=AsyncMock(),
        )
        second = await resolve_tool_context(
            tool_calls=[current],
            state=state,
            emitter=emitter,
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: 101.0,
            create_request_fn=create_request,
            wait_result_fn=AsyncMock(),
        )

        self.assertEqual(first.blocked_calls["local-current"].status, "unavailable")
        self.assertEqual(second.executable_calls, [])
        create_request.assert_awaited_once()
        emitter.context_required.assert_not_awaited()

    async def test_named_and_city_searches_do_not_request_geolocation(self):
        from app.services.stream.tool_context import resolve_tool_context

        create_request = AsyncMock()
        result = await resolve_tool_context(
            tool_calls=[
                {
                    "id": "local-named",
                    "name": "local_place_search",
                    "arguments": '{"query":"咖啡","near":"民治地铁站","anchor_source":"named"}',
                },
                {
                    "id": "local-city",
                    "name": "local_place_search",
                    "arguments": '{"query":"咖啡","city":"深圳","anchor_source":"none"}',
                },
                {
                    "id": "route-named",
                    "name": "route_compare",
                    "arguments": '{"origin":"民治","destination":"市民中心"}',
                },
            ],
            state=AgentLoopState(),
            emitter=AsyncMock(),
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: 100.0,
            create_request_fn=create_request,
            wait_result_fn=AsyncMock(),
        )

        self.assertEqual([call["id"] for call in result.executable_calls], ["local-named", "local-city", "route-named"])
        self.assertEqual(result.blocked_calls, {})
        create_request.assert_not_awaited()

    async def test_current_location_waits_in_same_state_caches_result_and_excludes_wait_time(self):
        from app.services.stream.tool_context import resolve_tool_context

        state = AgentLoopState()
        emitter = AsyncMock()
        create_request = AsyncMock(return_value=object())
        wait_result = AsyncMock(
            return_value=ResolvedContext(
                request_id="ctx-1",
                status="provided",
                location=Geolocation(latitude=22.616, longitude=114.031, accuracy_m=25, acquired_at=99),
            )
        )
        times = iter((100.0, 145.0))
        call = {
            "id": "local-current",
            "name": "local_place_search",
            "arguments": '{"query":"咖啡","anchor_source":"current_location"}',
        }

        first = await resolve_tool_context(
            tool_calls=[call],
            state=state,
            emitter=emitter,
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: next(times),
            request_id_factory=lambda: "ctx-1",
            create_request_fn=create_request,
            wait_result_fn=wait_result,
        )

        self.assertEqual(first.executable_calls, [call])
        self.assertEqual(first.runtime_context.geolocation.latitude, 22.616)
        self.assertEqual(state.context_wait_seconds, 45.0)
        emitter.context_required.assert_awaited_once()
        emitter.context_result.assert_awaited_once_with(
            request_id="ctx-1",
            context_type="geolocation",
            status="provided",
        )

        second = await resolve_tool_context(
            tool_calls=[call],
            state=state,
            emitter=emitter,
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: 200.0,
            create_request_fn=create_request,
            wait_result_fn=wait_result,
        )
        self.assertEqual(second.runtime_context.geolocation.latitude, 22.616)
        create_request.assert_awaited_once()
        wait_result.assert_awaited_once()

    async def test_denied_blocks_only_dependent_calls_without_reprompting(self):
        from app.services.stream.tool_context import resolve_tool_context

        state = AgentLoopState()
        create_request = AsyncMock(return_value=object())
        wait_result = AsyncMock(
            return_value=ResolvedContext(request_id="ctx-1", status="denied", reason="permission_denied")
        )
        current = {
            "id": "route-current",
            "name": "route_compare",
            "arguments": (
                '{"origin":"当前位置","origin_source":"current_location",'
                '"destination":"市民中心","destination_source":"named"}'
            ),
        }
        independent = {
            "id": "local-city",
            "name": "local_place_search",
            "arguments": '{"query":"咖啡","city":"深圳","anchor_source":"none"}',
        }
        times = iter((100.0, 101.0))

        first = await resolve_tool_context(
            tool_calls=[current, independent],
            state=state,
            emitter=AsyncMock(),
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: next(times),
            request_id_factory=lambda: "ctx-1",
            create_request_fn=create_request,
            wait_result_fn=wait_result,
        )

        self.assertEqual(first.executable_calls, [independent])
        self.assertEqual(first.blocked_calls["route-current"].status, "denied")
        self.assertEqual(state.context_wait_seconds, 1.0)

        second = await resolve_tool_context(
            tool_calls=[current],
            state=state,
            emitter=AsyncMock(),
            user_id="user-1",
            conversation_id="conv-1",
            message_id="msg-1",
            run_id="run-1",
            task_id="task-1",
            clock=lambda: 200.0,
            create_request_fn=create_request,
            wait_result_fn=wait_result,
        )
        self.assertEqual(second.executable_calls, [])
        self.assertEqual(second.blocked_calls["route-current"].status, "denied")
        create_request.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
