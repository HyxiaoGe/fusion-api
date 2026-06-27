"""agent-loop 跨模块契约测试。"""

import json
import unittest
from contextlib import ExitStack
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import SearchBlock
from app.services.stream import StreamHandler
from app.services.tool_handlers.base import ToolResult


@dataclass
class AgentLoopContractResult:
    events: list[dict] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    append_calls: list[dict] = field(default_factory=list)
    persist_calls: list[dict] = field(default_factory=list)
    finalize_calls: list[dict] = field(default_factory=list)
    session_started_calls: list[dict] = field(default_factory=list)
    session_status_calls: list[dict] = field(default_factory=list)
    step_started_calls: list[dict] = field(default_factory=list)
    step_completed_calls: list[dict] = field(default_factory=list)
    step_terminal_calls: list[dict] = field(default_factory=list)
    tool_execute_calls: list[dict] = field(default_factory=list)
    redis_entries: list = field(default_factory=list)
    redis_entry_types: list[str] = field(default_factory=list)
    redis_meta: dict = field(default_factory=dict)


class AgentLoopContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = StreamHandler()
        self.last_result = AgentLoopContractResult()

    async def _run_agent_contract(
        self,
        *,
        rounds,
        execute_tools_result=None,
        use_real_tool_executor=False,
        fake_handler=None,
        use_real_redis_stream=False,
        capabilities=None,
        options=None,
    ) -> AgentLoopContractResult:
        result = AgentLoopContractResult()
        self.last_result = result

        async def _capture_append(conversation_id, chunk_type, content, block_id, **extras):
            call = {
                "conversation_id": conversation_id,
                "chunk_type": chunk_type,
                "content": content,
                "block_id": block_id,
                **extras,
            }
            result.append_calls.append(call)
            if chunk_type == "agent_event":
                event = json.loads(content)
                result.events.append(event)
                result.event_types.append(event["type"])
            return f"{len(result.append_calls)}-0"

        def _capture_persist(db, msg_id, conv_id, model_id, content_blocks, usage_data=None, partial=False):
            result.persist_calls.append(
                {
                    "message_id": msg_id,
                    "conversation_id": conv_id,
                    "model_id": model_id,
                    "partial": partial,
                    "block_types": [getattr(block, "type", None) for block in content_blocks],
                    "block_ids": [getattr(block, "id", None) for block in content_blocks],
                    "usage": usage_data,
                }
            )

        async def _capture_finalize(conversation_id, success, error_msg="", task_id="", **kwargs):
            result.finalize_calls.append(
                {
                    "conversation_id": conversation_id,
                    "success": success,
                    "task_id": task_id,
                    "error_msg": error_msg,
                    **kwargs,
                }
            )

        async def _capture_session_started(**kwargs):
            result.session_started_calls.append(dict(kwargs))

        async def _capture_session_status(**kwargs):
            result.session_status_calls.append(dict(kwargs))

        async def _capture_step_started(**kwargs):
            result.step_started_calls.append(dict(kwargs))

        async def _capture_step_completed(**kwargs):
            result.step_completed_calls.append(dict(kwargs))

        async def _capture_step_terminal(**kwargs):
            result.step_terminal_calls.append(dict(kwargs))

        async def _capture_execute_tools(*args, **kwargs):
            result.tool_execute_calls.append({"args": args, **kwargs})
            return execute_tools_result or []

        async def _capture_and_execute_tools(*args, **kwargs):
            from app.services.stream import tool_executor

            result.tool_execute_calls.append({"args": args, **kwargs})
            return await tool_executor.execute_tools_parallel(*args, **kwargs)

        async def _raise_round(*_args, **_kwargs):
            raise rounds

        stream_round_side_effect = _raise_round if isinstance(rounds, BaseException) else list(rounds)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        fake_redis = None
        caught_exc = None

        with ExitStack() as stack:
            if use_real_redis_stream:
                import fakeredis.aioredis

                from app.services.stream_state_service import init_stream

                fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
                stack.enter_context(
                    patch(
                        "app.services.stream_state_service.get_redis_pool",
                        return_value=fake_redis,
                    )
                )
                await init_stream("conv-contract", "user-contract", "gpt-4", "msg-contract", "task-contract")

            stack.enter_context(patch("app.services.stream.runner.SessionLocal", return_value=mock_db))
            if not use_real_redis_stream:
                stack.enter_context(patch("app.services.stream.runner.append_chunk", side_effect=_capture_append))
                stack.enter_context(
                    patch("app.services.stream.tool_executor.append_chunk", side_effect=_capture_append)
                )
                stack.enter_context(patch("app.services.stream.runner.finalize_stream", side_effect=_capture_finalize))
            stack.enter_context(patch("app.services.stream.runner.persist_message", side_effect=_capture_persist))
            stack.enter_context(
                patch(
                    "app.services.stream.runner.build_llm_messages",
                    AsyncMock(return_value=[{"role": "user", "content": "hi"}]),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.stream.runner.llm_call_with_retry",
                    AsyncMock(return_value=MagicMock()),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.stream.runner.stream_round",
                    AsyncMock(side_effect=stream_round_side_effect),
                )
            )
            if use_real_tool_executor:
                stack.enter_context(
                    patch(
                        "app.services.stream.runner.execute_tools_parallel",
                        AsyncMock(side_effect=_capture_and_execute_tools),
                    )
                )
                stack.enter_context(patch("app.services.tool_handlers.get_handler", return_value=fake_handler))
            else:
                stack.enter_context(
                    patch(
                        "app.services.stream.runner.execute_tools_parallel",
                        AsyncMock(side_effect=_capture_execute_tools),
                    )
                )
            stack.enter_context(
                patch("app.services.agent.session_cache.write_session_started", side_effect=_capture_session_started)
            )
            stack.enter_context(
                patch("app.services.agent.session_cache.write_session_status", side_effect=_capture_session_status)
            )
            stack.enter_context(
                patch("app.services.agent.session_cache.write_step_started", side_effect=_capture_step_started)
            )
            stack.enter_context(
                patch("app.services.agent.session_cache.write_step_completed", side_effect=_capture_step_completed)
            )
            stack.enter_context(
                patch("app.services.agent.session_cache.write_step_terminal", side_effect=_capture_step_terminal)
            )

            try:
                await self.handler.generate_to_redis(
                    conversation_id="conv-contract",
                    user_id="user-contract",
                    model_id="gpt-4",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    provider="openai",
                    raw_messages=[{"role": "user", "content": "hi"}],
                    has_vision=False,
                    file_ids=None,
                    original_message="hi",
                    assistant_message_id="msg-contract",
                    task_id="task-contract",
                    options=options or {"use_reasoning": False},
                    capabilities=capabilities or {},
                    trace_id="run-contract",
                )
            except BaseException as exc:
                caught_exc = exc
            finally:
                if fake_redis is not None:
                    result.redis_entries = await fake_redis.xrange("stream:chunks:conv-contract")
                    result.redis_entry_types = [entry[1].get("type") for entry in result.redis_entries]
                    result.redis_meta = await fake_redis.hgetall("stream:meta:conv-contract")
                    await fake_redis.flushall()
                    await fake_redis.aclose()

        if caught_exc is not None:
            raise caught_exc

        return result

    async def test_contract_harness_records_events_persist_and_session_state(self):
        result = await self._run_agent_contract(
            rounds=[("", "Final answer", [], "stop", None)],
        )

        self.assertEqual(
            result.event_types,
            ["run_started", "step_started", "step_completed", "run_completed"],
        )
        self.assertEqual(
            result.finalize_calls[-1],
            {
                "conversation_id": "conv-contract",
                "success": True,
                "task_id": "task-contract",
                "error_msg": "",
            },
        )

    async def test_tool_round_contract_records_event_order_and_partial_persist(self):
        class FakeSearchHandler:
            tool_name = "web_search"

            async def execute(self, args):
                return ToolResult(
                    status="success",
                    data={"query": args["query"], "sources": [{"title": "OpenAI", "url": "https://openai.com"}]},
                    duration_ms=11,
                )

            async def log(self, **_kwargs):
                return None

            def _build_result_summary(self, result):
                return {"kind": "search", "status": result.status, "source_count": 1}

            def format_llm_context(self, result):
                return "搜索结果：OpenAI"

            def build_content_block(self, result, block_id, log_id):
                return SearchBlock(
                    type="search",
                    id=block_id,
                    query=result.data["query"],
                    tool_call_log_id=log_id,
                    sources=[],
                    status="success",
                    source_count=1,
                )

        tool_call = {"id": "tc-contract", "name": "web_search", "arguments": '{"query":"OpenAI"}'}

        result = await self._run_agent_contract(
            rounds=[
                ("需要搜索", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            use_real_tool_executor=True,
            fake_handler=FakeSearchHandler(),
            options={"use_reasoning": True},
            capabilities={"functionCalling": True, "deepThinking": True},
        )

        self.assertEqual(
            result.event_types,
            [
                "run_started",
                "step_started",
                "tool_call_started",
                "tool_call_completed",
                "step_completed",
                "step_started",
                "step_completed",
                "run_completed",
            ],
        )
        self.assertEqual([event["sequence"] for event in result.events], list(range(len(result.events))))
        self.assertEqual(result.persist_calls[0]["partial"], True)
        self.assertEqual(result.persist_calls[0]["block_types"], ["thinking"])
        self.assertEqual(result.tool_execute_calls[0]["message_id"], "msg-contract")
        self.assertEqual(result.tool_execute_calls[0]["trace_id"], "run-contract")
        self.assertEqual(result.tool_execute_calls[0]["step_number"], 1)
        self.assertEqual(result.persist_calls[-1]["partial"], False)

    async def test_redis_stream_contract_writes_done_terminal_entry(self):
        result = await self._run_agent_contract(
            rounds=[("", "Final answer", [], "stop", None)],
            use_real_redis_stream=True,
        )

        self.assertEqual(result.redis_entry_types[0], "start")
        self.assertIn("preparing", result.redis_entry_types)
        self.assertEqual(result.redis_entry_types[-1], "done")
        self.assertEqual(result.redis_meta["status"], "done")

    async def test_redis_stream_contract_writes_error_terminal_entry_on_failure(self):
        with self.assertRaises(RuntimeError):
            await self._run_agent_contract(
                rounds=RuntimeError("LLM 5xx"),
                use_real_redis_stream=True,
            )

        result = self.last_result
        self.assertEqual(result.redis_entry_types[0], "start")
        self.assertEqual(result.redis_entry_types[-1], "error")
        self.assertEqual(result.redis_entries[-1][1]["content"], "LLM 5xx")
        self.assertEqual(result.redis_meta["status"], "error")

    async def test_session_and_step_contract_records_completed_tool_run(self):
        class FakeSearchHandler:
            tool_name = "web_search"

            async def execute(self, args):
                return ToolResult(status="success", data={"query": args["query"]}, duration_ms=9)

            async def log(self, **_kwargs):
                return None

            def _build_result_summary(self, result):
                return {"kind": "search", "status": result.status}

            def format_llm_context(self, result):
                return "搜索结果"

            def build_content_block(self, result, block_id, log_id):
                return SearchBlock(
                    type="search",
                    id=block_id,
                    query=result.data["query"],
                    tool_call_log_id=log_id,
                    sources=[],
                    status="success",
                    source_count=1,
                )

        tool_call = {"id": "tc-contract", "name": "web_search", "arguments": '{"query":"OpenAI"}'}

        result = await self._run_agent_contract(
            rounds=[
                ("需要搜索", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            use_real_tool_executor=True,
            fake_handler=FakeSearchHandler(),
            options={"use_reasoning": True},
            capabilities={"functionCalling": True, "deepThinking": True},
        )

        self.assertEqual(result.session_started_calls[0]["run_id"], "run-contract")
        self.assertEqual(result.session_started_calls[0]["message_id"], "msg-contract")
        self.assertEqual(result.session_status_calls[-1]["status"], "completed")
        self.assertEqual(result.session_status_calls[-1]["total_steps"], 2)
        self.assertEqual(result.session_status_calls[-1]["total_tool_calls"], 1)
        self.assertEqual([call["step_number"] for call in result.step_started_calls], [1, 2])
        self.assertEqual([call["tool_names"] for call in result.step_completed_calls], [["web_search"], []])
        self.assertEqual([call["tool_calls_count"] for call in result.step_completed_calls], [1, 0])
        self.assertEqual(result.step_terminal_calls, [])

    async def test_session_and_step_contract_marks_active_step_failed_on_error(self):
        with self.assertRaises(RuntimeError):
            await self._run_agent_contract(rounds=RuntimeError("LLM 5xx"))

        result = self.last_result
        self.assertEqual([call["step_number"] for call in result.step_started_calls], [1])
        self.assertEqual(result.step_completed_calls, [])
        self.assertEqual(result.step_terminal_calls[-1]["step_id"], result.step_started_calls[0]["step_id"])
        self.assertEqual(result.step_terminal_calls[-1]["status"], "failed")
        self.assertEqual(result.session_status_calls[-1]["status"], "error")
        self.assertEqual(result.session_status_calls[-1]["total_steps"], 1)
        self.assertEqual(result.session_status_calls[-1]["total_tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
