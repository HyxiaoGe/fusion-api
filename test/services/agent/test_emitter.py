"""AgentEventEmitter 单元测试"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.services.agent.emitter import AgentEventEmitter


class EmitterEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_started_envelope(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.run_started(message_id="m1", model="gpt", tools=["web_search"], config={"max_steps": 8})
        writer.append_chunk.assert_awaited_once()
        args, kwargs = writer.append_chunk.call_args
        self.assertEqual(args[0], "c1")
        self.assertEqual(args[1], "task-1")
        self.assertEqual(args[2], "agent_event")
        self.assertEqual(args[3]["type"], "run_started")
        self.assertEqual(args[3]["sequence"], 0)
        self.assertEqual(args[3]["run_id"], "r1")
        self.assertEqual(args[3]["trace_id"], "r1")
        self.assertEqual(args[3]["message_id"], "m1")

    async def test_step_started_returns_step_id_and_persists_context(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)
        self.assertIsInstance(step_id, str)
        await em.tool_call_started(tool_call_id="t1", tool_name="web_search", arguments={"query": "x"})
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(last_args[3]["step_id"], step_id)

    async def test_tool_call_started_url_read_arguments_use_strict_allowlist(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        await em.tool_call_started(
            tool_call_id="t1",
            tool_name="url_read",
            arguments={
                "url": "https://example.com/page?token=secret",
                "reason": "  核实原文  ",
                "api_key": "secret-key",
                "nested": {"secret": "value"},
            },
        )

        payload = writer.append_chunk.await_args.args[3]
        self.assertEqual(
            payload["arguments"],
            {
                "url": "https://example.com/page",
                "reason": "核实原文",
                "url_policy_reason": "sensitive_query",
            },
        )
        self.assertNotIn("secret", str(payload["arguments"]))

    async def test_tool_call_started_amap_product_redacts_inline_credentials_before_redis(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        await em.tool_call_started(
            tool_call_id="t1",
            tool_name="local_place_search",
            arguments={
                "query": "api key: 如何申请地图服务",
                "near": (
                    "民治 api_key=LEAK_SENTINEL authorization=Bearer BEARER_SENTINEL "
                    "Proxy-Authorization: Token PROXY_SENTINEL cookie=COOKIE_SENTINEL "
                    "access_token=ACCESS_SENTINEL session_id=SESSION_SENTINEL"
                ),
            },
        )

        payload = writer.append_chunk.await_args.args[3]
        serialized = str(payload["arguments"])
        self.assertEqual(payload["arguments"]["query"], "api key: 如何申请地图服务")
        self.assertNotIn("LEAK_SENTINEL", serialized)
        self.assertNotIn("BEARER_SENTINEL", serialized)
        self.assertNotIn("PROXY_SENTINEL", serialized)
        self.assertNotIn("COOKIE_SENTINEL", serialized)
        self.assertNotIn("ACCESS_SENTINEL", serialized)
        self.assertNotIn("SESSION_SENTINEL", serialized)

    async def test_step_completed_clears_step_context(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)
        await em.step_completed(step_number=1, tool_call_count=0, duration_ms=10)
        # step_completed 自己的事件必须仍带本 step 的 step_id（清空在 emit 之后）
        step_completed_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(step_completed_args[3]["type"], "step_completed")
        self.assertEqual(step_completed_args[3]["step_id"], step_id)
        # 之后 _current_step_id 已被清空（白盒）
        self.assertIsNone(em._current_step_id)

    async def test_content_block_discarded_uses_current_step_envelope(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="r1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )
        step_id = await em.step_started(step_number=1)

        await em.content_block_discarded(block_id="blk-tool-preamble")

        payload = writer.append_chunk.await_args.args[3]
        self.assertEqual(payload["type"], "content_block_discarded")
        self.assertEqual(payload["protocol_version"], 2)
        self.assertEqual(payload["block_id"], "blk-tool-preamble")
        self.assertEqual(payload["step_id"], step_id)

    async def test_run_level_events_have_step_id_none_even_with_active_step(self):
        """run_failed/interrupted/limit_reached/completed 不能继承 _current_step_id"""
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.step_started(step_number=1)  # 设 _current_step_id

        # 不调 step_completed，模拟"step 中途异常"场景
        await em.run_failed(error_code="x", message="boom")
        failed_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(failed_args[3]["type"], "run_failed")
        self.assertIsNone(failed_args[3]["step_id"])

        await em.run_interrupted(reason="user_cancelled")
        interrupted_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(interrupted_args[3]["step_id"])

        await em.run_limit_reached(reason="max_steps")
        limit_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(limit_args[3]["step_id"])

        await em.run_completed(total_steps=1, total_tool_calls=0, finish_reason="stop")
        completed_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(completed_args[3]["step_id"])

    async def test_sequence_monotonic_under_concurrency(self):
        writer = AsyncMock()

        # 让 append_chunk 真正 yield 一次，确保即使去掉 lock 也会 reschedule
        async def slow_append(*args, **kwargs):
            await asyncio.sleep(0)

        writer.append_chunk = AsyncMock(side_effect=slow_append)

        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.step_started(step_number=1)

        async def parallel_call(i: int):
            await em.tool_call_started(tool_call_id=f"t{i}", tool_name="web_search", arguments={"i": i})

        await asyncio.gather(*[parallel_call(i) for i in range(20)])
        seqs = [c.args[3]["sequence"] for c in writer.append_chunk.call_args_list]
        self.assertEqual(seqs, list(range(len(seqs))))

    async def test_sanitizer_called(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.step_started(step_number=1)
        await em.tool_call_started(tool_call_id="t1", tool_name="web_search", arguments={"query": "x"})
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(last_args[3]["arguments"], {"query": "x"})

    async def test_result_summary_capped(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.step_started(step_number=1)
        big = {"kind": "search", "title": "x" * 5000, "count": 1, "truncated": False}
        await em.tool_call_completed(
            tool_call_id="t1", tool_name="web_search", status="success", duration_ms=10, result_summary=big
        )
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertTrue(last_args[3]["result_summary"]["truncated"])

    async def test_v2_events_use_same_sequence_stream(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        await em.run_started(message_id="m1", model="gpt", tools=["web_search"], config={"max_steps": 8})
        await em.run_progress_updated(
            phase="planning",
            label="正在理解问题",
            completed_steps=0,
            total_steps=4,
            completed_tool_calls=0,
            max_tool_calls=20,
        )
        await em.plan_snapshot(
            plan_id="plan-r1",
            revision=1,
            items=[
                {
                    "id": "understand",
                    "title": "理解问题",
                    "status": "running",
                    "kind": "reasoning",
                    "tool_names": [],
                    "evidence_item_ids": [],
                }
            ],
        )

        events = [call.args[3] for call in writer.append_chunk.call_args_list]
        self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
        self.assertEqual(events[1]["type"], "run_progress_updated")
        self.assertEqual(events[1]["protocol_version"], 2)
        self.assertIsNone(events[1]["step_id"])
        self.assertEqual(events[2]["type"], "plan_snapshot")

    async def test_context_status_update_is_safe_and_replayable(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1", trace_id="trace-1", conversation_id="c1", task_id="task-1", redis_writer=writer
        )
        await em.run_started(message_id="msg-1", model="gpt", tools=[], config={})
        await em.step_started(step_number=1)

        await em.context_status_updated(
            phase="final",
            status="trimmed",
            round_index=1,
            window_tokens=258_000,
            estimated_tokens_before=220_000,
            estimated_tokens_after=180_000,
            actual_prompt_tokens=179_500,
            removed_turns=2,
            removed_messages=5,
            removed_tool_transactions=1,
        )

        payload = writer.append_chunk.call_args_list[-1].args[3]
        self.assertEqual(payload["type"], "context_status_updated")
        self.assertEqual(payload["protocol_version"], 2)
        self.assertEqual(payload["phase"], "final")
        self.assertEqual(payload["message_id"], "msg-1")
        self.assertEqual(payload["round_index"], 1)
        self.assertEqual(payload["window_tokens"], 258_000)
        self.assertEqual(payload["actual_prompt_tokens"], 179_500)
        self.assertEqual(payload["sequence"], 2)
        self.assertIsNotNone(payload["step_id"])
        self.assertNotIn("messages", payload)
        self.assertNotIn("context_window_source", payload)

    async def test_context_status_rejects_unknown_status_and_negative_numbers(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="trace-1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )
        await em.run_started(message_id="msg-1", model="gpt", tools=[], config={})

        with self.assertRaises(ValidationError):
            await em.context_status_updated(
                phase="final",
                status="private-future-status",
                round_index=1,
                window_tokens=-1,
                estimated_tokens_before=None,
                estimated_tokens_after=None,
                actual_prompt_tokens=None,
                removed_turns=0,
                removed_messages=0,
                removed_tool_transactions=0,
            )

        self.assertEqual(writer.append_chunk.await_count, 1)

    async def test_step_level_v2_events_inherit_current_step(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)

        await em.plan_step_updated(
            plan_id="plan-r1",
            revision=2,
            item={
                "id": "search",
                "title": "搜索资料",
                "status": "completed",
                "kind": "search",
                "tool_names": ["web_search"],
                "evidence_item_ids": [],
            },
        )
        await em.tool_result_digest(
            tool_call_id="tc1",
            tool_name="web_search",
            status="success",
            title="找到 2 条结果",
            summary="优先保留官方来源。",
            key_findings=["官方页面确认发布时间。"],
            source_refs=[],
            truncated=False,
        )

        plan_payload = writer.append_chunk.call_args_list[-2].args[3]
        digest_payload = writer.append_chunk.call_args_list[-1].args[3]
        self.assertEqual(plan_payload["type"], "plan_step_updated")
        self.assertEqual(plan_payload["step_id"], step_id)
        self.assertEqual(digest_payload["type"], "tool_result_digest")
        self.assertEqual(digest_payload["step_id"], step_id)
        self.assertEqual(digest_payload["tool_call_id"], "tc1")

    async def test_geolocation_context_events_share_sequence_and_never_contain_coordinates(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="r1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )
        step_id = await em.step_started(step_number=1)

        await em.context_required(
            request_id="ctx-1",
            context_type="geolocation",
            purpose="nearby_search",
            reason="搜索当前位置附近的地点",
            expires_at=123.5,
        )
        await em.context_result(
            request_id="ctx-1",
            context_type="geolocation",
            status="provided",
        )

        required = writer.append_chunk.call_args_list[-2].args[3]
        result = writer.append_chunk.call_args_list[-1].args[3]
        self.assertEqual(required["type"], "context_required")
        self.assertEqual(result["type"], "context_result")
        self.assertEqual([required["sequence"], result["sequence"]], [1, 2])
        self.assertEqual(required["step_id"], step_id)
        self.assertEqual(result["step_id"], step_id)
        serialized = str([required, result])
        self.assertNotIn("latitude", serialized)
        self.assertNotIn("longitude", serialized)
