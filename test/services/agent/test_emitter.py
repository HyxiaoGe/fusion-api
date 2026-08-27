"""AgentEventEmitter 单元测试"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.services.agent.emitter import AgentEventEmitter
from app.services.agent.events import StepStarted

CAPABILITY_RESOLUTION = {
    "schema_version": 1,
    "router_version": "2026-08-27.1",
    "package_id": "fresh_web",
    "confidence": "high",
    "resolution_mode": "routed",
    "reason_codes": ["fresh_external_fact"],
    "external_tool_names": ["web_search"],
    "effective_plan_mode": "off",
    "include_current_date": True,
    "network_boundary_required": False,
    "bundle_fingerprint": "sha256:" + "a" * 64,
}


class EmitterEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_prompt_result_and_request_fingerprint_share_run_envelope(self):
        emitted = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                emitted.append(payload)

        em = AgentEventEmitter(
            run_id="r1", trace_id="r1", conversation_id="c1", task_id="t1", redis_writer=CaptureWriter()
        )
        await em.run_started(message_id="m1", model="gpt", tools=[], config={})
        await em.system_prompt_prepared(
            status="ready",
            source="code",
            template_version="1",
            section_ids=["core", "current_date"],
            fingerprint="a" * 64,
            char_count=100,
            duration_ms=0,
        )
        await em.llm_round_started(
            llm_round_id="q1",
            round_index=1,
            model="gpt",
            provider="openai",
            system_prompt_fingerprint="b" * 64,
        )

        self.assertEqual(
            [event["type"] for event in emitted], ["run_started", "system_prompt_prepared", "llm_round_started"]
        )
        self.assertEqual(emitted[1]["protocol_version"], 2)
        self.assertEqual(emitted[1]["status"], "ready")
        self.assertEqual(emitted[1]["section_ids"], ["core", "current_date"])
        self.assertEqual(emitted[1]["sequence"], 1)
        self.assertIsNone(emitted[1]["step_id"])
        self.assertEqual(emitted[2]["system_prompt_fingerprint"], "b" * 64)
        self.assertEqual(emitted[2]["run_id"], "r1")
        self.assertEqual(emitted[2]["llm_round_id"], "q1")

    async def test_run_started_envelope(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        await em.run_started(
            message_id="m1",
            model="gpt",
            tools=["web_search"],
            config={"max_steps": 8, "capability_resolution": CAPABILITY_RESOLUTION},
        )
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
        self.assertEqual(args[3]["task_id"], "task-1")
        self.assertEqual(args[3]["capability_resolution"], CAPABILITY_RESOLUTION)
        self.assertEqual(args[3]["tools"], args[3]["capability_resolution"]["external_tool_names"])
        self.assertNotIn("update_plan", args[3]["tools"])

    async def test_invalid_current_run_resolution_is_not_emitted(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="r1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )
        invalid_resolutions = (
            {
                **CAPABILITY_RESOLUTION,
                "package_id": "mcp_explicit",
                "reason_codes": ["explicit_authorized_tool_alias"],
                "external_tool_names": ["update_plan"],
                "include_current_date": False,
            },
            {
                **CAPABILITY_RESOLUTION,
                "package_id": "direct",
                "reason_codes": ["direct_greeting"],
                "external_tool_names": ["web_search"],
                "include_current_date": False,
            },
        )

        for invalid_resolution in invalid_resolutions:
            with self.subTest(invalid_resolution=invalid_resolution), self.assertRaises(ValidationError):
                await em.run_started(
                    message_id="m1",
                    model="gpt",
                    tools=invalid_resolution["external_tool_names"],
                    config={"capability_resolution": invalid_resolution},
                )

        writer.append_chunk.assert_not_awaited()

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

    async def test_tool_events_carry_plan_item_link_without_exposing_it_as_argument(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="r1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )

        await em.tool_call_started(
            tool_call_id="t1",
            tool_name="web_search",
            arguments={"query": "深圳天气"},
            plan_item_id="weather",
        )
        await em.tool_call_completed(
            tool_call_id="t1",
            tool_name="web_search",
            status="success",
            duration_ms=10,
            result_summary={},
            plan_item_id="weather",
        )

        started = writer.append_chunk.call_args_list[-2].args[3]
        completed = writer.append_chunk.call_args_list[-1].args[3]
        self.assertEqual(started["plan_item_id"], "weather")
        self.assertNotIn("plan_item_id", started["arguments"])
        self.assertEqual(completed["plan_item_id"], "weather")

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

    async def test_suggested_questions_pending_follows_run_completed_in_same_event_stream(self):
        writer = AsyncMock()
        em = AgentEventEmitter(
            run_id="r1",
            trace_id="r1",
            conversation_id="c1",
            task_id="task-1",
            redis_writer=writer,
        )
        await em.step_started(step_number=1)
        await em.run_completed(total_steps=1, total_tool_calls=0, finish_reason="stop")

        await em.suggested_questions_pending(message_id="msg-1", revision=2)

        events = [call.args[3] for call in writer.append_chunk.call_args_list]
        self.assertEqual(
            [event["type"] for event in events],
            ["step_started", "run_completed", "suggested_questions_pending"],
        )
        self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
        self.assertEqual(events[-1]["protocol_version"], 2)
        self.assertEqual(events[-1]["message_id"], "msg-1")
        self.assertEqual(events[-1]["revision"], 2)
        self.assertEqual(events[-1]["status"], "pending")
        self.assertIsNone(events[-1]["step_id"])

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
                    "phase_id": "phase-understand",
                    "phase_title": "分析任务并制定方案",
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
        self.assertEqual(events[2]["items"][0]["phase_id"], "phase-understand")
        self.assertEqual(events[2]["items"][0]["phase_title"], "分析任务并制定方案")

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
            repair_state="retrying",
            repair_id="repair_0123456789abcdef",
            plan_item_id="search",
        )

        plan_payload = writer.append_chunk.call_args_list[-2].args[3]
        digest_payload = writer.append_chunk.call_args_list[-1].args[3]
        self.assertEqual(plan_payload["type"], "plan_step_updated")
        self.assertEqual(plan_payload["step_id"], step_id)
        self.assertEqual(digest_payload["type"], "tool_result_digest")
        self.assertEqual(digest_payload["step_id"], step_id)
        self.assertEqual(digest_payload["tool_call_id"], "tc1")
        self.assertEqual(digest_payload["repair_state"], "retrying")
        self.assertEqual(digest_payload["repair_id"], "repair_0123456789abcdef")
        self.assertEqual(digest_payload["plan_item_id"], "search")

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

    async def test_cancelled_writer_reserves_sequence_before_await(self):
        writer = AsyncMock()
        writer.append_chunk.side_effect = [asyncio.CancelledError(), None]
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        with self.assertRaises(asyncio.CancelledError):
            await em.step_started(step_number=1)
        await em.step_started(step_number=2)

        payloads = [call.args[3] for call in writer.append_chunk.call_args_list]
        self.assertEqual([payload["sequence"] for payload in payloads], [0, 1])

    async def test_payload_validation_failure_does_not_consume_sequence(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        oversized = StepStarted(type="step_started", step_number=1, **em._envelope())

        with self.assertRaises(ValueError):
            await em._emit(oversized, max_payload_bytes=1)
        await em.step_started(step_number=1)

        self.assertEqual(writer.append_chunk.await_args.args[3]["sequence"], 0)

    async def test_seal_returns_last_reserved_sequence_and_rejects_later_emit(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        self.assertEqual(await em.seal_and_get_last_sequence(), -1)
        with self.assertRaises(RuntimeError):
            await em.run_completed(total_steps=0, total_tool_calls=0, finish_reason="stop")

        open_emitter = AgentEventEmitter(
            run_id="r2", trace_id="r2", conversation_id="c2", task_id="task-2", redis_writer=AsyncMock()
        )
        await open_emitter.step_started(step_number=1)
        self.assertEqual(await open_emitter.seal_and_get_last_sequence(), 0)

    async def test_new_lifecycle_helpers_preserve_parent_and_emit_opaque_query_summary(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)
        parent_step_id = await em.step_started(step_number=1)

        await em.llm_round_started(
            llm_round_id="llm-1",
            round_index=1,
            model="deepseek/deepseek-chat",
            provider="deepseek",
            parent_step_id=parent_step_id,
        )
        await em.llm_round_failed(
            llm_round_id="llm-1",
            error_code="timeout",
            message="上游错误 token=LEAK_TOKEN https://example.com/fail?secret=LEAK_SECRET",
            parent_step_id=parent_step_id,
        )
        await em.retrieval_started(
            retrieval_id="ret-1",
            query_summary="私密短查询 LEAK_QUERY token=LEAK_TOKEN https://example.com/search?q=LEAK_QUERY#fragment",
            parent_step_id=parent_step_id,
        )
        await em.tool_attempt_started(
            tool_attempt_id="attempt-1",
            tool_call_id="tool-1",
            tool_name="web_search",
            attempt_index=1,
            parent_step_id=parent_step_id,
        )

        payloads = [call.args[3] for call in writer.append_chunk.call_args_list[-4:]]
        self.assertEqual([payload["parent_step_id"] for payload in payloads], [parent_step_id] * 4)
        self.assertEqual(payloads[1]["message"], "模型服务响应超时，请稍后重试。")
        self.assertEqual(payloads[2]["query_summary"], "已发起知识库检索")
        serialized = str(payloads)
        self.assertNotIn("LEAK_TOKEN", serialized)
        self.assertNotIn("LEAK_SECRET", serialized)
        self.assertNotIn("LEAK_QUERY", serialized)

    async def test_lifecycle_errors_only_expose_known_controlled_error_code(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        await em.retrieval_failed(
            retrieval_id="ret-1",
            error_code="provider-private-token=LEAK_TOKEN",
            message="provider response https://example.com/error?api_key=LEAK_KEY",
        )

        payload = writer.append_chunk.await_args.args[3]
        self.assertIsNone(payload["error_code"])
        self.assertIsNone(payload["message"])
        self.assertNotIn("LEAK_TOKEN", str(payload))
        self.assertNotIn("LEAK_KEY", str(payload))

    async def test_remaining_lifecycle_helpers_publish_their_protocol_payloads(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1", conversation_id="c1", task_id="task-1", redis_writer=writer)

        await em.llm_round_first_output_delta(llm_round_id="llm-1", delta_kind="tool_call", ttft_ms=10)
        await em.llm_round_completed(
            llm_round_id="llm-1",
            finish_reason="stop",
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            cache_read_tokens=0,
            cache_write_tokens=None,
            ttft_ms=10,
            duration_ms=20,
        )
        await em.llm_round_cancelled(llm_round_id="llm-2", reason="superseded")
        await em.retrieval_completed(retrieval_id="ret-1", document_count=2, duration_ms=20)
        await em.retrieval_failed(retrieval_id="ret-2", error_code="timeout", message="已脱敏摘要")
        await em.retrieval_cancelled(retrieval_id="ret-3", reason="shutdown")
        await em.tool_attempt_completed(
            tool_attempt_id="attempt-1", tool_call_id="tool-1", status="timeout", error_code="timeout", duration_ms=20
        )

        payloads = [call.args[3] for call in writer.append_chunk.call_args_list]
        self.assertEqual(
            [payload["type"] for payload in payloads],
            [
                "llm_round_first_output_delta",
                "llm_round_completed",
                "llm_round_cancelled",
                "retrieval_completed",
                "retrieval_failed",
                "retrieval_cancelled",
                "tool_attempt_completed",
            ],
        )
        self.assertEqual(payloads[0]["delta_kind"], "tool_call")
        self.assertEqual(payloads[1]["total_tokens"], 3)
        self.assertEqual(payloads[-1]["tool_call_id"], "tool-1")
