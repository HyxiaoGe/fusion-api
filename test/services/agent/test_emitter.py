"""AgentEventEmitter 单元测试"""
import asyncio
import unittest
from unittest.mock import AsyncMock

from app.services.agent.emitter import AgentEventEmitter


class EmitterEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_started_envelope(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        await em.run_started(model="gpt", tools=["web_search"],
                             config={"max_steps": 8})
        writer.append_chunk.assert_awaited_once()
        args, kwargs = writer.append_chunk.call_args
        self.assertEqual(args[0], "c1")
        self.assertEqual(args[1], "agent_event")
        self.assertEqual(args[2]["type"], "run_started")
        self.assertEqual(args[2]["sequence"], 0)
        self.assertEqual(args[2]["run_id"], "r1")
        self.assertEqual(args[2]["trace_id"], "r1")

    async def test_step_started_returns_step_id_and_persists_context(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)
        self.assertIsInstance(step_id, str)
        await em.tool_call_started(tool_call_id="t1", tool_name="web_search",
                                   arguments={"query": "x"})
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(last_args[2]["step_id"], step_id)

    async def test_step_completed_clears_step_context(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        step_id = await em.step_started(step_number=1)
        await em.step_completed(step_number=1, tool_call_count=0, duration_ms=10)
        # step_completed 自己的事件必须仍带本 step 的 step_id（清空在 emit 之后）
        step_completed_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(step_completed_args[2]["type"], "step_completed")
        self.assertEqual(step_completed_args[2]["step_id"], step_id)
        # 之后 _current_step_id 已被清空（白盒）
        self.assertIsNone(em._current_step_id)

    async def test_run_level_events_have_step_id_none_even_with_active_step(self):
        """run_failed/interrupted/limit_reached/completed 不能继承 _current_step_id"""
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        await em.step_started(step_number=1)  # 设 _current_step_id

        # 不调 step_completed，模拟"step 中途异常"场景
        await em.run_failed(error_code="x", message="boom")
        failed_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(failed_args[2]["type"], "run_failed")
        self.assertIsNone(failed_args[2]["step_id"])

        await em.run_interrupted(reason="user_cancelled")
        interrupted_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(interrupted_args[2]["step_id"])

        await em.run_limit_reached(reason="max_steps")
        limit_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(limit_args[2]["step_id"])

        await em.run_completed(total_steps=1, total_tool_calls=0, finish_reason="stop")
        completed_args = writer.append_chunk.call_args_list[-1].args
        self.assertIsNone(completed_args[2]["step_id"])

    async def test_sequence_monotonic_under_concurrency(self):
        writer = AsyncMock()
        # 让 append_chunk 真正 yield 一次，确保即使去掉 lock 也会 reschedule
        async def slow_append(*args, **kwargs):
            await asyncio.sleep(0)
        writer.append_chunk = AsyncMock(side_effect=slow_append)

        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        await em.step_started(step_number=1)

        async def parallel_call(i: int):
            await em.tool_call_started(tool_call_id=f"t{i}",
                                       tool_name="web_search",
                                       arguments={"i": i})

        await asyncio.gather(*[parallel_call(i) for i in range(20)])
        seqs = [c.args[2]["sequence"] for c in writer.append_chunk.call_args_list]
        self.assertEqual(seqs, list(range(len(seqs))))

    async def test_sanitizer_called(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        await em.step_started(step_number=1)
        await em.tool_call_started(tool_call_id="t1", tool_name="web_search",
                                   arguments={"query": "x"})
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertEqual(last_args[2]["arguments"], {"query": "x"})

    async def test_result_summary_capped(self):
        writer = AsyncMock()
        em = AgentEventEmitter(run_id="r1", trace_id="r1",
                               conversation_id="c1", redis_writer=writer)
        await em.step_started(step_number=1)
        big = {"kind": "search", "title": "x" * 5000, "count": 1, "truncated": False}
        await em.tool_call_completed(tool_call_id="t1", tool_name="web_search",
                                     status="success", duration_ms=10,
                                     result_summary=big)
        last_args = writer.append_chunk.call_args_list[-1].args
        self.assertTrue(last_args[2]["result_summary"]["truncated"])
