"""
stream_handler 单元测试（Redis Stream 架构）

- SseEnvelopeFormatterTests: spec §4.6 SSE 顶层 envelope 形态
- AgentLoopFourPathsTests: spec §8.1 集成测试，覆盖 4 路径
  (normal / cancelled / failed / limit_reached)
"""

import asyncio
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.stream_handler import StreamHandler


class SseEnvelopeFormatterTests(unittest.TestCase):
    """spec §4.6 SSE 顶层 envelope 形态测试"""

    def test_agent_event_entry_to_envelope(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "agent_event",
            "content": '{"type":"run_started","run_id":"r1","sequence":0}',
            "block_id": "",
        })
        self.assertEqual(env["chunk_type"], "agent_event")
        self.assertEqual(env["data"]["type"], "run_started")
        self.assertEqual(env["data"]["sequence"], 0)
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_reasoning_entry_carries_run_step_ids(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "reasoning",
            "content": "hello",
            "block_id": "b1",
            "run_id": "r1",
            "step_id": "s1",
        })
        self.assertEqual(env["chunk_type"], "reasoning")
        self.assertEqual(env["data"], {
            "block_id": "b1", "delta": "hello",
            "run_id": "r1", "step_id": "s1",
        })

    def test_reasoning_entry_without_run_step_ids(self):
        """旧消息或缺失 run_id/step_id 时，data 不含这两键"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "reasoning",
            "content": "hello",
            "block_id": "b1",
        })
        self.assertEqual(env["data"], {"block_id": "b1", "delta": "hello"})

    def test_answering_entry(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "answering",
            "content": "world",
            "block_id": "b2",
            "run_id": "r1",
            "step_id": "s1",
        })
        self.assertEqual(env["chunk_type"], "answering")
        self.assertEqual(env["data"]["delta"], "world")
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_done_entry_empty_data(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "done", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "done", "data": {}})

    def test_preparing_entry_empty_data(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "preparing", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "preparing", "data": {}})

    def test_thinking_pending_entry(self):
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "thinking_pending", "content": "", "block_id": "b1"})
        self.assertEqual(env, {"chunk_type": "thinking_pending", "data": {"block_id": "b1"}})

    def test_error_entry_byok_structured_promoted(self):
        """BYOK 结构化 error_code: content 是 JSON 时升入 data"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "error",
            "content": '{"code":"provider_offline","message":"offline","retryable":true}',
            "block_id": "",
        })
        self.assertEqual(env["chunk_type"], "error")
        self.assertEqual(env["data"]["code"], "provider_offline")
        self.assertEqual(env["data"]["message"], "offline")
        self.assertEqual(env["data"]["retryable"], True)

    def test_error_entry_non_json_content_wrapped_as_message(self):
        """error content 不是 JSON dict 时兜底为 {code: stream_error, message: <content>}

        修复 P2：避免 finalize_stream(error_msg='用户中止' / '被新请求取代') 这类纯
        字符串 error 在 FE 端全丢成 {data: {}}。
        """
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "error",
            "content": "用户中止",
            "block_id": "",
        })
        self.assertEqual(env["chunk_type"], "error")
        self.assertEqual(env["data"], {"code": "stream_error", "message": "用户中止"})

    def test_error_entry_empty_content_empty_data(self):
        """error content 为空时 data 也为空"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "error",
            "content": "",
            "block_id": "",
        })
        self.assertEqual(env, {"chunk_type": "error", "data": {}})

    def test_unknown_type_falls_back_empty_data(self):
        """未知 chunk type 不抛，返回 {chunk_type: <type>, data: {}}"""
        from app.services.stream_handler import _entry_to_sse_envelope

        env = _entry_to_sse_envelope({
            "type": "future_unknown_type",
            "content": "anything",
            "block_id": "x",
        })
        self.assertEqual(env, {"chunk_type": "future_unknown_type", "data": {}})


class AgentLoopFourPathsTests(unittest.IsolatedAsyncioTestCase):
    """spec §8.1 集成测试：normal / cancelled / failed / limit_reached(max_steps)

    走 generate_to_redis 完整 agent loop，捕获 emitter 写入 Redis 的事件序列，
    验证：
      1) 事件类型按时序正确
      2) sequence 严格连续 0..N
      3) agent_sessions.status 终态正确
    """

    def setUp(self):
        self.handler = StreamHandler()

        # 捕获 append_chunk 全部调用（事件序列断言用）
        self.append_chunk_calls = []

        async def _capture_append(conversation_id, chunk_type, content, block_id, **extras):
            self.append_chunk_calls.append({
                "chunk_type": chunk_type,
                "content": content,
                "block_id": block_id,
                **extras,
            })
            return "1-0"

        # mock 顶层依赖：
        # - stream_handler.append_chunk: _stream_round 写 reasoning/answering 用
        # - stream_state_service.append_chunk: _AgentEventRedisWriter 写 agent_event 用
        # - finalize_stream / check_lock_owner: 防真写 Redis
        # - build_llm_messages: raw_messages 用 dict 占位，绕过真实 message 对象 schema
        self._patchers = [
            patch("app.services.stream_handler.append_chunk", side_effect=_capture_append),
            patch("app.services.stream_state_service.append_chunk", side_effect=_capture_append),
            patch("app.services.stream_handler.finalize_stream", AsyncMock()),
            patch("app.services.stream_handler.check_lock_owner", AsyncMock(return_value=True)),
            patch(
                "app.services.stream_handler.build_llm_messages",
                AsyncMock(return_value=[{"role": "user", "content": "hi"}]),
            ),
        ]
        for p in self._patchers:
            p.start()

        # mock SessionLocal（generate_to_redis 内 db.add / db.query）
        self.mock_db = MagicMock()
        self.db_patchers = [
            patch("app.services.stream_handler.SessionLocal", return_value=self.mock_db),
            patch(
                "app.services.agent.session_cache.SessionLocal",
                return_value=MagicMock(),
            ),
        ]
        for p in self.db_patchers:
            p.start()

        # 终态 status 捕获器（write_session_status 的最后一次调用即为终态）
        self.session_statuses = []

        async def _capture_status(*, run_id, status, total_steps, total_tool_calls, **kw):
            self.session_statuses.append({
                "run_id": run_id, "status": status,
                "total_steps": total_steps, "total_tool_calls": total_tool_calls,
            })

        # session_cache 写入全部 mock 掉，避免命中真 SQLAlchemy 路径；
        # write_session_status 用 side_effect 捕获参数。
        self.session_cache_patchers = [
            patch("app.services.agent.session_cache.write_session_started", AsyncMock()),
            patch("app.services.agent.session_cache.write_step_started", AsyncMock()),
            patch("app.services.agent.session_cache.write_step_completed", AsyncMock()),
            patch("app.services.agent.session_cache.write_step_terminal", AsyncMock()),
            patch(
                "app.services.agent.session_cache.write_session_status",
                side_effect=_capture_status,
            ),
        ]
        for p in self.session_cache_patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers + self.db_patchers + self.session_cache_patchers:
            p.stop()

    def _agent_events(self):
        """从 append_chunk 调用列表里抽出 agent_event 解出来。"""
        events = []
        for call in self.append_chunk_calls:
            if call["chunk_type"] == "agent_event":
                events.append(json.loads(call["content"]))
        return events

    async def _invoke(self, *, stream_round_side_effect, execute_tools_result=None,
                      patch_extra=None):
        """通用启动器：mock _stream_round + _execute_tools_parallel 后跑 generate_to_redis。

        stream_round_side_effect: callable 或 list；list 时按序消费每次 _stream_round 返回值
        execute_tools_result: _execute_tools_parallel 的返回值
        patch_extra: 额外 context manager 列表
        """
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                self.handler, "_stream_round",
                AsyncMock(side_effect=stream_round_side_effect),
            ))
            stack.enter_context(patch.object(
                self.handler, "_execute_tools_parallel",
                AsyncMock(return_value=execute_tools_result or []),
            ))
            stack.enter_context(patch.object(
                self.handler, "_llm_call_with_retry",
                AsyncMock(return_value=MagicMock()),
            ))
            for cm in patch_extra or []:
                stack.enter_context(cm)

            await self.handler.generate_to_redis(
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                raw_messages=[{"role": "user", "content": "hi"}],
                has_vision=False,
                file_ids=None,
                original_message="hi",
                assistant_message_id="msg-1",
                task_id="task-1",
                options={"use_reasoning": False},
                trace_id="trace-1",
            )

    async def test_normal_path_event_sequence(self):
        """正常 stop 路径：1 round（直接回答）→ run_completed(stop)"""
        await self._invoke(
            stream_round_side_effect=[
                ("", "Hello world", [], "stop", None),
            ],
        )

        events = self._agent_events()
        types = [e["type"] for e in events]
        self.assertEqual(types, [
            "run_started", "step_started", "step_completed", "run_completed",
        ])
        # sequence 严格连续 0..3
        seqs = [e["sequence"] for e in events]
        self.assertEqual(seqs, [0, 1, 2, 3])
        # run_completed.finish_reason
        self.assertEqual(events[-1]["finish_reason"], "stop")
        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "completed")

    async def test_normal_path_with_tool_calls(self):
        """正常 tool_calls + stop：2 round → run_completed(stop)"""
        tool_call = {"id": "tc1", "name": "web_search", "arguments": '{"query":"x"}'}

        await self._invoke(
            stream_round_side_effect=[
                ("", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            execute_tools_result=[
                # _execute_tools_parallel 返回 [(tc, result, handler, block_id, log_id), ...]
                (
                    tool_call,
                    SimpleNamespace(
                        status="success",
                        error_message=None,
                        duration_ms=10,
                    ),
                    None,  # handler=None → 走 else 分支不调用 build_content_block
                    "blk_aaa",
                    "log_aaa",
                ),
            ],
        )

        events = self._agent_events()
        types = [e["type"] for e in events]
        self.assertIn("run_started", types)
        self.assertIn("step_started", types)
        self.assertIn("step_completed", types)
        self.assertIn("run_completed", types)
        self.assertEqual(types[-1], "run_completed")
        # sequence 严格连续 0..N
        seqs = [e["sequence"] for e in events]
        self.assertEqual(seqs, list(range(len(seqs))))
        # finish_reason='stop'
        self.assertEqual(events[-1]["finish_reason"], "stop")
        self.assertEqual(self.session_statuses[-1]["status"], "completed")

    async def test_cancelled_path(self):
        """CancelledError 路径：发 run_interrupted + status='interrupted'"""
        async def _raise_cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await self._invoke(
                stream_round_side_effect=_raise_cancel,
            )

        events = self._agent_events()
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "run_started")
        self.assertIn("run_interrupted", types)
        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "interrupted")

    async def test_failed_path(self):
        """LLM 抛非 Cancelled 异常：发 run_failed + status='error'"""
        async def _raise_runtime(*args, **kwargs):
            raise RuntimeError("upstream LLM 5xx")

        # generate_to_redis 内 except Exception 块吞掉异常（不再 raise），
        # 走 run_failed + finalize_stream，函数正常返回。
        await self._invoke(
            stream_round_side_effect=_raise_runtime,
        )

        events = self._agent_events()
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "run_started")
        self.assertIn("run_failed", types)
        # 最后一个 agent_event 是 run_failed
        run_failed = [e for e in events if e["type"] == "run_failed"][0]
        self.assertIn("upstream LLM 5xx", run_failed["message"])
        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "error")

    async def test_limit_reached_max_steps(self):
        """触顶 max_steps：发 run_limit_reached(max_steps) → 强制总结 → run_completed(limit_reached)"""
        from app.services import stream_handler as sh

        tool_call = {"id": "tc1", "name": "web_search", "arguments": '{"query":"x"}'}

        # max_steps=2 → 需要 2 轮 tool_calls 让循环顶端触顶；之后强制总结再来 1 轮 stop
        rounds = [("", "", [tool_call], "tool_calls", None)] * 2
        rounds.append(("", "summary", [], "stop", None))

        with patch.object(sh, "AGENT_MAX_STEPS", 2):
            await self._invoke(
                stream_round_side_effect=rounds,
                execute_tools_result=[
                    (
                        tool_call,
                        SimpleNamespace(
                            status="success",
                            error_message=None,
                            duration_ms=10,
                        ),
                        None,
                        "blk_aaa",
                        "log_aaa",
                    ),
                ],
            )

        events = self._agent_events()
        types = [e["type"] for e in events]

        # 必须出现 run_limit_reached(reason='max_steps')
        limit_events = [e for e in events if e["type"] == "run_limit_reached"]
        self.assertEqual(len(limit_events), 1)
        self.assertEqual(limit_events[0]["reason"], "max_steps")

        # 触顶后还有强制总结的 step_started + step_completed
        limit_idx = types.index("run_limit_reached")
        post_limit = types[limit_idx + 1:]
        self.assertIn("step_started", post_limit)
        self.assertIn("step_completed", post_limit)

        # 最后是 run_completed(finish_reason='limit_reached')
        self.assertEqual(types[-1], "run_completed")
        self.assertEqual(events[-1]["finish_reason"], "limit_reached")

        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "limit_reached")


if __name__ == "__main__":
    unittest.main()
