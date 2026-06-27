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

from app.services.stream import StreamHandler
from app.services.stream.tool_execution_result import ToolExecutionRecord


class SseEnvelopeFormatterTests(unittest.TestCase):
    """spec §4.6 SSE 顶层 envelope 形态测试"""

    def test_agent_event_entry_to_envelope(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "agent_event",
                "content": '{"type":"run_started","run_id":"r1","sequence":0}',
                "block_id": "",
            }
        )
        self.assertEqual(env["chunk_type"], "agent_event")
        self.assertEqual(env["data"]["type"], "run_started")
        self.assertEqual(env["data"]["sequence"], 0)
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_reasoning_entry_carries_run_step_ids(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "reasoning",
                "content": "hello",
                "block_id": "b1",
                "run_id": "r1",
                "step_id": "s1",
            }
        )
        self.assertEqual(env["chunk_type"], "reasoning")
        self.assertEqual(
            env["data"],
            {
                "block_id": "b1",
                "delta": "hello",
                "run_id": "r1",
                "step_id": "s1",
            },
        )

    def test_reasoning_entry_without_run_step_ids(self):
        """旧消息或缺失 run_id/step_id 时，data 不含这两键"""
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "reasoning",
                "content": "hello",
                "block_id": "b1",
            }
        )
        self.assertEqual(env["data"], {"block_id": "b1", "delta": "hello"})

    def test_answering_entry(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "answering",
                "content": "world",
                "block_id": "b2",
                "run_id": "r1",
                "step_id": "s1",
            }
        )
        self.assertEqual(env["chunk_type"], "answering")
        self.assertEqual(env["data"]["delta"], "world")
        self.assertEqual(env["data"]["run_id"], "r1")

    def test_done_entry_empty_data(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "done", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "done", "data": {}})

    def test_preparing_entry_empty_data(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "preparing", "content": "", "block_id": ""})
        self.assertEqual(env, {"chunk_type": "preparing", "data": {}})

    def test_thinking_pending_entry(self):
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope({"type": "thinking_pending", "content": "", "block_id": "b1"})
        self.assertEqual(env, {"chunk_type": "thinking_pending", "data": {"block_id": "b1"}})

    def test_error_entry_byok_structured_promoted(self):
        """BYOK 结构化 error_code: content 是 JSON 时升入 data"""
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "error",
                "content": '{"code":"provider_offline","message":"offline","retryable":true}',
                "block_id": "",
            }
        )
        self.assertEqual(env["chunk_type"], "error")
        self.assertEqual(env["data"]["code"], "provider_offline")
        self.assertEqual(env["data"]["message"], "offline")
        self.assertEqual(env["data"]["retryable"], True)

    def test_error_entry_non_json_content_wrapped_as_message(self):
        """error content 不是 JSON dict 时兜底为 {code: stream_error, message: <content>}

        修复 P2：避免 finalize_stream(error_msg='用户中止' / '被新请求取代') 这类纯
        字符串 error 在 FE 端全丢成 {data: {}}。
        """
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "error",
                "content": "用户中止",
                "block_id": "",
            }
        )
        self.assertEqual(env["chunk_type"], "error")
        self.assertEqual(env["data"], {"code": "stream_error", "message": "用户中止"})

    def test_error_entry_empty_content_empty_data(self):
        """error content 为空时 data 也为空"""
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "error",
                "content": "",
                "block_id": "",
            }
        )
        self.assertEqual(env, {"chunk_type": "error", "data": {}})

    def test_unknown_type_falls_back_empty_data(self):
        """未知 chunk type 不抛，返回 {chunk_type: <type>, data: {}}"""
        from app.services.stream.sse_encoder import entry_to_sse_envelope as _entry_to_sse_envelope

        env = _entry_to_sse_envelope(
            {
                "type": "future_unknown_type",
                "content": "anything",
                "block_id": "x",
            }
        )
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
            self.append_chunk_calls.append(
                {
                    "chunk_type": chunk_type,
                    "content": content,
                    "block_id": block_id,
                    **extras,
                }
            )
            return "1-0"

        # mock 顶层依赖：
        # - stream.runner.append_chunk: _stream_round 写 reasoning/answering 用
        # - stream_state_service.append_chunk: _AgentEventRedisWriter 写 agent_event 用
        # - finalize_stream / check_lock_owner: 防真写 Redis
        # - build_llm_messages: raw_messages 用 dict 占位，绕过真实 message 对象 schema
        # finalize_stream mock 暴露到 self.finalize_mock，便于 test_failed_path 等用例
        # 在 raise 之后断言 SSE 收尾已先于异常传播完成。
        self.finalize_mock = AsyncMock()
        self._patchers = [
            patch("app.services.stream.runner.append_chunk", side_effect=_capture_append),
            # AgentEventRedisWriter (tool_executor.py:38) 通过本地 import 引用 append_chunk，
            # patch 必须打在 tool_executor 命名空间才会生效，patch stream_state_service 是无效的。
            patch("app.services.stream.tool_executor.append_chunk", side_effect=_capture_append),
            patch("app.services.stream.runner.finalize_stream", self.finalize_mock),
            patch("app.services.stream.llm_stream.check_lock_owner", AsyncMock(return_value=True)),
            patch(
                "app.services.stream.runner.build_llm_messages",
                AsyncMock(return_value=[{"role": "user", "content": "hi"}]),
            ),
        ]
        for p in self._patchers:
            p.start()

        # mock SessionLocal（generate_to_redis 内 db.add / db.query）
        self.mock_db = MagicMock()
        self.db_patchers = [
            patch("app.services.stream.runner.SessionLocal", return_value=self.mock_db),
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
            self.session_statuses.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "total_steps": total_steps,
                    "total_tool_calls": total_tool_calls,
                }
            )

        # session_cache 写入全部 mock 掉，避免命中真 SQLAlchemy 路径；
        # write_session_status 用 side_effect 捕获参数。
        self.write_step_started_mock = AsyncMock()
        self.write_step_completed_mock = AsyncMock()
        self.write_step_terminal_mock = AsyncMock()
        self.session_cache_patchers = [
            patch("app.services.agent.session_cache.write_session_started", AsyncMock()),
            patch("app.services.agent.session_cache.write_step_started", self.write_step_started_mock),
            patch("app.services.agent.session_cache.write_step_completed", self.write_step_completed_mock),
            patch("app.services.agent.session_cache.write_step_terminal", self.write_step_terminal_mock),
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

    async def _invoke(
        self,
        *,
        stream_round_side_effect,
        execute_tools_result=None,
        patch_extra=None,
        capabilities=None,
        options=None,
    ):
        """通用启动器：mock _stream_round + _execute_tools_parallel 后跑 generate_to_redis。

        stream_round_side_effect: callable 或 list；list 时按序消费每次 _stream_round 返回值
        execute_tools_result: _execute_tools_parallel 的返回值
        patch_extra: 额外 context manager 列表
        """
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "app.services.stream.runner.stream_round",
                    AsyncMock(side_effect=stream_round_side_effect),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.stream.runner.execute_tools_parallel",
                    AsyncMock(return_value=execute_tools_result or []),
                )
            )
            stack.enter_context(
                patch(
                    "app.services.stream.runner.llm_call_with_retry",
                    AsyncMock(return_value=MagicMock()),
                )
            )
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
                options=options or {"use_reasoning": False},
                capabilities=capabilities,
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
        self.assertEqual(
            types,
            [
                "run_started",
                "step_started",
                "step_completed",
                "run_completed",
            ],
        )
        # sequence 严格连续 0..3
        seqs = [e["sequence"] for e in events]
        self.assertEqual(seqs, [0, 1, 2, 3])
        # run_completed.finish_reason
        self.assertEqual(events[-1]["finish_reason"], "stop")
        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "completed")

    async def test_tool_mode_injects_web_search_contract_prompt(self):
        """工具模式：调用 LLM 前注入契约，避免 thinking 口头搜索但不发 tool_call。"""
        captured_messages = []

        async def _capture_llm_call(_model, _kwargs, messages, **_call_kwargs):
            captured_messages.append(messages)
            return MagicMock()

        await self._invoke(
            stream_round_side_effect=[
                ("", "Hello world", [], "stop", None),
            ],
            capabilities={"functionCalling": True, "deepThinking": True},
            patch_extra=[
                patch(
                    "app.services.stream.runner.llm_call_with_retry",
                    AsyncMock(side_effect=_capture_llm_call),
                )
            ],
        )

        system_prompts = [m["content"] for m in captured_messages[0] if m.get("role") == "system"]
        contract = "\n".join(system_prompts)
        self.assertIn("必须调用 web_search", contract)
        self.assertIn("不要在思考过程或最终回答中声称", contract)
        self.assertIn("没有调用工具", contract)

    async def test_round_summary_log_records_finish_reason_and_counts(self):
        """每轮 LLM 结束写诊断日志，后续可直接区分口头搜索和真实 tool_call。"""
        with patch("app.services.stream.runner.logger.info") as mock_info:
            await self._invoke(
                stream_round_side_effect=[
                    ("想搜索但没有工具调用", "Hello world", [], "stop", None),
                ],
                capabilities={"functionCalling": True, "deepThinking": True},
            )

        log_lines = [str(call.args[0]) for call in mock_info.call_args_list if call.args]
        round_logs = [line for line in log_lines if "AGENT_ROUND_SUMMARY" in line]
        self.assertEqual(len(round_logs), 1)
        self.assertIn("finish_reason=stop", round_logs[0])
        self.assertIn("tool_calls=0", round_logs[0])
        self.assertIn("reasoning_chars=10", round_logs[0])
        self.assertIn("content_chars=11", round_logs[0])

    async def test_normal_path_with_tool_calls(self):
        """正常 tool_calls + stop：2 round → run_completed(stop)"""
        tool_call = {"id": "tc1", "name": "web_search", "arguments": '{"query":"x"}'}

        await self._invoke(
            stream_round_side_effect=[
                ("", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            execute_tools_result=[
                ToolExecutionRecord(
                    tool_call=tool_call,
                    result=SimpleNamespace(
                        status="success",
                        error_message=None,
                        duration_ms=10,
                    ),
                    handler=None,  # handler=None → 走 else 分支不调用 build_content_block
                    block_id="blk_aaa",
                    log_id="log_aaa",
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

    async def test_tool_call_checkpoint_persists_message_before_tool_execution(self):
        """工具日志带 message_id 时，assistant 消息必须先落库，避免 FK 竞态丢诊断。"""
        tool_call = {"id": "tc1", "name": "web_search", "arguments": '{"query":"x"}'}
        order = []

        def _capture_persist(db, msg_id, conv_id, model_id, content_blocks, usage_data=None, partial=False):
            order.append(("persist", msg_id, partial, len(content_blocks)))

        async def _capture_execute(*_args, **kwargs):
            order.append(("execute", kwargs.get("message_id")))
            return [
                ToolExecutionRecord(
                    tool_call=tool_call,
                    result=SimpleNamespace(
                        status="success",
                        error_message=None,
                        duration_ms=10,
                    ),
                    handler=None,
                    block_id="blk_aaa",
                    log_id="log_aaa",
                )
            ]

        await self._invoke(
            stream_round_side_effect=[
                ("需要搜索", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            patch_extra=[
                patch(
                    "app.services.stream.runner.persist_message",
                    side_effect=_capture_persist,
                ),
                patch(
                    "app.services.stream.runner.execute_tools_parallel",
                    AsyncMock(side_effect=_capture_execute),
                ),
            ],
        )

        self.assertGreaterEqual(len(order), 2)
        self.assertEqual(order[0], ("persist", "msg-1", True, 1))
        self.assertEqual(order[1], ("execute", "msg-1"))

    async def test_degraded_url_read_injects_safe_tool_context(self):
        """url_read 降级时，下一轮 LLM 不能看到内部失败原因或被诱导无依据回答。"""
        from app.services.tool_handlers.base import ToolResult
        from app.services.tool_handlers.url_read import UrlReadHandler

        tool_call = {"id": "tc-url", "name": "url_read", "arguments": '{"url":"https://example.com"}'}
        captured_messages = []

        async def _capture_llm_call(_model, _kwargs, messages, **_call_kwargs):
            captured_messages.append([dict(message) for message in messages])
            return MagicMock()

        await self._invoke(
            stream_round_side_effect=[
                ("", "", [tool_call], "tool_calls", None),
                ("", "Final answer", [], "stop", None),
            ],
            execute_tools_result=[
                ToolExecutionRecord(
                    tool_call=tool_call,
                    result=ToolResult(
                        status="degraded",
                        error_message="reader-service 读取超时，已降级跳过",
                        data={"url": "https://example.com", "content": ""},
                    ),
                    handler=UrlReadHandler(),
                    block_id="blk_url",
                    log_id="log_url",
                ),
            ],
            patch_extra=[
                patch(
                    "app.services.stream.runner.llm_call_with_retry",
                    AsyncMock(side_effect=_capture_llm_call),
                )
            ],
        )

        self.assertGreaterEqual(len(captured_messages), 2)
        tool_messages = [message for message in captured_messages[1] if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        tool_context = tool_messages[0]["content"]
        self.assertIn("不能把该网页作为依据", tool_context)
        self.assertNotIn("reader-service", tool_context)
        self.assertNotIn("请基于你的知识回答", tool_context)

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
        """LLM 抛非 Cancelled 异常：发 run_failed + status='error' + re-raise 给上层"""

        async def _raise_runtime(*args, **kwargs):
            raise RuntimeError("upstream LLM 5xx")

        # generate_to_redis 内 except Exception 块完成协议层 + SSE 收尾后 re-raise，
        # 让 background task scheduler 拿到失败信号（与 CancelledError 行为对齐）。
        with self.assertRaises(RuntimeError) as cm:
            await self._invoke(
                stream_round_side_effect=_raise_runtime,
            )
        self.assertIn("upstream LLM 5xx", str(cm.exception))

        # 协议终态仍正确写入（即使 raise 也要先收尾）
        events = self._agent_events()
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "run_started")
        self.assertIn("run_failed", types)
        run_failed = [e for e in events if e["type"] == "run_failed"][0]
        self.assertIn("upstream LLM 5xx", run_failed["message"])

        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "error")

        # SSE finalize 已被调（finalize_stream 在 raise 之前完成）
        self.finalize_mock.assert_awaited()

    async def test_step_started_cache_failure_marks_active_step_failed(self):
        """step_started 事件发出后 cache 写入失败时，异常收尾必须标记 active step failed。"""
        self.write_step_started_mock.side_effect = RuntimeError("step cache boom")

        with self.assertRaises(RuntimeError) as cm:
            await self._invoke(
                stream_round_side_effect=[
                    ("", "不会执行到 stream_round", [], "stop", None),
                ],
            )
        self.assertIn("step cache boom", str(cm.exception))

        events = self._agent_events()
        step_started = [e for e in events if e["type"] == "step_started"][0]
        self.write_step_terminal_mock.assert_awaited_once_with(
            step_id=step_started["step_id"],
            status="failed",
        )
        self.assertIn("run_failed", [e["type"] for e in events])
        self.assertEqual(self.session_statuses[-1]["status"], "error")

    async def test_limit_reached_max_steps(self):
        """触顶 max_steps：发 run_limit_reached(max_steps) → 强制总结 → run_completed(limit_reached)"""
        from app.services.stream import runner as sh

        tool_call = {"id": "tc1", "name": "web_search", "arguments": '{"query":"x"}'}

        # max_steps=2 → 需要 2 轮 tool_calls 让循环顶端触顶；之后强制总结再来 1 轮 stop
        rounds = [("", "", [tool_call], "tool_calls", None)] * 2
        rounds.append(("", "summary", [], "stop", None))

        with patch.object(sh, "AGENT_MAX_STEPS", 2):
            await self._invoke(
                stream_round_side_effect=rounds,
                execute_tools_result=[
                    ToolExecutionRecord(
                        tool_call=tool_call,
                        result=SimpleNamespace(
                            status="success",
                            error_message=None,
                            duration_ms=10,
                        ),
                        handler=None,
                        block_id="blk_aaa",
                        log_id="log_aaa",
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
        post_limit = types[limit_idx + 1 :]
        self.assertIn("step_started", post_limit)
        self.assertIn("step_completed", post_limit)

        # 最后是 run_completed(finish_reason='limit_reached')
        self.assertEqual(types[-1], "run_completed")
        self.assertEqual(events[-1]["finish_reason"], "limit_reached")

        # session 终态
        self.assertEqual(self.session_statuses[-1]["status"], "limit_reached")

    async def test_tool_calls_empty_list_preserves_content(self):
        """雷点 3 修复后正向验证：finish_reason=tool_calls + 空 tool_calls_list 退化时，
        content_buf 应当被 append 到 content_blocks，且 run finish_reason 报告为 incomplete。
        """
        # 截获 persist_message 最终落库时传入的 content_blocks
        persist_calls = []

        def _capture_persist(db, msg_id, conv_id, model_id, content_blocks, usage_data=None, partial=False):
            if not partial:
                persist_calls.append(list(content_blocks))

        await self._invoke(
            stream_round_side_effect=[
                # 退化场景：content_buf 有文本，但 tool_calls_list 为空，finish_reason="tool_calls"
                ("", "Hello world", [], "tool_calls", None),
            ],
            patch_extra=[
                patch(
                    "app.services.stream.runner.persist_message",
                    side_effect=_capture_persist,
                ),
            ],
        )

        # 1. 走到 run_completed
        events = self._agent_events()
        run_completed_events = [e for e in events if e["type"] == "run_completed"]
        self.assertEqual(len(run_completed_events), 1)

        # 修复后：finish_reason 应为 incomplete（不是 stop）
        self.assertEqual(run_completed_events[0]["finish_reason"], "incomplete")

        # 2. content_buf "Hello world" 应当被保留在 content_blocks（雷点 3 修复）
        final_blocks = persist_calls[-1] if persist_calls else []
        text_blocks = [b for b in final_blocks if getattr(b, "type", None) == "text"]
        has_hello = any("Hello" in getattr(b, "text", "") for b in text_blocks)
        self.assertTrue(
            has_hello,
            "雷点 3 修复后：退化分支应保留 content_buf。"
            "如果这条断言挂了说明修复回归，请检查 runner.py unknown 退化分支。",
        )

    async def test_limit_reached_summary_timeout_falls_through(self):
        """雷点 2 修复验证：触顶总结超时时不卡死，落库已有内容 + 走 limit_reached 收尾。

        模拟：max_tool_calls=1 触顶，触顶总结的 LLM 调用 hang 住，
        asyncio.wait_for(remaining=2s) 超时后应 graceful 收尾，
        emit run_limit_reached + run_completed(limit_reached)，不 raise。
        """
        from app.services.stream import runner as run_mod

        llm_call_count = 0

        async def _hang_on_summary(*_a, **_kw):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                return MagicMock()
            await asyncio.sleep(120)  # 远超触顶总结预算

        tool_call = {"id": "tc1", "name": "web_search", "arguments": "{}"}

        with patch.object(run_mod, "AGENT_MAX_TOOL_CALLS", 1):
            await self._invoke(
                stream_round_side_effect=[
                    # 第 1 步：返回 1 个 tool_call，触发 max_tool_calls 触顶
                    ("", "", [tool_call], "tool_calls", None),
                    # 触顶总结的 stream_round 不会被消费（_do_summary 内部 hang 住了）
                ],
                execute_tools_result=[
                    ToolExecutionRecord(
                        tool_call=tool_call,
                        result=SimpleNamespace(
                            status="success",
                            error_message=None,
                            duration_ms=10,
                        ),
                        handler=None,  # handler=None → 走 else 分支
                        block_id="blk_x",
                        log_id="log_x",
                    ),
                ],
                patch_extra=[
                    # 压缩总 budget 为 2s，让 wait_for 迅速超时
                    patch("app.services.stream.runner.AGENT_TOTAL_TIMEOUT", 2),
                    # 首轮 LLM 立即返回；触顶总结 LLM hang，验证总结 wait_for 超时收尾
                    patch(
                        "app.services.stream.runner.llm_call_with_retry",
                        AsyncMock(side_effect=_hang_on_summary),
                    ),
                ],
            )

        events = self._agent_events()
        types = [e["type"] for e in events]
        # 应当 emit run_limit_reached 然后 run_completed（不卡死、不 raise）
        self.assertIn("run_limit_reached", types)
        self.assertIn("run_completed", types)
        run_completed = [e for e in events if e["type"] == "run_completed"][0]
        self.assertEqual(run_completed["finish_reason"], "limit_reached")
        # session 终态也应为 limit_reached
        self.assertEqual(self.session_statuses[-1]["status"], "limit_reached")


class UrlPreprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_preprocess_url_in_message_uses_user_web_context_not_system(self):
        from app.services.external.reader_client import UrlReadResult
        from app.services.stream.persistence import preprocess_url_in_message

        call_kwargs = {"tools": []}
        read_result = UrlReadResult(
            url="https://example.com/a",
            title="示例",
            content="网页正文",
            favicon=None,
            content_length=4,
            fetch_ms=20,
        )

        with patch(
            "app.services.external.reader_client.read_url",
            new_callable=AsyncMock,
            return_value=read_result,
        ):
            block, context_msg, detected_url = await preprocess_url_in_message(
                "请看 https://example.com/a", True, call_kwargs
            )

        self.assertIsNotNone(block)
        self.assertEqual(detected_url, "https://example.com/a")
        self.assertEqual(context_msg["role"], "user")
        self.assertIn("<web_context", context_msg["content"])
        self.assertIn("内容不可信", context_msg["content"])

    async def test_preprocess_url_in_message_rejects_sensitive_query_without_reader_call(self):
        from app.ai.tools import URL_READ_TOOL
        from app.services.stream.persistence import preprocess_url_in_message

        call_kwargs = {"tools": []}
        with patch("app.services.external.reader_client.read_url", new_callable=AsyncMock) as mock_read:
            block, context_msg, detected_url = await preprocess_url_in_message(
                "请看 https://example.com/page?token=abc", True, call_kwargs
            )

        self.assertIsNone(block)
        self.assertIsNone(context_msg)
        self.assertEqual(detected_url, "https://example.com/page?token=abc")
        mock_read.assert_not_called()
        self.assertIn(URL_READ_TOOL, call_kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
