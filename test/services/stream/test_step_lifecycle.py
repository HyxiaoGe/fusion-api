import unittest
from unittest.mock import AsyncMock, Mock, call

from app.services.stream import step_lifecycle as step_lifecycle_module
from app.services.stream.step_lifecycle import (
    AgentStepContext,
    complete_agent_step,
    start_agent_step,
)


class StepLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_agent_step_emits_started_before_cache_write_and_returns_context(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-1")
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()
        order = Mock()
        order.attach_mock(emitter.step_started, "step_started")
        order.attach_mock(session_cache.write_step_started, "write_step_started")

        context = await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=3,
            clock=Mock(return_value=10.5),
            block_id_factory=Mock(side_effect=["blk_thinking", "blk_text"]),
        )

        self.assertEqual(
            context,
            AgentStepContext(
                step_id="step-1",
                run_id="run-1",
                step_number=3,
                started_at=10.5,
                thinking_block_id="blk_thinking",
                text_block_id="blk_text",
            ),
        )
        self.assertEqual(
            order.mock_calls,
            [
                call.step_started(step_number=3),
                call.write_step_started(run_id="run-1", step_id="step-1", step_number=3),
            ],
        )

    async def test_start_agent_step_calls_callback_after_started_before_cache_write(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-1")
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()
        on_step_started = Mock()
        order = Mock()
        order.attach_mock(emitter.step_started, "step_started")
        order.attach_mock(on_step_started, "on_step_started")
        order.attach_mock(session_cache.write_step_started, "write_step_started")

        await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=3,
            clock=Mock(return_value=10.5),
            block_id_factory=Mock(side_effect=["blk_thinking", "blk_text"]),
            on_step_started=on_step_started,
        )

        self.assertEqual(
            order.mock_calls,
            [
                call.step_started(step_number=3),
                call.on_step_started("step-1"),
                call.write_step_started(run_id="run-1", step_id="step-1", step_number=3),
            ],
        )

    async def test_start_agent_step_default_block_ids_keep_blk_prefix_shape(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-1")
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()

        context = await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=1,
            clock=Mock(return_value=1.0),
        )

        self.assertRegex(context.thinking_block_id, r"^blk_[0-9a-f]{12}$")
        self.assertRegex(context.text_block_id, r"^blk_[0-9a-f]{12}$")
        self.assertNotEqual(context.thinking_block_id, context.text_block_id)

    async def test_start_agent_step_records_duration_origin_before_step_event_for_all_steps(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-1")
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()
        clock = Mock(return_value=10.5)
        order = Mock()
        order.attach_mock(clock, "clock")
        order.attach_mock(emitter.step_started, "step_started")
        order.attach_mock(session_cache.write_step_started, "write_step_started")

        context = await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=3,
            clock=clock,
            block_id_factory=Mock(side_effect=["blk_thinking", "blk_text"]),
        )

        self.assertEqual(context.started_at, 10.5)
        self.assertEqual(
            order.mock_calls,
            [
                call.clock(),
                call.step_started(step_number=3),
                call.write_step_started(run_id="run-1", step_id="step-1", step_number=3),
            ],
        )

    async def test_start_agent_step_updates_progress_plan_when_emitter_supports_v2(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-1")
        emitter.plan_step_updated = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()

        await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=1,
            clock=Mock(return_value=10.5),
            block_id_factory=Mock(side_effect=["blk_thinking", "blk_text"]),
        )

        emitter.plan_step_updated.assert_awaited_once_with(
            plan_id="plan-run-1",
            revision=2,
            item={
                "id": "understand",
                "title": "理解问题",
                "status": "running",
                "kind": "reasoning",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )

    async def test_start_agent_step_updates_read_and_answer_plan_for_followup_step(self):
        emitter = Mock()
        emitter.step_started = AsyncMock(return_value="step-2")
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_started = AsyncMock()

        await start_agent_step(
            emitter=emitter,
            session_cache=session_cache,
            run_id="run-1",
            step_number=2,
            completed_tool_calls=1,
            max_tool_calls=20,
            clock=Mock(return_value=10.5),
            block_id_factory=Mock(side_effect=["blk_thinking", "blk_text"]),
        )

        self.assertEqual(
            emitter.plan_step_updated.await_args_list,
            [
                call(
                    plan_id="plan-run-1",
                    revision=12,
                    item={
                        "id": "read",
                        "title": "读取关键来源",
                        "status": "completed",
                        "kind": "read",
                        "summary": "已完成关键来源读取",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
                call(
                    plan_id="plan-run-1",
                    revision=13,
                    item={
                        "id": "answer",
                        "title": "整理回答",
                        "status": "running",
                        "kind": "answer",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
            ],
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="synthesizing",
            label="正在整理回答",
            completed_steps=3,
            total_steps=None,
            completed_tool_calls=1,
            max_tool_calls=20,
        )

    async def test_mark_tool_round_started_for_url_read_reverts_answer_and_runs_read(self):
        emitter = Mock()
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        context = AgentStepContext(
            step_id="step-3",
            run_id="run-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        await step_lifecycle_module.mark_tool_round_started(
            context=context,
            emitter=emitter,
            tool_call_count=1,
            tool_names=("url_read",),
            completed_tool_calls=2,
            max_tool_calls=20,
        )

        self.assertEqual(
            emitter.plan_step_updated.await_args_list,
            [
                call(
                    plan_id="plan-run-1",
                    revision=24,
                    item={
                        "id": "answer",
                        "title": "整理回答",
                        "status": "pending",
                        "kind": "answer",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
                call(
                    plan_id="plan-run-1",
                    revision=25,
                    item={
                        "id": "read",
                        "title": "读取关键来源",
                        "status": "running",
                        "kind": "read",
                        "summary": "正在读取 1 个关键来源",
                        "tool_names": ["url_read"],
                        "evidence_item_ids": [],
                    },
                ),
            ],
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="reading",
            label="正在读取关键来源",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=2,
            max_tool_calls=20,
        )

    async def test_complete_agent_step_emits_completed_before_cache_write_and_returns_duration(self):
        emitter = Mock()
        emitter.step_completed = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_completed = AsyncMock()
        order = Mock()
        order.attach_mock(emitter.step_completed, "step_completed")
        order.attach_mock(session_cache.write_step_completed, "write_step_completed")
        context = AgentStepContext(
            step_id="step-1",
            run_id="run-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        duration_ms = await complete_agent_step(
            context=context,
            emitter=emitter,
            session_cache=session_cache,
            tool_names=("web_search",),
            tool_call_count=1,
            completed_tool_calls=3,
            max_tool_calls=20,
            clock=Mock(return_value=10.25),
        )

        self.assertEqual(duration_ms, 250)
        self.assertEqual(
            order.mock_calls,
            [
                call.step_completed(step_number=3, tool_call_count=1, duration_ms=250),
                call.write_step_completed(
                    step_id="step-1",
                    tool_names=["web_search"],
                    tool_calls_count=1,
                    duration_ms=250,
                ),
            ],
        )

    async def test_mark_tool_round_started_switches_understand_to_search_before_tool_execution(self):
        emitter = Mock()
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        context = AgentStepContext(
            step_id="step-1",
            run_id="run-1",
            step_number=1,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        await step_lifecycle_module.mark_tool_round_started(
            context=context,
            emitter=emitter,
            tool_call_count=2,
            completed_tool_calls=0,
            max_tool_calls=20,
        )

        self.assertEqual(
            emitter.plan_step_updated.await_args_list,
            [
                call(
                    plan_id="plan-run-1",
                    revision=3,
                    item={
                        "id": "understand",
                        "title": "理解问题",
                        "status": "completed",
                        "kind": "reasoning",
                        "summary": "已完成问题理解",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
                call(
                    plan_id="plan-run-1",
                    revision=4,
                    item={
                        "id": "search",
                        "title": "查找资料",
                        "status": "running",
                        "kind": "search",
                        "summary": "正在执行 2 个工具调用",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
            ],
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="researching",
            label="正在查找资料",
            completed_steps=1,
            total_steps=None,
            completed_tool_calls=0,
            max_tool_calls=20,
        )

    async def test_complete_agent_step_updates_search_and_read_plan_when_emitter_supports_v2(self):
        emitter = Mock()
        emitter.step_completed = AsyncMock()
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_completed = AsyncMock()
        context = AgentStepContext(
            step_id="step-1",
            run_id="run-1",
            step_number=1,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        await complete_agent_step(
            context=context,
            emitter=emitter,
            session_cache=session_cache,
            tool_names=("web_search",),
            tool_call_count=1,
            completed_tool_calls=3,
            max_tool_calls=20,
            clock=Mock(return_value=10.25),
        )

        self.assertEqual(
            emitter.plan_step_updated.await_args_list,
            [
                call(
                    plan_id="plan-run-1",
                    revision=5,
                    item={
                        "id": "search",
                        "title": "查找资料",
                        "status": "completed",
                        "kind": "search",
                        "summary": "完成 3 个工具调用",
                        "tool_names": ["web_search"],
                        "evidence_item_ids": [],
                    },
                ),
                call(
                    plan_id="plan-run-1",
                    revision=6,
                    item={
                        "id": "read",
                        "title": "读取关键来源",
                        "status": "running",
                        "kind": "read",
                        "summary": "正在整理关键来源",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    },
                ),
            ],
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="reading",
            label="正在读取关键来源",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=3,
            max_tool_calls=20,
        )

    async def test_complete_agent_step_without_tools_updates_answer_not_search(self):
        emitter = Mock()
        emitter.step_completed = AsyncMock()
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_completed = AsyncMock()
        context = AgentStepContext(
            step_id="step-2",
            run_id="run-1",
            step_number=2,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        await complete_agent_step(
            context=context,
            emitter=emitter,
            session_cache=session_cache,
            tool_names=(),
            tool_call_count=0,
            completed_tool_calls=1,
            max_tool_calls=20,
            clock=Mock(return_value=10.25),
        )

        emitter.plan_step_updated.assert_awaited_once_with(
            plan_id="plan-run-1",
            revision=19,
            item={
                "id": "answer",
                "title": "整理回答",
                "status": "completed",
                "kind": "answer",
                "summary": "已完成回答整理",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="answering",
            label="已完成回答整理",
            completed_steps=4,
            total_steps=None,
            completed_tool_calls=1,
            max_tool_calls=20,
        )

    async def test_complete_agent_step_for_url_read_completes_read_without_reopening_search(self):
        emitter = Mock()
        emitter.step_completed = AsyncMock()
        emitter.plan_step_updated = AsyncMock()
        emitter.run_progress_updated = AsyncMock()
        session_cache = Mock()
        session_cache.write_step_completed = AsyncMock()
        context = AgentStepContext(
            step_id="step-3",
            run_id="run-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )

        await complete_agent_step(
            context=context,
            emitter=emitter,
            session_cache=session_cache,
            tool_names=("url_read",),
            tool_call_count=1,
            completed_tool_calls=3,
            max_tool_calls=20,
            clock=Mock(return_value=10.25),
        )

        emitter.plan_step_updated.assert_awaited_once_with(
            plan_id="plan-run-1",
            revision=26,
            item={
                "id": "read",
                "title": "读取关键来源",
                "status": "completed",
                "kind": "read",
                "summary": "已完成关键来源读取",
                "tool_names": ["url_read"],
                "evidence_item_ids": [],
            },
        )
        emitter.run_progress_updated.assert_awaited_once_with(
            phase="reading",
            label="已完成关键来源读取",
            completed_steps=2,
            total_steps=None,
            completed_tool_calls=3,
            max_tool_calls=20,
        )
