import asyncio
import concurrent.futures
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta
from app.schemas.chat import (
    KnowledgeEvidenceBlock,
    KnowledgeSourceReference,
    SearchBlock,
    SearchSourceSummary,
    SourceReference,
    TextBlock,
    UrlBlock,
)
from app.services.agent.trajectory_recorder import TrajectoryRecorder
from app.services.knowledge.chat_grounding import KnowledgeGroundingResult
from app.services.stream.agent_loop_driver import AgentLoopExit, AgentLoopOutcome
from app.services.stream.agent_loop_execution import (
    AgentLoopDependencies as ExecutionDependencies,
)
from app.services.stream.agent_loop_execution import (
    AgentLoopExecutionRequest,
    build_agent_loop_execution,
)
from app.services.stream.agent_loop_lifecycle import (
    AgentLoopLifecycleDependencies,
    AgentLoopLifecycleRequest,
    _prepare_knowledge_grounding,
    commit_trajectory_barrier,
    configure_research_state,
    run_agent_loop_lifecycle,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import AgentLoopPreparedMessages
from app.services.stream.agent_loop_run_completion import (
    finalize_completed_run,
    write_fallback_run_error,
)
from app.services.stream.research_evidence import validate_research_completion
from app.services.stream.run_finalizer import interrupt_agent_run
from app.services.stream.tool_executor import AgentEventCompositeWriter
from app.services.stream_state_service import StreamOwnershipLostError, StreamWriteTerminalError


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


async def _start_event_run(**kwargs):
    await kwargs["emitter"].run_started(
        message_id=kwargs["message_id"], model=kwargs["model_id"], tools=kwargs["tools"], config=kwargs["config"]
    )


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


class AgentLoopLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_prompt_ready_is_emitted_once_before_driver(self):
        from app.ai.prompts.system_prompt import assemble_system_prompt

        emitted = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                emitted.append(payload)

        assembly = assemble_system_prompt(user_system_prompt="不能进入事件的规则原文")
        prepared = SimpleNamespace(
            messages=assembly.messages,
            initial_content_blocks=[],
            final_tool_names=[],
            prompt_assembly=assembly.metadata,
        )

        async def run_loop(**_kwargs):
            self.assertEqual([event["type"] for event in emitted].count("system_prompt_prepared"), 1)
            return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

        await run_agent_loop_lifecycle(
            request=self._request(),
            execution=self._execution(redis_writer=CaptureWriter()),
            dependencies=self._dependencies(
                start_agent_run_fn=_start_event_run,
                prepare_messages_fn=AsyncMock(return_value=prepared),
                run_agent_loop_fn=run_loop,
            ),
        )
        prompt_events = [event for event in emitted if event["type"] == "system_prompt_prepared"]
        self.assertEqual(len(prompt_events), 1)
        self.assertEqual(prompt_events[0]["fingerprint"], assembly.metadata["fingerprint"])
        self.assertNotIn("不能进入事件的规则原文", str(emitted))

    async def test_only_actual_assembly_error_emits_failed_prompt_result(self):
        from app.ai.prompts.system_prompt import SystemPromptAssemblyError

        metadata = {
            "status": "failed",
            "source": "code",
            "template_version": "1",
            "section_ids": [],
            "duration_ms": 0,
            "error_code": "assembly_failed",
            "message": "不应披露的偏好内容",
        }
        for error, expected_count in [(SystemPromptAssemblyError(metadata), 1), (RuntimeError("附件读取失败"), 0)]:
            with self.subTest(error_type=type(error).__name__):
                emitted = []

                class CaptureWriter:
                    async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                        emitted.append(payload)

                with self.assertRaises(type(error)):
                    await run_agent_loop_lifecycle(
                        request=self._request(),
                        execution=self._execution(redis_writer=CaptureWriter()),
                        dependencies=self._dependencies(
                            start_agent_run_fn=_start_event_run, prepare_messages_fn=AsyncMock(side_effect=error)
                        ),
                    )
                prompt_events = [event for event in emitted if event["type"] == "system_prompt_prepared"]
                self.assertEqual(len(prompt_events), expected_count)
                if prompt_events:
                    self.assertEqual(prompt_events[0]["status"], "failed")
                    self.assertEqual(prompt_events[0]["error_code"], "assembly_failed")
                self.assertNotIn("不应披露的偏好内容", str(emitted))

    async def test_knowledge_retrieval_emits_started_and_completed_around_real_call(self):
        emitted = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                emitted.append(payload)

        execution = self._execution(redis_writer=CaptureWriter())
        request = AgentLoopLifecycleRequest(
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="不能写入轨迹的原始检索问题",
            call_config=self._call_config(),
            limits=self._limits(),
            knowledge_base_ids=["kb-1"],
        )
        grounding = SimpleNamespace(evidence_block=SimpleNamespace(source_count=2))

        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(return_value=grounding),
        ):
            result = await _prepare_knowledge_grounding(request=request, execution=execution)

        self.assertIs(result, grounding)
        retrieval_events = [event for event in emitted if event["type"].startswith("retrieval_")]
        self.assertEqual(
            [event["type"] for event in retrieval_events],
            ["retrieval_started", "retrieval_completed"],
        )
        self.assertEqual(retrieval_events[0]["query_summary"], "已发起知识库检索")
        self.assertEqual(retrieval_events[1]["document_count"], 2)
        self.assertGreaterEqual(retrieval_events[1]["duration_ms"], 0)

    async def test_knowledge_retrieval_error_and_cancel_close_the_span(self):
        emitted = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                emitted.append(payload)

        execution = self._execution(redis_writer=CaptureWriter())
        request = AgentLoopLifecycleRequest(
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="敏感问题",
            call_config=self._call_config(),
            limits=self._limits(),
            knowledge_base_ids=["kb-1"],
        )

        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(side_effect=RuntimeError("上游敏感报错")),
        ):
            with self.assertRaisesRegex(Exception, "knowledge_retrieval_unavailable"):
                await _prepare_knowledge_grounding(request=request, execution=execution)

        retrieval_events = [event for event in emitted if event["type"].startswith("retrieval_")]
        self.assertEqual(
            [event["type"] for event in retrieval_events],
            ["retrieval_started", "retrieval_failed"],
        )
        self.assertNotIn("上游敏感报错", str(retrieval_events[-1]))

        emitted.clear()
        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _prepare_knowledge_grounding(request=request, execution=execution)

        retrieval_events = [event for event in emitted if event["type"].startswith("retrieval_")]
        self.assertEqual(
            [event["type"] for event in retrieval_events],
            ["retrieval_started", "retrieval_cancelled"],
        )

    async def test_retrieval_terminal_sink_failure_does_not_replace_primary_error_or_cancel(self):
        execution = self._execution()
        request = AgentLoopLifecycleRequest(
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="敏感问题",
            call_config=self._call_config(),
            limits=self._limits(),
            knowledge_base_ids=["kb-1"],
        )
        execution.emitter.retrieval_failed = AsyncMock(side_effect=RuntimeError("sink failed"))
        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(side_effect=RuntimeError("upstream failed")),
        ):
            with self.assertRaisesRegex(Exception, "knowledge_retrieval_unavailable"):
                await _prepare_knowledge_grounding(request=request, execution=execution)

        execution.emitter.retrieval_cancelled = AsyncMock(side_effect=RuntimeError("sink failed"))
        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _prepare_knowledge_grounding(request=request, execution=execution)

    def test_deep_research_with_files_still_requires_network_gate(self):
        execution = self._execution(
            call_config=SimpleNamespace(
                should_use_reasoning=False,
                call_kwargs={},
                announced_tools=["web_search", "url_read"],
                task_mode="deep_research",
                network_profile="deep_research",
                evidence_policy="deep_research_v1",
                plan_mode="on",
            )
        )

        configure_research_state(
            state=execution.state,
            call_config=execution.runtime,
            file_ids=["file-1"],
            content_blocks=[],
        )

        self.assertTrue(execution.state.research_network_required)
        self.assertEqual(
            execution.state.plan_coordinator.required_initial_tool_counts,
            {
                "web_search": 1,
                "url_read": 2,
            },
        )
        result = execution.state.plan_coordinator.apply_model_update(
            {
                "reason": "按研究阶段组织计划",
                "items": [
                    {
                        "id": "search",
                        "title": "查找可靠来源",
                        "status": "running",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "read-1",
                        "title": "核验来源一原文",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "read-2",
                        "title": "核验来源二原文",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "answer",
                        "title": "整理结论与建议",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read-1", "read-2"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(result.accepted)

    def test_continuation_rebuilds_research_workset_from_persisted_source_blocks(self):
        execution = self._execution(
            call_config=SimpleNamespace(
                should_use_reasoning=False,
                call_kwargs={},
                announced_tools=["web_search", "url_read"],
                task_mode="deep_research",
                network_profile="deep_research",
                evidence_policy="deep_research_v1",
                plan_mode="on",
            )
        )
        historical_blocks = [
            SearchBlock(
                type="search",
                query="研究",
                sources=[SearchSourceSummary(title="来源", url="https://example.com/report")],
                source_refs=[
                    SourceReference(
                        kind="search",
                        title="来源",
                        url="https://example.com/report",
                        evidence_id="ev-report",
                        citation_index=1,
                    )
                ],
                source_count=1,
            ),
            UrlBlock(
                type="url_read",
                url="https://example.com/report",
                title="来源",
                source_refs=[
                    SourceReference(
                        kind="url_read",
                        title="来源",
                        url="https://example.com/report",
                        evidence_id="ev-report",
                        citation_index=1,
                    )
                ],
                source_count=1,
            ),
            UrlBlock(
                type="url_read",
                url="https://example.com/second",
                title="第二来源",
                source_refs=[
                    SourceReference(
                        kind="url_read",
                        title="第二来源",
                        url="https://example.com/second",
                        evidence_id="ev-second",
                        citation_index=2,
                    )
                ],
                source_count=1,
            ),
        ]

        configure_research_state(
            state=execution.state,
            call_config=execution.runtime,
            file_ids=None,
            content_blocks=historical_blocks,
            allow_read_success=False,
        )

        self.assertEqual(execution.state.research_workset.successful_searches, 1)
        self.assertEqual(execution.state.research_workset.successful_read_urls, set())
        self.assertFalse(
            validate_research_completion(
                execution.state.research_workset,
                "历史来源不能直接复用。[1][2]",
            ).is_valid
        )

        execution.state.record_research_content_blocks(
            [
                UrlBlock(
                    type="url_read",
                    url="https://example.com/report",
                    title="来源",
                    source_refs=historical_blocks[1].source_refs,
                    source_count=1,
                ),
                UrlBlock(
                    type="url_read",
                    url="https://example.com/second",
                    title="第二来源",
                    source_refs=historical_blocks[2].source_refs,
                    source_count=1,
                ),
            ]
        )

        self.assertEqual(execution.state.research_workset.valid_citation_indexes, {1, 2})
        self.assertTrue(
            validate_research_completion(
                execution.state.research_workset,
                "重新读取后可继续引用。[1][2]",
            ).is_valid
        )

    def _call_config(self):
        return SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
            announced_tools=["web_search"],
        )

    def _limits(self):
        return AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=30)

    def _execution(self, *, call_config=None, limits=None, redis_writer=None):
        call_config = call_config or self._call_config()
        limits = limits or self._limits()

        class NoopWriter:
            async def append_chunk(self, *_args, **_kwargs):
                return None

        execution = build_agent_loop_execution(
            request=AgentLoopExecutionRequest(
                db="db",
                conversation_id="conv-life",
                user_id="user-life",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                assistant_message_id="msg-life",
                turn_message_id="turn-life",
                previous_run_id="run-previous",
                run_attempt_kind="continue",
                task_id="task-life",
                call_config=call_config,
                trace_id="run-life",
            ),
            limits=limits,
            dependencies=ExecutionDependencies(
                session_cache="session-cache",
                redis_writer=redis_writer or NoopWriter(),
                start_step_fn=_unused_async,
                complete_step_fn=_unused_async,
                run_round_fn=_unused_async,
                handle_tool_calls_round_fn=_unused_async,
                run_limit_summary_step_fn=_unused_async,
                llm_call_fn=_unused_async,
                stream_round_fn=_unused_async,
                execute_tools_fn=_unused_async,
                persist_message_fn=_unused_sync,
                log_round_summary_fn=lambda **_kwargs: None,
                warning_fn=lambda _message: None,
                clock=lambda: 10.0,
            ),
        )
        # lifecycle 测试不连接真实账本数据库；这里只验证编排层接纳与 finalize 契约。
        execution.trajectory_recorder.record_chunk = AsyncMock()
        execution.trajectory_recorder.finalize = AsyncMock()
        return execution

    def _request(self, *, call_config=None, limits=None):
        return AgentLoopLifecycleRequest(
            raw_messages=[{"role": "user", "content": "hi"}],
            has_vision=False,
            file_ids=None,
            original_message="hi",
            call_config=call_config or self._call_config(),
            limits=limits or self._limits(),
            initial_content_blocks=[],
            extra_system_prompts=[],
            preprocess_user_input=True,
        )

    def _dependencies(self, **overrides):
        async def append_chunk_fn(*_args, **_kwargs):
            return None

        async def start_agent_run_fn(**_kwargs):
            return None

        async def prepare_messages_fn(**_kwargs):
            return AgentLoopPreparedMessages(messages=[{"role": "user", "content": "hi"}])

        async def run_agent_loop_fn(**_kwargs):
            return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

        async def finalize_completed_run_fn(**_kwargs):
            return None

        async def finalize_superseded_run_fn(**_kwargs):
            return None

        async def finalize_cancelled_run_fn(**_kwargs):
            return None

        async def finalize_failed_run_fn(**_kwargs):
            return None

        async def write_fallback_run_error_fn(**_kwargs):
            return None

        values = {
            "append_chunk_fn": append_chunk_fn,
            "start_agent_run_fn": start_agent_run_fn,
            "prepare_messages_fn": prepare_messages_fn,
            "run_agent_loop_fn": run_agent_loop_fn,
            "finalize_completed_run_fn": finalize_completed_run_fn,
            "finalize_superseded_run_fn": finalize_superseded_run_fn,
            "finalize_cancelled_run_fn": finalize_cancelled_run_fn,
            "finalize_failed_run_fn": finalize_failed_run_fn,
            "write_fallback_run_error_fn": write_fallback_run_error_fn,
            "persist_message_fn": _unused_sync,
            "complete_agent_run_fn": _unused_async,
            "interrupt_agent_run_fn": _unused_async,
            "fail_agent_run_fn": _unused_async,
            "finalize_stream_fn": _unused_async,
            "write_fallback_error_status_fn": _unused_async,
            "info_fn": lambda _message: None,
            "error_fn": lambda _message: None,
            "warning_fn": lambda _message: None,
        }
        values.update(overrides)
        return AgentLoopLifecycleDependencies(**values)

    async def test_completed_path_prepares_runs_and_finalizes_in_order(self):
        call_order = []
        call_config = self._call_config()
        limits = self._limits()
        execution = self._execution(call_config=call_config, limits=limits)
        initial_block = TextBlock(type="text", id="txt-initial", text="初始块")

        async def append_chunk_fn(conversation_id, chunk_type, content, block_id, *, task_id):
            call_order.append(("append", conversation_id, task_id, chunk_type, content, block_id))

        async def start_agent_run_fn(**kwargs):
            call_order.append(
                (
                    "start",
                    kwargs["run_id"],
                    kwargs["turn_message_id"],
                    kwargs["previous_run_id"],
                    kwargs["run_attempt_kind"],
                    kwargs["tools"],
                    kwargs["config"],
                )
            )

        async def prepare_messages_fn(**kwargs):
            call_order.append(
                (
                    "prepare",
                    kwargs["db"],
                    kwargs["raw_messages"],
                    kwargs["call_config"],
                    kwargs["user_id"],
                    kwargs["conversation_id"],
                )
            )
            return AgentLoopPreparedMessages(
                messages=[{"role": "user", "content": "prepared"}],
                initial_content_blocks=[initial_block],
            )

        async def run_agent_loop_fn(**kwargs):
            call_order.append(("run", kwargs["messages"], list(kwargs["state"].content_blocks)))
            return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

        async def finalize_completed_run_fn(**kwargs):
            call_order.append(("completed", kwargs["context"], kwargs["terminal_state"].session_status))

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        await run_agent_loop_lifecycle(
            request=self._request(call_config=call_config, limits=limits),
            execution=execution,
            dependencies=self._dependencies(
                append_chunk_fn=append_chunk_fn,
                start_agent_run_fn=start_agent_run_fn,
                prepare_messages_fn=prepare_messages_fn,
                run_agent_loop_fn=run_agent_loop_fn,
                finalize_completed_run_fn=finalize_completed_run_fn,
                write_fallback_run_error_fn=write_fallback_run_error_fn,
            ),
        )

        self.assertEqual(
            [item[0] for item in call_order],
            ["append", "start", "prepare", "run", "completed", "fallback"],
        )
        self.assertEqual(call_order[0], ("append", "conv-life", "task-life", "preparing", "", ""))
        self.assertEqual(call_order[1][1], "run-life")
        self.assertEqual(call_order[1][2:5], ("turn-life", "run-previous", "continue"))
        self.assertEqual(call_order[1][5], ["web_search"])
        self.assertEqual(
            call_order[1][6],
            {
                "max_steps": 3,
                "max_tool_calls": 5,
                "timeout_s": 30,
                "plan_mode": "auto",
                "task_mode": "standard",
                "network_profile": "standard",
                "evidence_policy": "standard",
                "runtime_config_versions": {
                    "agent_strategy/default": "code-default",
                },
            },
        )
        self.assertIs(call_order[2][3], call_config)
        self.assertEqual(call_order[2][4], "user-life")
        self.assertEqual(call_order[2][5], "conv-life")
        self.assertEqual(call_order[3][1], [{"role": "user", "content": "prepared"}])
        self.assertEqual(call_order[3][2], [initial_block])
        self.assertIs(call_order[4][1], execution.completion_context)

    async def test_completed_persist_superseded_emits_terminal_and_skips_error_fallback(self):
        events = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                events.append(payload)

        class SessionCache:
            def __init__(self):
                self.status = "running"

            async def write_session_status(self, **kwargs):
                self.status = kwargs["status"]

        session_cache = SessionCache()
        execution = self._execution(redis_writer=CaptureWriter())
        execution = replace(
            execution,
            completion_context=replace(
                execution.completion_context,
                session_cache=session_cache,
            ),
        )
        fallback_status = AsyncMock()
        finalize_stream = AsyncMock()
        complete_run = AsyncMock()
        finalized = []

        async def finalize(expected_last_sequence):
            finalized.append(expected_last_sequence)
            self.assertEqual([event["type"] for event in events], ["run_interrupted"])

        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)
        real_seal = execution.emitter.seal_and_get_last_sequence
        execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=real_seal)

        await run_agent_loop_lifecycle(
            request=self._request(),
            execution=execution,
            dependencies=self._dependencies(
                finalize_completed_run_fn=finalize_completed_run,
                write_fallback_run_error_fn=write_fallback_run_error,
                persist_message_fn=lambda *_args: False,
                complete_agent_run_fn=complete_run,
                interrupt_agent_run_fn=interrupt_agent_run,
                finalize_stream_fn=finalize_stream,
                write_fallback_error_status_fn=fallback_status,
            ),
        )

        self.assertEqual(session_cache.status, "interrupted")
        self.assertTrue(execution.state.terminal_emitted)
        self.assertEqual(finalized, [0])
        self.assertEqual([event["type"] for event in events], ["run_interrupted"])
        complete_run.assert_not_awaited()
        fallback_status.assert_not_awaited()
        execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
        execution.trajectory_recorder.finalize.assert_awaited_once_with(0)

    async def test_completed_persist_superseded_terminal_failure_keeps_business_status_and_no_double_terminal(self):
        for terminal_error in (
            StreamWriteTerminalError("required writer failed"),
            StreamOwnershipLostError("ownership lost"),
        ):
            with self.subTest(error_type=type(terminal_error).__name__):
                events = []

                class FailingTerminalWriter:
                    async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                        if payload["type"] == "run_interrupted":
                            raise terminal_error
                        events.append(payload)

                class SessionCache:
                    def __init__(self):
                        self.status = "running"

                    async def write_session_status(self, **kwargs):
                        self.status = kwargs["status"]

                session_cache = SessionCache()
                execution = self._execution(redis_writer=FailingTerminalWriter())
                execution = replace(
                    execution,
                    completion_context=replace(
                        execution.completion_context,
                        session_cache=session_cache,
                    ),
                )
                fallback_status = AsyncMock()
                finalize_cancelled = AsyncMock()
                finalize_failed = AsyncMock()
                execution.trajectory_recorder.finalize = AsyncMock()
                real_seal = execution.emitter.seal_and_get_last_sequence
                execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=real_seal)
                dependencies = self._dependencies(
                    finalize_completed_run_fn=finalize_completed_run,
                    finalize_cancelled_run_fn=finalize_cancelled,
                    finalize_failed_run_fn=finalize_failed,
                    write_fallback_run_error_fn=write_fallback_run_error,
                    persist_message_fn=lambda *_args: False,
                    complete_agent_run_fn=AsyncMock(),
                    interrupt_agent_run_fn=interrupt_agent_run,
                    finalize_stream_fn=AsyncMock(),
                    write_fallback_error_status_fn=fallback_status,
                )

                if isinstance(terminal_error, StreamOwnershipLostError):
                    await run_agent_loop_lifecycle(
                        request=self._request(),
                        execution=execution,
                        dependencies=dependencies,
                    )
                else:
                    with self.assertRaises(StreamWriteTerminalError) as raised:
                        await run_agent_loop_lifecycle(
                            request=self._request(),
                            execution=execution,
                            dependencies=dependencies,
                        )
                    self.assertIs(raised.exception, terminal_error)

                self.assertEqual(session_cache.status, "interrupted")
                self.assertFalse(execution.state.terminal_emitted)
                self.assertEqual(events, [])
                fallback_status.assert_not_awaited()
                finalize_cancelled.assert_not_awaited()
                finalize_failed.assert_not_awaited()
                execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
                execution.trajectory_recorder.finalize.assert_awaited_once_with(0)

    async def test_completed_persist_superseded_cancel_during_terminal_has_no_second_terminal(self):
        terminal_entered = asyncio.Event()
        terminal_calls = 0

        class BlockingTerminalWriter:
            async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                nonlocal terminal_calls
                if payload["type"] == "run_interrupted":
                    terminal_calls += 1
                    terminal_entered.set()
                    await asyncio.Event().wait()

        class SessionCache:
            def __init__(self):
                self.status = "running"

            async def write_session_status(self, **kwargs):
                self.status = kwargs["status"]

        session_cache = SessionCache()
        execution = self._execution(redis_writer=BlockingTerminalWriter())
        execution = replace(
            execution,
            completion_context=replace(
                execution.completion_context,
                session_cache=session_cache,
            ),
        )
        finalize_cancelled = AsyncMock()
        fallback_status = AsyncMock()
        execution.trajectory_recorder.finalize = AsyncMock()
        real_seal = execution.emitter.seal_and_get_last_sequence
        execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=real_seal)
        task = asyncio.create_task(
            run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    finalize_completed_run_fn=finalize_completed_run,
                    finalize_cancelled_run_fn=finalize_cancelled,
                    write_fallback_run_error_fn=write_fallback_run_error,
                    persist_message_fn=lambda *_args: False,
                    complete_agent_run_fn=AsyncMock(),
                    interrupt_agent_run_fn=interrupt_agent_run,
                    finalize_stream_fn=AsyncMock(),
                    write_fallback_error_status_fn=fallback_status,
                ),
            )
        )
        await asyncio.wait_for(terminal_entered.wait(), timeout=0.5)

        self.assertTrue(task.cancel())
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)

        self.assertEqual(session_cache.status, "interrupted")
        self.assertEqual(terminal_calls, 1)
        self.assertFalse(execution.state.terminal_emitted)
        fallback_status.assert_not_awaited()
        finalize_cancelled.assert_not_awaited()
        execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
        execution.trajectory_recorder.finalize.assert_awaited_once_with(0)

    async def test_completed_persist_superseded_real_ledger_terminal_matrix(self):
        scenarios = (
            ("accepted", None, None, True, "complete"),
            (
                "redis_terminal",
                StreamWriteTerminalError("required writer failed"),
                None,
                False,
                "degraded",
            ),
            (
                "ownership",
                StreamOwnershipLostError("ownership lost"),
                None,
                False,
                "degraded",
            ),
            ("trajectory_sink", None, "write_failed", True, "degraded"),
        )

        for scenario, writer_error, sink_reason, terminal_emitted, trajectory_status in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp_dir:
                database_path = Path(temp_dir) / "trajectory.sqlite3"
                engine = create_engine(
                    f"sqlite:///{database_path}",
                    connect_args={"check_same_thread": False},
                )
                session_factory = sessionmaker(bind=engine, expire_on_commit=False)
                Base.metadata.create_all(engine)
                with session_factory() as db:
                    db.add(
                        AgentSession(
                            id="run-life",
                            conversation_id="conv-life",
                            message_id="msg-life",
                            user_id="user-life",
                            model_id="gpt-4",
                            provider="openai",
                            status="running",
                        )
                    )
                    db.commit()

                sink_state = {"fail_next": False}

                def recorder_session_factory():
                    if sink_state["fail_next"]:
                        sink_state["fail_next"] = False
                        raise RuntimeError("trajectory terminal sink failed")
                    return session_factory()

                class RedisWriter:
                    def __init__(self):
                        self.events = []

                    async def append_chunk(self, _conversation_id, _task_id, _chunk_type, payload):
                        if payload["type"] == "run_interrupted":
                            if writer_error is not None:
                                raise writer_error
                            if sink_reason is not None:
                                sink_state["fail_next"] = True
                        self.events.append(payload)

                class SessionCache:
                    async def write_session_status(self, **kwargs):
                        with session_factory() as db:
                            session = db.get(AgentSession, kwargs["run_id"])
                            session.status = kwargs["status"]
                            session.total_steps = kwargs["total_steps"]
                            session.total_tool_calls = kwargs["total_tool_calls"]
                            session.total_duration_ms = kwargs["total_duration_ms"]
                            session.terminal_at = datetime.now(UTC)
                            db.commit()

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                try:
                    recorder = TrajectoryRecorder(
                        run_id="run-life",
                        conversation_id="conv-life",
                        message_id="msg-life",
                        session_factory=recorder_session_factory,
                        executor=executor,
                        semaphore=threading.BoundedSemaphore(4),
                    )
                    redis_writer = RedisWriter()
                    execution = self._execution(redis_writer=redis_writer)
                    execution.emitter._writer = AgentEventCompositeWriter(
                        redis_writer=redis_writer,
                        trajectory_recorder=recorder,
                    )
                    execution = replace(
                        execution,
                        trajectory_recorder=recorder,
                        completion_context=replace(
                            execution.completion_context,
                            session_cache=SessionCache(),
                            trajectory_recorder=recorder,
                        ),
                    )
                    real_seal = execution.emitter.seal_and_get_last_sequence
                    execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=real_seal)
                    real_finalize = recorder.finalize
                    recorder.finalize = AsyncMock(side_effect=real_finalize)
                    await execution.emitter.run_started(
                        message_id="msg-life",
                        model="gpt-4",
                        tools=[],
                        config={},
                    )
                    fallback_status = AsyncMock()
                    finalize_stream = AsyncMock()
                    dependencies = self._dependencies(
                        finalize_completed_run_fn=finalize_completed_run,
                        finalize_cancelled_run_fn=AsyncMock(),
                        finalize_failed_run_fn=AsyncMock(),
                        write_fallback_run_error_fn=write_fallback_run_error,
                        persist_message_fn=lambda *_args: False,
                        complete_agent_run_fn=AsyncMock(),
                        interrupt_agent_run_fn=interrupt_agent_run,
                        finalize_stream_fn=finalize_stream,
                        write_fallback_error_status_fn=fallback_status,
                    )

                    if isinstance(writer_error, StreamOwnershipLostError):
                        await run_agent_loop_lifecycle(
                            request=self._request(),
                            execution=execution,
                            dependencies=dependencies,
                        )
                    elif writer_error is not None:
                        with self.assertRaises(type(writer_error)) as raised:
                            await run_agent_loop_lifecycle(
                                request=self._request(),
                                execution=execution,
                                dependencies=dependencies,
                            )
                        self.assertIs(raised.exception, writer_error)
                    else:
                        await run_agent_loop_lifecycle(
                            request=self._request(),
                            execution=execution,
                            dependencies=dependencies,
                        )

                    with session_factory() as db:
                        session = db.get(AgentSession, "run-life")
                        ledger_events = list(
                            db.scalars(
                                select(AgentEvent).where(AgentEvent.run_id == "run-life").order_by(AgentEvent.sequence)
                            )
                        )
                        event_types = [event.event_type for event in ledger_events]
                        meta = db.get(RunTrajectoryMeta, "run-life")
                        self.assertEqual(session.status, "interrupted")
                        self.assertIsNotNone(session.terminal_at)
                        self.assertEqual(meta.trajectory_status, trajectory_status)
                        if trajectory_status == "degraded":
                            self.assertEqual(
                                meta.degraded_reason,
                                sink_reason or "finalize_mismatch",
                            )
                            self.assertIsNone(meta.finalized_at)
                        else:
                            self.assertIsNone(meta.degraded_reason)
                            self.assertIsNotNone(meta.finalized_at)

                    expected_events = ["run_started", "run_interrupted"] if scenario == "accepted" else ["run_started"]
                    self.assertEqual(event_types, expected_events)
                    if scenario == "accepted":
                        self.assertEqual(ledger_events[-1].payload["reason"], "superseded")
                    required_event_types = [event["type"] for event in redis_writer.events]
                    self.assertEqual(
                        required_event_types,
                        (["run_started", "run_interrupted"] if writer_error is None else ["run_started"]),
                    )
                    self.assertEqual(execution.state.terminal_emitted, terminal_emitted)
                    fallback_status.assert_not_awaited()
                    dependencies.finalize_cancelled_run_fn.assert_not_awaited()
                    dependencies.finalize_failed_run_fn.assert_not_awaited()
                    execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
                    recorder.finalize.assert_awaited_once_with(1)
                    if terminal_emitted:
                        finalize_stream.assert_awaited_once()
                    else:
                        finalize_stream.assert_not_awaited()
                finally:
                    executor.shutdown(wait=True)
                    engine.dispose()

    async def test_terminal_matrix_commits_trajectory_barrier_once_after_fallback(self):
        scenarios = (
            ("completed", AgentLoopOutcome(exit=AgentLoopExit.COMPLETED), None, None),
            ("limit_reached", AgentLoopOutcome(exit=AgentLoopExit.COMPLETED), "max_steps", None),
            ("superseded", AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED), None, None),
            ("interrupted", None, None, StreamOwnershipLostError("external stop")),
            ("failed", None, None, RuntimeError("LLM failed")),
            ("user_cancelled", None, None, asyncio.CancelledError()),
        )

        for scenario, outcome, limit_reason, raised_error in scenarios:
            with self.subTest(scenario=scenario):
                execution = self._execution()
                calls = []

                async def run_agent_loop_fn(**_kwargs):
                    if limit_reason is not None:
                        execution.state.limit_reason = limit_reason
                    if raised_error is not None:
                        raise raised_error
                    return outcome

                async def write_fallback_run_error_fn(**_kwargs):
                    calls.append("fallback")

                async def seal_and_get_last_sequence():
                    calls.append("seal")
                    return 7

                async def finalize(expected_last_sequence):
                    calls.append(("finalize", expected_last_sequence))

                execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=seal_and_get_last_sequence)
                execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)
                dependencies = self._dependencies(
                    run_agent_loop_fn=run_agent_loop_fn,
                    write_fallback_run_error_fn=write_fallback_run_error_fn,
                )

                if isinstance(raised_error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await run_agent_loop_lifecycle(
                            request=self._request(),
                            execution=execution,
                            dependencies=dependencies,
                        )
                elif raised_error is not None and not isinstance(
                    raised_error,
                    StreamOwnershipLostError,
                ):
                    with self.assertRaises(type(raised_error)):
                        await run_agent_loop_lifecycle(
                            request=self._request(),
                            execution=execution,
                            dependencies=dependencies,
                        )
                else:
                    await run_agent_loop_lifecycle(
                        request=self._request(),
                        execution=execution,
                        dependencies=dependencies,
                    )

                self.assertEqual(calls[-3:], ["fallback", "seal", ("finalize", 7)])
                execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
                execution.trajectory_recorder.finalize.assert_awaited_once_with(7)

    async def test_suggested_questions_pending_is_emitted_before_fallback_and_seal(self):
        execution = self._execution()
        calls = []

        async def finalize_completed_run_fn(**_kwargs):
            calls.append("run_completed")
            await execution.emitter.suggested_questions_pending(
                message_id="msg-life",
                revision=1,
            )

        async def pending_event(**_kwargs):
            calls.append("suggested_questions_pending")

        async def write_fallback_run_error_fn(**_kwargs):
            calls.append("fallback")

        async def seal_and_get_last_sequence():
            calls.append("seal")
            return 3

        async def finalize(_expected_last_sequence):
            calls.append("finalize")

        execution.emitter.suggested_questions_pending = AsyncMock(side_effect=pending_event)
        execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=seal_and_get_last_sequence)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)

        await run_agent_loop_lifecycle(
            request=self._request(),
            execution=execution,
            dependencies=self._dependencies(
                finalize_completed_run_fn=finalize_completed_run_fn,
                write_fallback_run_error_fn=write_fallback_run_error_fn,
            ),
        )

        self.assertEqual(
            calls,
            [
                "run_completed",
                "suggested_questions_pending",
                "fallback",
                "seal",
                "finalize",
            ],
        )

    async def test_start_before_session_failure_still_seals_and_attempts_finalize(self):
        execution = self._execution()
        calls = []

        async def append_chunk_fn(*_args, **_kwargs):
            raise RuntimeError("start unavailable")

        async def seal_and_get_last_sequence():
            calls.append("seal")
            return -1

        async def finalize(expected_last_sequence):
            calls.append(("finalize", expected_last_sequence))

        execution.emitter.seal_and_get_last_sequence = AsyncMock(side_effect=seal_and_get_last_sequence)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)

        with self.assertRaisesRegex(RuntimeError, "start unavailable"):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(append_chunk_fn=append_chunk_fn),
            )

        self.assertEqual(calls, ["seal", ("finalize", -1)])

    async def test_barrier_failure_preserves_original_business_exception(self):
        execution = self._execution()
        warnings = []
        primary_error = RuntimeError("primary failure")

        async def run_agent_loop_fn(**_kwargs):
            raise primary_error

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=2)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=ValueError("barrier failure"))

        with self.assertRaises(RuntimeError) as raised:
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    run_agent_loop_fn=run_agent_loop_fn,
                    warning_fn=warnings.append,
                ),
            )

        self.assertIs(raised.exception, primary_error)
        self.assertEqual(len(warnings), 1)
        self.assertIn("轨迹完整性提交屏障失败", warnings[0])

    async def test_shutdown_second_cancellation_cannot_interrupt_bounded_barrier(self):
        execution = self._execution()
        loop_entered = asyncio.Event()
        barrier_entered = asyncio.Event()
        release_barrier = asyncio.Event()
        barrier_finished = asyncio.Event()

        async def run_agent_loop_fn(**_kwargs):
            loop_entered.set()
            await asyncio.Event().wait()

        async def finalize(_expected_last_sequence):
            barrier_entered.set()
            await release_barrier.wait()
            barrier_finished.set()

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=4)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)

        task = asyncio.create_task(
            run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(run_agent_loop_fn=run_agent_loop_fn),
            )
        )
        await asyncio.wait_for(loop_entered.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(barrier_entered.wait(), timeout=0.5)

        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release_barrier.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)

        self.assertTrue(barrier_finished.is_set())
        execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
        execution.trajectory_recorder.finalize.assert_awaited_once_with(4)

    async def _assert_ownership_lost_barrier_cancellation(self, *, cancel_count: int):
        execution = self._execution()
        barrier_entered = asyncio.Event()
        release_barrier = asyncio.Event()
        barrier_finished = asyncio.Event()

        async def run_agent_loop_fn(**_kwargs):
            raise StreamOwnershipLostError("external stop")

        async def finalize(_expected_last_sequence):
            barrier_entered.set()
            await release_barrier.wait()
            barrier_finished.set()

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=8)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)
        task = asyncio.create_task(
            run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(run_agent_loop_fn=run_agent_loop_fn),
            )
        )
        await asyncio.wait_for(barrier_entered.wait(), timeout=0.5)

        for _ in range(cancel_count):
            self.assertTrue(task.cancel())
            await asyncio.sleep(0)
            self.assertFalse(task.done())

        release_barrier.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)

        self.assertTrue(task.cancelled())
        self.assertEqual(task.cancelling(), 0)
        self.assertTrue(barrier_finished.is_set())
        execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
        execution.trajectory_recorder.finalize.assert_awaited_once_with(8)

    async def test_ownership_lost_then_one_barrier_cancel_restores_cancelled_error(self):
        await self._assert_ownership_lost_barrier_cancellation(cancel_count=1)

    async def test_ownership_lost_then_two_barrier_cancels_restore_cancelled_error(self):
        await self._assert_ownership_lost_barrier_cancellation(cancel_count=2)

    async def test_business_error_stays_primary_when_cancel_arrives_during_barrier(self):
        execution = self._execution()
        primary_error = RuntimeError("primary failure")
        barrier_entered = asyncio.Event()
        release_barrier = asyncio.Event()

        async def run_agent_loop_fn(**_kwargs):
            raise primary_error

        async def finalize(_expected_last_sequence):
            barrier_entered.set()
            await release_barrier.wait()

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=9)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)
        task = asyncio.create_task(
            run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(run_agent_loop_fn=run_agent_loop_fn),
            )
        )
        await asyncio.wait_for(barrier_entered.wait(), timeout=0.5)

        self.assertTrue(task.cancel())
        await asyncio.sleep(0)
        release_barrier.set()
        with self.assertRaises(RuntimeError) as raised:
            await asyncio.wait_for(task, timeout=0.5)

        self.assertIs(raised.exception, primary_error)
        self.assertFalse(task.cancelled())
        self.assertEqual(task.cancelling(), 0)

    async def test_fallback_error_stays_primary_when_cancel_arrives_during_barrier(self):
        execution = self._execution()
        fallback_error = RuntimeError("fallback failure")
        barrier_entered = asyncio.Event()
        release_barrier = asyncio.Event()

        async def write_fallback_run_error_fn(**_kwargs):
            raise fallback_error

        async def finalize(_expected_last_sequence):
            barrier_entered.set()
            await release_barrier.wait()

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=10)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)
        task = asyncio.create_task(
            run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    write_fallback_run_error_fn=write_fallback_run_error_fn,
                ),
            )
        )
        await asyncio.wait_for(barrier_entered.wait(), timeout=0.5)

        self.assertTrue(task.cancel())
        await asyncio.sleep(0)
        release_barrier.set()
        with self.assertRaises(RuntimeError) as raised:
            await asyncio.wait_for(task, timeout=0.5)

        self.assertIs(raised.exception, fallback_error)
        self.assertFalse(task.cancelled())
        self.assertEqual(task.cancelling(), 0)

    async def test_repeated_barrier_entry_never_seals_or_finalizes_twice(self):
        execution = self._execution()
        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=5)
        execution.trajectory_recorder.finalize = AsyncMock()

        await commit_trajectory_barrier(
            execution=execution,
            warning_fn=lambda _message: None,
        )
        await commit_trajectory_barrier(
            execution=execution,
            warning_fn=lambda _message: None,
        )

        execution.emitter.seal_and_get_last_sequence.assert_awaited_once_with()
        execution.trajectory_recorder.finalize.assert_awaited_once_with(5)

    async def test_barrier_timeout_is_bounded_and_does_not_cancel_late_finalize(self):
        execution = self._execution()
        barrier_entered = asyncio.Event()
        release_barrier = asyncio.Event()
        barrier_finished = asyncio.Event()
        warnings = []

        async def finalize(_expected_last_sequence):
            barrier_entered.set()
            await release_barrier.wait()
            barrier_finished.set()

        execution.emitter.seal_and_get_last_sequence = AsyncMock(return_value=6)
        execution.trajectory_recorder.finalize = AsyncMock(side_effect=finalize)

        with patch(
            "app.services.stream.agent_loop_lifecycle.TRAJECTORY_BARRIER_TIMEOUT_SECONDS",
            0.01,
        ):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(warning_fn=warnings.append),
            )

        self.assertTrue(barrier_entered.is_set())
        self.assertFalse(barrier_finished.is_set())
        self.assertEqual(len(warnings), 1)
        self.assertIn("轨迹完整性提交屏障超时", warnings[0])

        release_barrier.set()
        await asyncio.wait_for(barrier_finished.wait(), timeout=0.5)
        execution.trajectory_recorder.finalize.assert_awaited_once_with(6)

    async def test_empty_knowledge_retrieval_completes_without_preparing_or_running_llm(self):
        call_config = SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
            announced_tools=[],
            task_mode="standard",
            network_profile="standard",
            evidence_policy="knowledge_grounded_v1",
            plan_mode="off",
        )
        execution = self._execution(call_config=call_config)
        evidence = KnowledgeEvidenceBlock(
            type="knowledge_evidence",
            query="未知问题",
            status="empty",
            source_count=0,
            knowledge_base_ids=["kb-1"],
            source_refs=[],
        )
        grounding = KnowledgeGroundingResult(
            evidence_block=evidence,
            context_messages=[],
            no_evidence=True,
            deterministic_answer="未在所选知识库中找到足够依据",
        )
        prepare_messages = AsyncMock(side_effect=AssertionError("无命中不应准备 LLM 上下文"))
        run_loop = AsyncMock(side_effect=AssertionError("无命中不应调用 LLM"))
        finalized = []

        async def finalize_completed_run_fn(**kwargs):
            finalized.append(kwargs)

        request = self._request(call_config=call_config)
        request = AgentLoopLifecycleRequest(
            raw_messages=request.raw_messages,
            has_vision=False,
            file_ids=None,
            original_message="未知问题",
            call_config=request.call_config,
            limits=request.limits,
            knowledge_base_ids=["kb-1"],
        )
        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(return_value=grounding),
        ):
            await run_agent_loop_lifecycle(
                request=request,
                execution=execution,
                dependencies=self._dependencies(
                    prepare_messages_fn=prepare_messages,
                    run_agent_loop_fn=run_loop,
                    finalize_completed_run_fn=finalize_completed_run_fn,
                    claim_suggested_questions_fn=lambda **_kwargs: object(),
                    generate_suggested_questions_fn=lambda **_kwargs: None,
                ),
            )

        prepare_messages.assert_not_awaited()
        run_loop.assert_not_awaited()
        self.assertEqual([block.type for block in execution.state.content_blocks], ["knowledge_evidence", "text"])
        self.assertIsNone(finalized[0]["claim_suggested_questions_fn"])
        self.assertIsNone(finalized[0]["generate_suggested_questions_fn"])

    async def test_successful_knowledge_run_also_disables_ungrounded_suggestions(self):
        call_config = SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
            announced_tools=[],
            task_mode="standard",
            network_profile="standard",
            evidence_policy="knowledge_grounded_v1",
            plan_mode="off",
        )
        execution = self._execution(call_config=call_config)
        evidence = KnowledgeEvidenceBlock(
            type="knowledge_evidence",
            query="怎么发布",
            status="success",
            source_count=1,
            knowledge_base_ids=["kb-1"],
            source_refs=[
                KnowledgeSourceReference(
                    evidence_id="ev-knowledge-1",
                    citation_index=1,
                    knowledge_base_id="kb-1",
                    knowledge_base_name="产品手册",
                    document_id="doc-1",
                    index_version="version-1",
                    chunk_id="chunk-1",
                    ordinal=1,
                    filename="manual.md",
                    char_start=0,
                    char_end=10,
                )
            ],
        )
        grounding = KnowledgeGroundingResult(
            evidence_block=evidence,
            context_messages=[{"role": "user", "content": "不可信知识上下文"}],
            no_evidence=False,
        )
        finalized = []

        async def finalize_completed_run_fn(**kwargs):
            finalized.append(kwargs)

        request = AgentLoopLifecycleRequest(
            raw_messages=[{"role": "user", "content": "怎么发布"}],
            has_vision=False,
            file_ids=None,
            original_message="怎么发布",
            call_config=call_config,
            limits=self._limits(),
            knowledge_base_ids=["kb-1"],
        )
        with patch(
            "app.services.stream.agent_loop_lifecycle.prepare_knowledge_grounding",
            new=AsyncMock(return_value=grounding),
        ):
            await run_agent_loop_lifecycle(
                request=request,
                execution=execution,
                dependencies=self._dependencies(
                    finalize_completed_run_fn=finalize_completed_run_fn,
                    claim_suggested_questions_fn=lambda **_kwargs: object(),
                    generate_suggested_questions_fn=lambda **_kwargs: None,
                ),
            )

        self.assertIsNone(finalized[0]["claim_suggested_questions_fn"])
        self.assertIsNone(finalized[0]["generate_suggested_questions_fn"])

    async def test_start_run_records_runtime_config_versions(self):
        configs = []

        async def start_agent_run_fn(**kwargs):
            configs.append(kwargs["config"])

        with patch(
            "app.services.stream.agent_loop_lifecycle.get_agent_strategy_config",
            return_value=({"search": {}}, {"source": "db", "version": "agent-strategy-v7"}),
            create=True,
        ):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=self._execution(),
                dependencies=self._dependencies(start_agent_run_fn=start_agent_run_fn),
            )

        self.assertEqual(
            configs[0],
            {
                "max_steps": 3,
                "max_tool_calls": 5,
                "timeout_s": 30,
                "plan_mode": "auto",
                "task_mode": "standard",
                "network_profile": "standard",
                "evidence_policy": "standard",
                "runtime_config_versions": {
                    "agent_strategy/default": "agent-strategy-v7",
                },
            },
        )

    async def test_start_run_records_safe_mcp_tool_binding_snapshot(self):
        configs = []
        call_config = SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
            announced_tools=["mcp_docs_a1b2c3d4"],
            tool_bindings=[
                {
                    "alias": "mcp_docs_a1b2c3d4",
                    "server_id": "server-1",
                    "remote_tool_name": "microsoft_docs_search",
                    "provider": "microsoft",
                    "config_version": 7,
                    "tool_label": "Microsoft Learn 文档搜索",
                    "definition_sha256": "abc123",
                    "endpoint_url": "https://secret.invalid/mcp",
                    "credential_ref": "MCP_SECRET",
                }
            ],
        )

        async def start_agent_run_fn(**kwargs):
            configs.append(kwargs["config"])

        await run_agent_loop_lifecycle(
            request=self._request(call_config=call_config),
            execution=self._execution(call_config=call_config),
            dependencies=self._dependencies(start_agent_run_fn=start_agent_run_fn),
        )

        self.assertEqual(
            configs[0]["mcp_tool_bindings"],
            [
                {
                    "alias": "mcp_docs_a1b2c3d4",
                    "server_id": "server-1",
                    "remote_tool_name": "microsoft_docs_search",
                    "provider": "microsoft",
                    "config_version": 7,
                    "tool_label": "Microsoft Learn 文档搜索",
                    "definition_sha256": "abc123",
                }
            ],
        )
        self.assertNotIn("endpoint_url", str(configs[0]))
        self.assertNotIn("credential_ref", str(configs[0]))

    async def test_start_run_records_active_prompt_bundle_revision(self):
        configs = []

        async def start_agent_run_fn(**kwargs):
            configs.append(kwargs["config"])

        with patch(
            "app.services.stream.agent_loop_lifecycle.get_active_prompt_bundle_revision",
            return_value="b" * 64,
        ):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=self._execution(),
                dependencies=self._dependencies(start_agent_run_fn=start_agent_run_fn),
            )

        self.assertEqual(
            configs[0]["runtime_config_versions"]["prompt_bundle/fusion"],
            "b" * 64,
        )

    async def test_start_run_does_not_emit_plan_before_tools_are_called(self):
        emitted = []

        class CaptureWriter:
            async def append_chunk(self, _conversation_id, _task_id, chunk_type, payload):
                if chunk_type == "agent_event":
                    emitted.append(payload)

        execution = self._execution(redis_writer=CaptureWriter())
        call_config = self._call_config()
        limits = self._limits()

        async def start_agent_run_fn(**kwargs):
            await kwargs["emitter"].run_started(
                message_id=kwargs["message_id"],
                model=kwargs["model_id"],
                tools=kwargs["tools"],
                config=kwargs["config"],
            )

        request = self._request(call_config=call_config, limits=limits)
        request = AgentLoopLifecycleRequest(
            raw_messages=request.raw_messages,
            has_vision=request.has_vision,
            file_ids=request.file_ids,
            original_message="你好啊，你是谁",
            call_config=request.call_config,
            limits=request.limits,
            initial_content_blocks=request.initial_content_blocks,
            extra_system_prompts=request.extra_system_prompts,
            preprocess_user_input=request.preprocess_user_input,
        )

        await run_agent_loop_lifecycle(
            request=request,
            execution=execution,
            dependencies=self._dependencies(start_agent_run_fn=start_agent_run_fn),
        )

        self.assertEqual([event["type"] for event in emitted], ["run_started"])
        self.assertFalse(hasattr(execution.state, "plan_items"))

    async def test_lifecycle_passes_continuation_inputs_and_preserves_existing_blocks_first(self):
        call_order = []
        execution = self._execution()
        existing_block = TextBlock(type="text", id="txt-existing", text="旧回答")
        prepared_block = TextBlock(type="text", id="txt-url", text="URL 摘要")

        async def prepare_messages_fn(**kwargs):
            call_order.append(
                (
                    "prepare",
                    kwargs["extra_system_prompts"],
                    kwargs["preprocess_user_input"],
                )
            )
            return AgentLoopPreparedMessages(
                messages=[{"role": "user", "content": "prepared"}],
                initial_content_blocks=[prepared_block],
            )

        async def run_agent_loop_fn(**kwargs):
            call_order.append(("run", list(kwargs["state"].content_blocks)))
            return AgentLoopOutcome(exit=AgentLoopExit.COMPLETED)

        request = self._request()
        request = AgentLoopLifecycleRequest(
            raw_messages=request.raw_messages,
            has_vision=request.has_vision,
            file_ids=request.file_ids,
            original_message=request.original_message,
            call_config=request.call_config,
            limits=request.limits,
            initial_content_blocks=[existing_block],
            extra_system_prompts=["继续执行，不要重写前文"],
            preprocess_user_input=False,
        )

        await run_agent_loop_lifecycle(
            request=request,
            execution=execution,
            dependencies=self._dependencies(
                prepare_messages_fn=prepare_messages_fn,
                run_agent_loop_fn=run_agent_loop_fn,
            ),
        )

        self.assertEqual(call_order[0], ("prepare", ["继续执行，不要重写前文"], False))
        self.assertEqual(call_order[1], ("run", [existing_block, prepared_block]))

    async def test_superseded_path_finalizes_superseded_without_completed_finalize(self):
        call_order = []
        execution = self._execution()

        async def run_agent_loop_fn(**_kwargs):
            return AgentLoopOutcome(exit=AgentLoopExit.SUPERSEDED, error_msg="被新请求取代")

        async def finalize_superseded_run_fn(**kwargs):
            call_order.append(("superseded", kwargs["context"], kwargs["error_msg"]))

        async def finalize_completed_run_fn(**_kwargs):
            raise AssertionError("superseded 路径不应 completed finalize")

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        await run_agent_loop_lifecycle(
            request=self._request(),
            execution=execution,
            dependencies=self._dependencies(
                run_agent_loop_fn=run_agent_loop_fn,
                finalize_superseded_run_fn=finalize_superseded_run_fn,
                finalize_completed_run_fn=finalize_completed_run_fn,
                write_fallback_run_error_fn=write_fallback_run_error_fn,
            ),
        )

        self.assertEqual(
            call_order,
            [
                ("superseded", execution.completion_context, "被新请求取代"),
                ("fallback", execution.completion_context),
            ],
        )

    async def test_prepare_failure_finalizes_failed_then_reraises_and_writes_fallback(self):
        call_order = []
        execution = self._execution()

        async def append_chunk_fn(*_args, **_kwargs):
            call_order.append("append")

        async def start_agent_run_fn(**_kwargs):
            call_order.append("start")

        async def prepare_messages_fn(**_kwargs):
            call_order.append("prepare")
            raise ValueError("prepare boom")

        async def run_agent_loop_fn(**_kwargs):
            raise AssertionError("prepare 失败后不应进入 agent loop")

        async def finalize_failed_run_fn(**kwargs):
            call_order.append(("failed", kwargs["context"], str(kwargs["error"])))

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        with self.assertRaises(ValueError):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    append_chunk_fn=append_chunk_fn,
                    start_agent_run_fn=start_agent_run_fn,
                    prepare_messages_fn=prepare_messages_fn,
                    run_agent_loop_fn=run_agent_loop_fn,
                    finalize_failed_run_fn=finalize_failed_run_fn,
                    write_fallback_run_error_fn=write_fallback_run_error_fn,
                ),
            )

        self.assertEqual(
            call_order,
            [
                "append",
                "start",
                "prepare",
                ("failed", execution.completion_context, "prepare boom"),
                ("fallback", execution.completion_context),
            ],
        )

    async def test_cancelled_path_finalizes_then_reraises_and_writes_fallback(self):
        call_order = []
        execution = self._execution()

        async def run_agent_loop_fn(**_kwargs):
            raise asyncio.CancelledError()

        async def finalize_cancelled_run_fn(**kwargs):
            call_order.append(("cancelled", kwargs["context"]))

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        with self.assertRaises(asyncio.CancelledError):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    run_agent_loop_fn=run_agent_loop_fn,
                    finalize_cancelled_run_fn=finalize_cancelled_run_fn,
                    write_fallback_run_error_fn=write_fallback_run_error_fn,
                ),
            )

        self.assertEqual(
            call_order,
            [
                ("cancelled", execution.completion_context),
                ("fallback", execution.completion_context),
            ],
        )

    async def test_ownership_lost_finalizes_as_cancelled_without_reporting_failure(self):
        call_order = []
        execution = self._execution()

        async def run_agent_loop_fn(**_kwargs):
            raise StreamOwnershipLostError("external stop")

        async def finalize_cancelled_run_fn(**kwargs):
            call_order.append(("cancelled", kwargs["context"]))

        async def finalize_failed_run_fn(**_kwargs):
            raise AssertionError("外部终态已接管时不应进入失败收尾")

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        errors = []
        await run_agent_loop_lifecycle(
            request=self._request(),
            execution=execution,
            dependencies=self._dependencies(
                run_agent_loop_fn=run_agent_loop_fn,
                finalize_cancelled_run_fn=finalize_cancelled_run_fn,
                finalize_failed_run_fn=finalize_failed_run_fn,
                write_fallback_run_error_fn=write_fallback_run_error_fn,
                error_fn=errors.append,
            ),
        )

        self.assertEqual(
            call_order,
            [
                ("cancelled", execution.completion_context),
                ("fallback", execution.completion_context),
            ],
        )
        self.assertEqual(errors, [])

    async def test_failed_path_finalizes_then_reraises_and_writes_fallback(self):
        call_order = []
        execution = self._execution()

        async def run_agent_loop_fn(**_kwargs):
            raise RuntimeError("LLM 5xx")

        async def finalize_failed_run_fn(**kwargs):
            call_order.append(("failed", kwargs["context"], str(kwargs["error"])))

        async def write_fallback_run_error_fn(**kwargs):
            call_order.append(("fallback", kwargs["context"]))

        with self.assertRaises(RuntimeError):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    run_agent_loop_fn=run_agent_loop_fn,
                    finalize_failed_run_fn=finalize_failed_run_fn,
                    write_fallback_run_error_fn=write_fallback_run_error_fn,
                ),
            )

        self.assertEqual(
            call_order,
            [
                ("failed", execution.completion_context, "LLM 5xx"),
                ("fallback", execution.completion_context),
            ],
        )

    async def test_failed_path_log_never_contains_exception_message(self):
        secret = "sk-secret-value"
        errors = []
        execution = self._execution()

        async def run_agent_loop_fn(**_kwargs):
            raise RuntimeError(f"Authorization: Bearer {secret}; upstream 503")

        with self.assertRaises(RuntimeError):
            await run_agent_loop_lifecycle(
                request=self._request(),
                execution=execution,
                dependencies=self._dependencies(
                    run_agent_loop_fn=run_agent_loop_fn,
                    error_fn=errors.append,
                ),
            )

        self.assertEqual(
            errors,
            ["Agent 生成异常: conv_id=conv-life, error_type=RuntimeError"],
        )
        self.assertNotIn(secret, " ".join(errors))


if __name__ == "__main__":
    unittest.main()
