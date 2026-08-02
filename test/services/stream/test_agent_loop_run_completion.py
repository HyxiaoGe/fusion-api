import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.chat import ContextUsage, TextBlock, Usage
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.stream.agent_loop_run_completion import (
    AgentLoopRunCompletionContext,
    finalize_cancelled_run,
    finalize_completed_run,
    finalize_failed_run,
    finalize_superseded_run,
    write_fallback_run_error,
)
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.itinerary_observability import ItineraryToolObservation
from app.services.stream.limit_summary import SUMMARY_PROTOCOL_FALLBACK_TEXT


def _context(state: AgentLoopState | None = None) -> AgentLoopRunCompletionContext:
    return AgentLoopRunCompletionContext(
        db="db",
        conversation_id="conv-1",
        task_id="task-1",
        run_id="run-1",
        model_id="gpt-4",
        provider="openai",
        assistant_message_id="msg-1",
        emitter="emitter",
        session_cache="session-cache",
        state=state or AgentLoopState(),
        duration_ms_factory=lambda: 1234,
    )


class AgentLoopRunCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_run_terminal_emits_closed_model_plan_before_run_terminal_event(self):
        for terminal_kind in ("completed", "superseded", "cancelled", "failed"):
            with self.subTest(terminal_kind=terminal_kind):
                state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id=f"run-{terminal_kind}", mode="on"))
                self.assertTrue(
                    state.plan_coordinator.apply_model_update(
                        {
                            "reason": "先搜索再回答",
                            "items": [
                                {
                                    "id": "search",
                                    "title": "搜索资料",
                                    "status": "running",
                                    "kind": "search",
                                    "depends_on": [],
                                    "planned_tools": ["web_search"],
                                },
                                {
                                    "id": "answer",
                                    "title": "整理回答",
                                    "status": "pending",
                                    "kind": "answer",
                                    "depends_on": ["search"],
                                    "planned_tools": [],
                                },
                            ],
                        }
                    ).accepted
                )
                events: list[tuple[str, dict]] = []
                emitter = AsyncMock()

                async def record_plan(**snapshot):
                    events.append(("plan_snapshot", snapshot))

                async def record_run_terminal(**payload):
                    events.append(("run_terminal", payload))

                emitter.plan_snapshot.side_effect = record_plan
                run_terminal = AsyncMock(side_effect=record_run_terminal)
                context = replace(_context(state), emitter=emitter)
                common = {
                    "context": context,
                    "persist_message_fn": lambda *_args: None,
                    "finalize_stream_fn": AsyncMock(),
                }

                if terminal_kind == "completed":
                    await finalize_completed_run(
                        **common,
                        terminal_state=SimpleNamespace(
                            session_status="completed",
                            run_finish_reason="stop",
                        ),
                        complete_agent_run_fn=run_terminal,
                    )
                elif terminal_kind == "superseded":
                    await finalize_superseded_run(
                        **common,
                        error_msg=None,
                        interrupt_agent_run_fn=run_terminal,
                    )
                elif terminal_kind == "cancelled":
                    await finalize_cancelled_run(
                        **common,
                        interrupt_agent_run_fn=run_terminal,
                        warning_fn=lambda _message: None,
                    )
                else:
                    await finalize_failed_run(
                        **common,
                        error=RuntimeError("upstream failed"),
                        fail_agent_run_fn=run_terminal,
                        warning_fn=lambda _message: None,
                    )

                self.assertEqual([event[0] for event in events], ["plan_snapshot", "run_terminal"])
                self.assertNotIn(
                    "running",
                    [item["status"] for item in events[0][1]["items"]],
                )

    async def test_finalize_cancelled_continues_session_interrupt_after_terminal_plan_ownership_lost(self):
        from app.services.stream_state_service import StreamOwnershipLostError

        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-1", mode="on"))
        state.plan_coordinator.apply_model_update(
            {
                "reason": "执行研究",
                "items": [
                    {
                        "id": "research",
                        "title": "研究",
                        "status": "running",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "title": "回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["research"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        emitter = AsyncMock()
        emitter.plan_snapshot.side_effect = StreamOwnershipLostError("external stop")
        interrupt_agent_run_fn = AsyncMock()
        finalize_stream_fn = AsyncMock()
        warnings = []

        await finalize_cancelled_run(
            context=replace(_context(state), emitter=emitter),
            persist_message_fn=lambda *_args: None,
            interrupt_agent_run_fn=interrupt_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=warnings.append,
        )

        interrupt_agent_run_fn.assert_awaited_once()
        finalize_stream_fn.assert_awaited_once_with(
            "conv-1",
            success=False,
            error_msg="用户中止",
            task_id="task-1",
            error_code="stream_interrupted",
            error_data={"reason": "user_cancelled"},
        )
        self.assertTrue(state.terminal_emitted)
        self.assertIn("terminal plan ownership lost", warnings[0])

    async def test_finalize_cancelled_terminal_plan_ownership_lost_still_raises_status_write_failure(self):
        from app.services.stream.run_finalizer import InterruptedStatusWriteError
        from app.services.stream_state_service import StreamOwnershipLostError

        state = AgentLoopState(plan_coordinator=PlanCoordinator(run_id="run-1", mode="on"))
        state.plan_coordinator.apply_model_update(
            {
                "reason": "执行研究",
                "items": [
                    {
                        "id": "research",
                        "title": "研究",
                        "status": "running",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "title": "回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["research"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        emitter = AsyncMock()
        emitter.plan_snapshot.side_effect = StreamOwnershipLostError("external stop")
        interrupt_agent_run_fn = AsyncMock(side_effect=InterruptedStatusWriteError("status write failed"))

        with self.assertRaises(InterruptedStatusWriteError):
            await finalize_cancelled_run(
                context=replace(_context(state), emitter=emitter),
                persist_message_fn=lambda *_args: None,
                interrupt_agent_run_fn=interrupt_agent_run_fn,
                finalize_stream_fn=AsyncMock(),
                warning_fn=lambda _message: None,
            )

        interrupt_agent_run_fn.assert_awaited_once()

    async def test_failed_run_persists_context_even_without_visible_content(self):
        state = AgentLoopState()
        state.update_context(
            ContextUsage(
                status="required_context_over_budget",
                round_index=1,
                window_tokens=100,
                estimated_tokens_before=120,
                estimated_tokens_after=120,
            )
        )
        calls = []

        async def fail_agent_run_fn(**_kwargs):
            return None

        async def finalize_stream_fn(*_args, **_kwargs):
            return None

        await finalize_failed_run(
            context=_context(state),
            error=ValueError("context failed"),
            persist_message_fn=lambda *args: calls.append(args),
            fail_agent_run_fn=fail_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=lambda _message: None,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][5].context.status, "required_context_over_budget")
        self.assertTrue(calls[0][6])

    async def test_finalize_cancelled_treats_ownership_lost_as_external_stop_terminal(self):
        from app.services.stream_state_service import StreamOwnershipLostError

        state = AgentLoopState()
        ctx = _context(state)
        finalized = []
        warnings = []

        async def interrupt_agent_run_fn(**_kwargs):
            raise StreamOwnershipLostError("ownership lost")

        async def finalize_stream_fn(*args, **kwargs):
            finalized.append((args, kwargs))
            return False

        await finalize_cancelled_run(
            context=ctx,
            persist_message_fn=lambda *_args: None,
            interrupt_agent_run_fn=interrupt_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=warnings.append,
        )

        self.assertTrue(state.terminal_emitted)
        self.assertEqual(
            finalized,
            [
                (
                    ("conv-1",),
                    {
                        "success": False,
                        "error_msg": "用户中止",
                        "task_id": "task-1",
                        "error_code": "stream_interrupted",
                        "error_data": {"reason": "user_cancelled"},
                    },
                )
            ],
        )
        self.assertIn("外部 stop 已接管流终态", warnings[0])

    async def test_finalize_cancelled_still_raises_other_stream_terminal_errors(self):
        from app.services.stream_state_service import StreamWriteUnavailableError

        finalized = []

        async def interrupt_agent_run_fn(**_kwargs):
            raise StreamWriteUnavailableError("redis down")

        async def finalize_stream_fn(*_args, **_kwargs):
            finalized.append(True)

        with self.assertRaises(StreamWriteUnavailableError):
            await finalize_cancelled_run(
                context=_context(),
                persist_message_fn=lambda *_args: None,
                interrupt_agent_run_fn=interrupt_agent_run_fn,
                finalize_stream_fn=finalize_stream_fn,
                warning_fn=lambda _message: None,
            )

        self.assertEqual(finalized, [])

    async def test_finalize_cancelled_does_not_swallow_interrupted_status_write_error(self):
        from app.services.stream.run_finalizer import InterruptedStatusWriteError

        finalized = []

        async def interrupt_agent_run_fn(**_kwargs):
            raise InterruptedStatusWriteError("status write failed")

        async def finalize_stream_fn(*_args, **_kwargs):
            finalized.append(True)

        with self.assertRaises(InterruptedStatusWriteError):
            await finalize_cancelled_run(
                context=_context(),
                persist_message_fn=lambda *_args: None,
                interrupt_agent_run_fn=interrupt_agent_run_fn,
                finalize_stream_fn=finalize_stream_fn,
                warning_fn=lambda _message: None,
            )

        self.assertEqual(finalized, [])

    async def test_finalize_failed_does_not_swallow_stream_ownership_lost(self):
        from app.services.stream_state_service import StreamOwnershipLostError

        finalized = []

        async def fail_agent_run_fn(**_kwargs):
            raise StreamOwnershipLostError("ownership lost")

        async def finalize_stream_fn(*_args, **_kwargs):
            finalized.append(True)

        with self.assertRaises(StreamOwnershipLostError):
            await finalize_failed_run(
                context=_context(),
                error=ValueError("LLM failed"),
                persist_message_fn=lambda *_args: None,
                fail_agent_run_fn=fail_agent_run_fn,
                finalize_stream_fn=finalize_stream_fn,
                warning_fn=lambda _message: None,
            )

        self.assertEqual(finalized, [])

    async def test_finalize_completed_persists_then_completes_then_finalizes_success(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="回答"))
        state.update_usage(Usage(input_tokens=3, output_tokens=5))
        ctx = _context(state)
        calls = []

        def persist_message_fn(*args):
            calls.append(("persist", args))

        async def complete_agent_run_fn(**kwargs):
            calls.append(("complete", kwargs))

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        await finalize_completed_run(
            context=ctx,
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=persist_message_fn,
            complete_agent_run_fn=complete_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
        )

        self.assertEqual([call[0] for call in calls], ["persist", "complete", "finalize"])
        self.assertEqual(calls[0][1][:4], ("db", "msg-1", "conv-1", "gpt-4"))
        self.assertEqual(calls[0][1][5], Usage(input_tokens=3, output_tokens=5))
        self.assertFalse(calls[0][1][6])
        self.assertEqual(calls[1][1]["session_status"], "completed")
        self.assertEqual(calls[1][1]["finish_reason"], "stop")
        self.assertEqual(calls[2], ("finalize", ("conv-1",), {"success": True, "task_id": "task-1"}))
        self.assertTrue(state.terminal_emitted)

    async def test_finalize_completed_claims_before_stream_finalization_and_generates_after(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="正式回答"))
        calls = []
        claim = SimpleNamespace(message_id="msg-1", revision=1)

        def claim_suggested_questions_fn(**kwargs):
            calls.append(("claim", kwargs))
            return claim

        def generate_suggested_questions_fn(**kwargs):
            calls.append(("dispatch", kwargs))

        async def complete_agent_run_fn(**_kwargs):
            calls.append(("complete", None))

        async def finalize_stream_fn(*_args, **_kwargs):
            calls.append(("finalize", None))

        emitter = AsyncMock()

        async def emit_suggested_questions_pending(**kwargs):
            calls.append(("pending_event", kwargs))

        emitter.suggested_questions_pending.side_effect = emit_suggested_questions_pending

        await finalize_completed_run(
            context=replace(_context(state), emitter=emitter),
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=lambda *_args: calls.append(("persist", None)),
            complete_agent_run_fn=complete_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            claim_suggested_questions_fn=claim_suggested_questions_fn,
            generate_suggested_questions_fn=generate_suggested_questions_fn,
            warning_fn=lambda message: calls.append(("warning", message)),
        )

        self.assertEqual(
            [name for name, _payload in calls],
            ["persist", "complete", "claim", "pending_event", "finalize", "dispatch"],
        )
        self.assertEqual(calls[2][1]["assistant_message_id"], "msg-1")
        self.assertEqual(calls[2][1]["run_id"], "run-1")
        self.assertEqual(
            calls[3][1],
            {"message_id": "msg-1", "revision": 1},
        )
        self.assertIs(calls[5][1]["claim"], claim)
        self.assertNotIn("db", calls[5][1])

    async def test_suggestion_claim_failure_does_not_block_completed_stream_terminal(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="正式回答"))
        calls = []
        emitter = AsyncMock()

        def claim_suggested_questions_fn(**_kwargs):
            calls.append("claim")
            raise RuntimeError("claim failed")

        async def finalize_stream_fn(*_args, **_kwargs):
            calls.append("finalize")

        await finalize_completed_run(
            context=replace(_context(state), emitter=emitter),
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=AsyncMock(),
            finalize_stream_fn=finalize_stream_fn,
            claim_suggested_questions_fn=claim_suggested_questions_fn,
            generate_suggested_questions_fn=lambda **_kwargs: calls.append("dispatch"),
            warning_fn=lambda _message: calls.append("warning"),
        )

        self.assertEqual(calls, ["claim", "warning", "finalize"])
        emitter.suggested_questions_pending.assert_not_awaited()

    async def test_suggestion_generation_failure_never_changes_completed_stream_terminal(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="正式回答"))
        calls = []

        async def finalize_stream_fn(*_args, **_kwargs):
            calls.append("finalize")

        def generate_suggested_questions_fn(**_kwargs):
            calls.append("dispatch")
            raise RuntimeError("suggestion upstream failed")

        await finalize_completed_run(
            context=_context(state),
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=AsyncMock(),
            finalize_stream_fn=finalize_stream_fn,
            claim_suggested_questions_fn=lambda **_kwargs: SimpleNamespace(revision=1),
            generate_suggested_questions_fn=generate_suggested_questions_fn,
            warning_fn=lambda _message: calls.append("warning"),
        )

        self.assertEqual(calls, ["finalize", "dispatch", "warning"])

    async def test_stream_finalize_failure_marks_claim_failed_and_never_dispatches(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="正式回答"))
        claim = SimpleNamespace(revision=1)
        calls = []

        async def finalize_stream_fn(*_args, **_kwargs):
            calls.append("finalize")
            raise RuntimeError("redis finalize failed")

        with self.assertRaises(RuntimeError):
            await finalize_completed_run(
                context=_context(state),
                terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
                persist_message_fn=lambda *_args: None,
                complete_agent_run_fn=AsyncMock(),
                finalize_stream_fn=finalize_stream_fn,
                claim_suggested_questions_fn=lambda **_kwargs: claim,
                generate_suggested_questions_fn=lambda **_kwargs: calls.append("dispatch"),
                fail_suggested_questions_fn=lambda **kwargs: calls.append(("failed", kwargs["claim"])),
            )

        self.assertEqual(calls, ["finalize", ("failed", claim)])

    async def test_finalize_completed_does_not_complete_answer_plan_without_nonempty_text(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="   "))
        state.plan_coordinator = PlanCoordinator(run_id="run-1", mode="on")
        state.plan_coordinator.apply_model_update(
            {
                "reason": "先分析再回答",
                "items": [
                    {
                        "id": "reasoning",
                        "title": "分析",
                        "status": "running",
                        "kind": "reasoning",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "answer",
                        "title": "回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["reasoning"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        emitter = AsyncMock()

        async def complete_agent_run_fn(**_kwargs):
            return None

        async def finalize_stream_fn(*_args, **_kwargs):
            return None

        await finalize_completed_run(
            context=replace(_context(state), emitter=emitter),
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=complete_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
        )

        snapshot = emitter.plan_snapshot.await_args.kwargs
        self.assertEqual([item["status"] for item in snapshot["items"]], ["blocked", "blocked"])

    async def test_incomplete_plan_synthesis_never_completes_answer_from_fallback_text(self):
        for case, content_blocks in (
            (
                "safe_fallback",
                [TextBlock(type="text", id="txt-fallback", text=SUMMARY_PROTOCOL_FALLBACK_TEXT)],
            ),
            ("empty_answer", []),
            (
                "timeout_fallback",
                [TextBlock(type="text", id="txt-timeout", text=SUMMARY_PROTOCOL_FALLBACK_TEXT)],
            ),
        ):
            with self.subTest(case=case):
                state = AgentLoopState(
                    plan_coordinator=PlanCoordinator(run_id=f"run-{case}", mode="on"),
                    content_blocks=list(content_blocks),
                )
                self.assertTrue(
                    state.plan_coordinator.apply_model_update(
                        {
                            "reason": "先搜索再综合",
                            "items": [
                                {
                                    "id": "search",
                                    "title": "搜索资料",
                                    "status": "pending",
                                    "kind": "search",
                                    "depends_on": [],
                                    "planned_tools": ["web_search"],
                                },
                                {
                                    "id": "answer",
                                    "title": "整理回答",
                                    "status": "pending",
                                    "kind": "answer",
                                    "depends_on": ["search"],
                                    "planned_tools": [],
                                },
                            ],
                        }
                    ).accepted
                )
                state.plan_coordinator.mark_tools_started(["search"])
                state.plan_coordinator.mark_tool_results({"search": "completed"})
                state.plan_coordinator.begin_synthesis()
                state.mark_unknown_terminated()
                emitter = AsyncMock()

                await finalize_completed_run(
                    context=replace(_context(state), emitter=emitter),
                    terminal_state=SimpleNamespace(
                        session_status="incomplete",
                        run_finish_reason="incomplete",
                    ),
                    persist_message_fn=lambda *_args: None,
                    complete_agent_run_fn=AsyncMock(),
                    finalize_stream_fn=AsyncMock(),
                )

                snapshot = emitter.plan_snapshot.await_args.kwargs
                self.assertEqual(
                    {item["id"]: item["status"] for item in snapshot["items"]},
                    {"search": "completed", "answer": "blocked"},
                )

    async def test_valid_plan_synthesis_still_completes_answer(self):
        state = AgentLoopState(
            plan_coordinator=PlanCoordinator(run_id="run-valid-synthesis", mode="on"),
            content_blocks=[TextBlock(type="text", id="txt-valid", text="有效综合回答")],
        )
        self.assertTrue(
            state.plan_coordinator.apply_model_update(
                {
                    "reason": "先搜索再综合",
                    "items": [
                        {
                            "id": "search",
                            "title": "搜索资料",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理回答",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        state.plan_coordinator.mark_tools_started(["search"])
        state.plan_coordinator.mark_tool_results({"search": "completed"})
        state.plan_coordinator.begin_synthesis()
        emitter = AsyncMock()

        await finalize_completed_run(
            context=replace(_context(state), emitter=emitter),
            terminal_state=SimpleNamespace(
                session_status="completed",
                run_finish_reason="stop",
            ),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=AsyncMock(),
            finalize_stream_fn=AsyncMock(),
        )

        snapshot = emitter.plan_snapshot.await_args.kwargs
        self.assertEqual(
            {item["id"]: item["status"] for item in snapshot["items"]},
            {"search": "completed", "answer": "completed"},
        )

    async def test_plan_repair_exhausted_preserves_successful_tools_and_completed_final_answer(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-final", text="基于已取得资料的最终回答"))
        state.plan_coordinator = PlanCoordinator(run_id="run-plan-repair", mode="on")
        state.plan_coordinator.apply_model_update(
            {
                "reason": "检索、核验并回答",
                "items": [
                    {
                        "id": "search-success",
                        "title": "检索可靠资料",
                        "status": "running",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "search-failed",
                        "title": "补充对照资料",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": ["search-success"],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "title": "整理最终回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["search-failed"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        state.plan_coordinator.mark_tool_results({"search-success": "completed"})
        state.plan_coordinator.mark_tools_started(["search-failed"])
        state.plan_coordinator.mark_tool_results({"search-failed": "failed"})
        emitter = AsyncMock()

        await finalize_completed_run(
            context=replace(_context(state), emitter=emitter),
            terminal_state=SimpleNamespace(session_status="incomplete", run_finish_reason="incomplete"),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=AsyncMock(),
            finalize_stream_fn=AsyncMock(),
        )

        snapshot = emitter.plan_snapshot.await_args.kwargs
        self.assertEqual(
            {item["id"]: item["status"] for item in snapshot["items"]},
            {
                "search-success": "completed",
                "search-failed": "failed",
                "answer": "completed",
            },
        )

    async def test_itinerary_observation_failure_never_changes_completed_terminal_result(self):
        state = AgentLoopState()
        state.record_itinerary_tool_observations(
            [
                ItineraryToolObservation(
                    "search_flights",
                    "success",
                    None,
                    100,
                    False,
                    False,
                    False,
                    False,
                )
            ]
        )

        def fail_duration() -> int:
            raise RuntimeError("observation failed")

        context = replace(
            _context(state),
            duration_ms_factory=fail_duration,
        )
        finalized = []

        async def complete_agent_run_fn(**_kwargs):
            return None

        async def finalize_stream_fn(*args, **kwargs):
            finalized.append((args, kwargs))

        await finalize_completed_run(
            context=context,
            terminal_state=SimpleNamespace(session_status="completed", run_finish_reason="stop"),
            persist_message_fn=lambda *_args: None,
            complete_agent_run_fn=complete_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
        )

        self.assertEqual(finalized, [(("conv-1",), {"success": True, "task_id": "task-1"})])
        self.assertTrue(state.terminal_emitted)

    async def test_finalize_superseded_persists_and_interrupts_before_error_finalize(self):
        state = AgentLoopState()
        state.current_step_id = "step-1"
        ctx = _context(state)
        calls = []

        def persist_message_fn(*args):
            calls.append(("persist", args))

        async def interrupt_agent_run_fn(**kwargs):
            calls.append(("interrupt", kwargs))

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        await finalize_superseded_run(
            context=ctx,
            error_msg="被新请求取代",
            persist_message_fn=persist_message_fn,
            interrupt_agent_run_fn=interrupt_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
        )

        self.assertEqual([call[0] for call in calls], ["persist", "interrupt", "finalize"])
        self.assertTrue(calls[0][1][6])
        self.assertEqual(calls[1][1]["current_step_id"], "step-1")
        self.assertEqual(calls[1][1]["reason"], "superseded")
        self.assertEqual(
            calls[2],
            (
                "finalize",
                ("conv-1",),
                {"success": False, "error_msg": "被新请求取代", "task_id": "task-1"},
            ),
        )
        self.assertTrue(state.terminal_emitted)

    async def test_finalize_cancelled_persists_only_with_content_and_swallows_emit_failure(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="半截回答"))
        ctx = _context(state)
        calls = []

        def persist_message_fn(*args):
            calls.append(("persist", args))

        async def interrupt_agent_run_fn(**_kwargs):
            raise RuntimeError("emit down")

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        warnings = []

        await finalize_cancelled_run(
            context=ctx,
            persist_message_fn=persist_message_fn,
            interrupt_agent_run_fn=interrupt_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=warnings.append,
        )

        self.assertEqual([call[0] for call in calls], ["persist", "finalize"])
        self.assertTrue(calls[0][1][6])
        self.assertIn("emit run_interrupted 失败: emit down", warnings)
        self.assertEqual(
            calls[1],
            (
                "finalize",
                ("conv-1",),
                {
                    "success": False,
                    "error_msg": "用户中止",
                    "task_id": "task-1",
                    "error_code": "stream_interrupted",
                    "error_data": {"reason": "user_cancelled"},
                },
            ),
        )
        self.assertFalse(state.terminal_emitted)

    async def test_finalize_failed_persists_content_and_swallows_emit_failure(self):
        state = AgentLoopState()
        state.content_blocks.append(TextBlock(type="text", id="txt-1", text="半截回答"))
        ctx = _context(state)
        calls = []

        def persist_message_fn(*args):
            calls.append(("persist", args))

        async def fail_agent_run_fn(**_kwargs):
            raise RuntimeError("emit failed down")

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        warnings = []

        await finalize_failed_run(
            context=ctx,
            error=ValueError("LLM 5xx"),
            persist_message_fn=persist_message_fn,
            fail_agent_run_fn=fail_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=warnings.append,
        )

        self.assertEqual([call[0] for call in calls], ["persist", "finalize"])
        self.assertTrue(calls[0][1][6])
        self.assertIn("emit run_failed 失败: error_type=RuntimeError", warnings)
        self.assertNotIn("emit failed down", " ".join(warnings))
        self.assertEqual(
            calls[1],
            (
                "finalize",
                ("conv-1",),
                {
                    "success": False,
                    "error_msg": "生成服务暂时不可用，请稍后重试",
                    "error_code": "agent_run_failed",
                    "task_id": "task-1",
                },
            ),
        )
        self.assertFalse(state.terminal_emitted)

    async def test_finalize_failed_preserves_safe_structured_context_error_code(self):
        class ContextError(RuntimeError):
            error_code = "context_budget_exceeded"

        calls = []

        async def fail_agent_run_fn(**kwargs):
            calls.append(("fail", kwargs))

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        await finalize_failed_run(
            context=_context(),
            error=ContextError("请缩短输入"),
            persist_message_fn=lambda *_args: None,
            fail_agent_run_fn=fail_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=lambda _message: None,
        )

        self.assertEqual(calls[0][1]["error_code"], "context_budget_exceeded")
        self.assertEqual(calls[1][2]["error_code"], "context_budget_exceeded")
        self.assertEqual(
            calls[0][1]["message"],
            "当前消息与必要上下文过长，请缩短本次输入或移除较大的文件后重试",
        )
        self.assertEqual(
            calls[1][2]["error_msg"],
            "当前消息与必要上下文过长，请缩短本次输入或移除较大的文件后重试",
        )

    async def test_finalize_failed_never_exposes_secret_bearing_exception_message(self):
        secret = "sk-secret-value"
        calls = []

        async def fail_agent_run_fn(**kwargs):
            calls.append(("fail", kwargs))

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        await finalize_failed_run(
            context=_context(),
            error=RuntimeError(f"Authorization: Bearer {secret}; upstream 503"),
            persist_message_fn=lambda *_args: None,
            fail_agent_run_fn=fail_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=lambda message: calls.append(("warning", message)),
        )

        rendered = repr(calls)
        self.assertNotIn(secret, rendered)
        self.assertEqual(calls[0][1]["error_code"], "agent_run_failed")
        self.assertEqual(calls[0][1]["message"], "生成服务暂时不可用，请稍后重试")
        self.assertEqual(calls[1][2]["error_code"], "agent_run_failed")
        self.assertEqual(calls[1][2]["error_msg"], "生成服务暂时不可用，请稍后重试")

    async def test_untrusted_structured_error_code_is_not_forwarded(self):
        class SecretCodeError(RuntimeError):
            error_code = "sk_secret_value"

        calls = []

        async def fail_agent_run_fn(**kwargs):
            calls.append(("fail", kwargs))

        async def finalize_stream_fn(*args, **kwargs):
            calls.append(("finalize", args, kwargs))

        await finalize_failed_run(
            context=_context(),
            error=SecretCodeError("upstream failed"),
            persist_message_fn=lambda *_args: None,
            fail_agent_run_fn=fail_agent_run_fn,
            finalize_stream_fn=finalize_stream_fn,
            warning_fn=lambda _message: None,
        )

        self.assertEqual(calls[0][1]["error_code"], "agent_run_failed")
        self.assertEqual(calls[1][2]["error_code"], "agent_run_failed")
        self.assertNotIn("sk_secret_value", repr(calls))

    async def test_fallback_failure_log_never_contains_exception_message(self):
        warnings = []

        async def write_fallback_error_status_fn(**_kwargs):
            raise RuntimeError("Authorization: Bearer sk-secret-value")

        await write_fallback_run_error(
            context=_context(),
            write_fallback_error_status_fn=write_fallback_error_status_fn,
            warning_fn=warnings.append,
        )

        self.assertEqual(
            warnings,
            ["finally 兜底 write_session_status 失败: error_type=RuntimeError"],
        )

    async def test_write_fallback_run_error_skips_when_terminal_already_emitted(self):
        state = AgentLoopState()
        state.mark_terminal_emitted()
        calls = []

        async def write_fallback_error_status_fn(**kwargs):
            calls.append(kwargs)

        await write_fallback_run_error(
            context=_context(state),
            write_fallback_error_status_fn=write_fallback_error_status_fn,
            warning_fn=calls.append,
        )

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
