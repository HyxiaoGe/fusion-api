import unittest
from types import SimpleNamespace

from app.schemas.chat import TextBlock, Usage
from app.services.stream.agent_loop_run_completion import (
    AgentLoopRunCompletionContext,
    finalize_cancelled_run,
    finalize_completed_run,
    finalize_failed_run,
    finalize_superseded_run,
    write_fallback_run_error,
)
from app.services.stream.agent_loop_state import AgentLoopState


def _context(state: AgentLoopState | None = None) -> AgentLoopRunCompletionContext:
    return AgentLoopRunCompletionContext(
        db="db",
        conversation_id="conv-1",
        task_id="task-1",
        run_id="run-1",
        model_id="gpt-4",
        assistant_message_id="msg-1",
        emitter="emitter",
        session_cache="session-cache",
        state=state or AgentLoopState(),
        duration_ms_factory=lambda: 1234,
    )


class AgentLoopRunCompletionTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(calls[1][1]["session_status"], "completed")
        self.assertEqual(calls[1][1]["finish_reason"], "stop")
        self.assertEqual(calls[2], ("finalize", ("conv-1",), {"success": True, "task_id": "task-1"}))
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
        self.assertIn("emit run_interrupted 失败: emit down", warnings)
        self.assertEqual(
            calls[1],
            ("finalize", ("conv-1",), {"success": False, "error_msg": "用户中止", "task_id": "task-1"}),
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
        self.assertIn("emit run_failed 失败: emit failed down", warnings)
        self.assertEqual(
            calls[1],
            ("finalize", ("conv-1",), {"success": False, "error_msg": "LLM 5xx", "task_id": "task-1"}),
        )
        self.assertFalse(state.terminal_emitted)

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
