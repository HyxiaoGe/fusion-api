import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

from app.services.agent.llm_round_detail_recorder import LlmRoundDetailDraft
from app.services.stream.llm_round_lifecycle import LLMRoundLifecycle


class LlmRoundLifecycleDetailTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_schedules_exact_round_detail_after_terminal_event_once(self):
        emitter = AsyncMock()
        scheduler = Mock()
        lifecycle = await LLMRoundLifecycle.start(
            emitter=emitter,
            observation=MagicMock(duration_ms=100),
            round_index=2,
            model="deepseek-chat",
            provider="deepseek",
            parent_step_id="step-2",
            conversation_id="conv-1",
            run_id="run-1",
            message_id="msg-1",
            detail_scheduler=scheduler,
        )
        self.assertIsNotNone(lifecycle)
        lifecycle.record_detail(reasoning_text="显式推理", content_text="候选回答")

        await lifecycle.finish_success()
        await lifecycle.finish_success()

        emitter.llm_round_completed.assert_awaited_once()
        scheduler.assert_called_once_with(
            LlmRoundDetailDraft(
                conversation_id="conv-1",
                run_id="run-1",
                message_id="msg-1",
                llm_round_id=lifecycle.llm_round_id,
                reasoning_text="显式推理",
                content_text="候选回答",
            )
        )

    async def test_failed_and_cancelled_rounds_persist_captured_partial_text(self):
        for terminal in ("failed", "cancelled"):
            with self.subTest(terminal=terminal):
                emitter = AsyncMock()
                scheduler = Mock()
                lifecycle = await LLMRoundLifecycle.start(
                    emitter=emitter,
                    observation=MagicMock(duration_ms=50),
                    round_index=1,
                    model="deepseek-chat",
                    provider="deepseek",
                    parent_step_id="step-1",
                    conversation_id="conv-1",
                    run_id=f"run-{terminal}",
                    message_id="msg-1",
                    detail_scheduler=scheduler,
                )
                lifecycle.record_detail(reasoning_text="部分推理", content_text="部分回答")

                if terminal == "failed":
                    await lifecycle.finish_failed(RuntimeError("provider failed"))
                else:
                    await lifecycle.finish_cancelled(reason="shutdown")

                scheduler.assert_called_once()
                draft = scheduler.call_args.args[0]
                self.assertEqual(draft.reasoning_text, "部分推理")
                self.assertEqual(draft.content_text, "部分回答")

    async def test_scheduler_failure_is_fail_open_after_terminal_event(self):
        emitter = AsyncMock()
        lifecycle = await LLMRoundLifecycle.start(
            emitter=emitter,
            observation=MagicMock(duration_ms=50),
            round_index=1,
            model="deepseek-chat",
            provider="deepseek",
            parent_step_id="step-1",
            conversation_id="conv-1",
            run_id="run-1",
            message_id="msg-1",
            detail_scheduler=Mock(side_effect=RuntimeError("scheduler closed")),
        )
        lifecycle.record_detail(reasoning_text="推理", content_text="回答")

        await lifecycle.finish_success()

        emitter.llm_round_completed.assert_awaited_once()
        self.assertTrue(lifecycle.terminal_emitted)


if __name__ == "__main__":
    unittest.main()
