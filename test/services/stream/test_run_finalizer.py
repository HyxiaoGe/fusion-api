import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

from app.services.stream.run_finalizer import (
    AgentRunStats,
    complete_agent_run,
    fail_agent_run,
    interrupt_agent_run,
    start_agent_run,
    write_fallback_error_status,
)


class RunFinalizerTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self):
        emitter = SimpleNamespace(
            run_started=AsyncMock(),
            run_completed=AsyncMock(),
            run_interrupted=AsyncMock(),
            run_failed=AsyncMock(),
        )
        cache = SimpleNamespace(
            write_session_started=AsyncMock(),
            write_step_terminal=AsyncMock(),
            write_session_status=AsyncMock(),
        )
        order = Mock()
        order.attach_mock(emitter.run_started, "run_started")
        order.attach_mock(emitter.run_completed, "run_completed")
        order.attach_mock(emitter.run_interrupted, "run_interrupted")
        order.attach_mock(emitter.run_failed, "run_failed")
        order.attach_mock(cache.write_session_started, "write_session_started")
        order.attach_mock(cache.write_step_terminal, "write_step_terminal")
        order.attach_mock(cache.write_session_status, "write_session_status")
        return emitter, cache, order

    def _stats(self):
        return AgentRunStats(
            run_id="run-1",
            total_steps=3,
            total_tool_calls=2,
        )

    def _duration_factory(self, order, duration_ms=1234):
        duration_factory = Mock(return_value=duration_ms)
        order.attach_mock(duration_factory, "duration_ms_factory")
        return duration_factory

    async def test_start_agent_run_writes_session_before_run_started(self):
        emitter, cache, order = self._deps()

        await start_agent_run(
            emitter=emitter,
            session_cache=cache,
            run_id="run-1",
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            message_id="msg-1",
            tools=["web_search"],
            config={"max_steps": 8},
        )

        self.assertEqual(
            order.mock_calls,
            [
                call.write_session_started(
                    run_id="run-1",
                    conversation_id="conv-1",
                    user_id="user-1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="msg-1",
                    run_config={"max_steps": 8},
                ),
                call.run_started(
                    message_id="msg-1",
                    model="gpt-4",
                    tools=["web_search"],
                    config={"max_steps": 8},
                ),
            ],
        )

    async def test_complete_agent_run_emits_completed_before_session_status(self):
        emitter, cache, order = self._deps()
        duration_ms_factory = self._duration_factory(order)

        await complete_agent_run(
            emitter=emitter,
            session_cache=cache,
            stats=self._stats(),
            duration_ms_factory=duration_ms_factory,
            session_status="limit_reached",
            finish_reason="limit_reached",
            limit_reason="max_steps",
        )

        self.assertEqual(
            order.mock_calls,
            [
                call.run_completed(total_steps=3, total_tool_calls=2, finish_reason="limit_reached"),
                call.duration_ms_factory(),
                call.write_session_status(
                    run_id="run-1",
                    status="limit_reached",
                    total_steps=3,
                    total_tool_calls=2,
                    total_duration_ms=1234,
                    limit_reason="max_steps",
                ),
            ],
        )

    async def test_interrupt_agent_run_closes_current_step_before_interrupted_status(self):
        emitter, cache, order = self._deps()
        duration_ms_factory = self._duration_factory(order)

        await interrupt_agent_run(
            emitter=emitter,
            session_cache=cache,
            stats=self._stats(),
            duration_ms_factory=duration_ms_factory,
            current_step_id="step-1",
            reason="superseded",
        )

        self.assertEqual(
            order.mock_calls,
            [
                call.write_step_terminal(step_id="step-1", status="interrupted"),
                call.run_interrupted(reason="superseded"),
                call.duration_ms_factory(),
                call.write_session_status(
                    run_id="run-1",
                    status="interrupted",
                    total_steps=3,
                    total_tool_calls=2,
                    total_duration_ms=1234,
                ),
            ],
        )

    async def test_interrupt_agent_run_skips_step_terminal_without_current_step(self):
        emitter, cache, order = self._deps()
        duration_ms_factory = self._duration_factory(order)

        await interrupt_agent_run(
            emitter=emitter,
            session_cache=cache,
            stats=self._stats(),
            duration_ms_factory=duration_ms_factory,
            current_step_id=None,
            reason="user_cancelled",
        )

        cache.write_step_terminal.assert_not_awaited()
        self.assertEqual(
            order.mock_calls,
            [
                call.run_interrupted(reason="user_cancelled"),
                call.duration_ms_factory(),
                call.write_session_status(
                    run_id="run-1",
                    status="interrupted",
                    total_steps=3,
                    total_tool_calls=2,
                    total_duration_ms=1234,
                ),
            ],
        )

    async def test_fail_agent_run_writes_failed_step_and_error_status(self):
        emitter, cache, order = self._deps()
        duration_ms_factory = self._duration_factory(order)

        await fail_agent_run(
            emitter=emitter,
            session_cache=cache,
            stats=self._stats(),
            duration_ms_factory=duration_ms_factory,
            current_step_id="step-1",
            error_code="RuntimeError",
            message="upstream LLM 5xx",
        )

        self.assertEqual(
            order.mock_calls,
            [
                call.write_step_terminal(step_id="step-1", status="failed"),
                call.run_failed(error_code="RuntimeError", message="upstream LLM 5xx"),
                call.duration_ms_factory(),
                call.write_session_status(
                    run_id="run-1",
                    status="error",
                    total_steps=3,
                    total_tool_calls=2,
                    total_duration_ms=1234,
                ),
            ],
        )

    async def test_write_fallback_error_status_only_writes_session_status(self):
        emitter, cache, order = self._deps()
        duration_ms_factory = self._duration_factory(order)

        await write_fallback_error_status(
            session_cache=cache,
            stats=self._stats(),
            duration_ms_factory=duration_ms_factory,
        )

        emitter.run_completed.assert_not_awaited()
        emitter.run_interrupted.assert_not_awaited()
        emitter.run_failed.assert_not_awaited()
        cache.write_step_terminal.assert_not_awaited()
        self.assertEqual(
            order.mock_calls,
            [
                call.duration_ms_factory(),
                call.write_session_status(
                    run_id="run-1",
                    status="error",
                    total_steps=3,
                    total_tool_calls=2,
                    total_duration_ms=1234,
                ),
            ],
        )
