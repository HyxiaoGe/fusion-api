import unittest
from types import SimpleNamespace

from app.services.stream.agent_loop_execution import (
    AgentLoopDependencies,
    AgentLoopExecutionParts,
    AgentLoopExecutionRequest,
    build_agent_loop_execution,
    build_agent_loop_runtime,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_state import AgentLoopState
from app.services.stream.network_budget import NetworkToolBudget


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


class AgentLoopExecutionTests(unittest.TestCase):
    def _dependencies(self, *, clock):
        return AgentLoopDependencies(
            session_cache="session-cache",
            redis_writer="redis-writer",
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
            clock=clock,
        )

    def test_build_execution_context_wires_runtime_and_completion_context(self):
        clock_values = iter([100.0, 102.5])

        def clock():
            return next(clock_values)

        call_kwargs = {"tools": ["web_search"]}
        call_config = SimpleNamespace(
            should_use_reasoning=True,
            call_kwargs=call_kwargs,
        )
        limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=30)
        dependencies = self._dependencies(clock=clock)

        execution = build_agent_loop_execution(
            request=AgentLoopExecutionRequest(
                db="db",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={"temperature": 0},
                provider="openai",
                assistant_message_id="msg-1",
                task_id="task-1",
                call_config=call_config,
                trace_id="trace-1",
            ),
            limits=limits,
            dependencies=dependencies,
        )

        self.assertEqual(execution.run_id, "trace-1")
        self.assertEqual(execution.run_start, 100.0)
        self.assertIsInstance(execution.state, AgentLoopState)
        self.assertIsInstance(execution.network_budget, NetworkToolBudget)
        self.assertIs(execution.runtime.call_kwargs, call_kwargs)
        self.assertTrue(execution.runtime.should_use_reasoning)
        self.assertEqual(execution.runtime.limits, limits)
        self.assertEqual(execution.runtime.run_id, "trace-1")
        self.assertEqual(execution.runtime.run_start, 100.0)
        self.assertEqual(execution.runtime.conversation_id, "conv-1")
        self.assertEqual(execution.runtime.task_id, "task-1")
        self.assertEqual(execution.runtime.user_id, "user-1")
        self.assertEqual(execution.runtime.model_id, "gpt-4")
        self.assertEqual(execution.runtime.provider, "openai")
        self.assertEqual(execution.runtime.litellm_model, "openai/gpt-4")
        self.assertEqual(execution.runtime.litellm_kwargs, {"temperature": 0})
        self.assertEqual(execution.runtime.assistant_message_id, "msg-1")
        self.assertEqual(execution.runtime.session_cache, "session-cache")
        self.assertIs(execution.runtime.emitter, execution.emitter)
        self.assertIs(execution.runtime.network_budget, execution.network_budget)
        self.assertIs(execution.completion_context.state, execution.state)
        self.assertIs(execution.completion_context.emitter, execution.emitter)
        self.assertEqual(execution.completion_context.db, "db")
        self.assertEqual(execution.completion_context.conversation_id, "conv-1")
        self.assertEqual(execution.completion_context.task_id, "task-1")
        self.assertEqual(execution.completion_context.model_id, "gpt-4")
        self.assertEqual(execution.completion_context.assistant_message_id, "msg-1")
        self.assertEqual(execution.completion_context.session_cache, "session-cache")
        self.assertEqual(execution.completion_context.duration_ms_factory(), 2500)

    def test_build_execution_context_generates_run_id_when_trace_id_missing(self):
        call_config = SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
        )

        execution = build_agent_loop_execution(
            request=AgentLoopExecutionRequest(
                db="db",
                conversation_id="conv-uuid",
                user_id="user-uuid",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                assistant_message_id="msg-uuid",
                task_id="task-uuid",
                call_config=call_config,
                trace_id=None,
            ),
            limits=AgentLoopLimits(max_steps=1, max_tool_calls=1, total_timeout_s=1),
            dependencies=self._dependencies(clock=lambda: 10.0),
        )

        self.assertTrue(execution.run_id)
        self.assertNotEqual(execution.run_id, "None")
        self.assertEqual(execution.runtime.run_id, execution.run_id)
        self.assertEqual(execution.completion_context.run_id, execution.run_id)

    def test_build_agent_loop_runtime_accepts_prebuilt_execution_parts(self):
        call_kwargs = {"temperature": 0.1}
        call_config = SimpleNamespace(
            should_use_reasoning=True,
            call_kwargs=call_kwargs,
        )
        request = AgentLoopExecutionRequest(
            db="db",
            conversation_id="conv-runtime",
            user_id="user-runtime",
            model_id="gpt-4",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"metadata": {"trace": "x"}},
            provider="openai",
            assistant_message_id="msg-runtime",
            task_id="task-runtime",
            call_config=call_config,
            trace_id="trace-runtime",
        )
        limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=30)
        dependencies = self._dependencies(clock=lambda: 100.0)
        parts = AgentLoopExecutionParts(
            run_id="run-runtime",
            run_start=100.0,
            state=AgentLoopState(),
            network_budget=NetworkToolBudget(),
            emitter=object(),
        )

        runtime = build_agent_loop_runtime(
            request=request,
            limits=limits,
            dependencies=dependencies,
            parts=parts,
        )

        self.assertEqual(runtime.run_id, "run-runtime")
        self.assertEqual(runtime.run_start, 100.0)
        self.assertEqual(runtime.conversation_id, "conv-runtime")
        self.assertEqual(runtime.task_id, "task-runtime")
        self.assertEqual(runtime.model_id, "gpt-4")
        self.assertIs(runtime.call_kwargs, call_kwargs)
        self.assertIs(runtime.emitter, parts.emitter)
        self.assertIs(runtime.network_budget, parts.network_budget)
        self.assertIs(runtime.session_cache, dependencies.session_cache)


if __name__ == "__main__":
    unittest.main()
