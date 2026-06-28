import unittest
from types import SimpleNamespace

from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_wiring import (
    AgentLoopRunInput,
    AgentLoopWiringDependencies,
    build_agent_loop_lifecycle_call,
)


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


class AgentLoopWiringTests(unittest.TestCase):
    def test_build_lifecycle_call_wires_execution_and_lifecycle_dependencies(self):
        captured = {}
        call_config = SimpleNamespace(
            should_use_reasoning=True,
            call_kwargs={"tools": ["web_search"]},
            announced_tools=["web_search"],
        )
        fake_execution = SimpleNamespace(run_id="run-wiring")
        redis_writer = object()

        async def append_chunk_fn(*_args, **_kwargs):
            return None

        async def start_agent_run_fn(**_kwargs):
            return None

        async def prepare_messages_fn(**_kwargs):
            return None

        async def run_agent_loop_fn(**_kwargs):
            return None

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

        async def complete_agent_run_fn(**_kwargs):
            return None

        async def interrupt_agent_run_fn(**_kwargs):
            return None

        async def fail_agent_run_fn(**_kwargs):
            return None

        async def finalize_stream_fn(*_args, **_kwargs):
            return None

        async def write_fallback_error_status_fn(**_kwargs):
            return None

        def build_call_config_fn(**kwargs):
            captured["call_config_kwargs"] = kwargs
            return call_config

        def build_execution_fn(**kwargs):
            captured["execution_kwargs"] = kwargs
            return fake_execution

        def redis_writer_factory():
            captured["redis_writer_factory_called"] = True
            return redis_writer

        def warning_fn(_message):
            return None

        def info_fn(_message):
            return None

        def error_fn(_message):
            return None

        limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=30)
        run_input = AgentLoopRunInput(
            conversation_id="conv-wiring",
            user_id="user-wiring",
            model_id="gpt-4",
            litellm_model="openai/gpt-4",
            litellm_kwargs={"temperature": 0},
            provider="openai",
            raw_messages=[{"role": "user", "content": "hi"}],
            has_vision=False,
            file_ids=["file-1"],
            original_message="hi",
            assistant_message_id="msg-wiring",
            task_id="task-wiring",
            options=None,
            capabilities=None,
            trace_id="trace-wiring",
        )

        lifecycle_call = build_agent_loop_lifecycle_call(
            run_input=run_input,
            db="db-wiring",
            limits=limits,
            dependencies=AgentLoopWiringDependencies(
                build_call_config_fn=build_call_config_fn,
                build_execution_fn=build_execution_fn,
                session_cache="session-cache",
                redis_writer_factory=redis_writer_factory,
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
                clock=lambda: 100.0,
                append_chunk_fn=append_chunk_fn,
                start_agent_run_fn=start_agent_run_fn,
                prepare_messages_fn=prepare_messages_fn,
                run_agent_loop_fn=run_agent_loop_fn,
                finalize_completed_run_fn=finalize_completed_run_fn,
                finalize_superseded_run_fn=finalize_superseded_run_fn,
                finalize_cancelled_run_fn=finalize_cancelled_run_fn,
                finalize_failed_run_fn=finalize_failed_run_fn,
                write_fallback_run_error_fn=write_fallback_run_error_fn,
                complete_agent_run_fn=complete_agent_run_fn,
                interrupt_agent_run_fn=interrupt_agent_run_fn,
                fail_agent_run_fn=fail_agent_run_fn,
                finalize_stream_fn=finalize_stream_fn,
                write_fallback_error_status_fn=write_fallback_error_status_fn,
                info_fn=info_fn,
                error_fn=error_fn,
                warning_fn=warning_fn,
            ),
        )

        self.assertEqual(
            captured["call_config_kwargs"],
            {
                "provider": "openai",
                "options": {},
                "capabilities": {},
            },
        )
        self.assertTrue(captured["redis_writer_factory_called"])
        execution_request = captured["execution_kwargs"]["request"]
        execution_dependencies = captured["execution_kwargs"]["dependencies"]
        self.assertEqual(captured["execution_kwargs"]["limits"], limits)
        self.assertEqual(execution_request.db, "db-wiring")
        self.assertEqual(execution_request.conversation_id, "conv-wiring")
        self.assertEqual(execution_request.user_id, "user-wiring")
        self.assertEqual(execution_request.model_id, "gpt-4")
        self.assertEqual(execution_request.litellm_model, "openai/gpt-4")
        self.assertEqual(execution_request.litellm_kwargs, {"temperature": 0})
        self.assertEqual(execution_request.provider, "openai")
        self.assertEqual(execution_request.assistant_message_id, "msg-wiring")
        self.assertEqual(execution_request.task_id, "task-wiring")
        self.assertIs(execution_request.call_config, call_config)
        self.assertEqual(execution_request.trace_id, "trace-wiring")
        self.assertEqual(execution_dependencies.session_cache, "session-cache")
        self.assertIs(execution_dependencies.redis_writer, redis_writer)
        self.assertIs(execution_dependencies.llm_call_fn, _unused_async)
        self.assertIs(execution_dependencies.stream_round_fn, _unused_async)
        self.assertIs(execution_dependencies.execute_tools_fn, _unused_async)
        self.assertIs(execution_dependencies.persist_message_fn, _unused_sync)
        self.assertIs(execution_dependencies.warning_fn, warning_fn)

        self.assertIs(lifecycle_call.execution, fake_execution)
        self.assertEqual(lifecycle_call.request.raw_messages, [{"role": "user", "content": "hi"}])
        self.assertFalse(lifecycle_call.request.has_vision)
        self.assertEqual(lifecycle_call.request.file_ids, ["file-1"])
        self.assertEqual(lifecycle_call.request.original_message, "hi")
        self.assertIs(lifecycle_call.request.call_config, call_config)
        self.assertEqual(lifecycle_call.request.limits, limits)
        self.assertIs(lifecycle_call.dependencies.append_chunk_fn, append_chunk_fn)
        self.assertIs(lifecycle_call.dependencies.start_agent_run_fn, start_agent_run_fn)
        self.assertIs(lifecycle_call.dependencies.prepare_messages_fn, prepare_messages_fn)
        self.assertIs(lifecycle_call.dependencies.run_agent_loop_fn, run_agent_loop_fn)
        self.assertIs(lifecycle_call.dependencies.finalize_completed_run_fn, finalize_completed_run_fn)
        self.assertIs(lifecycle_call.dependencies.finalize_superseded_run_fn, finalize_superseded_run_fn)
        self.assertIs(lifecycle_call.dependencies.finalize_cancelled_run_fn, finalize_cancelled_run_fn)
        self.assertIs(lifecycle_call.dependencies.finalize_failed_run_fn, finalize_failed_run_fn)
        self.assertIs(lifecycle_call.dependencies.write_fallback_run_error_fn, write_fallback_run_error_fn)
        self.assertIs(lifecycle_call.dependencies.complete_agent_run_fn, complete_agent_run_fn)
        self.assertIs(lifecycle_call.dependencies.interrupt_agent_run_fn, interrupt_agent_run_fn)
        self.assertIs(lifecycle_call.dependencies.fail_agent_run_fn, fail_agent_run_fn)
        self.assertIs(lifecycle_call.dependencies.finalize_stream_fn, finalize_stream_fn)
        self.assertIs(lifecycle_call.dependencies.write_fallback_error_status_fn, write_fallback_error_status_fn)
        self.assertIs(lifecycle_call.dependencies.info_fn, info_fn)
        self.assertIs(lifecycle_call.dependencies.error_fn, error_fn)
        self.assertIs(lifecycle_call.dependencies.warning_fn, warning_fn)


if __name__ == "__main__":
    unittest.main()
