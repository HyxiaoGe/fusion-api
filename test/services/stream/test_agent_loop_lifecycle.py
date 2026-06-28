import asyncio
import unittest
from types import SimpleNamespace

from app.schemas.chat import TextBlock
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
    run_agent_loop_lifecycle,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import AgentLoopPreparedMessages


async def _unused_async(**_kwargs):
    raise AssertionError("不应调用这个依赖")


def _unused_sync(*_args, **_kwargs):
    raise AssertionError("不应调用这个依赖")


class AgentLoopLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _call_config(self):
        return SimpleNamespace(
            should_use_reasoning=False,
            call_kwargs={},
            announced_tools=["web_search"],
        )

    def _limits(self):
        return AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=30)

    def _execution(self, *, call_config=None, limits=None):
        call_config = call_config or self._call_config()
        limits = limits or self._limits()
        return build_agent_loop_execution(
            request=AgentLoopExecutionRequest(
                db="db",
                conversation_id="conv-life",
                user_id="user-life",
                model_id="gpt-4",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                provider="openai",
                assistant_message_id="msg-life",
                task_id="task-life",
                call_config=call_config,
                trace_id="run-life",
            ),
            limits=limits,
            dependencies=ExecutionDependencies(
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
                clock=lambda: 10.0,
            ),
        )

    def _request(self, *, call_config=None, limits=None):
        return AgentLoopLifecycleRequest(
            raw_messages=[{"role": "user", "content": "hi"}],
            has_vision=False,
            file_ids=None,
            original_message="hi",
            call_config=call_config or self._call_config(),
            limits=limits or self._limits(),
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

        async def append_chunk_fn(conversation_id, chunk_type, content, block_id):
            call_order.append(("append", conversation_id, chunk_type, content, block_id))

        async def start_agent_run_fn(**kwargs):
            call_order.append(("start", kwargs["run_id"], kwargs["tools"], kwargs["config"]))

        async def prepare_messages_fn(**kwargs):
            call_order.append(("prepare", kwargs["db"], kwargs["raw_messages"], kwargs["call_config"]))
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
        self.assertEqual(call_order[0], ("append", "conv-life", "preparing", "", ""))
        self.assertEqual(call_order[1][1], "run-life")
        self.assertEqual(call_order[1][2], ["web_search"])
        self.assertEqual(call_order[1][3], {"max_steps": 3, "max_tool_calls": 5, "timeout_s": 30})
        self.assertIs(call_order[2][3], call_config)
        self.assertEqual(call_order[3][1], [{"role": "user", "content": "prepared"}])
        self.assertEqual(call_order[3][2], [initial_block])
        self.assertIs(call_order[4][1], execution.completion_context)

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


if __name__ == "__main__":
    unittest.main()
