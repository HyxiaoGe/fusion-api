import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def _execution(self, *, call_config=None, limits=None, redis_writer=None):
        call_config = call_config or self._call_config()
        limits = limits or self._limits()

        class NoopWriter:
            async def append_chunk(self, *_args, **_kwargs):
                return None

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
            call_order.append(("start", kwargs["run_id"], kwargs["tools"], kwargs["config"]))

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
        self.assertEqual(call_order[1][2], ["web_search"])
        self.assertEqual(
            call_order[1][3],
            {
                "max_steps": 3,
                "max_tool_calls": 5,
                "timeout_s": 30,
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
        self.assertEqual(execution.state.plan_items, {})

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
