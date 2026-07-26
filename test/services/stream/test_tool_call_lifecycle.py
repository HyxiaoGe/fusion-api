"""tool_call 生命周期事件与执行结果状态转换测试。"""

import asyncio
import unittest
from unittest.mock import patch

from app.services.stream import tool_call_lifecycle as lifecycle_module
from app.services.stream.tool_call_lifecycle import execute_tool_with_lifecycle
from app.services.tool_handlers.base import ToolResult


class RecordingEmitter:
    def __init__(self):
        self.events = []

    async def tool_call_started(self, **kwargs):
        self.events.append(("started", kwargs))

    async def tool_call_completed(self, **kwargs):
        self.events.append(("completed", kwargs))


class ToolCallLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_tool_attempt_returns_success_result_and_measured_duration(self):
        attempt_cls = getattr(lifecycle_module, "ToolLifecycleAttempt")
        run_tool_attempt = getattr(lifecycle_module, "run_tool_attempt")
        result = ToolResult(status="success")
        target = object()
        args = {"query": "redis"}
        calls = []

        async def execute(received_target, received_args):
            calls.append((received_target, received_args))
            return result

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[100.0, 100.25]):
            attempt = await run_tool_attempt(target=target, args=args, execute=execute)

        self.assertEqual(attempt, attempt_cls(result=result, duration_ms=250, cancelled_error=None))
        self.assertEqual(calls, [(target, args)])

    async def test_run_tool_attempt_maps_exception_to_failed_result_without_raising(self):
        run_tool_attempt = getattr(lifecycle_module, "run_tool_attempt")

        async def execute(_target, _args):
            raise ValueError("参数非法")

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[200.0, 200.03]):
            attempt = await run_tool_attempt(target=object(), args={}, execute=execute)

        self.assertEqual(attempt.duration_ms, 30)
        self.assertIsNone(attempt.cancelled_error)
        self.assertEqual(attempt.result.status, "failed")
        self.assertEqual(attempt.result.error_message, "工具执行未完成")
        self.assertEqual(
            attempt.result.data,
            {"error_code": "tool_execution_failed", "retryable": False},
        )

    async def test_complete_tool_lifecycle_sets_duration_and_emits_completed_once(self):
        complete_tool_lifecycle = getattr(lifecycle_module, "complete_tool_lifecycle")
        emitter = RecordingEmitter()
        result = ToolResult(status="success")
        summaries = []

        def build_summary(received_result):
            summaries.append(received_result)
            return {"kind": "search"}

        await complete_tool_lifecycle(
            emitter=emitter,
            tool_call_id="call-0",
            tool_name="web_search",
            result=result,
            duration_ms=12,
            result_summary_builder=build_summary,
        )
        self.assertEqual(result.duration_ms, 12)
        self.assertEqual(summaries, [result])
        self.assertEqual(
            emitter.events,
            [
                (
                    "completed",
                    {
                        "tool_call_id": "call-0",
                        "tool_name": "web_search",
                        "status": "success",
                        "duration_ms": 12,
                        "result_summary": {"kind": "search"},
                        "error": None,
                    },
                )
            ],
        )

    async def test_repair_metadata_is_projected_without_argument_values(self):
        complete_tool_lifecycle = getattr(lifecycle_module, "complete_tool_lifecycle")
        emitter = RecordingEmitter()
        result = ToolResult(
            status="failed",
            data={
                "repair": {
                    "repair_id": "repair_0123456789abcdef",
                    "retryable": True,
                    "allowed_values": {"city": ["深圳市"]},
                }
            },
        )

        await complete_tool_lifecycle(
            emitter=emitter,
            tool_call_id="call-repair",
            tool_name="weather_forecast",
            result=result,
            duration_ms=0,
            result_summary_builder=lambda _result: {"kind": "weather", "truncated": False},
        )

        completed = emitter.events[0][1]
        self.assertEqual(completed["status"], "degraded")
        self.assertEqual(
            completed["result_summary"],
            {
                "kind": "weather",
                "truncated": False,
                "repair_state": "retrying",
                "repair_id": "repair_0123456789abcdef",
            },
        )
        self.assertNotIn("深圳市", str(completed))

    async def test_success_metadata_resolves_only_matching_repair(self):
        complete_tool_lifecycle = getattr(lifecycle_module, "complete_tool_lifecycle")
        emitter = RecordingEmitter()
        result = ToolResult(
            status="success",
            data={"resolves_repair_id": "repair_0123456789abcdef"},
        )

        await complete_tool_lifecycle(
            emitter=emitter,
            tool_call_id="call-resolved",
            tool_name="weather_forecast",
            result=result,
            duration_ms=10,
            result_summary_builder=lambda _result: {"kind": "weather", "truncated": False},
        )

        self.assertEqual(
            emitter.events[0][1]["result_summary"],
            {
                "kind": "weather",
                "truncated": False,
                "repair_state": "resolved",
                "resolves_repair_id": "repair_0123456789abcdef",
            },
        )

    async def test_execute_tool_with_lifecycle_emits_success_and_sets_measured_duration(self):
        emitter = RecordingEmitter()
        calls = []
        result = ToolResult(status="success", data={"items": [1]})
        args = {"query": "redis stream"}
        target = object()

        async def execute(received_target, received_args):
            calls.append(("execute", received_target, received_args))
            return result

        def build_summary(received_result):
            calls.append(("summary", received_result))
            return {"kind": "search", "count": 1}

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[10.0, 10.125]):
            returned = await execute_tool_with_lifecycle(
                tool_call_id="call-1",
                tool_name="web_search",
                args=args,
                target=target,
                execute=execute,
                result_summary_builder=build_summary,
                emitter=emitter,
            )

        self.assertIs(returned, result)
        self.assertEqual(result.duration_ms, 125)
        self.assertEqual(calls, [("execute", target, args), ("summary", result)])
        self.assertEqual(
            emitter.events,
            [
                (
                    "started",
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "web_search",
                        "arguments": args,
                    },
                ),
                (
                    "completed",
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "web_search",
                        "status": "success",
                        "duration_ms": 125,
                        "result_summary": {"kind": "search", "count": 1},
                        "error": None,
                    },
                ),
            ],
        )

    async def test_execute_tool_with_lifecycle_returns_failed_result_after_exception(self):
        emitter = RecordingEmitter()
        summaries = []

        async def execute(_target, _args):
            raise ValueError("参数非法")

        def build_summary(result):
            summaries.append(result)
            return {"kind": "search", "status": result.status}

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[20.0, 20.03]):
            result = await execute_tool_with_lifecycle(
                tool_call_id="call-2",
                tool_name="web_search",
                args={"query": "redis"},
                target=object(),
                execute=execute,
                result_summary_builder=build_summary,
                emitter=emitter,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_message, "工具执行未完成")
        self.assertEqual(
            result.data,
            {"error_code": "tool_execution_failed", "retryable": False},
        )
        self.assertEqual(result.duration_ms, None)
        self.assertEqual(len(summaries), 1)
        self.assertIs(summaries[0], result)
        self.assertEqual(emitter.events[0][0], "started")
        self.assertEqual(
            emitter.events[1],
            (
                "completed",
                {
                    "tool_call_id": "call-2",
                    "tool_name": "web_search",
                    "status": "failed",
                    "duration_ms": 30,
                    "result_summary": {"kind": "search", "status": "failed"},
                    "error": "工具执行未完成",
                },
            ),
        )

    async def test_execute_tool_with_lifecycle_sets_duration_on_returned_failed_result(self):
        emitter = RecordingEmitter()
        result = ToolResult(status="failed", error_message="reader-service 暂时不可用")
        summaries = []

        async def execute(_target, _args):
            return result

        def build_summary(received_result):
            summaries.append(received_result)
            return {"kind": "url_read", "status": received_result.status}

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[25.0, 25.125]):
            returned = await execute_tool_with_lifecycle(
                tool_call_id="call-returned-failed",
                tool_name="url_read",
                args={"url": "https://example.com"},
                target=object(),
                execute=execute,
                result_summary_builder=build_summary,
                emitter=emitter,
            )

        self.assertIs(returned, result)
        self.assertEqual(result.duration_ms, 125)
        self.assertEqual(summaries, [result])
        self.assertEqual(emitter.events[0][0], "started")
        self.assertEqual(
            emitter.events[1],
            (
                "completed",
                {
                    "tool_call_id": "call-returned-failed",
                    "tool_name": "url_read",
                    "status": "failed",
                    "duration_ms": 125,
                    "result_summary": {"kind": "url_read", "status": "failed"},
                    "error": "reader-service 暂时不可用",
                },
            ),
        )

    async def test_execute_tool_with_lifecycle_emits_failed_before_reraising_cancelled_error(self):
        emitter = RecordingEmitter()

        async def execute(_target, _args):
            raise asyncio.CancelledError("用户中止")

        def build_summary(result):
            return {"kind": "search", "status": result.status}

        with patch("app.services.stream.tool_call_lifecycle.time.monotonic", side_effect=[30.0, 30.041]):
            with self.assertRaises(asyncio.CancelledError):
                await execute_tool_with_lifecycle(
                    tool_call_id="call-3",
                    tool_name="web_search",
                    args={"query": "redis"},
                    target=object(),
                    execute=execute,
                    result_summary_builder=build_summary,
                    emitter=emitter,
                )

        self.assertEqual(emitter.events[0][0], "started")
        self.assertEqual(
            emitter.events[1],
            (
                "completed",
                {
                    "tool_call_id": "call-3",
                    "tool_name": "web_search",
                    "status": "failed",
                    "duration_ms": 41,
                    "result_summary": {"kind": "search", "status": "failed"},
                    "error": "工具执行未完成",
                },
            ),
        )

    async def test_execute_tool_with_lifecycle_without_emitter_only_executes_tool(self):
        result = ToolResult(status="success")
        calls = []

        async def execute(target, args):
            calls.append((target, args))
            return result

        def build_summary(_result):
            raise AssertionError("emitter 为空时不应构造事件摘要")

        target = object()
        args = {"query": "redis"}

        returned = await execute_tool_with_lifecycle(
            tool_call_id="call-4",
            tool_name="web_search",
            args=args,
            target=target,
            execute=execute,
            result_summary_builder=build_summary,
            emitter=None,
        )

        self.assertIs(returned, result)
        self.assertEqual(calls, [(target, args)])
        self.assertIsNone(result.duration_ms)


if __name__ == "__main__":
    unittest.main()
