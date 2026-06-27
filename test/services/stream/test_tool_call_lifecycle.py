"""tool_call 生命周期事件与执行结果状态转换测试。"""

import asyncio
import unittest
from unittest.mock import patch

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
        self.assertEqual(result.error_message, "ValueError: 参数非法")
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
                    "error": "ValueError: 参数非法",
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
                    "error": "CancelledError: 用户中止",
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
