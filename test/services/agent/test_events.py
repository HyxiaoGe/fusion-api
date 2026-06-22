"""agent.events 模型测试"""

import unittest

from pydantic import ValidationError

from app.services.agent.events import (
    AgentEventBase,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunLimitReached,
    RunStarted,
    StepCompleted,
    StepStarted,
    ToolCallCompleted,
    ToolCallDelta,
    ToolCallStarted,
)


class AgentEventModelTests(unittest.TestCase):
    def _common(self):
        return dict(run_id="r1", step_id="s1", tool_call_id=None, sequence=0, trace_id="t1", ts=1.0)

    def test_envelope_required_fields(self):
        with self.assertRaises(ValidationError):
            RunStarted(type="run_started", model="m", tools=[], config={})

    def test_run_started_payload(self):
        ev = RunStarted(
            type="run_started",
            conversation_id="c1",
            message_id="msg-1",
            model="gpt",
            tools=["web_search"],
            config={"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300},
            **self._common(),
        )
        self.assertEqual(ev.type, "run_started")
        self.assertEqual(ev.tools, ["web_search"])
        self.assertEqual(ev.message_id, "msg-1")

    def test_run_started_message_id_required(self):
        """RunStarted 缺 message_id 必须抛 ValidationError"""
        with self.assertRaises(ValidationError):
            RunStarted(
                type="run_started",
                conversation_id="c1",
                # message_id 漏了
                model="gpt",
                tools=[],
                config={},
                **self._common(),
            )

    def test_tool_call_completed_status_enum(self):
        ev = ToolCallCompleted(
            type="tool_call_completed",
            tool_name="web_search",
            status="success",
            duration_ms=12,
            result_summary={"kind": "search", "truncated": False},
            **self._common(),
        )
        self.assertEqual(ev.status, "success")
        with self.assertRaises(ValidationError):
            ToolCallCompleted(
                type="tool_call_completed",
                tool_name="x",
                status="bogus",
                duration_ms=1,
                result_summary={},
                **self._common(),
            )

    def test_run_limit_reached_reason_enum(self):
        for r in ("max_steps", "max_tool_calls", "timeout"):
            RunLimitReached(type="run_limit_reached", reason=r, **self._common())
        with self.assertRaises(ValidationError):
            RunLimitReached(type="run_limit_reached", reason="bogus", **self._common())

    def test_run_completed_finish_reason_enum(self):
        for fr in ("stop", "limit_reached", "incomplete"):
            RunCompleted(type="run_completed", total_steps=1, total_tool_calls=0, finish_reason=fr, **self._common())

    def test_all_events_serialize_to_dict(self):
        ev = StepStarted(type="step_started", step_number=1, **self._common())
        d = ev.model_dump()
        self.assertEqual(d["sequence"], 0)
        self.assertEqual(d["run_id"], "r1")

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            StepStarted(type="step_started", step_number=1, bogus_field="x", **self._common())


class AgentEventExportTests(unittest.TestCase):
    def test_all_event_classes_importable(self):
        """smoke: 11 个公开类（AgentEventBase + 10 个事件）全部成功导入"""
        classes = [
            AgentEventBase,
            RunStarted,
            StepStarted,
            ToolCallStarted,
            ToolCallDelta,
            ToolCallCompleted,
            StepCompleted,
            RunLimitReached,
            RunInterrupted,
            RunFailed,
            RunCompleted,
        ]
        self.assertEqual(len(classes), 11)
        # 同时校验每个类都是 AgentEventBase 子类（除了 base 本身）
        for cls in classes[1:]:
            self.assertTrue(issubclass(cls, AgentEventBase))


if __name__ == "__main__":
    unittest.main()
