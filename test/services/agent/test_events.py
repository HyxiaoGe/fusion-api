"""agent.events 模型测试"""

import unittest

from pydantic import ValidationError

from app.services.agent.events import (
    AgentEventBase,
    EvidenceItemUpserted,
    PlanSnapshot,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunLimitReached,
    RunProgressUpdated,
    RunStarted,
    StepCompleted,
    StepStarted,
    ToolCallCompleted,
    ToolCallDelta,
    ToolCallStarted,
    ToolResultDigest,
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


class AgentProgressV2EventModelTests(unittest.TestCase):
    def _common(self):
        return dict(run_id="r1", step_id=None, tool_call_id=None, sequence=0, trace_id="r1", ts=1.0)

    def test_run_progress_updated_requires_protocol_version_2(self):
        ev = RunProgressUpdated(
            type="run_progress_updated",
            protocol_version=2,
            phase="researching",
            label="正在搜索相关资料",
            completed_steps=1,
            total_steps=4,
            completed_tool_calls=2,
            max_tool_calls=20,
            **self._common(),
        )

        self.assertEqual(ev.protocol_version, 2)
        self.assertEqual(ev.phase, "researching")

        with self.assertRaises(ValidationError):
            RunProgressUpdated(
                type="run_progress_updated",
                protocol_version=1,
                phase="researching",
                label="正在搜索相关资料",
                **self._common(),
            )

    def test_plan_snapshot_forbids_unknown_fields(self):
        with self.assertRaises(ValidationError):
            PlanSnapshot(
                type="plan_snapshot",
                protocol_version=2,
                plan_id="plan-r1",
                revision=1,
                items=[],
                unexpected=True,
                **self._common(),
            )

    def test_tool_result_digest_model(self):
        ev = ToolResultDigest(
            type="tool_result_digest",
            protocol_version=2,
            step_id="s1",
            tool_call_id="tc1",
            tool_name="web_search",
            status="success",
            title="找到 2 条结果",
            summary="优先保留官方来源。",
            key_findings=["官方页面确认发布时间。"],
            source_refs=["ev-1"],
            truncated=False,
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )

        self.assertEqual(ev.tool_call_id, "tc1")
        self.assertEqual(ev.key_findings, ["官方页面确认发布时间。"])

    def test_evidence_item_upserted_model(self):
        ev = EvidenceItemUpserted(
            type="evidence_item_upserted",
            protocol_version=2,
            step_id="s1",
            tool_call_id="tc1",
            evidence={
                "id": "ev-1",
                "kind": "web",
                "status": "candidate",
                "title": "官方发布页",
                "url": "https://example.com/news",
                "domain": "example.com",
                "claim": "官方发布页确认发布时间。",
                "snippet": "页面摘要。",
                "used_by_final_answer": False,
            },
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )

        self.assertEqual(ev.evidence.id, "ev-1")
        self.assertEqual(ev.evidence.kind, "web")


if __name__ == "__main__":
    unittest.main()
