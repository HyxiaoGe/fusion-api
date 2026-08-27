"""agent.events 模型测试"""

import unittest

from pydantic import TypeAdapter, ValidationError

from app.services.agent.events import (
    AgentEventBase,
    AnyAgentEvent,
    ContentBlockDiscarded,
    ContextRequired,
    ContextResult,
    EvidenceItemUpserted,
    LLMRoundCompleted,
    LLMRoundFirstOutputDelta,
    LLMRoundStarted,
    PlanSnapshot,
    RetrievalCompleted,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunLimitReached,
    RunProgressUpdated,
    RunStarted,
    StepCompleted,
    StepStarted,
    SuggestedQuestionsPending,
    ToolAttemptCompleted,
    ToolAttemptStarted,
    ToolCallCompleted,
    ToolCallDelta,
    ToolCallStarted,
    ToolResultDigest,
)

CAPABILITY_RESOLUTION = {
    "schema_version": 1,
    "router_version": "2026-08-27.1",
    "package_id": "fresh_web",
    "confidence": "high",
    "resolution_mode": "routed",
    "reason_codes": ["fresh_external_fact"],
    "external_tool_names": ["web_search"],
    "effective_plan_mode": "off",
    "include_current_date": True,
    "network_boundary_required": False,
    "bundle_fingerprint": "sha256:" + "a" * 64,
}


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
            task_id="task-1",
            model="gpt",
            tools=["web_search"],
            config={"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300},
            capability_resolution=CAPABILITY_RESOLUTION,
            **self._common(),
        )
        self.assertEqual(ev.type, "run_started")
        self.assertEqual(ev.tools, ["web_search"])
        self.assertEqual(ev.message_id, "msg-1")
        self.assertEqual(ev.task_id, "task-1")
        self.assertEqual(ev.capability_resolution.model_dump(), CAPABILITY_RESOLUTION)

    def test_run_started_rejects_unsafe_or_invalid_capability_resolution(self):
        base = {
            "type": "run_started",
            "conversation_id": "c1",
            "message_id": "msg-1",
            "task_id": "task-1",
            "model": "gpt",
            "tools": ["web_search"],
            "config": {},
            **self._common(),
        }
        with self.assertRaises(ValidationError):
            RunStarted(
                **base,
                capability_resolution={
                    **CAPABILITY_RESOLUTION,
                    "original_message": "用户原文禁止进入事件协议",
                },
            )
        with self.assertRaises(ValidationError):
            RunStarted(
                **base,
                capability_resolution={
                    **CAPABILITY_RESOLUTION,
                    "bundle_fingerprint": "a" * 64,
                },
            )

    def test_run_started_tools_must_match_external_resolution_without_update_plan(self):
        with self.assertRaises(ValidationError):
            RunStarted(
                type="run_started",
                conversation_id="c1",
                message_id="msg-1",
                task_id="task-1",
                model="gpt",
                tools=["web_search", "update_plan"],
                config={"capability_resolution": CAPABILITY_RESOLUTION},
                capability_resolution=CAPABILITY_RESOLUTION,
                **self._common(),
            )

    def test_run_started_message_id_required(self):
        """RunStarted 缺 message_id 必须抛 ValidationError"""
        with self.assertRaises(ValidationError):
            RunStarted(
                type="run_started",
                conversation_id="c1",
                task_id="task-1",
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

    def test_system_prompt_result_rejects_loading_and_full_prompt_payload(self):
        adapter = TypeAdapter(AnyAgentEvent)
        payload = {
            **self._common(), "type": "system_prompt_prepared", "protocol_version": 2,
            "status": "ready", "source": "code", "template_version": "1", "section_ids": ["app_identity"],
            "fingerprint": "a" * 64, "char_count": 200, "duration_ms": 0,
        }
        self.assertEqual(adapter.validate_python(payload).status, "ready")
        self.assertEqual(adapter.validate_python({**payload, "status": "failed"}).status, "failed")
        with self.assertRaises(ValidationError):
            adapter.validate_python({**payload, "status": "loading"})
        with self.assertRaises(ValidationError):
            adapter.validate_python({**payload, "prompt": "不能暴露的规则"})

    def test_existing_event_defaults_to_schema_version_1(self):
        event = StepStarted(type="step_started", step_number=1, **self._common())

        self.assertEqual(event.model_dump()["schema_version"], 1)

    def test_new_lifecycle_events_are_discriminated_by_type(self):
        payloads = [
            {"type": "llm_round_started", "llm_round_id": "llm-1", "round_index": 1, "model": "deepseek/deepseek-chat", "provider": "deepseek"},
            {"type": "llm_round_first_output_delta", "llm_round_id": "llm-1", "delta_kind": "content", "ttft_ms": 12},
            {"type": "llm_round_completed", "llm_round_id": "llm-1", "status": "success", "finish_reason": "stop", "input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "cache_read_tokens": None, "cache_write_tokens": None, "ttft_ms": 12, "duration_ms": 15},
            {"type": "llm_round_failed", "llm_round_id": "llm-1", "status": "failed", "error_code": "timeout", "message": "已脱敏错误摘要"},
            {"type": "llm_round_cancelled", "llm_round_id": "llm-1", "status": "cancelled", "reason": "user_cancelled"},
            {"type": "retrieval_started", "retrieval_id": "ret-1", "query_summary": "查天气"},
            {"type": "retrieval_completed", "retrieval_id": "ret-1", "status": "success", "document_count": 2, "duration_ms": 11},
            {"type": "retrieval_failed", "retrieval_id": "ret-1", "status": "failed", "error_code": "timeout", "message": "已脱敏错误摘要"},
            {"type": "retrieval_cancelled", "retrieval_id": "ret-1", "status": "cancelled", "reason": "shutdown"},
            {"type": "tool_attempt_started", "tool_attempt_id": "attempt-1", "tool_call_id": "tool-1", "tool_name": "web_search", "attempt_index": 1},
            {"type": "tool_attempt_completed", "tool_attempt_id": "attempt-1", "status": "timeout", "error_code": "timeout", "duration_ms": 12},
        ]
        adapter = TypeAdapter(AnyAgentEvent)

        parsed = [adapter.validate_python({**self._common(), **payload}) for payload in payloads]

        self.assertEqual([event.type for event in parsed], [payload["type"] for payload in payloads])
        self.assertTrue(all(event.schema_version == 1 for event in parsed))

    def test_new_lifecycle_events_reject_invalid_boundaries_and_statuses(self):
        with self.assertRaises(ValidationError):
            LLMRoundStarted(type="llm_round_started", llm_round_id="llm-1", round_index=0, model="m", provider="p", **self._common())
        with self.assertRaises(ValidationError):
            LLMRoundFirstOutputDelta(
                type="llm_round_first_output_delta", llm_round_id="llm-1", delta_kind="content", ttft_ms=-1, **self._common()
            )
        with self.assertRaises(ValidationError):
            LLMRoundCompleted(
                type="llm_round_completed", llm_round_id="llm-1", status="success", finish_reason=None,
                input_tokens=-1, output_tokens=0, total_tokens=0, cache_read_tokens=None,
                cache_write_tokens=None, ttft_ms=None, duration_ms=0, **self._common()
            )
        with self.assertRaises(ValidationError):
            RetrievalCompleted(
                type="retrieval_completed", retrieval_id="ret-1", status="success", document_count=0,
                duration_ms=-1, **self._common()
            )
        with self.assertRaises(ValidationError):
            ToolAttemptStarted(
                type="tool_attempt_started", tool_attempt_id="attempt-1", tool_call_id="tool-1",
                tool_name="web_search", attempt_index=0,
                **{key: value for key, value in self._common().items() if key != "tool_call_id"},
            )
        with self.assertRaises(ValidationError):
            ToolAttemptCompleted(
                type="tool_attempt_completed", tool_attempt_id="attempt-1", status="unknown",
                error_code=None, duration_ms=0, **self._common()
            )

    def test_content_block_discarded_requires_explicit_block_id(self):
        event = ContentBlockDiscarded(
            type="content_block_discarded",
            protocol_version=2,
            block_id="blk-tool-preamble",
            **self._common(),
        )

        self.assertEqual(event.block_id, "blk-tool-preamble")
        with self.assertRaises(ValidationError):
            ContentBlockDiscarded(
                type="content_block_discarded",
                protocol_version=2,
                **self._common(),
            )

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            StepStarted(type="step_started", step_number=1, bogus_field="x", **self._common())

    def test_geolocation_context_events_have_strict_safe_contract(self):
        required = ContextRequired(
            type="context_required",
            protocol_version=2,
            context_type="geolocation",
            request_id="ctx-1",
            purpose="nearby_search",
            reason="搜索当前位置附近的地点",
            expires_at=123.5,
            **self._common(),
        )
        result = ContextResult(
            type="context_result",
            protocol_version=2,
            context_type="geolocation",
            request_id="ctx-1",
            status="provided",
            **self._common(),
        )

        self.assertEqual(required.purpose, "nearby_search")
        self.assertEqual(result.status, "provided")
        self.assertNotIn("location", result.model_dump())
        with self.assertRaises(ValidationError):
            ContextRequired(
                type="context_required",
                protocol_version=2,
                context_type="geolocation",
                request_id="ctx-1",
                purpose="bogus",
                reason="x",
                expires_at=123.5,
                **self._common(),
            )

    def test_suggested_questions_pending_has_strict_revision_contract(self):
        event = SuggestedQuestionsPending(
            type="suggested_questions_pending",
            protocol_version=2,
            message_id="msg-1",
            revision=3,
            status="pending",
            **self._common(),
        )

        self.assertEqual(
            event.model_dump(),
            {
                **self._common(),
                "schema_version": 1,
                "parent_run_id": None,
                "parent_step_id": None,
                "type": "suggested_questions_pending",
                "protocol_version": 2,
                "message_id": "msg-1",
                "revision": 3,
                "status": "pending",
            },
        )
        with self.assertRaises(ValidationError):
            SuggestedQuestionsPending(
                type="suggested_questions_pending",
                protocol_version=2,
                message_id="msg-1",
                revision=0,
                status="pending",
                **self._common(),
            )
        with self.assertRaises(ValidationError):
            SuggestedQuestionsPending(
                type="suggested_questions_pending",
                protocol_version=2,
                message_id="msg-1",
                revision=3,
                status="ready",
                **self._common(),
            )


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

    def test_plan_snapshot_supports_model_plan_metadata_and_item_linkage(self):
        event = PlanSnapshot(
            type="plan_snapshot",
            protocol_version=2,
            plan_id="plan-r1",
            mode="on",
            source="model",
            revision=2,
            reason="model_update",
            items=[
                {
                    "id": "route",
                    "title": "查询路线",
                    "phase_id": "phase-route",
                    "phase_title": "查询路线",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["route_compare"],
                }
            ],
            **self._common(),
        )

        self.assertEqual(event.source, "model")
        self.assertEqual(event.items[0].planned_tools, ["route_compare"])
        self.assertEqual(event.items[0].phase_id, "phase-route")
        self.assertEqual(event.items[0].phase_title, "查询路线")

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
            repair_state="resolved",
            repair_id="repair_0123456789abcdef",
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )

        self.assertEqual(ev.tool_call_id, "tc1")
        self.assertEqual(ev.key_findings, ["官方页面确认发布时间。"])
        self.assertEqual(ev.repair_state, "resolved")

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

    def test_evidence_item_upserted_accepts_read_lifecycle_status(self):
        ev = EvidenceItemUpserted(
            type="evidence_item_upserted",
            protocol_version=2,
            step_id="s1",
            tool_call_id="tc1",
            evidence={
                "id": "ev-1",
                "kind": "web",
                "status": "read_success",
                "title": "官方发布页",
                "url": "https://example.com/news",
                "domain": "example.com",
                "claim": "官方发布页已读取。",
                "snippet": "页面摘要。",
                "used_by_final_answer": False,
            },
            **{k: v for k, v in self._common().items() if k not in {"step_id", "tool_call_id"}},
        )

        self.assertEqual(ev.evidence.status, "read_success")


if __name__ == "__main__":
    unittest.main()
