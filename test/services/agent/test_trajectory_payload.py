import unittest

from app.services.agent.trajectory_payload import (
    MAX_LEDGER_LIST_ITEMS,
    MAX_LEDGER_TEXT_LENGTH,
    UnsupportedTrajectoryEventError,
    build_trajectory_payload,
)

COMMON = {
    "schema_version": 1,
    "run_id": "run-1",
    "parent_run_id": None,
    "step_id": "step-1",
    "parent_step_id": None,
    "tool_call_id": "call-1",
    "sequence": 3,
    "trace_id": "trace-1",
    "ts": 1_700_000_000.25,
}

COMMON_KEYS = set(COMMON) | {"type"}

EVENT_FIELDS = {
    "run_started": {
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "task_id": "task-1",
        "model": "gpt-4",
        "tools": ["web_search"],
        "config": {"prompt": "禁止落库"},
    },
    "step_started": {"step_number": 1},
    "tool_call_started": {
        "tool_name": "web_search",
        "arguments": {"query": "完整工具输入"},
        "plan_item_id": "plan-item-1",
    },
    "tool_call_delta": {"tool_name": "web_search", "delta": {"arguments": "完整工具增量"}},
    "tool_call_completed": {
        "tool_name": "web_search",
        "status": "success",
        "duration_ms": 20,
        "result_summary": {"content": "完整工具输出"},
        "error": None,
        "plan_item_id": "plan-item-1",
    },
    "step_completed": {"step_number": 1, "tool_call_count": 1, "duration_ms": 30},
    "run_limit_reached": {"reason": "max_steps"},
    "run_interrupted": {"reason": "user_cancelled"},
    "run_failed": {"error_code": "upstream_error", "message": "安全错误摘要"},
    "run_completed": {"total_steps": 1, "total_tool_calls": 1, "finish_reason": "stop"},
    "llm_round_started": {
        "llm_round_id": "round-1",
        "round_index": 1,
        "model": "gpt-4",
        "provider": "openai",
        "system_prompt_fingerprint": "b" * 64,
    },
    "system_prompt_prepared": {
        "protocol_version": 2,
        "status": "ready",
        "source": "code",
        "template_version": "1",
        "section_ids": ["core", "current_date", "user_preferences"],
        "fingerprint": "a" * 64,
        "char_count": 300,
        "duration_ms": 1,
        "error_code": None,
        "message": None,
    },
    "llm_round_first_output_delta": {
        "llm_round_id": "round-1",
        "delta_kind": "content",
        "ttft_ms": 45,
    },
    "llm_round_completed": {
        "llm_round_id": "round-1",
        "status": "success",
        "finish_reason": "stop",
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "cache_read_tokens": 2,
        "cache_write_tokens": 1,
        "ttft_ms": 45,
        "duration_ms": 100,
    },
    "llm_round_failed": {
        "llm_round_id": "round-1",
        "status": "failed",
        "error_code": "provider_error",
        "message": "安全错误摘要",
    },
    "llm_round_cancelled": {
        "llm_round_id": "round-1",
        "status": "cancelled",
        "reason": "shutdown",
    },
    "retrieval_started": {"retrieval_id": "retrieval-1", "query_summary": "安全查询摘要"},
    "retrieval_completed": {
        "retrieval_id": "retrieval-1",
        "status": "success",
        "document_count": 2,
        "duration_ms": 80,
    },
    "retrieval_failed": {
        "retrieval_id": "retrieval-1",
        "status": "failed",
        "error_code": "retrieval_error",
        "message": "安全错误摘要",
    },
    "retrieval_cancelled": {
        "retrieval_id": "retrieval-1",
        "status": "cancelled",
        "reason": "superseded",
    },
    "tool_attempt_started": {
        "tool_attempt_id": "attempt-1",
        "tool_name": "web_search",
        "attempt_index": 1,
    },
    "tool_attempt_completed": {
        "tool_attempt_id": "attempt-1",
        "status": "success",
        "error_code": None,
        "duration_ms": 50,
    },
    "suggested_questions_pending": {
        "protocol_version": 2,
        "message_id": "msg-1",
        "revision": 1,
        "status": "pending",
    },
    "run_progress_updated": {
        "protocol_version": 2,
        "phase": "thinking",
        "label": "正在分析",
        "completed_steps": 1,
        "total_steps": 2,
        "completed_tool_calls": 1,
        "max_tool_calls": 5,
    },
    "plan_snapshot": {
        "protocol_version": 2,
        "plan_id": "plan-1",
        "mode": "auto",
        "source": "observed",
        "revision": 1,
        "reason": "initial",
        "items": [
            {
                "id": "item-1",
                "title": "检索资料",
                "phase_id": "phase-1",
                "phase_title": "研究",
                "status": "running",
                "kind": "search",
                "summary": "摘要",
                "tool_names": ["web_search"],
                "evidence_item_ids": ["evidence-1"],
                "depends_on": [],
                "planned_tools": ["web_search"],
            }
        ],
    },
    "plan_step_updated": {
        "protocol_version": 2,
        "plan_id": "plan-1",
        "mode": "auto",
        "source": "observed",
        "revision": 2,
        "reason": "tool_finished",
        "item": {
            "id": "item-1",
            "title": "检索资料",
            "status": "completed",
            "kind": "search",
            "summary": "完成",
            "tool_names": ["web_search"],
        },
    },
    "tool_result_digest": {
        "protocol_version": 2,
        "tool_name": "web_search",
        "status": "success",
        "title": "检索完成",
        "summary": "安全摘要",
        "key_findings": ["事实一"],
        "source_refs": ["evidence-1"],
        "truncated": False,
        "repair_state": None,
        "repair_id": None,
        "plan_item_id": "item-1",
    },
    "evidence_item_upserted": {
        "protocol_version": 2,
        "evidence": {
            "id": "evidence-1",
            "kind": "web",
            "status": "used",
            "title": "资料",
            "url": "https://example.com/report?token=secret#fragment",
            "domain": "example.com",
            "claim": "安全结论",
            "snippet": "安全摘录",
            "used_by_final_answer": True,
            "citation_index": 1,
        },
    },
    "content_block_upserted": {
        "protocol_version": 2,
        "content_block": {"type": "text", "text": "完整 content 禁止落库"},
    },
    "content_block_discarded": {"protocol_version": 2, "block_id": "block-1"},
    "context_status_updated": {
        "protocol_version": 2,
        "message_id": "msg-1",
        "phase": "final",
        "status": "compacted",
        "round_index": 1,
        "window_tokens": 100,
        "estimated_tokens_before": 120,
        "estimated_tokens_after": 80,
        "actual_prompt_tokens": 82,
        "removed_turns": 1,
        "removed_messages": 2,
        "removed_tool_transactions": 1,
    },
    "context_required": {
        "protocol_version": 2,
        "context_type": "geolocation",
        "request_id": "request-1",
        "purpose": "nearby_search",
        "reason": "需要附近结果",
        "expires_at": 1_700_000_060.0,
    },
    "context_result": {
        "protocol_version": 2,
        "context_type": "geolocation",
        "request_id": "request-1",
        "status": "provided",
    },
}

EVENT_ALLOWED_FIELDS = {
    "run_started": {"conversation_id", "message_id", "task_id", "model", "tools"},
    "step_started": {"step_number"},
    "tool_call_started": {"tool_name", "plan_item_id"},
    "tool_call_delta": {"tool_name"},
    "tool_call_completed": {"tool_name", "status", "duration_ms", "plan_item_id"},
    "step_completed": {"step_number", "tool_call_count", "duration_ms"},
    "run_limit_reached": {"reason"},
    "run_interrupted": {"reason"},
    "run_failed": {"error_code", "message"},
    "run_completed": {"total_steps", "total_tool_calls", "finish_reason"},
    "llm_round_started": {"llm_round_id", "round_index", "model", "provider", "system_prompt_fingerprint"},
    "system_prompt_prepared": {
        "protocol_version",
        "status",
        "source",
        "template_version",
        "section_ids",
        "fingerprint",
        "char_count",
        "duration_ms",
        "error_code",
        "message",
    },
    "llm_round_first_output_delta": {"llm_round_id", "delta_kind", "ttft_ms"},
    "llm_round_completed": {
        "llm_round_id",
        "status",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "ttft_ms",
        "duration_ms",
    },
    "llm_round_failed": {"llm_round_id", "status", "error_code", "message"},
    "llm_round_cancelled": {"llm_round_id", "status", "reason"},
    "retrieval_started": {"retrieval_id", "query_summary"},
    "retrieval_completed": {"retrieval_id", "status", "document_count", "duration_ms"},
    "retrieval_failed": {"retrieval_id", "status", "error_code", "message"},
    "retrieval_cancelled": {"retrieval_id", "status", "reason"},
    "tool_attempt_started": {"tool_attempt_id", "tool_name", "attempt_index"},
    "tool_attempt_completed": {"tool_attempt_id", "status", "error_code", "duration_ms"},
    "suggested_questions_pending": {"protocol_version", "message_id", "revision", "status"},
    "run_progress_updated": {
        "protocol_version",
        "phase",
        "label",
        "completed_steps",
        "total_steps",
        "completed_tool_calls",
        "max_tool_calls",
    },
    "plan_snapshot": {"protocol_version", "plan_id", "mode", "source", "revision", "reason", "items"},
    "plan_step_updated": {"protocol_version", "plan_id", "mode", "source", "revision", "reason", "item"},
    "tool_result_digest": {
        "protocol_version",
        "tool_name",
        "status",
        "title",
        "summary",
        "key_findings",
        "source_refs",
        "truncated",
        "repair_state",
        "repair_id",
        "plan_item_id",
    },
    "evidence_item_upserted": {"protocol_version", "evidence"},
    "content_block_upserted": {"protocol_version"},
    "content_block_discarded": {"protocol_version", "block_id"},
    "context_status_updated": {
        "protocol_version",
        "message_id",
        "phase",
        "status",
        "round_index",
        "window_tokens",
        "estimated_tokens_before",
        "estimated_tokens_after",
        "actual_prompt_tokens",
        "removed_turns",
        "removed_messages",
        "removed_tool_transactions",
    },
    "context_required": {
        "protocol_version",
        "context_type",
        "request_id",
        "purpose",
        "reason",
        "expires_at",
    },
    "context_result": {"protocol_version", "context_type", "request_id", "status"},
}


def _assert_event_type_uses_an_explicit_top_level_allowlist(event_type):
    payload = {
        **COMMON,
        "type": event_type,
        **EVENT_FIELDS[event_type],
        "prompt": "system prompt",
        "tool_schema": {"parameters": {"password": "secret"}},
        "unexpected": "禁止落库",
    }

    stored = build_trajectory_payload(payload)

    assert set(stored) == COMMON_KEYS | EVENT_ALLOWED_FIELDS[event_type]
    assert "prompt" not in stored
    assert "tool_schema" not in stored
    assert "unexpected" not in stored


def _assert_nested_payloads_drop_full_inputs_outputs_and_strip_url_query():
    started = build_trajectory_payload({**COMMON, "type": "tool_call_started", **EVENT_FIELDS["tool_call_started"]})
    completed = build_trajectory_payload(
        {**COMMON, "type": "tool_call_completed", **EVENT_FIELDS["tool_call_completed"]}
    )
    content = build_trajectory_payload(
        {**COMMON, "type": "content_block_upserted", **EVENT_FIELDS["content_block_upserted"]}
    )
    evidence = build_trajectory_payload(
        {**COMMON, "type": "evidence_item_upserted", **EVENT_FIELDS["evidence_item_upserted"]}
    )

    assert "arguments" not in started
    assert "result_summary" not in completed
    assert "content_block" not in content
    assert evidence["evidence"]["url"] == "https://example.com/report"
    assert set(evidence["evidence"]) == {
        "id",
        "kind",
        "status",
        "title",
        "url",
        "domain",
        "claim",
        "snippet",
        "used_by_final_answer",
        "citation_index",
    }


def _assert_evidence_url_drops_embedded_credentials_as_well_as_query_and_fragment():
    payload = {
        **COMMON,
        "type": "evidence_item_upserted",
        **EVENT_FIELDS["evidence_item_upserted"],
    }
    payload["evidence"] = {
        **payload["evidence"],
        "url": "https://alice:password@example.com/report?api_key=secret#details",
    }

    stored = build_trajectory_payload(payload)

    assert stored["evidence"]["url"] == "https://example.com/report"


def _assert_text_and_lists_are_bounded_and_secret_like_error_text_is_redacted():
    long_text = "x" * (MAX_LEDGER_TEXT_LENGTH + 50)
    error = build_trajectory_payload(
        {
            **COMMON,
            "type": "run_failed",
            "error_code": "provider_error",
            "message": f"请求失败 api_key=sk-super-secret {long_text}",
        }
    )
    digest = build_trajectory_payload(
        {
            **COMMON,
            "type": "tool_result_digest",
            **EVENT_FIELDS["tool_result_digest"],
            "key_findings": [long_text] * (MAX_LEDGER_LIST_ITEMS + 3),
        }
    )

    assert "sk-super-secret" not in error["message"]
    assert "[REDACTED]" in error["message"]
    assert len(error["message"]) <= MAX_LEDGER_TEXT_LENGTH
    assert len(digest["key_findings"]) == MAX_LEDGER_LIST_ITEMS
    assert all(len(item) <= MAX_LEDGER_TEXT_LENGTH for item in digest["key_findings"])


class TrajectoryPayloadTests(unittest.TestCase):
    def test_every_event_type_uses_an_explicit_top_level_allowlist(self):
        for event_type in sorted(EVENT_FIELDS):
            with self.subTest(event_type=event_type):
                _assert_event_type_uses_an_explicit_top_level_allowlist(event_type)

    def test_nested_payloads_drop_full_inputs_outputs_and_strip_url_query(self):
        _assert_nested_payloads_drop_full_inputs_outputs_and_strip_url_query()

    def test_evidence_url_drops_embedded_credentials_as_well_as_query_and_fragment(self):
        _assert_evidence_url_drops_embedded_credentials_as_well_as_query_and_fragment()

    def test_text_and_lists_are_bounded_and_secret_like_error_text_is_redacted(self):
        _assert_text_and_lists_are_bounded_and_secret_like_error_text_is_redacted()

    def test_unknown_event_type_is_rejected_instead_of_storing_raw_payload(self):
        with self.assertRaises(UnsupportedTrajectoryEventError):
            build_trajectory_payload({**COMMON, "type": "future_event", "secret": "raw"})
