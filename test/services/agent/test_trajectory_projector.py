"""轨迹账本读侧投影器的纯函数契约。"""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.trajectory import TrajectoryEventRecord
from app.services.agent.trajectory_projector import project_trajectory

BASE_TIME = datetime(2026, 8, 22, 1, 0, tzinfo=UTC)


def event(
    sequence: int,
    event_type: str,
    *,
    run_id: str = "run-1",
    offset_ms: int = 0,
    schema_version: int | None = 1,
    step_id: str | None = None,
    tool_call_id: str | None = None,
    parent_step_id: str | None = None,
    payload: dict | None = None,
) -> TrajectoryEventRecord:
    return TrajectoryEventRecord(
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        schema_version=schema_version,
        timestamp=BASE_TIME + timedelta(milliseconds=offset_ms),
        step_id=step_id,
        tool_call_id=tool_call_id,
        parent_step_id=parent_step_id,
        trace_id="trace-1",
        payload=payload or {},
    )


def span_by_id(projection, span_id: str):
    return next(span for span in projection.spans if span.span_id == span_id)


def record_by_sequence(projection, sequence: int):
    return next(record for record in projection.records if record.sequence == sequence)


def test_projects_paired_lifecycle_spans_with_hierarchy_timing_ttft_and_records():
    records = [
        event(0, "run_started", payload={"model": "gpt-4"}),
        event(1, "step_started", offset_ms=10, step_id="step-1", payload={"step_number": 1}),
        event(
            2,
            "tool_call_started",
            offset_ms=20,
            step_id="step-1",
            tool_call_id="tool-1",
            payload={"tool_name": "web_search"},
        ),
        event(
            3,
            "tool_attempt_started",
            offset_ms=25,
            step_id="step-1",
            tool_call_id="tool-1",
            payload={"tool_attempt_id": "attempt-1", "tool_name": "web_search", "attempt_index": 1},
        ),
        event(
            4,
            "tool_attempt_completed",
            offset_ms=45,
            step_id="step-1",
            tool_call_id="tool-1",
            payload={"tool_attempt_id": "attempt-1", "status": "success", "duration_ms": 20},
        ),
        event(
            5,
            "llm_round_started",
            offset_ms=50,
            step_id="step-1",
            parent_step_id="step-1",
            payload={"llm_round_id": "llm-1", "model": "gpt-4", "provider": "openai"},
        ),
        event(
            6,
            "llm_round_first_output_delta",
            offset_ms=80,
            step_id="step-1",
            parent_step_id="step-1",
            payload={"llm_round_id": "llm-1", "ttft_ms": 30},
        ),
        event(
            7,
            "llm_round_completed",
            offset_ms=150,
            step_id="step-1",
            parent_step_id="step-1",
            payload={"llm_round_id": "llm-1", "status": "success", "ttft_ms": 31, "duration_ms": 100},
        ),
        event(
            8,
            "retrieval_started",
            offset_ms=155,
            step_id="step-1",
            parent_step_id="step-1",
            payload={"retrieval_id": "retrieval-1", "query_summary": "安全摘要"},
        ),
        event(
            9,
            "retrieval_completed",
            offset_ms=235,
            step_id="step-1",
            parent_step_id="step-1",
            payload={"retrieval_id": "retrieval-1", "status": "success", "duration_ms": 80},
        ),
        event(
            10,
            "tool_call_completed",
            offset_ms=240,
            step_id="step-1",
            tool_call_id="tool-1",
            payload={"tool_name": "web_search", "status": "success", "duration_ms": 100},
        ),
        event(11, "step_completed", offset_ms=250, step_id="step-1", payload={"duration_ms": 240}),
        event(12, "run_completed", offset_ms=260),
    ]

    projection = project_trajectory(
        records, run_status="completed", run_ended_at=BASE_TIME + timedelta(milliseconds=260), truncated=False
    )

    assert [span.span_id for span in projection.spans] == [
        "run:run-1",
        "step:step-1",
        "tool:tool-1",
        "tool_attempt:attempt-1",
        "llm:llm-1",
        "retrieval:retrieval-1",
    ]
    assert span_by_id(projection, "step:step-1").parent_span_id == "run:run-1"
    assert span_by_id(projection, "tool:tool-1").parent_span_id == "step:step-1"
    assert span_by_id(projection, "tool_attempt:attempt-1").parent_span_id == "tool:tool-1"
    assert span_by_id(projection, "llm:llm-1").parent_span_id == "step:step-1"
    assert span_by_id(projection, "retrieval:retrieval-1").parent_span_id == "step:step-1"
    assert span_by_id(projection, "tool:tool-1").duration_ms == 100
    assert span_by_id(projection, "llm:llm-1").duration_ms == 100
    assert span_by_id(projection, "llm:llm-1").ttft_ms == 31
    assert span_by_id(projection, "retrieval:retrieval-1").duration_ms == 80
    assert span_by_id(projection, "tool_attempt:attempt-1").duration_ms == 20
    assert span_by_id(projection, "tool:tool-1").record_sequences == [2, 10]
    assert record_by_sequence(projection, 6).span_id == "llm:llm-1"
    assert record_by_sequence(projection, 9).span_id == "retrieval:retrieval-1"


def test_projects_recorded_tool_and_step_summaries_without_started_events():
    records = [
        event(0, "run_started"),
        event(
            1,
            "tool_call_completed",
            offset_ms=100,
            step_id="step-1",
            tool_call_id="tool-1",
            payload={"tool_name": "web_search", "status": "success", "duration_ms": 20},
        ),
        event(2, "step_completed", offset_ms=150, step_id="step-1", payload={"duration_ms": 30}),
        event(3, "run_completed", offset_ms=160),
    ]

    projection = project_trajectory(
        records, run_status="completed", run_ended_at=BASE_TIME + timedelta(milliseconds=160), truncated=False
    )

    tool = span_by_id(projection, "tool:tool-1")
    step = span_by_id(projection, "step:step-1")
    assert (tool.start_sequence, tool.end_sequence, tool.duration_ms, tool.terminal_source) == (1, 1, 20, "recorded")
    assert tool.started_at == BASE_TIME + timedelta(milliseconds=80)
    assert (step.start_sequence, step.end_sequence, step.duration_ms, step.terminal_source) == (2, 2, 30, "recorded")
    assert step.started_at == BASE_TIME + timedelta(milliseconds=120)


def test_annotations_do_not_create_spans_and_attach_to_most_precise_parent():
    records = [
        event(0, "run_started"),
        event(1, "step_started", step_id="step-1"),
        event(2, "tool_call_started", step_id="step-1", tool_call_id="tool-1", payload={"tool_name": "web_search"}),
        event(3, "plan_snapshot", step_id="step-1"),
        event(4, "evidence_item_upserted", step_id="step-1"),
        event(5, "context_status_updated", parent_step_id="step-1"),
        event(6, "suggested_questions_pending"),
    ]

    projection = project_trajectory(records, run_status="running", run_ended_at=None, truncated=False)

    assert [span.span_id for span in projection.spans] == ["run:run-1", "step:step-1", "tool:tool-1"]
    assert record_by_sequence(projection, 3).span_id == "step:step-1"
    assert record_by_sequence(projection, 4).span_id == "step:step-1"
    assert record_by_sequence(projection, 5).span_id == "step:step-1"
    assert record_by_sequence(projection, 6).span_id == "run:run-1"
    assert span_by_id(projection, "step:step-1").record_sequences == [1, 3, 4, 5]


@pytest.mark.parametrize(
    ("run_status", "event_type", "expected_status", "reason"),
    [
        ("completed", "run_completed", "unknown", "run_completed_without_close"),
        ("failed", "run_failed", "failed", "run_failed_without_close"),
        ("interrupted", "run_interrupted", "cancelled", "run_interrupted_without_close"),
    ],
)
def test_terminal_runs_infer_orphan_closure(run_status, event_type, expected_status, reason):
    records = [
        event(0, "run_started"),
        event(1, "step_started", offset_ms=10, step_id="step-1"),
        event(2, event_type, offset_ms=100),
    ]

    projection = project_trajectory(
        records, run_status=run_status, run_ended_at=BASE_TIME + timedelta(milliseconds=120), truncated=False
    )

    step = span_by_id(projection, "step:step-1")
    assert (step.status, step.terminal_source, step.inferred_reason) == (expected_status, "inferred", reason)
    assert step.ended_at == BASE_TIME + timedelta(milliseconds=120)


def test_truncated_prefix_never_uses_full_run_terminal_to_close_open_spans():
    records = [event(0, "run_started"), event(1, "step_started", offset_ms=10, step_id="step-1")]

    projection = project_trajectory(
        records, run_status="completed", run_ended_at=BASE_TIME + timedelta(milliseconds=999), truncated=True
    )

    step = span_by_id(projection, "step:step-1")
    assert (step.status, step.terminal_source, step.inferred_reason) == ("unknown", "inferred", "truncated_prefix")
    assert step.ended_at == BASE_TIME + timedelta(milliseconds=10)


def test_legacy_record_without_schema_version_projects_as_zero_without_mutation():
    records = [event(0, "run_started", schema_version=None, payload={"model": "gpt-4", "nested": {"value": 1}})]
    before = deepcopy([record.model_dump() for record in records])

    projection = project_trajectory(records, run_status="running", run_ended_at=None, truncated=False)

    assert projection.records[0].schema_version == 0
    assert [record.model_dump() for record in records] == before


def test_running_open_span_keeps_running_status_without_terminal_source():
    projection = project_trajectory(
        [event(0, "run_started"), event(1, "step_started", step_id="step-1")],
        run_status="running",
        run_ended_at=None,
        truncated=False,
    )

    step = span_by_id(projection, "step:step-1")
    assert (step.status, step.terminal_source, step.inferred_reason, step.ended_at) == ("running", None, None, None)
