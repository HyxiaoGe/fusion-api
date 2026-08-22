"""将有序脱敏账本事件投影为稳定的轨迹 span。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.schemas.trajectory import (
    TrajectoryEventRecord,
    TrajectoryProjection,
    TrajectoryRecord,
    TrajectorySpan,
)

_RUN_START = "run_started"
_RUN_TERMINALS = frozenset({"run_completed", "run_failed", "run_interrupted", "run_limit_reached"})
_STEP_START = "step_started"
_STEP_TERMINAL = "step_completed"
_TOOL_START = "tool_call_started"
_TOOL_TERMINAL = "tool_call_completed"
_LLM_START = "llm_round_started"
_LLM_TERMINALS = frozenset({"llm_round_completed", "llm_round_failed", "llm_round_cancelled"})
_RETRIEVAL_START = "retrieval_started"
_RETRIEVAL_TERMINALS = frozenset({"retrieval_completed", "retrieval_failed", "retrieval_cancelled"})
_ATTEMPT_START = "tool_attempt_started"
_ATTEMPT_TERMINAL = "tool_attempt_completed"


@dataclass
class _SpanBuilder:
    span_id: str
    kind: str
    name: str
    parent_span_id: str | None
    parent_span_candidates: tuple[str, ...]
    start_sequence: int
    started_at: datetime
    end_sequence: int | None = None
    ended_at: datetime | None = None
    duration_ms: int | None = None
    status: str = "running"
    terminal_source: str | None = None
    inferred_reason: str | None = None
    ttft_ms: int | None = None
    record_sequences: list[int] = field(default_factory=list)

    def to_dto(self) -> TrajectorySpan:
        return TrajectorySpan(
            span_id=self.span_id,
            kind=self.kind,
            name=self.name,
            parent_span_id=self.parent_span_id,
            start_sequence=self.start_sequence,
            end_sequence=self.end_sequence,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_ms=self.duration_ms,
            status=self.status,
            terminal_source=self.terminal_source,
            inferred_reason=self.inferred_reason,
            ttft_ms=self.ttft_ms,
            record_sequences=self.record_sequences,
        )


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _duration_ms(started_at: datetime, ended_at: datetime) -> int:
    return max(0, int((ended_at - started_at).total_seconds() * 1000))


def _terminal_status(event_type: str, payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, str):
        return status
    if event_type.endswith("_completed") or event_type == "run_completed":
        return "success"
    if event_type.endswith("_failed") or event_type == "run_failed":
        return "failed"
    return "cancelled"


def _orphan_outcome(run_status: str) -> tuple[str, str] | None:
    normalized = run_status.lower()
    if normalized in {"complete", "completed", "incomplete", "success"}:
        return "unknown", "run_completed_without_close"
    if normalized in {"failed", "error"}:
        return "failed", "run_failed_without_close"
    if normalized in {"interrupted", "cancelled", "canceled", "limit_reached"}:
        return "cancelled", "run_interrupted_without_close"
    return None


def _merge_sequences(first: list[int], second: list[int]) -> list[int]:
    """在线性时间内合并两个按 sequence 升序的关联列表。"""

    merged: list[int] = []
    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        if first[first_index] <= second[second_index]:
            merged.append(first[first_index])
            first_index += 1
        else:
            merged.append(second[second_index])
            second_index += 1
    merged.extend(first[first_index:])
    merged.extend(second[second_index:])
    return merged


def project_trajectory(
    records: list[TrajectoryEventRecord],
    *,
    run_status: str,
    run_ended_at: datetime | None,
    truncated: bool,
) -> TrajectoryProjection:
    """单遍投影有序账本；不访问数据库，也不修改输入事件。"""

    builders: dict[str, _SpanBuilder] = {}
    ordered_span_ids: list[str] = []
    projected_records: list[TrajectoryRecord] = []
    pending_record_associations: list[tuple[TrajectoryRecord, tuple[str, ...]]] = []
    deferred_record_sequences: dict[str, list[int]] = {}

    def ensure_span(
        span_id: str,
        *,
        kind: str,
        name: str,
        parent_span_candidates: tuple[str, ...],
        record: TrajectoryEventRecord,
        started_at: datetime | None = None,
    ) -> _SpanBuilder:
        builder = builders.get(span_id)
        if builder is None:
            builder = _SpanBuilder(
                span_id=span_id,
                kind=kind,
                name=name,
                parent_span_id=parent_span_candidates[0] if parent_span_candidates else None,
                parent_span_candidates=parent_span_candidates,
                start_sequence=record.sequence,
                started_at=started_at or record.timestamp,
            )
            builders[span_id] = builder
            ordered_span_ids.append(span_id)
        return builder

    def ensure_run(record: TrajectoryEventRecord) -> _SpanBuilder:
        return ensure_span(
            f"run:{record.run_id}",
            kind="run",
            name=record.run_id,
            parent_span_candidates=(),
            record=record,
        )

    def step_or_run_candidates(record: TrajectoryEventRecord) -> tuple[str, ...]:
        step_id = record.parent_step_id or record.step_id
        run_span_id = f"run:{record.run_id}"
        return (f"step:{step_id}", run_span_id) if step_id is not None else (run_span_id,)

    def tool_step_or_run_candidates(record: TrajectoryEventRecord) -> tuple[str, ...]:
        candidates: list[str] = []
        if record.tool_call_id is not None:
            candidates.append(f"tool:{record.tool_call_id}")
        candidates.extend(step_or_run_candidates(record))
        return tuple(candidates)

    def append_record(
        record: TrajectoryEventRecord,
        span_id: str | None,
        fallback_span_candidates: tuple[str, ...],
    ) -> None:
        projected_record = TrajectoryRecord(
            sequence=record.sequence,
            event_type=record.event_type,
            schema_version=record.schema_version or 0,
            timestamp=record.timestamp,
            step_id=record.step_id,
            tool_call_id=record.tool_call_id,
            parent_step_id=record.parent_step_id,
            trace_id=record.trace_id,
            span_id=span_id,
            payload=deepcopy(record.payload),
        )
        projected_records.append(projected_record)
        if span_id is not None:
            builders[span_id].record_sequences.append(record.sequence)
        else:
            pending_record_associations.append((projected_record, fallback_span_candidates))

    for record in records:
        run = ensure_run(record)
        payload = record.payload
        event_type = record.event_type
        span_id: str | None = None
        fallback_span_candidates = tool_step_or_run_candidates(record)

        if event_type == _RUN_START:
            span_id = run.span_id
        elif event_type in _RUN_TERMINALS:
            span_id = run.span_id
            run.end_sequence = record.sequence
            run.ended_at = record.timestamp
            duration = _payload_int(payload, "duration_ms")
            run.duration_ms = duration if duration is not None else _duration_ms(run.started_at, record.timestamp)
            run.status = _terminal_status(event_type, payload)
            run.terminal_source = "recorded"
            run.inferred_reason = None
        elif event_type == _STEP_START and record.step_id is not None:
            span_id = f"step:{record.step_id}"
            ensure_span(
                span_id,
                kind="step",
                name=str(payload.get("step_number", record.step_id)),
                parent_span_candidates=(run.span_id,),
                record=record,
            )
        elif event_type == _STEP_TERMINAL and record.step_id is not None:
            span_id = f"step:{record.step_id}"
            duration = _payload_int(payload, "duration_ms")
            step = builders.get(span_id)
            if step is None and duration is not None:
                step = ensure_span(
                    span_id,
                    kind="step",
                    name=str(payload.get("step_number", record.step_id)),
                    parent_span_candidates=(run.span_id,),
                    record=record,
                    started_at=record.timestamp - timedelta(milliseconds=duration),
                )
            if step is not None:
                step.end_sequence = record.sequence
                step.ended_at = record.timestamp
                step.duration_ms = duration if duration is not None else _duration_ms(step.started_at, record.timestamp)
                step.status = _terminal_status(event_type, payload)
                step.terminal_source = "recorded"
                step.inferred_reason = None
            else:
                span_id = None
        elif event_type == _TOOL_START and record.tool_call_id is not None:
            span_id = f"tool:{record.tool_call_id}"
            ensure_span(
                span_id,
                kind="tool",
                name=str(payload.get("tool_name", record.tool_call_id)),
                parent_span_candidates=step_or_run_candidates(record),
                record=record,
            )
        elif event_type == _TOOL_TERMINAL and record.tool_call_id is not None:
            span_id = f"tool:{record.tool_call_id}"
            duration = _payload_int(payload, "duration_ms")
            tool = builders.get(span_id)
            if tool is None and duration is not None:
                tool = ensure_span(
                    span_id,
                    kind="tool",
                    name=str(payload.get("tool_name", record.tool_call_id)),
                    parent_span_candidates=step_or_run_candidates(record),
                    record=record,
                    started_at=record.timestamp - timedelta(milliseconds=duration),
                )
            if tool is not None:
                tool.end_sequence = record.sequence
                tool.ended_at = record.timestamp
                tool.duration_ms = duration if duration is not None else _duration_ms(tool.started_at, record.timestamp)
                tool.status = _terminal_status(event_type, payload)
                tool.terminal_source = "recorded"
                tool.inferred_reason = None
            else:
                span_id = None
        elif event_type == _LLM_START and isinstance(payload.get("llm_round_id"), str):
            span_id = f"llm:{payload['llm_round_id']}"
            ensure_span(
                span_id,
                kind="llm",
                name=str(payload.get("model", payload["llm_round_id"])),
                parent_span_candidates=step_or_run_candidates(record),
                record=record,
            )
        elif event_type == "llm_round_first_output_delta" and isinstance(payload.get("llm_round_id"), str):
            candidate = f"llm:{payload['llm_round_id']}"
            if candidate in builders:
                span_id = candidate
                ttft = _payload_int(payload, "ttft_ms")
                if ttft is not None:
                    builders[candidate].ttft_ms = ttft
            else:
                fallback_span_candidates = step_or_run_candidates(record)
        elif event_type in _LLM_TERMINALS and isinstance(payload.get("llm_round_id"), str):
            candidate = f"llm:{payload['llm_round_id']}"
            llm = builders.get(candidate)
            if llm is not None:
                span_id = candidate
                duration = _payload_int(payload, "duration_ms")
                llm.end_sequence = record.sequence
                llm.ended_at = record.timestamp
                llm.duration_ms = duration if duration is not None else _duration_ms(llm.started_at, record.timestamp)
                llm.status = _terminal_status(event_type, payload)
                llm.terminal_source = "recorded"
                llm.inferred_reason = None
                ttft = _payload_int(payload, "ttft_ms")
                if ttft is not None:
                    llm.ttft_ms = ttft
            else:
                fallback_span_candidates = step_or_run_candidates(record)
        elif event_type == _RETRIEVAL_START and isinstance(payload.get("retrieval_id"), str):
            span_id = f"retrieval:{payload['retrieval_id']}"
            ensure_span(
                span_id,
                kind="retrieval",
                name=str(payload.get("query_summary", payload["retrieval_id"])),
                parent_span_candidates=step_or_run_candidates(record),
                record=record,
            )
        elif event_type in _RETRIEVAL_TERMINALS and isinstance(payload.get("retrieval_id"), str):
            candidate = f"retrieval:{payload['retrieval_id']}"
            retrieval = builders.get(candidate)
            if retrieval is not None:
                span_id = candidate
                duration = _payload_int(payload, "duration_ms")
                retrieval.end_sequence = record.sequence
                retrieval.ended_at = record.timestamp
                retrieval.duration_ms = (
                    duration if duration is not None else _duration_ms(retrieval.started_at, record.timestamp)
                )
                retrieval.status = _terminal_status(event_type, payload)
                retrieval.terminal_source = "recorded"
                retrieval.inferred_reason = None
            else:
                fallback_span_candidates = step_or_run_candidates(record)
        elif event_type == _ATTEMPT_START and isinstance(payload.get("tool_attempt_id"), str):
            span_id = f"tool_attempt:{payload['tool_attempt_id']}"
            ensure_span(
                span_id,
                kind="tool_attempt",
                name=str(payload.get("tool_name", payload["tool_attempt_id"])),
                parent_span_candidates=tool_step_or_run_candidates(record),
                record=record,
            )
        elif event_type == _ATTEMPT_TERMINAL and isinstance(payload.get("tool_attempt_id"), str):
            candidate = f"tool_attempt:{payload['tool_attempt_id']}"
            attempt = builders.get(candidate)
            if attempt is not None:
                span_id = candidate
                duration = _payload_int(payload, "duration_ms")
                attempt.end_sequence = record.sequence
                attempt.ended_at = record.timestamp
                attempt.duration_ms = (
                    duration if duration is not None else _duration_ms(attempt.started_at, record.timestamp)
                )
                attempt.status = _terminal_status(event_type, payload)
                attempt.terminal_source = "recorded"
                attempt.inferred_reason = None
            else:
                fallback_span_candidates = tool_step_or_run_candidates(record)
        append_record(record, span_id, fallback_span_candidates)

    for builder in builders.values():
        if builder.parent_span_candidates:
            builder.parent_span_id = next(
                (candidate for candidate in builder.parent_span_candidates if candidate in builders),
                None,
            )

    for projected_record, candidates in pending_record_associations:
        span_id = next(candidate for candidate in candidates if candidate in builders)
        projected_record.span_id = span_id
        deferred_record_sequences.setdefault(span_id, []).append(projected_record.sequence)

    for builder in builders.values():
        deferred_sequences = deferred_record_sequences.get(builder.span_id)
        if deferred_sequences:
            builder.record_sequences = _merge_sequences(builder.record_sequences, deferred_sequences)

    if truncated and records:
        close_time = records[-1].timestamp
        for builder in builders.values():
            if builder.terminal_source is None:
                builder.end_sequence = records[-1].sequence
                builder.ended_at = close_time
                builder.duration_ms = _duration_ms(builder.started_at, close_time)
                builder.status = "unknown"
                builder.terminal_source = "inferred"
                builder.inferred_reason = "truncated_prefix"
    elif not truncated:
        outcome = _orphan_outcome(run_status)
        if outcome is not None and records:
            status, reason = outcome
            close_time = run_ended_at or records[-1].timestamp
            close_sequence = records[-1].sequence
            for builder in builders.values():
                if builder.terminal_source is None:
                    builder.end_sequence = close_sequence
                    builder.ended_at = close_time
                    builder.duration_ms = _duration_ms(builder.started_at, close_time)
                    builder.status = status
                    builder.terminal_source = "inferred"
                    builder.inferred_reason = reason

    return TrajectoryProjection(
        records=projected_records,
        spans=[builders[span_id].to_dto() for span_id in ordered_span_ids],
    )
