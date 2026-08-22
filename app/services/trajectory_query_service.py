"""普通用户轨迹快照组装；只投影脱敏事件账本。"""

from __future__ import annotations

from app.db.models import AgentEvent, AgentSession
from app.db.trajectory_repository import TrajectoryRepository
from app.schemas.trajectory import (
    TrajectoryCompleteness,
    TrajectoryEventRecord,
    TrajectoryRunListResponse,
    TrajectoryRunSummary,
    TrajectorySnapshot,
)
from app.services.agent.trajectory_projector import project_trajectory
from app.services.agent.trajectory_reconciliation import resolve_user_trajectory_status_from_rows


class TrajectoryQueryService:
    """受鉴权 repository 支撑的只读轨迹查询服务。"""

    def __init__(
        self,
        repository: TrajectoryRepository,
        *,
        max_events_per_run: int,
        max_runs_per_conversation: int,
    ) -> None:
        if max_events_per_run <= 0 or max_runs_per_conversation <= 0:
            raise ValueError("轨迹查询上限必须大于 0")
        self._repository = repository
        self._max_events_per_run = max_events_per_run
        self._max_runs_per_conversation = max_runs_per_conversation

    def list_runs(self, conversation_id: str, user_id: str) -> TrajectoryRunListResponse | None:
        rows = self._repository.list_runs(conversation_id, user_id, self._max_runs_per_conversation + 1)
        if rows is None:
            return None
        truncated = len(rows) > self._max_runs_per_conversation
        bounded_rows = rows[: self._max_runs_per_conversation]
        watermark = self._repository.resolve_ledger_watermark()
        items = [
            self._run_summary(
                run, resolve_user_trajectory_status_from_rows(run.created_at, meta, watermark).trajectory_status
            )
            for run, meta in bounded_rows
        ]
        return TrajectoryRunListResponse(items=self._grouping_order(items), truncated=truncated)

    def get_user_snapshot(self, conversation_id: str, run_id: str, user_id: str) -> TrajectorySnapshot | None:
        row = self._repository.get_run(conversation_id, run_id, user_id)
        if row is None:
            return None
        run, meta = row
        event_rows = self._repository.list_events(conversation_id, run_id, self._max_events_per_run + 1)
        truncated = len(event_rows) > self._max_events_per_run
        loaded_events = event_rows[: self._max_events_per_run]
        assessment = resolve_user_trajectory_status_from_rows(
            run.created_at,
            meta,
            self._repository.resolve_ledger_watermark(),
        )
        records = [self._event_record(event) for event in loaded_events]
        projection = project_trajectory(
            records,
            run_status=run.status,
            run_ended_at=run.terminal_at,
            truncated=truncated,
        )
        return TrajectorySnapshot(
            run=self._run_summary(run, assessment.trajectory_status),
            records=projection.records,
            spans=projection.spans,
            completeness=TrajectoryCompleteness(
                status=assessment.trajectory_status,
                degraded_reason=assessment.degraded_reason,
                event_count=meta.event_count if meta is not None else None,
                expected_last_sequence=meta.expected_last_sequence if meta is not None else None,
                loaded_event_count=len(loaded_events),
                first_sequence=loaded_events[0].sequence if loaded_events else None,
                last_sequence=loaded_events[-1].sequence if loaded_events else None,
            ),
            truncated=truncated,
        )

    @staticmethod
    def _run_summary(run: AgentSession, trajectory_status: str) -> TrajectoryRunSummary:
        return TrajectoryRunSummary(
            run_id=run.id,
            message_id=run.message_id,
            turn_message_id=run.turn_message_id,
            attempt_index=run.attempt_index,
            status=run.status,
            trajectory_status=trajectory_status,
            total_steps=run.total_steps or 0,
            total_tool_calls=run.total_tool_calls or 0,
            duration_ms=run.total_duration_ms,
            started_at=run.created_at,
            ended_at=run.terminal_at,
        )

    @staticmethod
    def _event_record(event: AgentEvent) -> TrajectoryEventRecord:
        return TrajectoryEventRecord(
            run_id=event.run_id,
            sequence=event.sequence,
            event_type=event.event_type,
            schema_version=event.schema_version if event.schema_version is not None else 0,
            timestamp=event.event_ts,
            step_id=event.step_id,
            tool_call_id=event.tool_call_id,
            parent_step_id=event.parent_step_id,
            trace_id=event.trace_id,
            payload=dict(event.payload or {}),
        )

    @staticmethod
    def _grouping_order(items: list[TrajectoryRunSummary]) -> list[TrajectoryRunSummary]:
        groups: dict[str, list[TrajectoryRunSummary]] = {}
        for item in items:
            group_key = item.turn_message_id or f"__unlinked__:{item.run_id}"
            groups.setdefault(group_key, []).append(item)
        ordered: list[TrajectoryRunSummary] = []
        for group in sorted(
            groups.values(),
            key=lambda values: max(value.started_at for value in values),
            reverse=True,
        ):
            ordered.extend(
                sorted(
                    group,
                    key=lambda value: (
                        value.attempt_index is None,
                        value.attempt_index if value.attempt_index is not None else 0,
                        value.started_at,
                        value.run_id,
                    ),
                )
            )
        return ordered
