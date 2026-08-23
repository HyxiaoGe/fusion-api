"""普通用户轨迹快照组装；只投影脱敏事件账本。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from app.db.models import AgentEvent, AgentSession
from app.db.trajectory_repository import TrajectoryRepository
from app.schemas.admin_trajectory import AdminTrajectorySnapshot, AdminTrajectoryToolCall
from app.schemas.trajectory import (
    ToolNodeDetail,
    TrajectoryCompleteness,
    TrajectoryEventRecord,
    TrajectoryNodeDetailResponse,
    TrajectoryRunListResponse,
    TrajectoryRunSummary,
    TrajectorySnapshot,
)
from app.services.admin_audit_service import AdminAuditService
from app.services.agent.trajectory_projector import project_trajectory
from app.services.agent.trajectory_reconciliation import (
    resolve_ledger_watermark,
    resolve_user_trajectory_status_from_rows,
)
from app.utils.time import as_utc, utc_now


class TrajectoryQueryService:
    """受鉴权 repository 支撑的只读轨迹查询服务。"""

    def __init__(
        self,
        repository: TrajectoryRepository,
        *,
        max_events_per_run: int,
        max_runs_per_conversation: int,
        detail_settle_grace_seconds: float = 5,
        now_provider: Callable[[], datetime] = utc_now,
    ) -> None:
        if max_events_per_run <= 0 or max_runs_per_conversation <= 0:
            raise ValueError("轨迹查询上限必须大于 0")
        if detail_settle_grace_seconds < 0:
            raise ValueError("轨迹详情收敛宽限期不能为负数")
        self._repository = repository
        self._max_events_per_run = max_events_per_run
        self._max_runs_per_conversation = max_runs_per_conversation
        self._detail_settle_grace = timedelta(seconds=detail_settle_grace_seconds)
        self._now_provider = now_provider

    def list_runs(self, conversation_id: str, user_id: str) -> TrajectoryRunListResponse | None:
        rows = self._repository.list_runs(conversation_id, user_id, self._max_runs_per_conversation + 1)
        if rows is None:
            return None
        truncated = len(rows) > self._max_runs_per_conversation
        bounded_rows = rows[: self._max_runs_per_conversation]
        watermark = self._ledger_watermark()
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
        return self._snapshot_from_row(conversation_id, run_id, row)

    def get_admin_snapshot(self, conversation_id: str, run_id: str) -> AdminTrajectorySnapshot | None:
        """构造管理员诊断，但不承担权限判断或访问审计。"""
        row = self._repository.get_run_for_admin(conversation_id, run_id)
        if row is None:
            return None
        snapshot = self._snapshot_from_row(conversation_id, run_id, row)
        tool_rows = self._repository.list_tool_diagnostics(run_id, self._max_events_per_run + 1)
        tool_calls_truncated = len(tool_rows) > self._max_events_per_run
        tools = [self._admin_tool_call(tool) for tool in tool_rows[: self._max_events_per_run]]
        return AdminTrajectorySnapshot(
            snapshot=snapshot,
            tool_calls=tools,
            tool_calls_truncated=tool_calls_truncated,
        )

    def get_user_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        user_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_tool_node_detail(conversation_id, run_id, tool_call_id, user_id=user_id)

    def get_admin_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_tool_node_detail(conversation_id, run_id, tool_call_id, user_id=None)

    def _get_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        *,
        user_id: str | None,
    ) -> TrajectoryNodeDetailResponse | None:
        run = self._repository.get_detail_run(conversation_id, run_id, user_id)
        if run is None:
            return None
        tool = self._repository.get_exact_tool_detail(
            conversation_id=conversation_id,
            user_id=run.user_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )
        if tool is not None:
            return self._available_tool_detail(tool_call_id, tool)

        watermark = self._repository.get_detail_watermark()
        if watermark is None or as_utc(run.created_at) < as_utc(watermark):
            return self._unavailable_tool_detail("not_recorded", "detail_not_recorded")
        if run.status == "running":
            return self._unavailable_tool_detail("pending", "run_in_progress")
        if run.terminal_at is not None:
            age = as_utc(self._now_provider()) - as_utc(run.terminal_at)
            if age <= self._detail_settle_grace:
                return self._unavailable_tool_detail("pending", "detail_settling")
        return self._unavailable_tool_detail("degraded", "tool_detail_missing")

    @staticmethod
    def _available_tool_detail(tool_call_id: str, tool) -> TrajectoryNodeDetailResponse:
        safe_item = AdminAuditService._tool_item(tool)
        payload = safe_item["arguments"] or None
        result = safe_item["result_preview"] or None
        sections = ["summary"]
        if payload is not None:
            sections.append("payload")
        if result is not None:
            sections.append("result")
        if safe_item["duration_ms"] is not None:
            sections.append("timing")
        redacted_fields = [
            field.replace("arguments", "payload", 1).replace("result_preview", "result", 1)
            for field in safe_item["redacted_fields"]
        ]
        return TrajectoryNodeDetailResponse(
            status="available",
            available_sections=sections,
            detail=ToolNodeDetail(
                tool_call_id=tool_call_id,
                tool_name=safe_item["tool_name"],
                status=safe_item["status"],
                duration_ms=safe_item["duration_ms"],
                payload=payload,
                result=result,
                error=safe_item["error"],
            ),
            redacted_fields=redacted_fields,
            reason=None,
        )

    @staticmethod
    def _unavailable_tool_detail(status: str, reason: str) -> TrajectoryNodeDetailResponse:
        return TrajectoryNodeDetailResponse(
            status=status,
            available_sections=[],
            detail=None,
            redacted_fields=[],
            reason=reason,
        )

    @staticmethod
    def _admin_tool_call(tool) -> AdminTrajectoryToolCall:
        item = {key: value for key, value in AdminAuditService._tool_item(tool).items() if key != "trace_id"}
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            item["created_at"] = TrajectoryQueryService._tool_call_log_created_at_as_utc(created_at)
        return AdminTrajectoryToolCall(association="run", **item)

    @staticmethod
    def _tool_call_log_created_at_as_utc(value: datetime | None) -> datetime | None:
        """按 ToolCallLog 的既有北京时间墙钟语义规范化，不能替代通用 as_utc。"""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(UTC)
        return value.astimezone(UTC)

    def _snapshot_from_row(
        self,
        conversation_id: str,
        run_id: str,
        row: tuple[AgentSession, object | None],
    ) -> TrajectorySnapshot:
        run, meta = row
        event_rows = self._repository.list_events(conversation_id, run_id, self._max_events_per_run + 1)
        truncated = len(event_rows) > self._max_events_per_run
        loaded_events = event_rows[: self._max_events_per_run]
        assessment = resolve_user_trajectory_status_from_rows(
            run.created_at,
            meta,
            self._ledger_watermark(),
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

    def _ledger_watermark(self):
        return resolve_ledger_watermark(self._repository.list_ledger_watermark_rows())

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
