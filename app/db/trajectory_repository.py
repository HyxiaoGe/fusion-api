"""轨迹历史读取的有界数据库访问。"""

from __future__ import annotations

from typing import TypeAlias

from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from app.db.models import AgentEvent, AgentSession, Conversation, RunTrajectoryMeta, TrajectoryLedgerSettings
from app.services.agent.trajectory_reconciliation import LedgerWatermarkResolution, resolve_ledger_watermark

RunWithMeta: TypeAlias = tuple[AgentSession, RunTrajectoryMeta | None]


class TrajectoryRepository:
    """普通用户轨迹读侧的 repository；所有上限由调用方显式传入。"""

    def __init__(self, session: Session):
        self._session = session

    def list_runs(self, conversation_id: str, user_id: str, limit: int) -> list[RunWithMeta] | None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        if not self._conversation_is_owned(conversation_id, user_id):
            return None
        rows = self._session.execute(
            select(AgentSession, RunTrajectoryMeta)
            .options(
                load_only(
                    AgentSession.id,
                    AgentSession.conversation_id,
                    AgentSession.user_id,
                    AgentSession.message_id,
                    AgentSession.turn_message_id,
                    AgentSession.attempt_index,
                    AgentSession.status,
                    AgentSession.total_steps,
                    AgentSession.total_tool_calls,
                    AgentSession.total_duration_ms,
                    AgentSession.terminal_at,
                    AgentSession.created_at,
                ),
                load_only(
                    RunTrajectoryMeta.run_id,
                    RunTrajectoryMeta.trajectory_status,
                    RunTrajectoryMeta.event_count,
                    RunTrajectoryMeta.expected_last_sequence,
                    RunTrajectoryMeta.degraded_reason,
                    RunTrajectoryMeta.terminal_intent_reason,
                    RunTrajectoryMeta.terminal_intent_pending_at,
                ),
            )
            .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
            .where(AgentSession.conversation_id == conversation_id)
            .where(AgentSession.user_id == user_id)
            .order_by(AgentSession.created_at.desc(), AgentSession.id.desc())
            .limit(limit)
        ).all()
        return [(row[0], row[1]) for row in rows]

    def get_run(self, conversation_id: str, run_id: str, user_id: str) -> RunWithMeta | None:
        row = self._session.execute(
            select(AgentSession, RunTrajectoryMeta)
            .options(
                load_only(
                    AgentSession.id,
                    AgentSession.conversation_id,
                    AgentSession.user_id,
                    AgentSession.message_id,
                    AgentSession.turn_message_id,
                    AgentSession.attempt_index,
                    AgentSession.status,
                    AgentSession.total_steps,
                    AgentSession.total_tool_calls,
                    AgentSession.total_duration_ms,
                    AgentSession.terminal_at,
                    AgentSession.created_at,
                ),
                load_only(
                    RunTrajectoryMeta.run_id,
                    RunTrajectoryMeta.trajectory_status,
                    RunTrajectoryMeta.event_count,
                    RunTrajectoryMeta.expected_last_sequence,
                    RunTrajectoryMeta.degraded_reason,
                    RunTrajectoryMeta.terminal_intent_reason,
                    RunTrajectoryMeta.terminal_intent_pending_at,
                ),
            )
            .join(Conversation, Conversation.id == AgentSession.conversation_id)
            .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
            .where(AgentSession.id == run_id)
            .where(AgentSession.user_id == user_id)
        ).one_or_none()
        return None if row is None else (row[0], row[1])

    def list_events(self, conversation_id: str, run_id: str, limit: int) -> list[AgentEvent]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        return list(
            self._session.execute(
                select(AgentEvent)
                .options(
                    load_only(
                        AgentEvent.run_id,
                        AgentEvent.sequence,
                        AgentEvent.event_type,
                        AgentEvent.schema_version,
                        AgentEvent.event_ts,
                        AgentEvent.step_id,
                        AgentEvent.tool_call_id,
                        AgentEvent.parent_step_id,
                        AgentEvent.trace_id,
                        AgentEvent.payload,
                    )
                )
                .where(AgentEvent.conversation_id == conversation_id)
                .where(AgentEvent.run_id == run_id)
                .order_by(AgentEvent.sequence.asc())
                .limit(limit)
            ).scalars()
        )

    def resolve_ledger_watermark(self) -> LedgerWatermarkResolution:
        rows = self._session.execute(
            select(
                TrajectoryLedgerSettings.singleton_key,
                TrajectoryLedgerSettings.ledger_enabled_at,
            )
        ).all()
        return resolve_ledger_watermark([(row.singleton_key, row.ledger_enabled_at) for row in rows])

    def _conversation_is_owned(self, conversation_id: str, user_id: str) -> bool:
        return (
            self._session.execute(
                select(Conversation.id).where(Conversation.id == conversation_id).where(Conversation.user_id == user_id)
            ).scalar_one_or_none()
            is not None
        )
