"""轨迹历史读取的有界数据库访问。"""

from __future__ import annotations

from typing import TypeAlias

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, load_only

from app.db.models import (
    AgentEvent,
    AgentLlmRoundDetail,
    AgentSession,
    AgentSystemPromptSnapshot,
    Conversation,
    RunTrajectoryMeta,
    ToolCallLog,
    TrajectoryLedgerSettings,
)
from app.schemas.trajectory import UserTrajectoryMetaRow

RunWithMeta: TypeAlias = tuple[AgentSession, UserTrajectoryMetaRow | None]


def _llm_round_count_subquery():
    return (
        select(func.count(func.distinct(AgentEvent.payload["llm_round_id"].as_string())))
        .where(AgentEvent.run_id == AgentSession.id)
        .where(AgentEvent.event_type == "llm_round_started")
        .correlate(AgentSession)
        .scalar_subquery()
    )


class TrajectoryRepository:
    """普通用户轨迹读侧的 repository；所有上限由调用方显式传入。"""

    def __init__(self, session: Session):
        self._session = session

    def list_runs(self, conversation_id: str, user_id: str, limit: int) -> list[RunWithMeta] | None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        rows = self._session.execute(
            select(
                Conversation.id,
                AgentSession,
                RunTrajectoryMeta.trajectory_status,
                RunTrajectoryMeta.event_count,
                RunTrajectoryMeta.expected_last_sequence,
                RunTrajectoryMeta.degraded_reason,
                RunTrajectoryMeta.terminal_intent_pending_at,
                RunTrajectoryMeta.llm_detail_schema_version,
                _llm_round_count_subquery(),
            )
            .select_from(Conversation)
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
            )
            .outerjoin(
                AgentSession,
                and_(AgentSession.conversation_id == Conversation.id, AgentSession.user_id == user_id),
            )
            .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
            .order_by(AgentSession.created_at.desc(), AgentSession.id.desc())
            .limit(limit)
        ).all()
        if not rows:
            return None
        return [(row[1], self._user_meta_from_columns(row[2:])) for row in rows if row[1] is not None]

    def get_run(self, conversation_id: str, run_id: str, user_id: str) -> RunWithMeta | None:
        row = self._session.execute(
            select(
                AgentSession,
                RunTrajectoryMeta.trajectory_status,
                RunTrajectoryMeta.event_count,
                RunTrajectoryMeta.expected_last_sequence,
                RunTrajectoryMeta.degraded_reason,
                RunTrajectoryMeta.terminal_intent_pending_at,
                RunTrajectoryMeta.llm_detail_schema_version,
                _llm_round_count_subquery(),
            )
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
            )
            .join(Conversation, Conversation.id == AgentSession.conversation_id)
            .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
            .where(AgentSession.id == run_id)
            .where(AgentSession.user_id == user_id)
        ).one_or_none()
        return None if row is None else (row[0], self._user_meta_from_columns(row[1:]))

    def get_run_for_admin(self, conversation_id: str, run_id: str) -> RunWithMeta | None:
        """管理员读取仍严格验证 run 属于指定会话，但不附加普通用户归属条件。"""
        row = self._session.execute(
            select(
                AgentSession,
                RunTrajectoryMeta.trajectory_status,
                RunTrajectoryMeta.event_count,
                RunTrajectoryMeta.expected_last_sequence,
                RunTrajectoryMeta.degraded_reason,
                RunTrajectoryMeta.terminal_intent_pending_at,
                RunTrajectoryMeta.llm_detail_schema_version,
                _llm_round_count_subquery(),
            )
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
            )
            .join(Conversation, Conversation.id == AgentSession.conversation_id)
            .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
            .where(Conversation.id == conversation_id)
            .where(AgentSession.id == run_id)
        ).one_or_none()
        return None if row is None else (row[0], self._user_meta_from_columns(row[1:]))

    def get_detail_run(
        self,
        conversation_id: str,
        run_id: str,
        user_id: str | None,
    ) -> AgentSession | None:
        """读取 Node Detail 所属 run；普通端点必须同时限定当前用户。"""
        query = (
            select(AgentSession)
            .options(
                load_only(
                    AgentSession.id,
                    AgentSession.conversation_id,
                    AgentSession.user_id,
                    AgentSession.status,
                    AgentSession.terminal_at,
                    AgentSession.created_at,
                )
            )
            .join(Conversation, Conversation.id == AgentSession.conversation_id)
            .where(Conversation.id == conversation_id)
            .where(AgentSession.id == run_id)
            .where(Conversation.user_id == AgentSession.user_id)
        )
        if user_id is not None:
            query = query.where(Conversation.user_id == user_id).where(AgentSession.user_id == user_id)
        return self._session.execute(query).scalar_one_or_none()

    def get_exact_tool_detail(
        self,
        *,
        conversation_id: str,
        user_id: str,
        run_id: str,
        tool_call_id: str,
    ) -> ToolCallLog | None:
        """只按完整归属与精确关联键读取工具日志，不提供任何启发式 fallback。"""
        return self._session.execute(
            select(ToolCallLog)
            .options(
                load_only(
                    ToolCallLog.id,
                    ToolCallLog.conversation_id,
                    ToolCallLog.message_id,
                    ToolCallLog.user_id,
                    ToolCallLog.tool_name,
                    ToolCallLog.status,
                    ToolCallLog.error_message,
                    ToolCallLog.duration_ms,
                    ToolCallLog.model_id,
                    ToolCallLog.provider,
                    ToolCallLog.input_params,
                    ToolCallLog.output_data,
                    ToolCallLog.trace_id,
                    ToolCallLog.tool_call_id,
                    ToolCallLog.step_number,
                    ToolCallLog.created_at,
                )
            )
            .where(ToolCallLog.conversation_id == conversation_id)
            .where(ToolCallLog.user_id == user_id)
            .where(ToolCallLog.trace_id == run_id)
            .where(ToolCallLog.tool_call_id == tool_call_id)
        ).scalar_one_or_none()

    def get_system_prompt_snapshot(self, conversation_id: str, run_id: str, user_id: str) -> object | None:
        """只从独立私有表读取正文，并重新验证 Run 与当前用户的完整归属。"""
        return self._session.execute(
            select(AgentSystemPromptSnapshot.snapshot)
            .join(AgentSession, AgentSession.id == AgentSystemPromptSnapshot.run_id)
            .join(Conversation, Conversation.id == AgentSession.conversation_id)
            .where(AgentSystemPromptSnapshot.conversation_id == conversation_id)
            .where(AgentSystemPromptSnapshot.user_id == user_id)
            .where(AgentSystemPromptSnapshot.run_id == run_id)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user_id)
            .where(AgentSession.id == run_id)
            .where(AgentSession.user_id == user_id)
        ).scalar_one_or_none()

    def get_system_prompt_prepared_event(self, conversation_id: str, run_id: str) -> AgentEvent | None:
        """每个 Run 只有一次组装；读取对应终态元数据，不加载完整事件列。"""
        return self._session.execute(
            select(AgentEvent)
            .options(load_only(AgentEvent.payload))
            .where(AgentEvent.conversation_id == conversation_id)
            .where(AgentEvent.run_id == run_id)
            .where(AgentEvent.event_type == "system_prompt_prepared")
            .order_by(AgentEvent.sequence.desc())
            .limit(1)
        ).scalar_one_or_none()

    def get_detail_watermark(self):
        return self._session.execute(
            select(TrajectoryLedgerSettings.trajectory_detail_enabled_at).where(
                TrajectoryLedgerSettings.singleton_key == "default"
            )
        ).scalar_one_or_none()

    def get_llm_detail_schema_version(self, run_id: str) -> int | None:
        return self._session.execute(
            select(RunTrajectoryMeta.llm_detail_schema_version).where(RunTrajectoryMeta.run_id == run_id)
        ).scalar_one_or_none()

    def get_exact_llm_detail(
        self,
        *,
        conversation_id: str,
        run_id: str,
        llm_round_id: str,
    ) -> AgentLlmRoundDetail | None:
        return self._session.execute(
            select(AgentLlmRoundDetail)
            .options(
                load_only(
                    AgentLlmRoundDetail.llm_round_id,
                    AgentLlmRoundDetail.reasoning_text,
                    AgentLlmRoundDetail.content_text,
                    AgentLlmRoundDetail.redacted_fields,
                    AgentLlmRoundDetail.truncated_fields,
                )
            )
            .where(AgentLlmRoundDetail.conversation_id == conversation_id)
            .where(AgentLlmRoundDetail.run_id == run_id)
            .where(AgentLlmRoundDetail.llm_round_id == llm_round_id)
        ).scalar_one_or_none()

    def list_llm_round_details(self, conversation_id: str, run_id: str) -> list[AgentLlmRoundDetail]:
        return list(
            self._session.execute(
                select(AgentLlmRoundDetail)
                .options(
                    load_only(
                        AgentLlmRoundDetail.llm_round_id,
                        AgentLlmRoundDetail.reasoning_preview,
                        AgentLlmRoundDetail.output_preview,
                        AgentLlmRoundDetail.recorded_at,
                    )
                )
                .where(AgentLlmRoundDetail.conversation_id == conversation_id)
                .where(AgentLlmRoundDetail.run_id == run_id)
                .order_by(AgentLlmRoundDetail.recorded_at.asc(), AgentLlmRoundDetail.llm_round_id.asc())
            ).scalars()
        )

    def list_llm_round_lifecycle_events(
        self,
        *,
        conversation_id: str,
        run_id: str,
        llm_round_id: str,
    ) -> list[AgentEvent]:
        return list(
            self._session.execute(
                select(AgentEvent)
                .options(load_only(AgentEvent.event_type, AgentEvent.event_ts, AgentEvent.sequence, AgentEvent.payload))
                .where(AgentEvent.conversation_id == conversation_id)
                .where(AgentEvent.run_id == run_id)
                .where(
                    AgentEvent.event_type.in_(
                        (
                            "llm_round_started",
                            "llm_round_completed",
                            "llm_round_failed",
                            "llm_round_cancelled",
                        )
                    )
                )
                .where(AgentEvent.payload["llm_round_id"].as_string() == llm_round_id)
                .order_by(AgentEvent.sequence.asc())
            ).scalars()
        )

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

    def list_tool_diagnostics(self, run_id: str, limit: int = 5001) -> list[ToolCallLog]:
        """返回仅可由 trace_id 可靠归属到 run 的工具日志。"""
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        return list(
            self._session.execute(
                select(ToolCallLog)
                .options(
                    load_only(
                        ToolCallLog.id,
                        ToolCallLog.message_id,
                        ToolCallLog.trace_id,
                        ToolCallLog.step_number,
                        ToolCallLog.tool_name,
                        ToolCallLog.status,
                        ToolCallLog.duration_ms,
                        ToolCallLog.model_id,
                        ToolCallLog.provider,
                        ToolCallLog.input_params,
                        ToolCallLog.output_data,
                        ToolCallLog.error_message,
                        ToolCallLog.created_at,
                    )
                )
                .where(ToolCallLog.trace_id == run_id)
                .order_by(
                    ToolCallLog.step_number.asc().nulls_last(),
                    ToolCallLog.created_at.asc().nulls_last(),
                    ToolCallLog.id.asc(),
                )
                .limit(limit)
            ).scalars()
        )

    def list_ledger_watermark_rows(self) -> list[tuple[object, object]]:
        """返回原始水位行；状态规则由 service 层在中立数据上解释。"""
        rows = self._session.execute(
            select(
                TrajectoryLedgerSettings.singleton_key,
                TrajectoryLedgerSettings.ledger_enabled_at,
            )
        ).all()
        return [(row.singleton_key, row.ledger_enabled_at) for row in rows]

    @staticmethod
    def _user_meta_from_columns(values: tuple[object, ...]) -> UserTrajectoryMetaRow | None:
        (
            trajectory_status,
            event_count,
            expected_last_sequence,
            degraded_reason,
            pending_at,
            llm_version,
            llm_round_count,
        ) = values
        if trajectory_status is None:
            return None
        return UserTrajectoryMetaRow(
            trajectory_status=str(trajectory_status),
            event_count=int(event_count),
            expected_last_sequence=expected_last_sequence if isinstance(expected_last_sequence, int) else None,
            degraded_reason=degraded_reason if isinstance(degraded_reason, str) else None,
            has_pending_terminal_intent=pending_at is not None,
            llm_detail_schema_version=llm_version if isinstance(llm_version, int) else None,
            llm_round_count=int(llm_round_count or 0),
        )
