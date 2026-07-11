"""管理员审计中心独立只读查询与审计写入仓储。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from sqlalchemy import Integer, cast, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import (
    AdminAuditEvent,
    AgentProgressSnapshot,
    AgentSession,
    AgentStep,
    Conversation,
    ConversationFile,
    File,
    Message,
    PerformanceRun,
    ToolCallLog,
    User,
)


def page_payload(items: list[Any], total: int, page: int, page_size: int) -> dict[str, Any]:
    total_pages = math.ceil(total / page_size) if total else 0
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


class AdminAuditRepository:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _offset(page: int, page_size: int) -> int:
        return (page - 1) * page_size

    def list_users(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        is_superuser: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        users_query = self.db.query(User)
        if query and query.strip():
            pattern = f"%{query.strip()}%"
            users_query = users_query.filter(
                or_(
                    User.id.ilike(pattern),
                    User.username.ilike(pattern),
                    User.nickname.ilike(pattern),
                    User.email.ilike(pattern),
                )
            )
        if is_superuser is not None:
            users_query = users_query.filter(User.is_superuser.is_(is_superuser))
        if created_from is not None:
            users_query = users_query.filter(User.created_at >= created_from)
        if created_to is not None:
            users_query = users_query.filter(User.created_at <= created_to)
        total = users_query.count()
        users = (
            users_query.order_by(User.created_at.desc(), User.id.desc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        user_ids = [row.id for row in users]
        if not user_ids:
            return [], total

        conversation_stats = {
            user_id: (count, last_active)
            for user_id, count, last_active in self.db.query(
                Conversation.user_id,
                func.count(Conversation.id),
                func.max(Conversation.updated_at),
            )
            .filter(Conversation.user_id.in_(user_ids))
            .group_by(Conversation.user_id)
            .all()
        }
        message_stats = {
            user_id: (count, input_tokens or 0, output_tokens or 0)
            for user_id, count, input_tokens, output_tokens in self.db.query(
                Conversation.user_id,
                func.count(Message.id),
                func.sum(cast(Message.usage["input_tokens"].as_string(), Integer)),
                func.sum(cast(Message.usage["output_tokens"].as_string(), Integer)),
            )
            .join(Message, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id.in_(user_ids))
            .group_by(Conversation.user_id)
            .all()
        }
        tool_counts = dict(
            self.db.query(ToolCallLog.user_id, func.count(ToolCallLog.id))
            .filter(ToolCallLog.user_id.in_(user_ids))
            .group_by(ToolCallLog.user_id)
            .all()
        )
        items = []
        for user in users:
            conversation_count, last_active = conversation_stats.get(user.id, (0, None))
            message_count, input_tokens, output_tokens = message_stats.get(user.id, (0, 0, 0))
            items.append(
                {
                    "user": user,
                    "last_active_at": last_active,
                    "conversation_count": conversation_count,
                    "message_count": message_count,
                    "tool_call_count": tool_counts.get(user.id, 0),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
        return items, total

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        user = self.db.query(User).filter(User.id == user_id).first()
        if user is None:
            return None
        conversation_count, last_active = (
            self.db.query(func.count(Conversation.id), func.max(Conversation.updated_at))
            .filter(Conversation.user_id == user.id)
            .one()
        )
        message_count, input_tokens, output_tokens = (
            self.db.query(
                func.count(Message.id),
                func.sum(cast(Message.usage["input_tokens"].as_string(), Integer)),
                func.sum(cast(Message.usage["output_tokens"].as_string(), Integer)),
            )
            .join(Conversation, Conversation.id == Message.conversation_id)
            .filter(Conversation.user_id == user.id)
            .one()
        )
        tool_count = self.db.query(func.count(ToolCallLog.id)).filter(ToolCallLog.user_id == user.id).scalar() or 0
        return {
            "user": user,
            "last_active_at": last_active,
            "conversation_count": conversation_count or 0,
            "message_count": message_count or 0,
            "tool_call_count": tool_count,
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
        }

    def list_conversations(
        self,
        *,
        page: int,
        page_size: int,
        user_id: str | None = None,
        query: str | None = None,
        model_id: str | None = None,
        has_tools: bool | None = None,
        has_files: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        conversations_query = self.db.query(Conversation).join(User, User.id == Conversation.user_id)
        if user_id:
            conversations_query = conversations_query.filter(Conversation.user_id == user_id)
        if model_id:
            conversations_query = conversations_query.filter(Conversation.model_id == model_id)
        if query and query.strip():
            pattern = f"%{query.strip()}%"
            conversations_query = conversations_query.filter(
                or_(
                    Conversation.id.ilike(pattern),
                    Conversation.title.ilike(pattern),
                    User.id.ilike(pattern),
                    User.username.ilike(pattern),
                    User.nickname.ilike(pattern),
                    User.email.ilike(pattern),
                )
            )
        if has_tools is not None:
            clause = self.db.query(ToolCallLog.id).filter(ToolCallLog.conversation_id == Conversation.id).exists()
            conversations_query = conversations_query.filter(clause if has_tools else ~clause)
        if has_files is not None:
            clause = (
                self.db.query(ConversationFile.conversation_id)
                .filter(ConversationFile.conversation_id == Conversation.id)
                .exists()
            )
            conversations_query = conversations_query.filter(clause if has_files else ~clause)
        if created_from is not None:
            conversations_query = conversations_query.filter(Conversation.created_at >= created_from)
        if created_to is not None:
            conversations_query = conversations_query.filter(Conversation.created_at <= created_to)
        if updated_from is not None:
            conversations_query = conversations_query.filter(Conversation.updated_at >= updated_from)
        if updated_to is not None:
            conversations_query = conversations_query.filter(Conversation.updated_at <= updated_to)

        total = conversations_query.count()
        conversations = (
            conversations_query.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return self._conversation_rows(conversations), total

    def _conversation_rows(self, conversations: list[Conversation]) -> list[dict[str, Any]]:
        conversation_ids = [row.id for row in conversations]
        if not conversation_ids:
            return []
        users = {
            row.id: row for row in self.db.query(User).filter(User.id.in_({c.user_id for c in conversations})).all()
        }
        message_stats = {
            conversation_id: (count, input_tokens or 0, output_tokens or 0)
            for conversation_id, count, input_tokens, output_tokens in self.db.query(
                Message.conversation_id,
                func.count(Message.id),
                func.sum(cast(Message.usage["input_tokens"].as_string(), Integer)),
                func.sum(cast(Message.usage["output_tokens"].as_string(), Integer)),
            )
            .filter(Message.conversation_id.in_(conversation_ids))
            .group_by(Message.conversation_id)
            .all()
        }
        tool_counts = dict(
            self.db.query(ToolCallLog.conversation_id, func.count(ToolCallLog.id))
            .filter(ToolCallLog.conversation_id.in_(conversation_ids))
            .group_by(ToolCallLog.conversation_id)
            .all()
        )
        file_counts = dict(
            self.db.query(ConversationFile.conversation_id, func.count(ConversationFile.file_id))
            .filter(ConversationFile.conversation_id.in_(conversation_ids))
            .group_by(ConversationFile.conversation_id)
            .all()
        )
        ranked_sessions = (
            self.db.query(
                AgentSession.conversation_id.label("conversation_id"),
                AgentSession.status.label("status"),
                func.row_number()
                .over(
                    partition_by=AgentSession.conversation_id,
                    order_by=(AgentSession.created_at.desc(), AgentSession.id.desc()),
                )
                .label("row_number"),
            )
            .filter(AgentSession.conversation_id.in_(conversation_ids))
            .subquery()
        )
        latest_status = dict(
            self.db.query(ranked_sessions.c.conversation_id, ranked_sessions.c.status)
            .filter(ranked_sessions.c.row_number == 1)
            .all()
        )
        result = []
        for conversation in conversations:
            message_count, input_tokens, output_tokens = message_stats.get(conversation.id, (0, 0, 0))
            result.append(
                {
                    "conversation": conversation,
                    "user": users.get(conversation.user_id),
                    "message_count": message_count,
                    "tool_call_count": tool_counts.get(conversation.id, 0),
                    "file_count": file_counts.get(conversation.id, 0),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latest_agent_status": latest_status.get(conversation.id),
                }
            )
        return result

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation is None:
            return None
        return self._conversation_rows([conversation])[0]

    def conversation_target_user_id(self, conversation_id: str) -> str | None:
        row = self.db.query(Conversation.user_id).filter(Conversation.id == conversation_id).first()
        return row[0] if row else None

    def list_messages(self, conversation_id: str, *, page: int, page_size: int) -> tuple[list[Message], int]:
        query = self.db.query(Message).filter(Message.conversation_id == conversation_id)
        total = query.count()
        rows = (
            query.order_by(Message.created_at.asc(), Message.id.asc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return rows, total

    def list_tool_calls(self, conversation_id: str, *, page: int, page_size: int) -> tuple[list[ToolCallLog], int]:
        query = self.db.query(ToolCallLog).filter(ToolCallLog.conversation_id == conversation_id)
        total = query.count()
        rows = (
            query.order_by(ToolCallLog.created_at.asc(), ToolCallLog.id.asc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return rows, total

    def list_agent_runs(self, conversation_id: str, *, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        query = self.db.query(AgentSession).filter(AgentSession.conversation_id == conversation_id)
        total = query.count()
        sessions = (
            query.order_by(AgentSession.created_at.desc(), AgentSession.id.desc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        run_ids = [row.id for row in sessions]
        if not run_ids:
            return [], total
        steps_by_run: dict[str, list[AgentStep]] = {run_id: [] for run_id in run_ids}
        for step in (
            self.db.query(AgentStep)
            .filter(AgentStep.trace_id.in_(run_ids))
            .order_by(AgentStep.trace_id.asc(), AgentStep.step_number.asc(), AgentStep.id.asc())
            .all()
        ):
            steps_by_run.setdefault(step.trace_id, []).append(step)
        snapshots = {
            row.run_id: row
            for row in self.db.query(AgentProgressSnapshot).filter(AgentProgressSnapshot.run_id.in_(run_ids)).all()
        }
        tools_by_run: dict[str, list[ToolCallLog]] = {run_id: [] for run_id in run_ids}
        for tool in (
            self.db.query(ToolCallLog)
            .filter(ToolCallLog.trace_id.in_(run_ids))
            .order_by(ToolCallLog.trace_id.asc(), ToolCallLog.step_number.asc(), ToolCallLog.created_at.asc())
            .all()
        ):
            tools_by_run.setdefault(tool.trace_id or "", []).append(tool)
        return [
            {
                "session": session,
                "steps": steps_by_run.get(session.id, []),
                "snapshot": snapshots.get(session.id),
                "tool_calls": tools_by_run.get(session.id, []),
            }
            for session in sessions
        ], total

    def list_files(self, conversation_id: str, *, page: int, page_size: int) -> tuple[list[File], int]:
        query = (
            self.db.query(File)
            .join(ConversationFile, ConversationFile.file_id == File.id)
            .filter(ConversationFile.conversation_id == conversation_id)
        )
        total = query.count()
        rows = (
            query.order_by(ConversationFile.created_at.asc(), File.id.asc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return rows, total

    def create_audit_event(self, *, commit: bool = True, **values: Any) -> AdminAuditEvent:
        event = AdminAuditEvent(**values)
        self.db.add(event)
        if commit:
            self.db.commit()
            self.db.refresh(event)
        else:
            self.db.flush()
        return event

    def list_audit_events(
        self,
        *,
        page: int,
        page_size: int,
        admin_user_id: str | None = None,
        target_user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[list[AdminAuditEvent], int]:
        query = self.db.query(AdminAuditEvent)
        if admin_user_id:
            query = query.filter(AdminAuditEvent.admin_user_id == admin_user_id)
        if target_user_id:
            query = query.filter(AdminAuditEvent.target_user_id == target_user_id)
        if action:
            query = query.filter(AdminAuditEvent.action == action)
        if resource_type:
            query = query.filter(AdminAuditEvent.resource_type == resource_type)
        if created_from is not None:
            query = query.filter(AdminAuditEvent.created_at >= created_from)
        if created_to is not None:
            query = query.filter(AdminAuditEvent.created_at <= created_to)
        total = query.count()
        rows = (
            query.order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return rows, total

    def get_performance_run(self, run_id: str) -> PerformanceRun | None:
        return self.db.query(PerformanceRun).filter(PerformanceRun.run_id == run_id).first()

    def import_performance_run(self, values: dict[str, Any]) -> tuple[PerformanceRun, bool]:
        existing = self.get_performance_run(values["run_id"])
        if existing is not None:
            return existing, False
        row = PerformanceRun(**values)
        self.db.add(row)
        try:
            self.db.flush()
            return row, True
        except IntegrityError:
            self.db.rollback()
            existing = self.get_performance_run(values["run_id"])
            if existing is None:
                raise
            return existing, False

    def list_performance_runs(
        self,
        *,
        page: int,
        page_size: int,
        environment: str | None = None,
        status: str | None = None,
        model_id: str | None = None,
    ) -> tuple[list[PerformanceRun], int]:
        query = self.db.query(PerformanceRun)
        if environment:
            query = query.filter(PerformanceRun.environment == environment)
        if status:
            query = query.filter(PerformanceRun.status == status)
        if model_id:
            query = query.filter(PerformanceRun.model_id == model_id)
        total = query.count()
        rows = (
            query.order_by(PerformanceRun.created_at.desc(), PerformanceRun.run_id.desc())
            .offset(self._offset(page, page_size))
            .limit(page_size)
            .all()
        )
        return rows, total
