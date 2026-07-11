"""add admin audit center

Revision ID: b8d4f7a1c2e6
Revises: 7d2f8a1c9b30
Create Date: 2026-07-11 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b8d4f7a1c2e6"
down_revision: Union[str, Sequence[str], None] = "7d2f8a1c9b30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("admin_user_id", sa.String(), nullable=False),
        sa.Column("admin_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("resource_type", sa.String(length=40), nullable=False),
        sa.Column("resource_id", sa.String(), nullable=True),
        sa.Column("target_user_id", sa.String(), nullable=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(length=300), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_admin_audit_events_created_id", "admin_audit_events", ["created_at", "id"])
    op.create_index(
        "ix_admin_audit_events_admin_created_id",
        "admin_audit_events",
        ["admin_user_id", "created_at", "id"],
    )
    op.create_index(
        "ix_admin_audit_events_target_created_id",
        "admin_audit_events",
        ["target_user_id", "created_at", "id"],
    )
    op.create_index(
        "ix_admin_audit_events_resource_created_id",
        "admin_audit_events",
        ["resource_type", "resource_id", "created_at", "id"],
    )

    op.create_table(
        "performance_runs",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("environment", sa.String(length=30), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("safe_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("imported_by_user_id", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_performance_runs_created_id", "performance_runs", ["created_at", "run_id"])
    op.create_index(
        "ix_performance_runs_environment_created_id",
        "performance_runs",
        ["environment", "created_at", "run_id"],
    )

    op.create_index("ix_conversations_updated_id", "conversations", ["updated_at", "id"])
    op.create_index(
        "ix_conversations_user_updated_id",
        "conversations",
        ["user_id", "updated_at", "id"],
    )
    op.create_index(
        "ix_messages_conversation_created_id",
        "messages",
        ["conversation_id", "created_at", "id"],
    )
    op.create_index(
        "ix_tool_call_logs_conversation_created_id",
        "tool_call_logs",
        ["conversation_id", "created_at", "id"],
    )
    op.create_index(
        "ix_tool_call_logs_user_created_id",
        "tool_call_logs",
        ["user_id", "created_at", "id"],
    )
    op.create_index(
        "ix_tool_call_logs_trace_step_created_id",
        "tool_call_logs",
        ["trace_id", "step_number", "created_at", "id"],
    )
    op.create_index(
        "ix_agent_sessions_conversation_created_id",
        "agent_sessions",
        ["conversation_id", "created_at", "id"],
    )
    op.create_index(
        "ix_agent_sessions_user_created_id",
        "agent_sessions",
        ["user_id", "created_at", "id"],
    )

    # PostgreSQL 的 NOT VALID 会立即阻止新增孤立数据，同时保留历史孤立 step。
    op.execute(
        "ALTER TABLE agent_steps "
        "ADD CONSTRAINT fk_agent_steps_trace_id_agent_sessions "
        "FOREIGN KEY (trace_id) REFERENCES agent_sessions(id) ON DELETE CASCADE NOT VALID"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_steps DROP CONSTRAINT IF EXISTS fk_agent_steps_trace_id_agent_sessions")
    op.drop_index("ix_agent_sessions_user_created_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_conversation_created_id", table_name="agent_sessions")
    op.drop_index("ix_tool_call_logs_trace_step_created_id", table_name="tool_call_logs")
    op.drop_index("ix_tool_call_logs_user_created_id", table_name="tool_call_logs")
    op.drop_index("ix_tool_call_logs_conversation_created_id", table_name="tool_call_logs")
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.drop_index("ix_conversations_user_updated_id", table_name="conversations")
    op.drop_index("ix_conversations_updated_id", table_name="conversations")

    op.drop_index("ix_performance_runs_environment_created_id", table_name="performance_runs")
    op.drop_index("ix_performance_runs_created_id", table_name="performance_runs")
    op.drop_table("performance_runs")

    op.drop_index("ix_admin_audit_events_resource_created_id", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_target_created_id", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_admin_created_id", table_name="admin_audit_events")
    op.drop_index("ix_admin_audit_events_created_id", table_name="admin_audit_events")
    op.drop_table("admin_audit_events")
