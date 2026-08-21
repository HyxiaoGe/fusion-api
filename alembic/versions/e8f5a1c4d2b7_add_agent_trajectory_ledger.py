"""增加 Agent 轨迹账本与 run attempt 层级。

Revision ID: e8f5a1c4d2b7
Revises: d7e4a9c2f1b6
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e8f5a1c4d2b7"
down_revision: Union[str, Sequence[str], None] = "d7e4a9c2f1b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_sessions", sa.Column("turn_message_id", sa.String(), nullable=True))
    op.add_column("agent_sessions", sa.Column("previous_run_id", sa.String(), nullable=True))
    op.add_column("agent_sessions", sa.Column("attempt_index", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_agent_sessions_previous_run_id",
        "agent_sessions",
        "agent_sessions",
        ["previous_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.execute(
        "UPDATE agent_sessions "
        "SET turn_message_id = message_id "
        "WHERE message_id IS NOT NULL"
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                row_number() OVER (
                    PARTITION BY turn_message_id
                    ORDER BY created_at, id
                ) AS allocated_attempt_index
            FROM agent_sessions
            WHERE turn_message_id IS NOT NULL
        )
        UPDATE agent_sessions AS target
        SET attempt_index = ranked.allocated_attempt_index
        FROM ranked
        WHERE target.id = ranked.id
        """
    )
    op.create_index(
        "uq_agent_sessions_turn_attempt",
        "agent_sessions",
        ["turn_message_id", "attempt_index"],
        unique=True,
        postgresql_where=sa.text("turn_message_id IS NOT NULL AND attempt_index IS NOT NULL"),
    )

    op.create_table(
        "agent_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("step_id", sa.String(), nullable=True),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("parent_step_id", sa.String(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("event_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_agent_events_run_sequence"),
    )
    op.create_index(
        "ix_agent_events_conversation_ts",
        "agent_events",
        ["conversation_id", "event_ts"],
    )
    op.create_index("ix_agent_events_run", "agent_events", ["run_id"])

    op.create_table(
        "run_trajectory_meta",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("trajectory_status", sa.String(), nullable=False),
        sa.Column("event_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("expected_last_sequence", sa.Integer(), nullable=True),
        sa.Column("first_event_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("degraded_reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "trajectory_ledger_settings",
        sa.Column("singleton_key", sa.String(), nullable=False),
        sa.Column("ledger_enabled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "singleton_key = 'default'",
            name="ck_trajectory_ledger_settings_singleton_key",
        ),
        sa.PrimaryKeyConstraint("singleton_key"),
    )
    op.execute(
        "INSERT INTO trajectory_ledger_settings "
        "(singleton_key, ledger_enabled_at, created_at) "
        "VALUES ('default', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )


def downgrade() -> None:
    op.drop_table("trajectory_ledger_settings")
    op.drop_table("run_trajectory_meta")
    op.drop_table("agent_events")
    op.drop_index("uq_agent_sessions_turn_attempt", table_name="agent_sessions")
    op.drop_constraint(
        "fk_agent_sessions_previous_run_id",
        "agent_sessions",
        type_="foreignkey",
    )
    op.drop_column("agent_sessions", "attempt_index")
    op.drop_column("agent_sessions", "previous_run_id")
    op.drop_column("agent_sessions", "turn_message_id")
