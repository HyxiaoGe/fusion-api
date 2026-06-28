"""add agent progress snapshots

Revision ID: 9c1f2a3b4d5e
Revises: 4c6a1f2b8d90
Create Date: 2026-06-28
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "9c1f2a3b4d5e"
down_revision: Union[str, Sequence[str], None] = "4c6a1f2b8d90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_progress_snapshots",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("protocol_version", sa.Integer(), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_agent_progress_snapshots_run_id"),
    )
    op.create_index(
        "ix_agent_progress_message_updated",
        "agent_progress_snapshots",
        ["conversation_id", "message_id", "updated_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_conversation_id"),
        "agent_progress_snapshots",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_created_at"),
        "agent_progress_snapshots",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_message_id"),
        "agent_progress_snapshots",
        ["message_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_run_id"),
        "agent_progress_snapshots",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_updated_at"),
        "agent_progress_snapshots",
        ["updated_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_progress_snapshots_user_id"),
        "agent_progress_snapshots",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_progress_snapshots_user_id"), table_name="agent_progress_snapshots")
    op.drop_index(op.f("ix_agent_progress_snapshots_updated_at"), table_name="agent_progress_snapshots")
    op.drop_index(op.f("ix_agent_progress_snapshots_run_id"), table_name="agent_progress_snapshots")
    op.drop_index(op.f("ix_agent_progress_snapshots_message_id"), table_name="agent_progress_snapshots")
    op.drop_index(op.f("ix_agent_progress_snapshots_created_at"), table_name="agent_progress_snapshots")
    op.drop_index(op.f("ix_agent_progress_snapshots_conversation_id"), table_name="agent_progress_snapshots")
    op.drop_index("ix_agent_progress_message_updated", table_name="agent_progress_snapshots")
    op.drop_table("agent_progress_snapshots")
