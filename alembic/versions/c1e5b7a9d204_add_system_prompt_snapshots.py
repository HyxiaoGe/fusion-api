"""增加系统提示词快照独立存储。

Revision ID: c1e5b7a9d204
Revises: a4d8c2e7f901
Create Date: 2026-08-27 00:00:00
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "c1e5b7a9d204"
down_revision = "a4d8c2e7f901"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_system_prompt_snapshots",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "ix_agent_system_prompt_snapshots_conversation_run",
        "agent_system_prompt_snapshots",
        ["conversation_id", "run_id"],
    )


def downgrade() -> None:
    op.drop_table("agent_system_prompt_snapshots")
