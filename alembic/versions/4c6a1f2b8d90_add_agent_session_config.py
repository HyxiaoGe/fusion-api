"""add agent session config

Revision ID: 4c6a1f2b8d90
Revises: 3b4c8a7d2f10
Create Date: 2026-06-28
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "4c6a1f2b8d90"
down_revision: Union[str, Sequence[str], None] = "3b4c8a7d2f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_sessions",
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_agent_sessions_conversation_message_created_at",
        "agent_sessions",
        ["conversation_id", "message_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_sessions_conversation_message_created_at", table_name="agent_sessions")
    op.drop_column("agent_sessions", "config")
