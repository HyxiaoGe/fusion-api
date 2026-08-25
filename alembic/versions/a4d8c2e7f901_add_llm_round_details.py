"""增加 LLM Round Detail 与 run 级能力版本。

Revision ID: a4d8c2e7f901
Revises: 9f2d6c1a8b43
Create Date: 2026-08-24 00:00:00
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "a4d8c2e7f901"
down_revision = "9f2d6c1a8b43"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "run_trajectory_meta",
        sa.Column("llm_detail_schema_version", sa.Integer(), nullable=True),
    )
    op.create_table(
        "agent_llm_round_details",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("llm_round_id", sa.String(), nullable=False),
        sa.Column("reasoning_text", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("reasoning_preview", sa.Text(), nullable=True),
        sa.Column("output_preview", sa.Text(), nullable=True),
        sa.Column(
            "redacted_fields",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "truncated_fields",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "llm_round_id", name="uq_agent_llm_round_details_run_round"),
    )


def downgrade() -> None:
    op.drop_table("agent_llm_round_details")
    op.drop_column("run_trajectory_meta", "llm_detail_schema_version")
