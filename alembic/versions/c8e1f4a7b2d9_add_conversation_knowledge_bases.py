"""新增会话知识库选择

Revision ID: c8e1f4a7b2d9
Revises: b7d1e4f6a920
"""

import sqlalchemy as sa

from alembic import op

revision: str = "c8e1f4a7b2d9"
down_revision: str | None = "b7d1e4f6a920"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_knowledge_bases",
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "position >= 0 AND position < 5",
            name="ck_conversation_knowledge_bases_position",
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("conversation_id", "knowledge_base_id"),
        sa.UniqueConstraint(
            "conversation_id",
            "position",
            name="uq_conversation_knowledge_bases_position",
        ),
    )
    op.create_index(
        "ix_conversation_knowledge_bases_knowledge_base_id",
        "conversation_knowledge_bases",
        ["knowledge_base_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_knowledge_bases_knowledge_base_id",
        table_name="conversation_knowledge_bases",
    )
    op.drop_table("conversation_knowledge_bases")
