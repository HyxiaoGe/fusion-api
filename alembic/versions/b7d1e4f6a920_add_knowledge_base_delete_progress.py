"""新增知识库删除持久进度

Revision ID: b7d1e4f6a920
Revises: 9c4e7a2b6d11
"""

import sqlalchemy as sa

from alembic import op

revision: str = "b7d1e4f6a920"
down_revision: str | None = "9c4e7a2b6d11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_index_tasks",
        sa.Column("cleanup_profile_cursor", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "knowledge_index_tasks",
        sa.Column("cleanup_cursor", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "knowledge_index_tasks",
        sa.Column("cleanup_vectors_completed", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("knowledge_index_tasks", "cleanup_vectors_completed")
    op.drop_column("knowledge_index_tasks", "cleanup_cursor")
    op.drop_column("knowledge_index_tasks", "cleanup_profile_cursor")
