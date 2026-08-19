"""增加消息轮次生成写栅栏。

Revision ID: d9f2a6c4e7b1
Revises: c8e1f4a7b2d9
"""

import sqlalchemy as sa

from alembic import op

revision: str = "d9f2a6c4e7b1"
down_revision: str | None = "c8e1f4a7b2d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("generation_task_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "generation_task_id")
