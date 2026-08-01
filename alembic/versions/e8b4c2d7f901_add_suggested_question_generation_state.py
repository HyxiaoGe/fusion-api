"""增加推荐问题生成版本与状态。

Revision ID: e8b4c2d7f901
Revises: d7a9c4e2f1b6
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "e8b4c2d7f901"
down_revision: Union[str, Sequence[str], None] = "d7a9c4e2f1b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "suggested_questions_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "messages",
        sa.Column(
            "suggested_questions_status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'idle'"),
        ),
    )
    op.add_column(
        "messages",
        sa.Column("suggested_questions_auto_run_id", sa.String(), nullable=True),
    )
    # 兼容历史数据：已有推荐问题直接视为 ready，不触发无意义的重新生成。
    op.execute(
        "UPDATE messages "
        "SET suggested_questions_status = 'ready' "
        "WHERE suggested_questions IS NOT NULL"
    )
    op.create_index(
        "ix_messages_suggested_questions_pending",
        "messages",
        ["created_at", "id"],
        unique=False,
        postgresql_where=sa.text("suggested_questions_status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_messages_suggested_questions_pending", table_name="messages")
    op.drop_column("messages", "suggested_questions_auto_run_id")
    op.drop_column("messages", "suggested_questions_status")
    op.drop_column("messages", "suggested_questions_revision")
