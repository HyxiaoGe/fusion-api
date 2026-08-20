"""统一 Agent 会话时间为 UTC aware。

Revision ID: d7e4a9c2f1b6
Revises: c4f8a2d1e6b9
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d7e4a9c2f1b6"
down_revision: Union[str, Sequence[str], None] = "d9f2a6c4e7b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 线上历史值由 PostgreSQL 按 UTC 墙上时间保存；这里只补齐时区语义，不移动实际时刻。
    op.execute(
        "ALTER TABLE agent_sessions ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE "
        "USING created_at AT TIME ZONE 'UTC'"
    )
    op.execute("UPDATE agent_sessions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    op.alter_column(
        "agent_sessions",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


def downgrade() -> None:
    op.alter_column(
        "agent_sessions",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
        server_default=None,
    )
    op.execute(
        "ALTER TABLE agent_sessions ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE "
        "USING created_at AT TIME ZONE 'UTC'"
    )
