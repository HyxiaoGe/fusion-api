"""为工具调用日志增加精确关联字段。

Revision ID: 4a7c9e2b6d81
Revises: e8f5a1c4d2b7
Create Date: 2026-08-24 00:00:00
"""

import sqlalchemy as sa

from alembic import op

revision = "4a7c9e2b6d81"
down_revision = "e8f5a1c4d2b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool_call_logs",
        sa.Column("tool_call_id", sa.String(), nullable=True),
    )
    op.create_index(
        "uq_tool_call_logs_trace_tool_call",
        "tool_call_logs",
        ["trace_id", "tool_call_id"],
        unique=True,
        postgresql_where=sa.text("trace_id IS NOT NULL AND tool_call_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_tool_call_logs_trace_tool_call",
        table_name="tool_call_logs",
    )
    op.drop_column("tool_call_logs", "tool_call_id")
