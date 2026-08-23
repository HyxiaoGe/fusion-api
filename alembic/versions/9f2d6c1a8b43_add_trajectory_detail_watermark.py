"""增加轨迹节点详情独立启用水位。

Revision ID: 9f2d6c1a8b43
Revises: 4a7c9e2b6d81
Create Date: 2026-08-24 00:00:00
"""

import sqlalchemy as sa

from alembic import op

revision = "9f2d6c1a8b43"
down_revision = "4a7c9e2b6d81"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trajectory_ledger_settings",
        sa.Column("trajectory_detail_enabled_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trajectory_ledger_settings", "trajectory_detail_enabled_at")
