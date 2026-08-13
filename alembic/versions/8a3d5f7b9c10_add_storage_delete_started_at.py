"""add storage delete started at

Revision ID: 8a3d5f7b9c10
Revises: 6f8b1d3c5a20
Create Date: 2026-08-13 09:20:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "8a3d5f7b9c10"
down_revision: Union[str, Sequence[str], None] = "6f8b1d3c5a20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "knowledge_storage_cleanup_tasks",
        sa.Column("delete_started_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_storage_cleanup_tasks", "delete_started_at")
