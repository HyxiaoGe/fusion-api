"""add knowledge upload outcome

Revision ID: 6f8b1d3c5a20
Revises: 5e7a9c2d4b10
Create Date: 2026-08-13 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "6f8b1d3c5a20"
down_revision: Union[str, Sequence[str], None] = "5e7a9c2d4b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "knowledge_storage_upload_intents",
        sa.Column("outcome", sa.String(length=20), server_default="uploading", nullable=False),
    )
    op.add_column(
        "knowledge_storage_upload_intents",
        sa.Column("outcome_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_knowledge_storage_upload_intents_outcome",
        "knowledge_storage_upload_intents",
        "outcome IN ('uploading', 'succeeded', 'failed', 'uncertain')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_knowledge_storage_upload_intents_outcome",
        "knowledge_storage_upload_intents",
        type_="check",
    )
    op.drop_column("knowledge_storage_upload_intents", "outcome_at")
    op.drop_column("knowledge_storage_upload_intents", "outcome")
