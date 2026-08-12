"""add knowledge storage cleanup tasks

Revision ID: 5e7a9c2d4b10
Revises: 1f4b7c9d2e60
Create Date: 2026-08-13 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "5e7a9c2d4b10"
down_revision: Union[str, Sequence[str], None] = "1f4b7c9d2e60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_storage_cleanup_tasks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("storage_backend", sa.String(length=20), nullable=False),
        sa.Column("storage_key", sa.String(length=600), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="5", nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("lease_owner", sa.String(length=120), nullable=True),
        sa.Column("lease_token_hash", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=120), nullable=True),
        sa.Column("error_summary", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("attempt_count >= 0", name="ck_knowledge_storage_cleanup_attempt_count"),
        sa.CheckConstraint("max_attempts > 0", name="ck_knowledge_storage_cleanup_max_attempts"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'retry', 'completed', 'failed')",
            name="ck_knowledge_storage_cleanup_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_backend", "storage_key", name="uq_knowledge_storage_cleanup_object"),
    )
    op.create_index(
        "ix_knowledge_storage_cleanup_claim",
        "knowledge_storage_cleanup_tasks",
        ["status", "available_at", "created_at", "id"],
    )

    op.create_table(
        "knowledge_storage_upload_intents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("cleanup_task_id", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cleanup_task_id"],
            ["knowledge_storage_cleanup_tasks.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_knowledge_storage_upload_intents_cleanup_task_id",
        "knowledge_storage_upload_intents",
        ["cleanup_task_id"],
    )
    op.create_index(
        "ix_knowledge_storage_upload_intents_guard",
        "knowledge_storage_upload_intents",
        ["cleanup_task_id", "expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_storage_upload_intents_guard",
        table_name="knowledge_storage_upload_intents",
    )
    op.drop_index(
        "ix_knowledge_storage_upload_intents_cleanup_task_id",
        table_name="knowledge_storage_upload_intents",
    )
    op.drop_table("knowledge_storage_upload_intents")
    op.drop_index("ix_knowledge_storage_cleanup_claim", table_name="knowledge_storage_cleanup_tasks")
    op.drop_table("knowledge_storage_cleanup_tasks")
