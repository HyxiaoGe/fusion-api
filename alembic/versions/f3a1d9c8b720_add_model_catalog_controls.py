"""add model catalog controls and admission operations

Revision ID: f3a1d9c8b720
Revises: e8b4c2d7f901
Create Date: 2026-08-04 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f3a1d9c8b720"
down_revision: Union[str, Sequence[str], None] = "e8b4c2d7f901"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_catalog_controls",
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("selectable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("routable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("reason", sa.String(length=300), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("routable IS TRUE", name="ck_model_catalog_controls_routable_true"),
        sa.CheckConstraint("revision > 0", name="ck_model_catalog_controls_revision_positive"),
        sa.PrimaryKeyConstraint("model_id"),
        sa.UniqueConstraint("model_id"),
    )
    op.create_table(
        "model_admission_operations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("candidate_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("governance_run_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("requested_by", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(length=300), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_token_hash", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("catalog_invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed')",
            name="ck_model_admission_operations_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_fingerprint",
            "governance_run_id",
            name="uq_model_admission_operation_candidate_run",
        ),
    )
    op.create_index(
        "ix_model_admission_operations_status_created",
        "model_admission_operations",
        ["status", "created_at", "id"],
    )
    op.create_index(
        "uq_model_admission_operations_single_running",
        "model_admission_operations",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_index("uq_model_admission_operations_single_running", table_name="model_admission_operations")
    op.drop_index("ix_model_admission_operations_status_created", table_name="model_admission_operations")
    op.drop_table("model_admission_operations")
    op.drop_table("model_catalog_controls")
