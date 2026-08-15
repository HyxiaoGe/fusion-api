"""add knowledge base v1

Revision ID: 1f4b7c9d2e60
Revises: a6c2f1d9e4b7
Create Date: 2026-08-12 21:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "1f4b7c9d2e60"
down_revision: Union[str, Sequence[str], None] = "a6c2f1d9e4b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("name_normalized", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("business_type", sa.String(length=60), server_default="general", nullable=False),
        sa.Column("status", sa.String(length=20), server_default="active", nullable=False),
        sa.Column("embedding_provider", sa.String(length=60), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("embedding_revision", sa.String(length=120), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("distance_metric", sa.String(length=20), server_default="COSINE", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("embedding_dimension > 1", name="ck_knowledge_bases_embedding_dimension"),
        sa.CheckConstraint("status IN ('active', 'deleting', 'deleted')", name="ck_knowledge_bases_status"),
        sa.CheckConstraint("distance_metric IN ('COSINE')", name="ck_knowledge_bases_distance_metric"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name_normalized", name="uq_knowledge_bases_user_name"),
    )
    op.create_index("ix_knowledge_bases_user_id", "knowledge_bases", ["user_id"])
    op.create_index(
        "ix_knowledge_bases_user_updated_id",
        "knowledge_bases",
        ["user_id", "updated_at", "id"],
    )

    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("mimetype", sa.String(length=120), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("dedupe_key", sa.String(length=120), nullable=False),
        sa.Column("storage_backend", sa.String(length=20), nullable=False),
        sa.Column("storage_key", sa.String(length=600), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="queued", nullable=False),
        sa.Column("parser_version", sa.String(length=40), nullable=False),
        sa.Column("chunker_version", sa.String(length=40), nullable=False),
        sa.Column("embedding_provider", sa.String(length=60), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("embedding_revision", sa.String(length=120), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("distance_metric", sa.String(length=20), server_default="COSINE", nullable=False),
        sa.Column("desired_index_version", sa.String(length=36), nullable=False),
        sa.Column("active_index_version", sa.String(length=36), nullable=True),
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(length=120), nullable=True),
        sa.Column("error_summary", sa.String(length=500), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("size >= 0", name="ck_knowledge_documents_size"),
        sa.CheckConstraint("embedding_dimension > 1", name="ck_knowledge_documents_embedding_dimension"),
        sa.CheckConstraint("chunk_count >= 0", name="ck_knowledge_documents_chunk_count"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_knowledge_documents_attempt_count"),
        sa.CheckConstraint(
            "status IN ('queued', 'parsing', 'chunking', 'embedding', 'writing', "
            "'ready', 'failed', 'deleting', 'deleted')",
            name="ck_knowledge_documents_status",
        ),
        sa.CheckConstraint("distance_metric IN ('COSINE')", name="ck_knowledge_documents_distance_metric"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("knowledge_base_id", "dedupe_key", name="uq_knowledge_documents_base_dedupe"),
    )
    op.create_index("ix_knowledge_documents_knowledge_base_id", "knowledge_documents", ["knowledge_base_id"])
    op.create_index("ix_knowledge_documents_user_id", "knowledge_documents", ["user_id"])
    op.create_index(
        "ix_knowledge_documents_base_updated_id",
        "knowledge_documents",
        ["knowledge_base_id", "updated_at", "id"],
    )
    op.create_index(
        "ix_knowledge_documents_user_status",
        "knowledge_documents",
        ["user_id", "status"],
    )

    op.create_table(
        "knowledge_index_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="building", nullable=False),
        sa.Column("parser_version", sa.String(length=40), nullable=False),
        sa.Column("chunker_version", sa.String(length=40), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("embedding_provider", sa.String(length=60), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("embedding_revision", sa.String(length=120), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("distance_metric", sa.String(length=20), server_default="COSINE", nullable=False),
        sa.Column("collection_name", sa.String(length=200), nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(length=120), nullable=True),
        sa.Column("error_summary", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("embedding_dimension > 1", name="ck_knowledge_index_versions_embedding_dimension"),
        sa.CheckConstraint("chunk_size >= 100", name="ck_knowledge_index_versions_chunk_size"),
        sa.CheckConstraint(
            "chunk_overlap >= 0 AND chunk_overlap < chunk_size",
            name="ck_knowledge_index_versions_chunk_overlap",
        ),
        sa.CheckConstraint("chunk_count >= 0", name="ck_knowledge_index_versions_chunk_count"),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'superseded', 'deleting', 'deleted', 'failed')",
            name="ck_knowledge_index_versions_status",
        ),
        sa.CheckConstraint("distance_metric IN ('COSINE')", name="ck_knowledge_index_versions_distance_metric"),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_index_versions_document_id", "knowledge_index_versions", ["document_id"])
    op.create_index("ix_knowledge_index_versions_knowledge_base_id", "knowledge_index_versions", ["knowledge_base_id"])
    op.create_index("ix_knowledge_index_versions_user_id", "knowledge_index_versions", ["user_id"])
    op.create_index(
        "ix_knowledge_index_versions_document_created",
        "knowledge_index_versions",
        ["document_id", "created_at", "id"],
    )
    op.create_index(
        "ix_knowledge_index_versions_base_status",
        "knowledge_index_versions",
        ["knowledge_base_id", "status"],
    )

    op.create_table(
        "knowledge_chunk_manifests",
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("knowledge_base_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("index_version", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_sha256", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_knowledge_chunk_manifests_ordinal"),
        sa.CheckConstraint(
            "char_start >= 0 AND char_end >= char_start",
            name="ck_knowledge_chunk_manifests_offsets",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["index_version"], ["knowledge_index_versions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint("index_version", "ordinal", name="uq_knowledge_chunk_manifests_version_ordinal"),
    )
    op.create_index(
        "ix_knowledge_chunk_manifests_auth",
        "knowledge_chunk_manifests",
        ["user_id", "knowledge_base_id", "document_id", "index_version"],
    )
    op.create_index("ix_knowledge_chunk_manifests_document_id", "knowledge_chunk_manifests", ["document_id"])
    op.create_index("ix_knowledge_chunk_manifests_index_version", "knowledge_chunk_manifests", ["index_version"])
    op.create_index("ix_knowledge_chunk_manifests_user_id", "knowledge_chunk_manifests", ["user_id"])

    op.create_table(
        "knowledge_index_tasks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("task_type", sa.String(length=30), nullable=False),
        sa.Column("index_version", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("phase", sa.String(length=20), server_default="queued", nullable=False),
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
        sa.CheckConstraint("attempt_count >= 0", name="ck_knowledge_index_tasks_attempt_count"),
        sa.CheckConstraint("max_attempts > 0", name="ck_knowledge_index_tasks_max_attempts"),
        sa.CheckConstraint(
            "task_type IN ('index_document', 'delete_document', 'delete_knowledge_base', 'delete_index_version')",
            name="ck_knowledge_index_tasks_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'retry', 'completed', 'failed', 'cancelled')",
            name="ck_knowledge_index_tasks_status",
        ),
        sa.CheckConstraint(
            "phase IN ('queued', 'parsing', 'chunking', 'embedding', 'writing', 'deleting', 'completed')",
            name="ck_knowledge_index_tasks_phase",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["knowledge_documents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["index_version"], ["knowledge_index_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_index_tasks_document_id", "knowledge_index_tasks", ["document_id"])
    op.create_index("ix_knowledge_index_tasks_knowledge_base_id", "knowledge_index_tasks", ["knowledge_base_id"])
    op.create_index("ix_knowledge_index_tasks_user_id", "knowledge_index_tasks", ["user_id"])
    op.create_index(
        "ix_knowledge_index_tasks_claim",
        "knowledge_index_tasks",
        ["status", "available_at", "created_at", "id"],
    )
    op.create_index(
        "ix_knowledge_index_tasks_base_status",
        "knowledge_index_tasks",
        ["knowledge_base_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_index_tasks_base_status", table_name="knowledge_index_tasks")
    op.drop_index("ix_knowledge_index_tasks_claim", table_name="knowledge_index_tasks")
    op.drop_index("ix_knowledge_index_tasks_user_id", table_name="knowledge_index_tasks")
    op.drop_index("ix_knowledge_index_tasks_knowledge_base_id", table_name="knowledge_index_tasks")
    op.drop_index("ix_knowledge_index_tasks_document_id", table_name="knowledge_index_tasks")
    op.drop_table("knowledge_index_tasks")

    op.drop_index("ix_knowledge_chunk_manifests_user_id", table_name="knowledge_chunk_manifests")
    op.drop_index("ix_knowledge_chunk_manifests_index_version", table_name="knowledge_chunk_manifests")
    op.drop_index("ix_knowledge_chunk_manifests_document_id", table_name="knowledge_chunk_manifests")
    op.drop_index("ix_knowledge_chunk_manifests_auth", table_name="knowledge_chunk_manifests")
    op.drop_table("knowledge_chunk_manifests")

    op.drop_index("ix_knowledge_index_versions_base_status", table_name="knowledge_index_versions")
    op.drop_index("ix_knowledge_index_versions_document_created", table_name="knowledge_index_versions")
    op.drop_index("ix_knowledge_index_versions_user_id", table_name="knowledge_index_versions")
    op.drop_index("ix_knowledge_index_versions_knowledge_base_id", table_name="knowledge_index_versions")
    op.drop_index("ix_knowledge_index_versions_document_id", table_name="knowledge_index_versions")
    op.drop_table("knowledge_index_versions")

    op.drop_index("ix_knowledge_documents_user_status", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_base_updated_id", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_user_id", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_knowledge_base_id", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")

    op.drop_index("ix_knowledge_bases_user_updated_id", table_name="knowledge_bases")
    op.drop_index("ix_knowledge_bases_user_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
