"""drop provider / model_sources / user_credentials tables

模型注册表彻底迁到 LiteLLM Proxy，fusion-api 改成薄代理。这一刀下完，
本地 DB 不再保留任何模型 / provider / BYOK 元数据。

降级路径见 downgrade()：会按 baseline schema 把三张表 + 索引 + 外键约束建回来，
但**数据不会回来**——降级只是让 schema 回到能跑旧代码的状态。
迁移生产环境前请先备份这三张表。

Revision ID: 3b4c8a7d2f10
Revises: 2ac863b60dd3
Create Date: 2026-05-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3b4c8a7d2f10"
down_revision: Union[str, Sequence[str], None] = "2ac863b60dd3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop three tables (user_credentials -> model_sources -> providers).

    用 IF EXISTS 兜底：不同环境的 baseline 索引可能不完全一致
    （dev 库就缺 ix_user_credentials_id），强删会炸。表本身也用 IF EXISTS
    保证幂等。
    """
    # user_credentials 引用 providers（外键），先删
    op.execute("DROP INDEX IF EXISTS ix_user_credentials_user_id")
    op.execute("DROP INDEX IF EXISTS ix_user_credentials_id")
    op.execute("DROP TABLE IF EXISTS user_credentials CASCADE")

    # model_sources 引用 providers（外键），再删
    op.execute("DROP INDEX IF EXISTS ix_model_sources_model_id")
    op.execute("DROP INDEX IF EXISTS ix_model_sources_id")
    op.execute("DROP TABLE IF EXISTS model_sources CASCADE")

    # providers 最后
    op.execute("DROP TABLE IF EXISTS providers CASCADE")


def downgrade() -> None:
    """Rebuild three tables to baseline schema (data NOT restored)."""
    op.create_table(
        "providers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "auth_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("litellm_prefix", sa.String(), nullable=False),
        sa.Column("custom_base_url", sa.Boolean(), server_default=sa.text("false"), nullable=True),
        sa.Column("priority", sa.Integer(), server_default="100", nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=True),
        sa.Column("status", sa.String(), server_default="ok", nullable=False),
        sa.Column("offline_reason", sa.String(), nullable=True),
        sa.Column("offline_message", sa.Text(), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_failure_at", sa.DateTime(), nullable=True),
        sa.Column("last_failure_kind", sa.String(), nullable=True),
        sa.Column("last_probe_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("knowledge_cutoff", sa.String(), nullable=True),
        sa.Column("capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("pricing", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["provider"], ["providers.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_sources_id"), "model_sources", ["id"], unique=False)
    op.create_index(op.f("ix_model_sources_model_id"), "model_sources", ["model_id"], unique=True)

    op.create_table(
        "user_credentials",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider_id", sa.String(), nullable=False),
        sa.Column("api_key", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("last_error_kind", sa.String(), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "provider_id", name="uq_user_credentials_user_provider"),
    )
    op.create_index(op.f("ix_user_credentials_id"), "user_credentials", ["id"], unique=False)
    op.create_index(op.f("ix_user_credentials_user_id"), "user_credentials", ["user_id"], unique=False)
