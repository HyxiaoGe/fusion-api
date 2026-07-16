"""增加远程 MCP 服务配置表。

Revision ID: d7a9c4e2f1b6
Revises: c4f8a2d1e6b9
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d7a9c4e2f1b6"
down_revision: Union[str, Sequence[str], None] = "c4f8a2d1e6b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("provider", sa.String(length=80), nullable=False),
        sa.Column("endpoint_url", sa.String(length=2048), nullable=False),
        sa.Column("transport", sa.String(length=32), server_default="streamable_http", nullable=False),
        sa.Column("auth_type", sa.String(length=20), server_default="none", nullable=False),
        sa.Column("auth_name", sa.String(length=128), nullable=True),
        sa.Column("credential_ref", sa.String(length=128), nullable=True),
        sa.Column("config_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("is_enabled", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "allowed_tools",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "discovered_tools",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("health_status", sa.String(length=20), server_default="disabled", nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=80), nullable=True),
        sa.Column("last_error_message", sa.String(length=300), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("transport = 'streamable_http'", name="ck_mcp_servers_transport"),
        sa.CheckConstraint(
            "auth_type IN ('none', 'bearer', 'header', 'query')",
            name="ck_mcp_servers_auth_type",
        ),
        sa.CheckConstraint(
            "health_status IN ('unknown', 'healthy', 'unhealthy', 'disabled')",
            name="ck_mcp_servers_health_status",
        ),
        sa.CheckConstraint(
            "(auth_type = 'none' AND auth_name IS NULL AND credential_ref IS NULL) OR "
            "(auth_type = 'bearer' AND auth_name IS NULL AND credential_ref IS NOT NULL) OR "
            "(auth_type IN ('header', 'query') AND auth_name IS NOT NULL AND credential_ref IS NOT NULL)",
            name="ck_mcp_servers_auth_shape",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_mcp_servers_name"),
    )
    op.create_index(op.f("ix_mcp_servers_provider"), "mcp_servers", ["provider"])
    op.create_index(op.f("ix_mcp_servers_is_enabled"), "mcp_servers", ["is_enabled"])
    op.create_index(op.f("ix_mcp_servers_health_status"), "mcp_servers", ["health_status"])
    op.create_index("ix_mcp_servers_provider_enabled", "mcp_servers", ["provider", "is_enabled"])
    op.create_index("ix_mcp_servers_health_updated", "mcp_servers", ["health_status", "updated_at"])


def downgrade() -> None:
    op.drop_index("ix_mcp_servers_health_updated", table_name="mcp_servers")
    op.drop_index("ix_mcp_servers_provider_enabled", table_name="mcp_servers")
    op.drop_index(op.f("ix_mcp_servers_health_status"), table_name="mcp_servers")
    op.drop_index(op.f("ix_mcp_servers_is_enabled"), table_name="mcp_servers")
    op.drop_index(op.f("ix_mcp_servers_provider"), table_name="mcp_servers")
    op.drop_table("mcp_servers")
