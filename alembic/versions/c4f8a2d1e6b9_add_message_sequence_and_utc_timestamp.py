"""为消息增加数据库顺序号并统一 UTC 时间。

Revision ID: c4f8a2d1e6b9
Revises: b8d4f7a1c2e6
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c4f8a2d1e6b9"
down_revision: Union[str, Sequence[str], None] = "b8d4f7a1c2e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 每次 nextval 原子预留一个奇数，紧随其后的偶数留给同轮 assistant。
    op.execute("CREATE SEQUENCE message_order_sequence START WITH 1 INCREMENT BY 2 NO CYCLE")
    op.add_column("messages", sa.Column("sequence", sa.BigInteger(), nullable=True))
    op.create_index("ux_messages_sequence", "messages", ["sequence"], unique=True)
    # 不回填历史行；但迁移窗口中的旧镜像若省略 sequence，仍由数据库分配非空顺序号。
    op.alter_column(
        "messages",
        "sequence",
        existing_type=sa.BigInteger(),
        existing_nullable=True,
        server_default=sa.text("nextval('message_order_sequence')"),
    )

    # 历史列虽然是无时区类型，但线上长期按 UTC 序列化；不在迁移中猜测少量本地坏数据。
    op.execute(
        "ALTER TABLE messages ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING created_at AT TIME ZONE 'UTC'"
    )
    op.execute("UPDATE messages SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    op.alter_column(
        "messages",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    for column_name in ("created_at", "updated_at"):
        op.execute(
            f"ALTER TABLE conversations ALTER COLUMN {column_name} TYPE TIMESTAMP WITH TIME ZONE "
            f"USING {column_name} AT TIME ZONE 'UTC'"
        )
        op.execute(f"UPDATE conversations SET {column_name} = CURRENT_TIMESTAMP WHERE {column_name} IS NULL")
        op.alter_column(
            "conversations",
            column_name,
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        )


def downgrade() -> None:
    for column_name in ("created_at", "updated_at"):
        op.alter_column(
            "conversations",
            column_name,
            existing_type=sa.DateTime(timezone=True),
            nullable=True,
            server_default=None,
        )
        op.execute(
            f"ALTER TABLE conversations ALTER COLUMN {column_name} TYPE TIMESTAMP WITHOUT TIME ZONE "
            f"USING {column_name} AT TIME ZONE 'UTC'"
        )

    op.alter_column(
        "messages",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
        server_default=None,
    )
    op.execute(
        "ALTER TABLE messages ALTER COLUMN created_at TYPE TIMESTAMP WITHOUT TIME ZONE "
        "USING created_at AT TIME ZONE 'UTC'"
    )
    op.drop_index("ux_messages_sequence", table_name="messages")
    op.drop_column("messages", "sequence")
    op.execute("DROP SEQUENCE message_order_sequence")
