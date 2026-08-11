"""backfill model catalog migration origin registry

Revision ID: a6c2f1d9e4b7
Revises: f3a1d9c8b720
Create Date: 2026-08-11 09:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "a6c2f1d9e4b7"
down_revision: Union[str, Sequence[str], None] = "f3a1d9c8b720"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DO $fusion$
            BEGIN
              IF to_regclass('_fusion_alembic_f3a1d9c8b720_origins') IS NULL THEN
                IF to_regclass('model_catalog_controls') IS NULL
                   OR to_regclass('model_admission_operations') IS NULL THEN
                  RAISE EXCEPTION '缺少模型管理表，无法回填迁移来源登记';
                END IF;

                CREATE TABLE _fusion_alembic_f3a1d9c8b720_origins (
                  object_name varchar(100) NOT NULL,
                  created_by_migration boolean NOT NULL,
                  CONSTRAINT ck_f3a1d9c8b720_origin_object_name CHECK (
                    object_name IN ('model_catalog_controls', 'model_admission_operations')
                  ),
                  PRIMARY KEY (object_name)
                );
                INSERT INTO _fusion_alembic_f3a1d9c8b720_origins (
                  object_name,
                  created_by_migration
                ) VALUES
                  ('model_catalog_controls', true),
                  ('model_admission_operations', true);
              END IF;
            END;
            $fusion$;
            """
        )
    )


def downgrade() -> None:
    # 来源登记属于父修订 f3a1d9c8b720 的回滚契约，必须保留给父修订消费。
    pass
