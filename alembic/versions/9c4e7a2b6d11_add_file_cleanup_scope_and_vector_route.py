"""新增 File cleanup 隔离状态与索引 Milvus 路由

Revision ID: 9c4e7a2b6d11
Revises: 8a3d5f7b9c10
"""

import sqlalchemy as sa

from alembic import op

revision: str = "9c4e7a2b6d11"
down_revision: str | None = "8a3d5f7b9c10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_storage_cleanup_tasks",
        sa.Column("file_status", sa.String(length=20), nullable=True),
    )
    op.create_check_constraint(
        "ck_knowledge_storage_cleanup_file_status",
        "knowledge_storage_cleanup_tasks",
        "file_status IS NULL OR file_status IN ('pending', 'running', 'retry', 'completed', 'failed')",
    )
    op.create_index(
        "ix_knowledge_storage_cleanup_file_claim",
        "knowledge_storage_cleanup_tasks",
        ["file_status", "available_at", "created_at", "id"],
    )
    # 兼容迁移执行前已经由旧 API 提交 File 引用、但尚未完成 cleanup fence 的极小窗口。
    # 这些行转换后物理 status 固定 completed，显式回滚到不认识 File 引用的旧 Worker
    # 也无法领取；新 Worker 继续通过 file_status 收敛。
    op.execute(
        sa.text(
            """
            UPDATE knowledge_storage_cleanup_tasks AS cleanup
            SET file_status = cleanup.status,
                status = 'completed'
            WHERE cleanup.file_status IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM files AS file
                  WHERE file.storage_backend = cleanup.storage_backend
                    AND (
                        file.storage_key = cleanup.storage_key
                        OR file.thumbnail_key = cleanup.storage_key
                    )
              )
            """
        )
    )
    # 迁移在旧 API 停止前执行。旧版在迁移之后才提交 File INSERT/UPDATE 时，
    # 一次性 backfill 无法看到其引用；触发器把这些在途 fence 也原子转换成旧
    # Worker 永远不会领取的物理 completed，并清除共享 task 的全部 intent。
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION fusion_classify_file_cleanup_scope()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                WITH referenced_tasks AS (
                    UPDATE knowledge_storage_cleanup_tasks AS cleanup
                    SET file_status = cleanup.status,
                        status = 'completed'
                    WHERE cleanup.file_status IS NULL
                      AND cleanup.storage_backend = NEW.storage_backend
                      AND (
                          cleanup.storage_key = NEW.storage_key
                          OR cleanup.storage_key = NEW.thumbnail_key
                      )
                    RETURNING cleanup.id
                )
                DELETE FROM knowledge_storage_upload_intents AS intent
                USING referenced_tasks
                WHERE intent.cleanup_task_id = referenced_tasks.id;
                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER tr_files_classify_cleanup_scope
            AFTER INSERT OR UPDATE OF storage_backend, storage_key, thumbnail_key ON files
            FOR EACH ROW
            EXECUTE FUNCTION fusion_classify_file_cleanup_scope()
            """
        )
    )
    op.add_column(
        "knowledge_index_versions",
        sa.Column("milvus_uri", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "knowledge_index_versions",
        sa.Column("milvus_database", sa.String(length=120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_index_versions", "milvus_database")
    op.drop_column("knowledge_index_versions", "milvus_uri")
    op.execute(sa.text("DROP TRIGGER IF EXISTS tr_files_classify_cleanup_scope ON files"))
    op.execute(sa.text("DROP FUNCTION IF EXISTS fusion_classify_file_cleanup_scope()"))
    op.drop_index(
        "ix_knowledge_storage_cleanup_file_claim",
        table_name="knowledge_storage_cleanup_tasks",
    )
    op.drop_constraint(
        "ck_knowledge_storage_cleanup_file_status",
        "knowledge_storage_cleanup_tasks",
        type_="check",
    )
    op.drop_column("knowledge_storage_cleanup_tasks", "file_status")
