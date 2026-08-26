import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.db import models as db_models

MIGRATION_PATH = Path(__file__).parent.parent / "alembic" / "versions" / "c1e5b7a9d204_add_system_prompt_snapshots.py"


def load_migration():
    if not MIGRATION_PATH.exists():
        raise AssertionError("缺少系统提示词快照独立存储迁移")
    spec = importlib.util.spec_from_file_location("system_prompt_snapshot_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class SystemPromptSnapshotMigrationTests(unittest.TestCase):
    def test_revision_extends_the_current_trajectory_detail_head(self):
        migration = load_migration()

        self.assertEqual(migration.revision, "c1e5b7a9d204")
        self.assertEqual(migration.down_revision, "a4d8c2e7f901")

    def test_upgrade_creates_private_run_scoped_snapshot_table_without_backfill(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        operation.create_table.assert_called_once()
        table_args = operation.create_table.call_args.args
        self.assertEqual(table_args[0], "agent_system_prompt_snapshots")
        columns = {item.name: item for item in table_args[1:] if isinstance(item, sa.Column)}
        self.assertEqual(
            set(columns),
            {"run_id", "conversation_id", "user_id", "snapshot", "recorded_at"},
        )
        self.assertFalse(columns["run_id"].nullable)
        self.assertFalse(columns["conversation_id"].nullable)
        self.assertFalse(columns["user_id"].nullable)
        self.assertFalse(columns["snapshot"].nullable)
        self.assertIsInstance(columns["snapshot"].type, postgresql.JSONB)

        foreign_keys = [item for item in table_args[1:] if isinstance(item, sa.ForeignKeyConstraint)]
        targets = {next(iter(item.elements)).target_fullname: item.ondelete for item in foreign_keys}
        self.assertEqual(targets["agent_sessions.id"], "CASCADE")
        self.assertEqual(targets["conversations.id"], "CASCADE")
        operation.create_index.assert_called_once_with(
            "ix_agent_system_prompt_snapshots_conversation_run",
            "agent_system_prompt_snapshots",
            ["conversation_id", "run_id"],
        )
        operation.execute.assert_not_called()

    def test_downgrade_removes_only_the_snapshot_table(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        operation.drop_table.assert_called_once_with("agent_system_prompt_snapshots")


class SystemPromptSnapshotModelTests(unittest.TestCase):
    def test_orm_exposes_run_scoped_private_snapshot(self):
        self.assertTrue(hasattr(db_models, "AgentSystemPromptSnapshot"))
        table = db_models.AgentSystemPromptSnapshot.__table__

        self.assertTrue(table.c.run_id.primary_key)
        self.assertEqual(next(iter(table.c.run_id.foreign_keys)).target_fullname, "agent_sessions.id")
        self.assertEqual(next(iter(table.c.conversation_id.foreign_keys)).target_fullname, "conversations.id")
        self.assertIsInstance(table.c.snapshot.type, sa.JSON)
        self.assertFalse(table.c.snapshot.nullable)


if __name__ == "__main__":
    unittest.main()
