import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa

from app.db.models import TrajectoryLedgerSettings

MIGRATION_PATH = (
    Path(__file__).parent.parent / "alembic" / "versions" / "9f2d6c1a8b43_add_trajectory_detail_watermark.py"
)


def load_migration():
    if not MIGRATION_PATH.exists():
        raise AssertionError("缺少 Trajectory Node Detail 水位迁移")
    spec = importlib.util.spec_from_file_location("trajectory_node_detail_watermark_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class TrajectoryNodeDetailMigrationTests(unittest.TestCase):
    def test_revision_extends_tool_call_linkage_head(self):
        migration = load_migration()

        self.assertEqual(migration.revision, "9f2d6c1a8b43")
        self.assertEqual(migration.down_revision, "4a7c9e2b6d81")

    def test_upgrade_adds_nullable_watermark_without_activating_it(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        operation.add_column.assert_called_once()
        table_name, column = operation.add_column.call_args.args
        self.assertEqual(table_name, "trajectory_ledger_settings")
        self.assertEqual(column.name, "trajectory_detail_enabled_at")
        self.assertIsInstance(column.type, sa.DateTime)
        self.assertTrue(column.type.timezone)
        self.assertTrue(column.nullable)
        operation.execute.assert_not_called()

    def test_downgrade_removes_only_detail_watermark(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        operation.drop_column.assert_called_once_with(
            "trajectory_ledger_settings",
            "trajectory_detail_enabled_at",
        )


class TrajectoryNodeDetailModelTests(unittest.TestCase):
    def test_orm_watermark_is_nullable_timezone_datetime(self):
        column = TrajectoryLedgerSettings.__table__.c.trajectory_detail_enabled_at

        self.assertTrue(column.nullable)
        self.assertIsInstance(column.type, sa.DateTime)
        self.assertTrue(column.type.timezone)


if __name__ == "__main__":
    unittest.main()
