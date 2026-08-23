import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.db.models import ToolCallLog

MIGRATION_PATH = Path(__file__).parent.parent / "alembic" / "versions" / "4a7c9e2b6d81_add_tool_call_log_linkage.py"


def load_migration():
    if not MIGRATION_PATH.exists():
        raise AssertionError("缺少 ToolCallLog 精确关联迁移")
    spec = importlib.util.spec_from_file_location("tool_call_log_linkage_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class ToolCallLogLinkageMigrationTests(unittest.TestCase):
    def test_revision_extends_current_single_head(self):
        migration = load_migration()

        self.assertEqual(migration.revision, "4a7c9e2b6d81")
        self.assertEqual(migration.down_revision, "e8f5a1c4d2b7")

    def test_upgrade_adds_nullable_column_and_partial_unique_index(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        operation.add_column.assert_called_once()
        table_name, column = operation.add_column.call_args.args
        self.assertEqual(table_name, "tool_call_logs")
        self.assertEqual(column.name, "tool_call_id")
        self.assertIsInstance(column.type, sa.String)
        self.assertTrue(column.nullable)

        operation.create_index.assert_called_once()
        index_call = operation.create_index.call_args
        self.assertEqual(
            index_call.args[:3],
            (
                "uq_tool_call_logs_trace_tool_call",
                "tool_call_logs",
                ["trace_id", "tool_call_id"],
            ),
        )
        self.assertTrue(index_call.kwargs["unique"])
        self.assertEqual(
            str(index_call.kwargs["postgresql_where"]),
            "trace_id IS NOT NULL AND tool_call_id IS NOT NULL",
        )

    def test_downgrade_removes_index_before_column(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        operation.drop_index.assert_called_once_with(
            "uq_tool_call_logs_trace_tool_call",
            table_name="tool_call_logs",
        )
        operation.drop_column.assert_called_once_with("tool_call_logs", "tool_call_id")


class ToolCallLogLinkageModelTests(unittest.TestCase):
    def test_model_matches_nullable_partial_unique_contract(self):
        column = ToolCallLog.__table__.c.tool_call_id
        self.assertTrue(column.nullable)
        self.assertIsInstance(column.type, sa.String)

        index = next(
            index for index in ToolCallLog.__table__.indexes if index.name == "uq_tool_call_logs_trace_tool_call"
        )
        self.assertTrue(index.unique)
        self.assertEqual([column.name for column in index.columns], ["trace_id", "tool_call_id"])
        predicate = index.dialect_options["postgresql"]["where"].compile(dialect=postgresql.dialect())
        self.assertEqual(
            str(predicate),
            "trace_id IS NOT NULL AND tool_call_id IS NOT NULL",
        )


if __name__ == "__main__":
    unittest.main()
