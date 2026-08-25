import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa

from app.db import models as db_models

MIGRATION_PATH = Path(__file__).parent.parent / "alembic" / "versions" / "a4d8c2e7f901_add_llm_round_details.py"


def load_migration():
    if not MIGRATION_PATH.exists():
        raise AssertionError("缺少 LLM Round Detail 迁移")
    spec = importlib.util.spec_from_file_location("trajectory_llm_round_detail_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class TrajectoryLlmRoundDetailMigrationTests(unittest.TestCase):
    def test_revision_extends_node_detail_watermark_head(self):
        migration = load_migration()

        self.assertEqual(migration.revision, "a4d8c2e7f901")
        self.assertEqual(migration.down_revision, "9f2d6c1a8b43")

    def test_upgrade_adds_capability_version_and_detail_table_without_backfill(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        operation.add_column.assert_called_once()
        table_name, version_column = operation.add_column.call_args.args
        self.assertEqual(table_name, "run_trajectory_meta")
        self.assertEqual(version_column.name, "llm_detail_schema_version")
        self.assertIsInstance(version_column.type, sa.Integer)
        self.assertTrue(version_column.nullable)

        operation.create_table.assert_called_once()
        table_args = operation.create_table.call_args.args
        self.assertEqual(table_args[0], "agent_llm_round_details")
        columns = {item.name: item for item in table_args[1:] if isinstance(item, sa.Column)}
        self.assertEqual(
            set(columns),
            {
                "id",
                "conversation_id",
                "run_id",
                "message_id",
                "llm_round_id",
                "reasoning_text",
                "content_text",
                "reasoning_preview",
                "output_preview",
                "redacted_fields",
                "truncated_fields",
                "recorded_at",
            },
        )
        self.assertFalse(columns["conversation_id"].nullable)
        self.assertFalse(columns["run_id"].nullable)
        self.assertFalse(columns["llm_round_id"].nullable)
        self.assertTrue(columns["message_id"].nullable)

        foreign_keys = [item for item in table_args[1:] if isinstance(item, sa.ForeignKeyConstraint)]
        targets = {next(iter(item.elements)).target_fullname: item.ondelete for item in foreign_keys}
        self.assertEqual(targets["conversations.id"], "CASCADE")
        self.assertEqual(targets["agent_sessions.id"], "CASCADE")
        self.assertEqual(targets["messages.id"], "SET NULL")

        unique_constraints = [item for item in table_args[1:] if isinstance(item, sa.UniqueConstraint)]
        self.assertEqual(len(unique_constraints), 1)
        self.assertEqual(unique_constraints[0].name, "uq_agent_llm_round_details_run_round")
        self.assertEqual(list(unique_constraints[0]._pending_colargs), ["run_id", "llm_round_id"])
        operation.execute.assert_not_called()

    def test_downgrade_removes_detail_table_then_capability_version(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        self.assertEqual(
            operation.method_calls,
            [
                unittest.mock.call.drop_table("agent_llm_round_details"),
                unittest.mock.call.drop_column("run_trajectory_meta", "llm_detail_schema_version"),
            ],
        )


class TrajectoryLlmRoundDetailModelTests(unittest.TestCase):
    def test_orm_exposes_capability_version_and_exact_round_identity(self):
        self.assertTrue(hasattr(db_models, "AgentLlmRoundDetail"))
        detail_model = db_models.AgentLlmRoundDetail

        version_column = db_models.RunTrajectoryMeta.__table__.c.llm_detail_schema_version
        self.assertTrue(version_column.nullable)
        self.assertIsInstance(version_column.type, sa.Integer)

        table = detail_model.__table__
        self.assertEqual(table.c.run_id.foreign_keys.pop().target_fullname, "agent_sessions.id")
        self.assertEqual(table.c.conversation_id.foreign_keys.pop().target_fullname, "conversations.id")
        unique = next(
            constraint
            for constraint in table.constraints
            if isinstance(constraint, sa.UniqueConstraint) and constraint.name == "uq_agent_llm_round_details_run_round"
        )
        self.assertEqual([column.name for column in unique.columns], ["run_id", "llm_round_id"])


if __name__ == "__main__":
    unittest.main()
