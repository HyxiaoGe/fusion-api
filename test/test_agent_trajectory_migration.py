import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta, TrajectoryLedgerSettings

MIGRATION_PATH = (
    Path(__file__).parent.parent
    / "alembic"
    / "versions"
    / "e8f5a1c4d2b7_add_agent_trajectory_ledger.py"
)


def load_migration():
    spec = importlib.util.spec_from_file_location("agent_trajectory_ledger_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class AgentTrajectoryMigrationTests(unittest.TestCase):
    def test_revision_extends_the_current_agent_session_head(self):
        migration = load_migration()

        self.assertEqual(migration.revision, "e8f5a1c4d2b7")
        self.assertEqual(migration.down_revision, "d7e4a9c2f1b6")

    def test_upgrade_expands_and_backfills_agent_session_attempt_hierarchy(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        added_columns = {
            call.args[1].name: call.args[1]
            for call in operation.add_column.call_args_list
            if call.args[0] == "agent_sessions"
        }
        self.assertEqual(set(added_columns), {"turn_message_id", "previous_run_id", "attempt_index"})
        self.assertTrue(all(column.nullable for column in added_columns.values()))
        operation.create_foreign_key.assert_any_call(
            "fk_agent_sessions_previous_run_id",
            "agent_sessions",
            "agent_sessions",
            ["previous_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        attempt_index_call = next(
            call for call in operation.create_index.call_args_list if call.args[0] == "uq_agent_sessions_turn_attempt"
        )
        self.assertEqual(attempt_index_call.args[1:], ("agent_sessions", ["turn_message_id", "attempt_index"]))
        self.assertTrue(attempt_index_call.kwargs["unique"])
        self.assertEqual(
            str(attempt_index_call.kwargs["postgresql_where"]),
            "turn_message_id IS NOT NULL AND attempt_index IS NOT NULL",
        )

        executed_sql = "\n".join(str(call.args[0]) for call in operation.execute.call_args_list)
        self.assertIn("SET turn_message_id = message_id", executed_sql)
        self.assertIn("WHERE message_id IS NOT NULL", executed_sql)
        self.assertIn("row_number() OVER", executed_sql)
        self.assertIn("PARTITION BY turn_message_id", executed_sql)
        self.assertIn("ORDER BY created_at, id", executed_sql)
        self.assertNotIn("SET previous_run_id", executed_sql)

    def test_upgrade_creates_ledger_tables_with_cascade_foreign_keys_and_environment_watermark(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        tables = {call.args[0]: call.args[1:] for call in operation.create_table.call_args_list}
        self.assertEqual(set(tables), {"agent_events", "run_trajectory_meta", "trajectory_ledger_settings"})
        for table_name in ("agent_events", "run_trajectory_meta"):
            foreign_keys = [item for item in tables[table_name] if isinstance(item, sa.ForeignKeyConstraint)]
            targets = {
                (source, element.target_fullname, constraint.ondelete)
                for constraint in foreign_keys
                for source, element in zip(constraint.column_keys, constraint.elements, strict=True)
            }
            self.assertIn(("conversation_id", "conversations.id", "CASCADE"), targets)
            self.assertIn(("run_id", "agent_sessions.id", "CASCADE"), targets)

        settings_columns = {
            item.name: item for item in tables["trajectory_ledger_settings"] if isinstance(item, sa.Column)
        }
        self.assertIsInstance(settings_columns["ledger_enabled_at"].type, sa.DateTime)
        self.assertTrue(settings_columns["ledger_enabled_at"].type.timezone)
        executed_sql = "\n".join(str(call.args[0]) for call in operation.execute.call_args_list)
        self.assertIn("INSERT INTO trajectory_ledger_settings", executed_sql)
        self.assertIn("CURRENT_TIMESTAMP", executed_sql)

    def test_downgrade_removes_only_the_objects_added_by_this_revision(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        self.assertEqual(
            [call.args[0] for call in operation.drop_table.call_args_list],
            ["trajectory_ledger_settings", "run_trajectory_meta", "agent_events"],
        )
        self.assertEqual(
            [call.args[1] for call in operation.drop_column.call_args_list],
            ["attempt_index", "previous_run_id", "turn_message_id"],
        )


class AgentTrajectoryModelContractTests(unittest.TestCase):
    def test_orm_models_match_the_ledger_contract(self):
        self.assertEqual(AgentEvent.__table__.c.event_id.type.python_type.__name__, "UUID")
        self.assertFalse(AgentEvent.__table__.c.payload.nullable)
        self.assertEqual(
            {column.name for column in AgentEvent.__table__.constraints if isinstance(column, sa.UniqueConstraint)},
            {"uq_agent_events_run_sequence"},
        )
        self.assertEqual(RunTrajectoryMeta.__table__.c.run_id.foreign_keys.pop().ondelete, "CASCADE")
        self.assertFalse(TrajectoryLedgerSettings.__table__.c.ledger_enabled_at.nullable)

        attempt_index = next(index for index in AgentSession.__table__.indexes if index.name == "uq_agent_sessions_turn_attempt")
        compiled_where = str(attempt_index.dialect_options["postgresql"]["where"].compile(dialect=postgresql.dialect()))
        self.assertTrue(attempt_index.unique)
        self.assertEqual(compiled_where, "turn_message_id IS NOT NULL AND attempt_index IS NOT NULL")


if __name__ == "__main__":
    unittest.main()
