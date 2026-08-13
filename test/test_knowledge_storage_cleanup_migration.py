import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
CLEANUP_MIGRATION_PATH = ROOT / "alembic" / "versions" / "5e7a9c2d4b10_add_knowledge_storage_cleanup.py"
OUTCOME_MIGRATION_PATH = ROOT / "alembic" / "versions" / "6f8b1d3c5a20_add_knowledge_upload_outcome.py"
FILE_SCOPE_MIGRATION_PATH = ROOT / "alembic" / "versions" / "9c4e7a2b6d11_add_file_cleanup_scope_and_vector_route.py"


def load_migration(path=CLEANUP_MIGRATION_PATH, name="knowledge_storage_cleanup_migration"):
    spec = importlib.util.spec_from_file_location(name, path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class KnowledgeStorageCleanupMigrationTests(unittest.TestCase):
    def test_migration_is_expand_only_and_drops_intents_before_tasks(self):
        migration = load_migration()
        with patch.object(migration, "op") as operation:
            migration.upgrade()
        self.assertEqual(
            [call.args[0] for call in operation.create_table.call_args_list],
            ["knowledge_storage_cleanup_tasks", "knowledge_storage_upload_intents"],
        )
        self.assertFalse(operation.drop_table.called)
        self.assertEqual(migration.down_revision, "1f4b7c9d2e60")

        with patch.object(migration, "op") as operation:
            migration.downgrade()
        self.assertEqual(
            [call.args[0] for call in operation.drop_table.call_args_list],
            ["knowledge_storage_upload_intents", "knowledge_storage_cleanup_tasks"],
        )

    def test_upload_outcome_migration_is_expand_only_and_matches_state_machine(self):
        migration = load_migration(OUTCOME_MIGRATION_PATH, "knowledge_upload_outcome_migration")
        with patch.object(migration, "op") as operation:
            migration.upgrade()

        columns = [call.args[1] for call in operation.add_column.call_args_list]
        self.assertEqual([column.name for column in columns], ["outcome", "outcome_at"])
        self.assertEqual(columns[0].server_default.arg, "uploading")
        constraint = operation.create_check_constraint.call_args.args[2]
        self.assertIn("'uncertain'", constraint)
        self.assertEqual(migration.down_revision, "5e7a9c2d4b10")
        self.assertFalse(operation.drop_table.called)

    def test_file_cleanup_scope_and_vector_route_migration_is_expand_only(self):
        migration = load_migration(FILE_SCOPE_MIGRATION_PATH, "knowledge_file_cleanup_scope_migration")
        with patch.object(migration, "op") as operation:
            migration.upgrade()

        added = [(call.args[0], call.args[1]) for call in operation.add_column.call_args_list]
        self.assertEqual(
            [(table, column.name, column.nullable) for table, column in added],
            [
                ("knowledge_storage_cleanup_tasks", "file_status", True),
                ("knowledge_index_versions", "milvus_uri", True),
                ("knowledge_index_versions", "milvus_database", True),
            ],
        )
        constraint = operation.create_check_constraint.call_args.args[2]
        self.assertIn("file_status IS NULL", constraint)
        operation.create_index.assert_called_once_with(
            "ix_knowledge_storage_cleanup_file_claim",
            "knowledge_storage_cleanup_tasks",
            ["file_status", "available_at", "created_at", "id"],
        )
        backfill_sql = str(operation.execute.call_args.args[0])
        self.assertIn("FROM files AS file", backfill_sql)
        self.assertIn("file.storage_key = cleanup.storage_key", backfill_sql)
        self.assertIn("file.thumbnail_key = cleanup.storage_key", backfill_sql)
        self.assertIn("status = 'completed'", backfill_sql)
        self.assertEqual(migration.down_revision, "8a3d5f7b9c10")
        self.assertFalse(operation.drop_column.called)

    def test_revision_chain_has_single_cleanup_head(self):
        revisions: dict[str, str | None] = {}
        for path in (ROOT / "alembic" / "versions").glob("*.py"):
            assignments: dict[str, str | None] = {}
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                target = node.target if isinstance(node, ast.AnnAssign) else None
                if isinstance(target, ast.Name) and target.id in {"revision", "down_revision"}:
                    assignments[target.id] = ast.literal_eval(node.value)
            if assignments:
                revisions[assignments["revision"]] = assignments["down_revision"]
        heads = set(revisions) - {parent for parent in revisions.values() if parent is not None}

        self.assertEqual(heads, {"9c4e7a2b6d11"})
        visited: set[str] = set()
        current: str | None = "9c4e7a2b6d11"
        while current is not None:
            self.assertNotIn(current, visited)
            visited.add(current)
            current = revisions[current]
        self.assertEqual(visited, set(revisions))


if __name__ == "__main__":
    unittest.main()
