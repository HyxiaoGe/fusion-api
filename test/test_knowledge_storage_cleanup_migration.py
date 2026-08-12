import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
MIGRATION_PATH = ROOT / "alembic" / "versions" / "5e7a9c2d4b10_add_knowledge_storage_cleanup.py"


def load_migration():
    spec = importlib.util.spec_from_file_location("knowledge_storage_cleanup_migration", MIGRATION_PATH)
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

        self.assertEqual(heads, {"5e7a9c2d4b10"})
        visited: set[str] = set()
        current: str | None = "5e7a9c2d4b10"
        while current is not None:
            self.assertNotIn(current, visited)
            visited.add(current)
            current = revisions[current]
        self.assertEqual(visited, set(revisions))


if __name__ == "__main__":
    unittest.main()
