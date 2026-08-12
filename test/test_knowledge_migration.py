import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy as sa

MIGRATION_PATH = (
    Path(__file__).parent.parent
    / "alembic"
    / "versions"
    / "1f4b7c9d2e60_add_knowledge_base_v1.py"
)


def load_migration():
    spec = importlib.util.spec_from_file_location("knowledge_base_v1_migration", MIGRATION_PATH)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class KnowledgeMigrationTests(unittest.TestCase):
    def test_upgrade_is_expand_only_and_creates_versioned_task_schema(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        created = [call.args[0] for call in operation.create_table.call_args_list]
        self.assertEqual(
            created,
            [
                "knowledge_bases",
                "knowledge_documents",
                "knowledge_index_versions",
                "knowledge_chunk_manifests",
                "knowledge_index_tasks",
            ],
        )
        self.assertFalse(operation.drop_table.called)
        task_args = operation.create_table.call_args_list[-1].args[1:]
        foreign_targets = {
            element.target_fullname
            for item in task_args
            if isinstance(item, sa.ForeignKeyConstraint)
            for element in item.elements
        }
        self.assertIn("knowledge_index_versions.id", foreign_targets)
        self.assertEqual(migration.down_revision, "a6c2f1d9e4b7")

    def test_downgrade_drops_dependents_before_owners(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        dropped = [call.args[0] for call in operation.drop_table.call_args_list]
        self.assertEqual(
            dropped,
            [
                "knowledge_index_tasks",
                "knowledge_chunk_manifests",
                "knowledge_index_versions",
                "knowledge_documents",
                "knowledge_bases",
            ],
        )


if __name__ == "__main__":
    unittest.main()
