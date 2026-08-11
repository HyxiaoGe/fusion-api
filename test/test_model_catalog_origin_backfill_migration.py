import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

MIGRATION_PATH = (
    Path(__file__).parent.parent / "alembic" / "versions" / "a6c2f1d9e4b7_backfill_model_catalog_origin_registry.py"
)


def load_migration():
    spec = importlib.util.spec_from_file_location("model_catalog_origin_backfill_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


class ModelCatalogOriginBackfillMigrationTest(unittest.TestCase):
    def test_upgrade_backfills_registry_only_when_missing(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.upgrade()

        operation.execute.assert_called_once()
        upgrade_sql = str(operation.execute.call_args.args[0])
        self.assertIn("to_regclass('_fusion_alembic_f3a1d9c8b720_origins') IS NULL", upgrade_sql)
        self.assertIn("CREATE TABLE _fusion_alembic_f3a1d9c8b720_origins", upgrade_sql)
        self.assertIn("'model_catalog_controls', true", upgrade_sql)
        self.assertIn("'model_admission_operations', true", upgrade_sql)

    def test_downgrade_preserves_registry_for_parent_revision(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        operation.execute.assert_not_called()
        operation.drop_table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
