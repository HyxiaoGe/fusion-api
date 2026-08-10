import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

MIGRATION_PATH = Path(__file__).parent.parent / "alembic" / "versions" / "f3a1d9c8b720_add_model_catalog_controls.py"


def load_migration():
    spec = importlib.util.spec_from_file_location("model_management_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def compatible_columns(table_name: str):
    if table_name == "model_catalog_controls":
        definitions = [
            ("model_id", sa.String(200), False, None),
            ("selectable", sa.Boolean(), False, "true"),
            ("routable", sa.Boolean(), False, "true"),
            ("revision", sa.Integer(), False, "1"),
            ("reason", sa.String(300), False, None),
            ("updated_by", sa.String(), False, None),
            ("created_at", sa.DateTime(timezone=True), False, "now()"),
            ("updated_at", sa.DateTime(timezone=True), False, "now()"),
        ]
    else:
        definitions = [
            ("id", sa.String(), False, None),
            ("model_id", sa.String(200), False, None),
            ("candidate_fingerprint", sa.String(64), False, None),
            ("governance_run_id", sa.String(32), False, None),
            ("status", sa.String(20), False, "'pending'::character varying"),
            ("requested_by", sa.String(), False, None),
            ("request_id", sa.String(), False, None),
            ("reason", sa.String(300), False, None),
            ("attempts", sa.Integer(), False, "0"),
            ("lease_token_hash", sa.String(64), True, None),
            ("lease_expires_at", sa.DateTime(timezone=True), True, None),
            ("catalog_invalidated_at", sa.DateTime(timezone=True), True, None),
            ("result", postgresql.JSONB(), False, "'{}'::jsonb"),
            ("created_at", sa.DateTime(timezone=True), False, "now()"),
            ("updated_at", sa.DateTime(timezone=True), False, "now()"),
            ("terminal_at", sa.DateTime(timezone=True), True, None),
        ]
    return [
        {
            "name": name,
            "type": column_type,
            "nullable": nullable,
            "default": default,
        }
        for name, column_type, nullable, default in definitions
    ]


def compatible_inspector() -> Mock:
    inspector = Mock()
    inspector.has_table.return_value = True
    inspector.get_columns.side_effect = compatible_columns
    inspector.get_pk_constraint.side_effect = lambda table_name: {
        "constrained_columns": ["model_id"] if table_name == "model_catalog_controls" else ["id"]
    }
    inspector.get_check_constraints.side_effect = lambda table_name: (
        [
            {"name": "ck_model_catalog_controls_routable_true", "sqltext": "routable IS TRUE"},
            {"name": "ck_model_catalog_controls_revision_positive", "sqltext": "revision > 0"},
        ]
        if table_name == "model_catalog_controls"
        else [
            {
                "name": "ck_model_admission_operations_status",
                "sqltext": (
                    "status::text = ANY (ARRAY["
                    "'pending'::character varying, "
                    "'running'::character varying, "
                    "'succeeded'::character varying, "
                    "'failed'::character varying"
                    "]::text[])"
                ),
            }
        ]
    )
    inspector.get_unique_constraints.side_effect = lambda table_name: (
        []
        if table_name == "model_catalog_controls"
        else [
            {
                "name": "uq_model_admission_operation_candidate_run",
                "column_names": ["candidate_fingerprint", "governance_run_id"],
            }
        ]
    )
    inspector.get_indexes.side_effect = lambda table_name: (
        []
        if table_name == "model_catalog_controls"
        else [
            {
                "name": "uq_model_admission_operation_candidate_run",
                "column_names": ["candidate_fingerprint", "governance_run_id"],
                "unique": True,
            },
            {
                "name": "ix_model_admission_operations_status_created",
                "column_names": ["status", "created_at", "id"],
                "unique": False,
            },
            {
                "name": "uq_model_admission_operations_single_running",
                "column_names": ["status"],
                "unique": True,
                "dialect_options": {"postgresql_where": "status = 'running'"},
            },
        ]
    )
    inspector.get_foreign_keys.return_value = []
    return inspector


class ModelManagementMigrationTest(unittest.TestCase):
    def test_fresh_database_creates_both_tables_and_indexes(self):
        migration = load_migration()
        inspector = Mock()
        inspector.has_table.return_value = False

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
        ):
            migration.upgrade()

        self.assertEqual(operation.create_table.call_count, 2)
        self.assertEqual(operation.create_index.call_count, 2)

    def test_compatible_existing_schema_is_adopted_without_ddl(self):
        migration = load_migration()
        inspector = compatible_inspector()

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
        ):
            migration.upgrade()

        operation.create_table.assert_not_called()
        operation.create_index.assert_not_called()

    def test_incompatible_existing_schema_fails_instead_of_blindly_stamping(self):
        migration = load_migration()
        inspector = compatible_inspector()
        catalog_columns = compatible_columns("model_catalog_controls")
        catalog_columns.pop()
        inspector.get_columns.side_effect = lambda table_name: (
            catalog_columns if table_name == "model_catalog_controls" else compatible_columns(table_name)
        )

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "字段集合与迁移定义不一致"),
        ):
            migration.upgrade()

        operation.create_table.assert_not_called()

    def test_offline_sql_generation_never_reflects_database(self):
        migration = load_migration()

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=True),
            patch.object(migration.sa, "inspect") as inspect_database,
        ):
            migration.upgrade()

        inspect_database.assert_not_called()
        self.assertEqual(operation.create_table.call_count, 2)
        self.assertEqual(operation.create_index.call_count, 2)

    def test_same_check_name_with_drifted_expression_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        inspector.get_check_constraints.side_effect = lambda table_name: (
            [
                {"name": "ck_model_catalog_controls_routable_true", "sqltext": "routable IS TRUE"},
                {"name": "ck_model_catalog_controls_revision_positive", "sqltext": "revision > 0"},
            ]
            if table_name == "model_catalog_controls"
            else [
                {
                    "name": "ck_model_admission_operations_status",
                    "sqltext": "status IN ('pending', 'running')",
                }
            ]
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "检查约束 .* 表达式不兼容"),
        ):
            migration.upgrade()

    def test_same_partial_index_name_with_opposite_predicate_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_indexes = inspector.get_indexes("model_admission_operations")
        operation_indexes[-1]["dialect_options"] = {"postgresql_where": "status <> 'running'"}
        inspector.get_indexes.side_effect = lambda table_name: (
            [] if table_name == "model_catalog_controls" else operation_indexes
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "索引 .* 不兼容"),
        ):
            migration.upgrade()

    def test_extra_restrictive_check_constraint_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_get_checks = inspector.get_check_constraints.side_effect
        inspector.get_check_constraints.side_effect = lambda table_name: (
            original_get_checks(table_name)
            if table_name == "model_catalog_controls"
            else [
                *original_get_checks(table_name),
                {"name": "ck_model_admission_operations_attempts", "sqltext": "attempts < 3"},
            ]
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "检查约束集合不兼容"),
        ):
            migration.upgrade()

    def test_full_index_with_partial_predicate_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_indexes = inspector.get_indexes("model_admission_operations")
        operation_indexes[1]["dialect_options"] = {"postgresql_where": "status = 'pending'"}
        inspector.get_indexes.side_effect = lambda table_name: (
            [] if table_name == "model_catalog_controls" else operation_indexes
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "索引 .* 不兼容"),
        ):
            migration.upgrade()

    def test_extra_unique_index_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        inspector.get_indexes.side_effect = lambda table_name: (
            [
                {
                    "name": "uq_model_catalog_controls_updated_by",
                    "column_names": ["updated_by"],
                    "unique": True,
                }
            ]
            if table_name == "model_catalog_controls"
            else compatible_inspector().get_indexes(table_name)
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "迁移定义外的唯一索引"),
        ):
            migration.upgrade()

    def test_extra_foreign_key_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        inspector.get_foreign_keys.side_effect = lambda table_name: (
            [{"name": "fk_model_catalog_controls_user"}] if table_name == "model_catalog_controls" else []
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "迁移定义外的外键"),
        ):
            migration.upgrade()

    def test_column_default_drift_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_columns = compatible_columns("model_admission_operations")
        next(column for column in operation_columns if column["name"] == "attempts")["default"] = "1"
        inspector.get_columns.side_effect = lambda table_name: (
            compatible_columns(table_name) if table_name == "model_catalog_controls" else operation_columns
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "默认值不兼容"),
        ):
            migration.upgrade()


if __name__ == "__main__":
    unittest.main()
