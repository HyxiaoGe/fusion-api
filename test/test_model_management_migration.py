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
    inspector.has_table.side_effect = lambda table_name: table_name != "_fusion_alembic_f3a1d9c8b720_origins"
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
                "dialect_options": {"postgresql_where": "(status)::text = 'running'::text"},
            },
        ]
    )
    inspector.get_foreign_keys.return_value = []

    def execute_catalog_query(statement, parameters):
        query = str(statement)
        result = Mock()
        if "relpersistence" in query:
            result.one_or_none.return_value = ("r", "p", False, False, False, False, False)
            return result
        if "uses_type_default_collation" in query:
            result.mappings.return_value = [
                {
                    "column_name": column["name"],
                    "uses_type_default_collation": True,
                }
                for column in compatible_columns(parameters["table_name"])
            ]
            return result
        table_name = parameters["table_name"]
        result.mappings.return_value = [
            {
                "index_name": index["name"],
                "indisvalid": True,
                "indisready": True,
                "indislive": True,
            }
            for index in inspector.get_indexes(table_name)
        ]
        return result

    inspector.bind.execute.side_effect = execute_catalog_query
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

        self.assertEqual(operation.create_table.call_count, 3)
        self.assertEqual(operation.create_index.call_count, 2)
        self.assertEqual(operation.execute.call_count, 1)

    def test_compatible_existing_schema_is_adopted_without_ddl(self):
        migration = load_migration()
        inspector = compatible_inspector()

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
        ):
            migration.upgrade()

        self.assertEqual(operation.create_table.call_count, 1)
        self.assertEqual(operation.create_table.call_args.args[0], migration.ORIGIN_REGISTRY_TABLE)
        operation.create_index.assert_not_called()
        operation.execute.assert_called_once()

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
        self.assertEqual(operation.create_table.call_count, 3)
        self.assertEqual(operation.create_index.call_count, 2)
        self.assertEqual(operation.execute.call_count, 1)

    def test_redundant_unique_constraint_on_catalog_primary_key_is_adopted(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_get_unique_constraints = inspector.get_unique_constraints.side_effect
        original_get_indexes = inspector.get_indexes.side_effect
        inspector.get_unique_constraints.side_effect = lambda table_name: (
            [
                {
                    "name": "model_catalog_controls_model_id_key",
                    "column_names": ["model_id"],
                }
            ]
            if table_name == "model_catalog_controls"
            else original_get_unique_constraints(table_name)
        )
        inspector.get_indexes.side_effect = lambda table_name: (
            [
                {
                    "name": "model_catalog_controls_model_id_key",
                    "column_names": ["model_id"],
                    "unique": True,
                    "duplicates_constraint": "model_catalog_controls_model_id_key",
                }
            ]
            if table_name == "model_catalog_controls"
            else original_get_indexes(table_name)
        )

        with (
            patch.object(migration, "op") as operation,
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
        ):
            migration.upgrade()

        self.assertEqual(operation.create_table.call_count, 1)
        self.assertEqual(operation.create_table.call_args.args[0], migration.ORIGIN_REGISTRY_TABLE)
        operation.create_index.assert_not_called()

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

    def test_semantic_cast_in_check_expression_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_get_checks = inspector.get_check_constraints.side_effect
        inspector.get_check_constraints.side_effect = lambda table_name: (
            original_get_checks(table_name)
            if table_name == "model_catalog_controls"
            else [
                {
                    "name": "ck_model_admission_operations_status",
                    "sqltext": (
                        "status::boolean::text = ANY (ARRAY["
                        "'pending'::character varying, "
                        "'running'::character varying, "
                        "'succeeded'::character varying, "
                        "'failed'::character varying"
                        "]::text[])"
                    ),
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

    def test_partial_index_string_literal_case_is_preserved(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_indexes = inspector.get_indexes("model_admission_operations")
        operation_indexes[-1]["dialect_options"] = {"postgresql_where": "status = 'RUNNING'"}
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
            self.assertRaisesRegex(RuntimeError, "迁移定义外的索引"),
        ):
            migration.upgrade()

    def test_extra_non_unique_expression_index_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_indexes = inspector.get_indexes("model_admission_operations")
        operation_indexes.append(
            {
                "name": "ix_model_admission_operations_divide_attempts",
                "column_names": [None],
                "expressions": ["1 / attempts"],
                "unique": False,
            }
        )
        inspector.get_indexes.side_effect = lambda table_name: (
            [] if table_name == "model_catalog_controls" else operation_indexes
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "迁移定义外的索引"),
        ):
            migration.upgrade()

    def test_invalid_index_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "pg_index.indisvalid" in str(statement) and parameters["table_name"] == "model_admission_operations":
                index_states = list(result.mappings())
                next(
                    state
                    for state in index_states
                    if state["index_name"] == "uq_model_admission_operations_single_running"
                )["indisvalid"] = False
                result.mappings.return_value = index_states
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "无效或未就绪索引"),
        ):
            migration.upgrade()

    def test_unlogged_table_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "relpersistence" in str(statement) and parameters["table_name"] == "model_catalog_controls":
                result.one_or_none.return_value = ("r", "u")
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "必须是 PostgreSQL 持久普通表"),
        ):
            migration.upgrade()

    def test_rls_enabled_table_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "relpersistence" in str(statement) and parameters["table_name"] == "model_catalog_controls":
                result.one_or_none.return_value = ("r", "p", True, True)
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能启用行级安全"),
        ):
            migration.upgrade()

    def test_table_with_user_trigger_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "relpersistence" in str(statement) and parameters["table_name"] == "model_admission_operations":
                result.one_or_none.return_value = ("r", "p", False, False, False, True)
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能包含用户触发器"),
        ):
            migration.upgrade()

    def test_table_with_inheritance_relationship_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "relpersistence" in str(statement) and parameters["table_name"] == "model_admission_operations":
                result.one_or_none.return_value = ("r", "p", False, False, True, False)
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能参与表继承"),
        ):
            migration.upgrade()

    def test_table_with_rewrite_rule_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "relpersistence" in str(statement) and parameters["table_name"] == "model_admission_operations":
                result.one_or_none.return_value = ("r", "p", False, False, False, False, True)
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能包含 rewrite rule"),
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

    def test_char_column_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        catalog_columns = compatible_columns("model_catalog_controls")
        next(column for column in catalog_columns if column["name"] == "model_id")["type"] = sa.CHAR(200)
        inspector.get_columns.side_effect = lambda table_name: (
            catalog_columns if table_name == "model_catalog_controls" else compatible_columns(table_name)
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能使用定长 CHAR 类型"),
        ):
            migration.upgrade()

    def test_small_integer_column_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        catalog_columns = compatible_columns("model_catalog_controls")
        next(column for column in catalog_columns if column["name"] == "revision")["type"] = sa.SmallInteger()
        inspector.get_columns.side_effect = lambda table_name: (
            catalog_columns if table_name == "model_catalog_controls" else compatible_columns(table_name)
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "整数类型宽度不兼容"),
        ):
            migration.upgrade()

    def test_non_default_string_collation_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_execute = inspector.bind.execute.side_effect

        def execute_catalog_query(statement, parameters):
            result = original_execute(statement, parameters)
            if "uses_type_default_collation" in str(statement):
                result.mappings.return_value = [
                    {
                        "column_name": column["name"],
                        "uses_type_default_collation": column["name"] != "status",
                    }
                    for column in compatible_columns(parameters["table_name"])
                ]
            return result

        inspector.bind.execute.side_effect = execute_catalog_query

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "排序规则不兼容"),
        ):
            migration.upgrade()

    def test_origin_registry_is_excluded_from_alembic_autogenerate(self):
        env_source = (MIGRATION_PATH.parent.parent / "env.py").read_text(encoding="utf-8")

        self.assertIn('name == "_fusion_alembic_f3a1d9c8b720_origins"', env_source)
        self.assertEqual(env_source.count("include_object=include_object"), 2)

    def test_not_valid_check_constraint_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        original_get_checks = inspector.get_check_constraints.side_effect
        catalog_checks = original_get_checks("model_catalog_controls")
        catalog_checks[0]["dialect_options"] = {"not_valid": True}
        inspector.get_check_constraints.side_effect = lambda table_name: (
            catalog_checks if table_name == "model_catalog_controls" else original_get_checks(table_name)
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "尚未完整验证"),
        ):
            migration.upgrade()

    def test_computed_column_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_columns = compatible_columns("model_admission_operations")
        next(column for column in operation_columns if column["name"] == "reason")["computed"] = {
            "sqltext": "'legacy'",
            "persisted": True,
        }
        inspector.get_columns.side_effect = lambda table_name: (
            compatible_columns(table_name) if table_name == "model_catalog_controls" else operation_columns
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能是生成列或 identity 列"),
        ):
            migration.upgrade()

    def test_identity_column_is_rejected(self):
        migration = load_migration()
        inspector = compatible_inspector()
        operation_columns = compatible_columns("model_admission_operations")
        next(column for column in operation_columns if column["name"] == "attempts")["identity"] = {"always": True}
        inspector.get_columns.side_effect = lambda table_name: (
            compatible_columns(table_name) if table_name == "model_catalog_controls" else operation_columns
        )

        with (
            patch.object(migration, "op"),
            patch.object(migration.context, "is_offline_mode", return_value=False),
            patch.object(migration.sa, "inspect", return_value=inspector),
            self.assertRaisesRegex(RuntimeError, "不能是生成列或 identity 列"),
        ):
            migration.upgrade()

    def test_downgrade_only_drops_tables_marked_as_created_by_this_migration(self):
        migration = load_migration()

        with patch.object(migration, "op") as operation:
            migration.downgrade()

        operation.drop_index.assert_not_called()
        operation.drop_table.assert_not_called()
        operation.execute.assert_called_once()
        downgrade_sql = str(operation.execute.call_args.args[0])
        self.assertIn(migration.ORIGIN_REGISTRY_TABLE, downgrade_sql)
        self.assertIn("缺少迁移来源登记表", downgrade_sql)
        self.assertIn("DROP TABLE model_admission_operations", downgrade_sql)
        self.assertIn("DROP TABLE model_catalog_controls", downgrade_sql)
        self.assertIn("END;\n            $fusion$;", downgrade_sql)


if __name__ == "__main__":
    unittest.main()
