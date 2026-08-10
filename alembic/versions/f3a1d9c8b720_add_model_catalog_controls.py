"""add model catalog controls and admission operations

Revision ID: f3a1d9c8b720
Revises: e8b4c2d7f901
Create Date: 2026-08-04 00:00:00.000000
"""

import re
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import context, op

revision: str = "f3a1d9c8b720"
down_revision: Union[str, Sequence[str], None] = "e8b4c2d7f901"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ORIGIN_REGISTRY_TABLE = "_fusion_alembic_f3a1d9c8b720_origins"
MANAGED_TABLES = ("model_catalog_controls", "model_admission_operations")


MODEL_CATALOG_CONTROL_COLUMNS = {
    "model_id": (sa.String, 200, False, None),
    "selectable": (sa.Boolean, None, False, "true"),
    "routable": (sa.Boolean, None, False, "true"),
    "revision": (sa.Integer, None, False, "1"),
    "reason": (sa.String, 300, False, None),
    "updated_by": (sa.String, None, False, None),
    "created_at": (sa.DateTime, None, False, "now()"),
    "updated_at": (sa.DateTime, None, False, "now()"),
}

MODEL_ADMISSION_OPERATION_COLUMNS = {
    "id": (sa.String, None, False, None),
    "model_id": (sa.String, 200, False, None),
    "candidate_fingerprint": (sa.String, 64, False, None),
    "governance_run_id": (sa.String, 32, False, None),
    "status": (sa.String, 20, False, "'pending'"),
    "requested_by": (sa.String, None, False, None),
    "request_id": (sa.String, None, False, None),
    "reason": (sa.String, 300, False, None),
    "attempts": (sa.Integer, None, False, "0"),
    "lease_token_hash": (sa.String, 64, True, None),
    "lease_expires_at": (sa.DateTime, None, True, None),
    "catalog_invalidated_at": (sa.DateTime, None, True, None),
    "result": (postgresql.JSONB, None, False, "'{}'"),
    "created_at": (sa.DateTime, None, False, "now()"),
    "updated_at": (sa.DateTime, None, False, "now()"),
    "terminal_at": (sa.DateTime, None, True, None),
}


def _normalize_unquoted_sql(segment: str) -> str:
    normalized = re.sub(
        r"::(?:character varying|text|boolean|integer|jsonb)(?:\[\])?",
        "",
        segment,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[\s()]", "", normalized.lower())


def _normalize_sql_expression(expression: object) -> str:
    sql = str(expression)
    normalized: list[str] = []
    unquoted_start = 0
    index = 0

    while index < len(sql):
        quote = sql[index]
        if quote not in {"'", '"'}:
            index += 1
            continue

        normalized.append(_normalize_unquoted_sql(sql[unquoted_start:index]))
        quoted_end = index + 1
        while quoted_end < len(sql):
            if sql[quoted_end] != quote:
                quoted_end += 1
                continue
            if quoted_end + 1 < len(sql) and sql[quoted_end + 1] == quote:
                quoted_end += 2
                continue
            quoted_end += 1
            break
        normalized.append(sql[index:quoted_end])
        index = quoted_end
        unquoted_start = index

    normalized.append(_normalize_unquoted_sql(sql[unquoted_start:]))
    return "".join(normalized)


def _validate_existing_table(
    inspector: sa.Inspector,
    table_name: str,
    expected_columns: dict[
        str,
        tuple[type[sa.types.TypeEngine], int | None, bool, str | None],
    ],
    *,
    expected_primary_key: set[str],
    expected_checks: dict[str, str],
    expected_unique_constraints: dict[str, set[str]] | None = None,
    allowed_redundant_unique_constraint_columns: set[frozenset[str]] | None = None,
    expected_indexes: dict[str, tuple[tuple[str, ...], bool, str | None]] | None = None,
) -> None:
    relation_state_row = inspector.bind.execute(
        sa.text(
            """
            SELECT relkind, relpersistence
            FROM pg_catalog.pg_class
            WHERE oid = to_regclass(:table_name)
            """
        ),
        {"table_name": table_name},
    ).one_or_none()
    relation_state = tuple(relation_state_row) if relation_state_row is not None else None
    if relation_state != ("r", "p"):
        raise RuntimeError(f"已有表 {table_name} 必须是 PostgreSQL 持久普通表：actual={relation_state}")

    actual_columns = {column["name"]: column for column in inspector.get_columns(table_name)}
    if set(actual_columns) != set(expected_columns):
        raise RuntimeError(
            f"已有表 {table_name} 的字段集合与迁移定义不一致："
            f"actual={sorted(actual_columns)} expected={sorted(expected_columns)}"
        )

    for column_name, (
        expected_type,
        expected_length,
        expected_nullable,
        expected_default,
    ) in expected_columns.items():
        column = actual_columns[column_name]
        actual_type = column["type"]
        if column.get("computed") is not None or column.get("identity") is not None:
            raise RuntimeError(f"已有表 {table_name}.{column_name} 不能是生成列或 identity 列")
        if issubclass(expected_type, sa.String) and isinstance(actual_type, sa.CHAR):
            raise RuntimeError(f"已有表 {table_name}.{column_name} 不能使用定长 CHAR 类型")
        if not isinstance(actual_type, expected_type):
            raise RuntimeError(
                f"已有表 {table_name}.{column_name} 的类型不兼容："
                f"actual={actual_type} expected={expected_type.__name__}"
            )
        if issubclass(expected_type, sa.String) and getattr(actual_type, "length", None) != expected_length:
            raise RuntimeError(
                f"已有表 {table_name}.{column_name} 的长度不兼容："
                f"actual={getattr(actual_type, 'length', None)} expected={expected_length}"
            )
        if issubclass(expected_type, sa.DateTime) and getattr(actual_type, "timezone", False) is not True:
            raise RuntimeError(f"已有表 {table_name}.{column_name} 必须使用带时区时间戳")
        if bool(column.get("nullable")) is not expected_nullable:
            raise RuntimeError(
                f"已有表 {table_name}.{column_name} 的 nullable 不兼容："
                f"actual={column.get('nullable')} expected={expected_nullable}"
            )
        actual_default = column.get("default")
        if (
            (expected_default is None and actual_default is not None)
            or (expected_default is not None and actual_default is None)
            or (
                expected_default is not None
                and actual_default is not None
                and _normalize_sql_expression(actual_default) != _normalize_sql_expression(expected_default)
            )
        ):
            raise RuntimeError(
                f"已有表 {table_name}.{column_name} 的默认值不兼容：actual={actual_default} expected={expected_default}"
            )

    primary_key = set(inspector.get_pk_constraint(table_name).get("constrained_columns") or [])
    if primary_key != expected_primary_key:
        raise RuntimeError(
            f"已有表 {table_name} 的主键不兼容：actual={sorted(primary_key)} expected={sorted(expected_primary_key)}"
        )

    reflected_checks = [
        constraint for constraint in inspector.get_check_constraints(table_name) if constraint.get("name")
    ]
    checks = {constraint.get("name"): constraint.get("sqltext") for constraint in reflected_checks}
    if set(checks) != set(expected_checks):
        raise RuntimeError(
            f"已有表 {table_name} 的检查约束集合不兼容：actual={sorted(checks)} expected={sorted(expected_checks)}"
        )
    for check_name, expected_expression in expected_checks.items():
        actual_expression = checks[check_name]
        if _normalize_sql_expression(actual_expression) != _normalize_sql_expression(expected_expression):
            raise RuntimeError(f"已有表 {table_name} 的检查约束 {check_name} 表达式不兼容")
    for constraint in reflected_checks:
        dialect_options = constraint.get("dialect_options") or {}
        if dialect_options.get("not_valid") or dialect_options.get("no_inherit"):
            raise RuntimeError(f"已有表 {table_name} 的检查约束 {constraint['name']} 尚未完整验证")

    if expected_unique_constraints is not None:
        unique_constraints = {
            constraint.get("name"): set(constraint.get("column_names") or [])
            for constraint in inspector.get_unique_constraints(table_name)
            if constraint.get("name")
        }
        allowed_redundant_columns = allowed_redundant_unique_constraint_columns or set()
        unexpected_unique_constraints = {
            constraint_name: column_names
            for constraint_name, column_names in unique_constraints.items()
            if constraint_name not in expected_unique_constraints
            and frozenset(column_names) not in allowed_redundant_columns
        }
        missing_unique_constraints = set(expected_unique_constraints) - set(unique_constraints)
        if unexpected_unique_constraints or missing_unique_constraints:
            raise RuntimeError(
                f"已有表 {table_name} 的唯一约束集合不兼容："
                f"unexpected={sorted(unexpected_unique_constraints)} "
                f"missing={sorted(missing_unique_constraints)}"
            )
        for constraint_name, expected_column_names in expected_unique_constraints.items():
            if unique_constraints.get(constraint_name) != expected_column_names:
                raise RuntimeError(f"已有表 {table_name} 的唯一约束 {constraint_name} 不兼容")

    indexes = {index.get("name"): index for index in inspector.get_indexes(table_name) if index.get("name")}
    index_states = {
        row["index_name"]: (
            bool(row["indisvalid"]),
            bool(row["indisready"]),
            bool(row["indislive"]),
        )
        for row in inspector.bind.execute(
            sa.text(
                """
                SELECT index_class.relname AS index_name,
                       pg_index.indisvalid,
                       pg_index.indisready,
                       pg_index.indislive
                FROM pg_catalog.pg_index
                JOIN pg_catalog.pg_class AS index_class
                  ON index_class.oid = pg_index.indexrelid
                WHERE pg_index.indrelid = to_regclass(:table_name)
                """
            ),
            {"table_name": table_name},
        ).mappings()
    }
    invalid_indexes = {index_name for index_name in indexes if index_states.get(index_name) != (True, True, True)}
    if invalid_indexes:
        raise RuntimeError(f"已有表 {table_name} 包含无效或未就绪索引：{sorted(invalid_indexes)}")
    if expected_indexes:
        for index_name, (expected_column_names, expected_unique, expected_predicate) in expected_indexes.items():
            index = indexes.get(index_name)
            actual_column_names = tuple(index.get("column_names") or []) if index else ()
            actual_unique = bool(index.get("unique")) if index else False
            dialect_options = index.get("dialect_options") or {} if index else {}
            predicate = dialect_options.get("postgresql_where")
            predicate_text = _normalize_sql_expression(predicate) if predicate is not None else ""
            expected_predicate_text = (
                _normalize_sql_expression(expected_predicate) if expected_predicate is not None else ""
            )
            if (
                actual_column_names != expected_column_names
                or actual_unique is not expected_unique
                or predicate_text != expected_predicate_text
            ):
                raise RuntimeError(f"已有表 {table_name} 的索引 {index_name} 不兼容")

    allowed_unique_indexes = {
        index_name for index_name, (_, unique, _) in (expected_indexes or {}).items() if unique
    } | set(unique_constraints if expected_unique_constraints is not None else {})
    unexpected_unique_indexes = {
        index_name
        for index_name, index in indexes.items()
        if bool(index.get("unique")) and index_name not in allowed_unique_indexes
    }
    if unexpected_unique_indexes:
        raise RuntimeError(f"已有表 {table_name} 包含迁移定义外的唯一索引：{sorted(unexpected_unique_indexes)}")

    if inspector.get_foreign_keys(table_name):
        raise RuntimeError(f"已有表 {table_name} 包含迁移定义外的外键")


def _create_model_catalog_controls() -> None:
    op.create_table(
        "model_catalog_controls",
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("selectable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("routable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("reason", sa.String(length=300), nullable=False),
        sa.Column("updated_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("routable IS TRUE", name="ck_model_catalog_controls_routable_true"),
        sa.CheckConstraint("revision > 0", name="ck_model_catalog_controls_revision_positive"),
        sa.PrimaryKeyConstraint("model_id"),
    )


def _create_model_admission_operations() -> None:
    op.create_table(
        "model_admission_operations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("candidate_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("governance_run_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("requested_by", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(length=300), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_token_hash", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("catalog_invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed')",
            name="ck_model_admission_operations_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_fingerprint",
            "governance_run_id",
            name="uq_model_admission_operation_candidate_run",
        ),
    )
    op.create_index(
        "ix_model_admission_operations_status_created",
        "model_admission_operations",
        ["status", "created_at", "id"],
    )
    op.create_index(
        "uq_model_admission_operations_single_running",
        "model_admission_operations",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def _create_origin_registry(created_tables: set[str]) -> None:
    op.create_table(
        ORIGIN_REGISTRY_TABLE,
        sa.Column("object_name", sa.String(length=100), nullable=False),
        sa.Column("created_by_migration", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "object_name IN ('model_catalog_controls', 'model_admission_operations')",
            name="ck_f3a1d9c8b720_origin_object_name",
        ),
        sa.PrimaryKeyConstraint("object_name"),
    )
    values = ", ".join(
        f"('{table_name}', {'true' if table_name in created_tables else 'false'})" for table_name in MANAGED_TABLES
    )
    op.execute(sa.text(f"INSERT INTO {ORIGIN_REGISTRY_TABLE} (object_name, created_by_migration) VALUES {values}"))


def upgrade() -> None:
    if context.is_offline_mode():
        _create_origin_registry(set(MANAGED_TABLES))
        _create_model_catalog_controls()
        _create_model_admission_operations()
        return

    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(ORIGIN_REGISTRY_TABLE):
        raise RuntimeError(f"迁移来源登记表已存在，拒绝覆盖：{ORIGIN_REGISTRY_TABLE}")

    catalog_exists = inspector.has_table("model_catalog_controls")
    operation_exists = inspector.has_table("model_admission_operations")

    if catalog_exists:
        _validate_existing_table(
            inspector,
            "model_catalog_controls",
            MODEL_CATALOG_CONTROL_COLUMNS,
            expected_primary_key={"model_id"},
            expected_checks={
                "ck_model_catalog_controls_routable_true": "routable IS TRUE",
                "ck_model_catalog_controls_revision_positive": "revision > 0",
            },
            expected_unique_constraints={},
            allowed_redundant_unique_constraint_columns={frozenset({"model_id"})},
        )

    if operation_exists:
        _validate_existing_table(
            inspector,
            "model_admission_operations",
            MODEL_ADMISSION_OPERATION_COLUMNS,
            expected_primary_key={"id"},
            expected_checks={
                "ck_model_admission_operations_status": (
                    "status::text = ANY (ARRAY["
                    "'pending'::character varying, "
                    "'running'::character varying, "
                    "'succeeded'::character varying, "
                    "'failed'::character varying"
                    "]::text[])"
                ),
            },
            expected_unique_constraints={
                "uq_model_admission_operation_candidate_run": {
                    "candidate_fingerprint",
                    "governance_run_id",
                },
            },
            expected_indexes={
                "ix_model_admission_operations_status_created": (
                    ("status", "created_at", "id"),
                    False,
                    None,
                ),
                "uq_model_admission_operations_single_running": (
                    ("status",),
                    True,
                    "status = 'running'",
                ),
            },
        )

    created_tables = {
        table_name
        for table_name, exists in (
            ("model_catalog_controls", catalog_exists),
            ("model_admission_operations", operation_exists),
        )
        if not exists
    }
    _create_origin_registry(created_tables)
    if not catalog_exists:
        _create_model_catalog_controls()
    if not operation_exists:
        _create_model_admission_operations()


def downgrade() -> None:
    op.execute(
        sa.text(
            f"""
            DO $fusion$
            DECLARE
              operation_created boolean;
              catalog_created boolean;
            BEGIN
              IF to_regclass('{ORIGIN_REGISTRY_TABLE}') IS NULL THEN
                RAISE EXCEPTION '缺少迁移来源登记表：{ORIGIN_REGISTRY_TABLE}';
              END IF;

              SELECT created_by_migration INTO operation_created
              FROM {ORIGIN_REGISTRY_TABLE}
              WHERE object_name = 'model_admission_operations';
              IF NOT FOUND THEN
                RAISE EXCEPTION '缺少 model_admission_operations 来源记录';
              END IF;

              SELECT created_by_migration INTO catalog_created
              FROM {ORIGIN_REGISTRY_TABLE}
              WHERE object_name = 'model_catalog_controls';
              IF NOT FOUND THEN
                RAISE EXCEPTION '缺少 model_catalog_controls 来源记录';
              END IF;

              IF operation_created THEN
                DROP TABLE model_admission_operations;
              END IF;
              IF catalog_created THEN
                DROP TABLE model_catalog_controls;
              END IF;
              DROP TABLE {ORIGIN_REGISTRY_TABLE};
            END
            $fusion$
            """
        )
    )
