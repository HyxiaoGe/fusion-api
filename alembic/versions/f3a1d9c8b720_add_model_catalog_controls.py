"""add model catalog controls and admission operations

Revision ID: f3a1d9c8b720
Revises: e8b4c2d7f901
Create Date: 2026-08-04 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f3a1d9c8b720"
down_revision: Union[str, Sequence[str], None] = "e8b4c2d7f901"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


MODEL_CATALOG_CONTROL_COLUMNS = {
    "model_id": (sa.String, 200, False),
    "selectable": (sa.Boolean, None, False),
    "routable": (sa.Boolean, None, False),
    "revision": (sa.Integer, None, False),
    "reason": (sa.String, 300, False),
    "updated_by": (sa.String, None, False),
    "created_at": (sa.DateTime, None, False),
    "updated_at": (sa.DateTime, None, False),
}

MODEL_ADMISSION_OPERATION_COLUMNS = {
    "id": (sa.String, None, False),
    "model_id": (sa.String, 200, False),
    "candidate_fingerprint": (sa.String, 64, False),
    "governance_run_id": (sa.String, 32, False),
    "status": (sa.String, 20, False),
    "requested_by": (sa.String, None, False),
    "request_id": (sa.String, None, False),
    "reason": (sa.String, 300, False),
    "attempts": (sa.Integer, None, False),
    "lease_token_hash": (sa.String, 64, True),
    "lease_expires_at": (sa.DateTime, None, True),
    "catalog_invalidated_at": (sa.DateTime, None, True),
    "result": (postgresql.JSONB, None, False),
    "created_at": (sa.DateTime, None, False),
    "updated_at": (sa.DateTime, None, False),
    "terminal_at": (sa.DateTime, None, True),
}


def _validate_existing_table(
    inspector: sa.Inspector,
    table_name: str,
    expected_columns: dict[str, tuple[type[sa.types.TypeEngine], int | None, bool]],
    *,
    expected_primary_key: set[str],
    expected_checks: set[str],
    expected_unique_constraints: dict[str, set[str]] | None = None,
    expected_indexes: dict[str, tuple[tuple[str, ...], bool, str | None]] | None = None,
) -> None:
    actual_columns = {column["name"]: column for column in inspector.get_columns(table_name)}
    if set(actual_columns) != set(expected_columns):
        raise RuntimeError(
            f"已有表 {table_name} 的字段集合与迁移定义不一致："
            f"actual={sorted(actual_columns)} expected={sorted(expected_columns)}"
        )

    for column_name, (expected_type, expected_length, expected_nullable) in expected_columns.items():
        column = actual_columns[column_name]
        actual_type = column["type"]
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

    primary_key = set(inspector.get_pk_constraint(table_name).get("constrained_columns") or [])
    if primary_key != expected_primary_key:
        raise RuntimeError(
            f"已有表 {table_name} 的主键不兼容：actual={sorted(primary_key)} expected={sorted(expected_primary_key)}"
        )

    check_names = {
        constraint.get("name") for constraint in inspector.get_check_constraints(table_name) if constraint.get("name")
    }
    missing_checks = expected_checks - check_names
    if missing_checks:
        raise RuntimeError(f"已有表 {table_name} 缺少检查约束：{sorted(missing_checks)}")

    if expected_unique_constraints:
        unique_constraints = {
            constraint.get("name"): set(constraint.get("column_names") or [])
            for constraint in inspector.get_unique_constraints(table_name)
            if constraint.get("name")
        }
        for constraint_name, expected_column_names in expected_unique_constraints.items():
            if unique_constraints.get(constraint_name) != expected_column_names:
                raise RuntimeError(f"已有表 {table_name} 的唯一约束 {constraint_name} 不兼容")

    if expected_indexes:
        indexes = {index.get("name"): index for index in inspector.get_indexes(table_name) if index.get("name")}
        for index_name, (expected_column_names, expected_unique, predicate_token) in expected_indexes.items():
            index = indexes.get(index_name)
            actual_column_names = tuple(index.get("column_names") or []) if index else ()
            actual_unique = bool(index.get("unique")) if index else False
            dialect_options = index.get("dialect_options") or {} if index else {}
            predicate = dialect_options.get("postgresql_where")
            predicate_text = str(predicate) if predicate is not None else ""
            if (
                actual_column_names != expected_column_names
                or actual_unique is not expected_unique
                or (predicate_token is not None and predicate_token not in predicate_text)
            ):
                raise RuntimeError(f"已有表 {table_name} 的索引 {index_name} 不兼容")


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
        sa.UniqueConstraint("model_id"),
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


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("model_catalog_controls"):
        _validate_existing_table(
            inspector,
            "model_catalog_controls",
            MODEL_CATALOG_CONTROL_COLUMNS,
            expected_primary_key={"model_id"},
            expected_checks={
                "ck_model_catalog_controls_routable_true",
                "ck_model_catalog_controls_revision_positive",
            },
        )
    else:
        _create_model_catalog_controls()

    if inspector.has_table("model_admission_operations"):
        _validate_existing_table(
            inspector,
            "model_admission_operations",
            MODEL_ADMISSION_OPERATION_COLUMNS,
            expected_primary_key={"id"},
            expected_checks={"ck_model_admission_operations_status"},
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
                "uq_model_admission_operations_single_running": (("status",), True, "running"),
            },
        )
    else:
        _create_model_admission_operations()


def downgrade() -> None:
    op.drop_index("uq_model_admission_operations_single_running", table_name="model_admission_operations")
    op.drop_index("ix_model_admission_operations_status_created", table_name="model_admission_operations")
    op.drop_table("model_admission_operations")
    op.drop_table("model_catalog_controls")
