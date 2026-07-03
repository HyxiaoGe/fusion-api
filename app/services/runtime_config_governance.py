"""Runtime Config 治理服务。

提供只读诊断、离线校验和 active 状态切换。聊天主链路仍通过
app.core.runtime_config 读取，治理服务只负责可观测和人工干预。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.runtime_config import (
    SessionFactory,
    clear_runtime_config_cache,
    deep_merge_config,
    get_runtime_config_payload,
)
from app.core.runtime_config_schema import validate_runtime_config_payload
from app.db.database import SessionLocal
from app.db.models import RuntimeConfigEntry
from app.schemas.response import ApiException
from app.services.runtime_config_defaults import (
    DEFAULT_AGENT_STRATEGY_CONFIG,
    DEFAULT_MODEL_PRESENTATION_CONFIG,
    DEFAULT_PROMPT_TEMPLATES,
)


def build_runtime_config_snapshot(
    *,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """构建 admin 只读诊断快照。"""

    entries = _load_runtime_config_entries(session_factory)
    defaults = get_runtime_config_defaults()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "effective": [
            _build_effective_entry(namespace, key, default_payload, session_factory=session_factory)
            for (namespace, key), default_payload in defaults.items()
        ],
        "entries": [_serialize_runtime_config_entry(row, defaults.get((row.namespace, row.key))) for row in entries],
    }


def validate_runtime_config_candidate(
    namespace: str,
    key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """校验待写入 payload；已知配置域按默认值合并后校验。"""

    default_payload = get_runtime_config_defaults().get((namespace, key))
    candidate_payload = deep_merge_config(default_payload, payload) if default_payload is not None else payload
    result = validate_runtime_config_payload(namespace, key, candidate_payload)
    return {
        "namespace": namespace,
        "key": key,
        "valid": result.valid,
        "issues": result.issues,
    }


def set_runtime_config_entry_active(
    entry_id: str,
    is_active: bool,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """切换某个 runtime config 版本的 active 状态。"""

    session: Session | None = None
    try:
        session = session_factory()
        row = session.query(RuntimeConfigEntry).filter(RuntimeConfigEntry.id == entry_id).first()
        if row is None:
            raise ApiException.not_found("运行时配置不存在")
        row.is_active = is_active
        session.commit()
        session.refresh(row)
        clear_runtime_config_cache()
        defaults = get_runtime_config_defaults()
        return _serialize_runtime_config_entry(row, defaults.get((row.namespace, row.key)))
    finally:
        if session is not None:
            session.close()


def get_runtime_config_defaults() -> dict[tuple[str, str], dict[str, Any]]:
    """返回当前代码内置默认配置索引。"""

    defaults: dict[tuple[str, str], dict[str, Any]] = {
        ("agent_strategy", "default"): DEFAULT_AGENT_STRATEGY_CONFIG,
        ("model_presentation", "default"): DEFAULT_MODEL_PRESENTATION_CONFIG,
    }
    for key, template in DEFAULT_PROMPT_TEMPLATES.items():
        defaults[("prompt_template", key)] = {"template": template}
    return defaults


def _load_runtime_config_entries(session_factory: SessionFactory) -> list[RuntimeConfigEntry]:
    session: Session | None = None
    try:
        session = session_factory()
        return (
            session.query(RuntimeConfigEntry)
            .order_by(
                RuntimeConfigEntry.namespace.asc(),
                RuntimeConfigEntry.key.asc(),
                RuntimeConfigEntry.updated_at.desc(),
                RuntimeConfigEntry.created_at.desc(),
            )
            .all()
        )
    finally:
        if session is not None:
            session.close()


def _build_effective_entry(
    namespace: str,
    key: str,
    default_payload: dict[str, Any],
    *,
    session_factory: SessionFactory,
) -> dict[str, Any]:
    payload, meta = get_runtime_config_payload(
        namespace,
        key,
        default_payload,
        session_factory=session_factory,
        use_cache=False,
    )
    validation = validate_runtime_config_payload(namespace, key, payload)
    return {
        "namespace": namespace,
        "key": key,
        "source": meta["source"],
        "version": meta["version"],
        "valid": validation.valid,
        "issues": validation.issues,
        "skipped_versions": meta.get("skipped_versions", []),
        "validation_warnings": meta.get("validation_warnings", {}),
        "payload": payload,
    }


def _serialize_runtime_config_entry(
    row: RuntimeConfigEntry,
    default_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(row.payload, dict):
        candidate_payload = (
            deep_merge_config(default_payload, row.payload) if default_payload is not None else row.payload
        )
        validation = validate_runtime_config_payload(row.namespace, row.key, candidate_payload)
    else:
        validation = validate_runtime_config_payload(row.namespace, row.key, row.payload)
    return {
        "id": row.id,
        "namespace": row.namespace,
        "key": row.key,
        "version": row.version,
        "is_active": row.is_active,
        "valid": validation.valid,
        "issues": validation.issues,
        "description": row.description,
        "created_at": _isoformat(row.created_at),
        "updated_at": _isoformat(row.updated_at),
        "payload": row.payload,
    }


def _isoformat(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None
