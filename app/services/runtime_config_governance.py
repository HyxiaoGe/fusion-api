"""Runtime Config 治理服务。

提供只读诊断、离线校验和 active 状态切换。聊天主链路仍通过
app.core.runtime_config 读取，治理服务只负责可观测和人工干预。
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
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


def create_runtime_config_entry(
    *,
    namespace: str,
    key: str,
    version: str,
    payload: dict[str, Any],
    description: str | None = None,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """创建一个已校验但默认不生效的 runtime config 版本。"""

    session: Session | None = None
    try:
        session = session_factory()
        validation = validate_runtime_config_candidate(namespace, key, payload)
        if not validation["valid"]:
            raise ApiException.bad_request(_format_validation_error(validation["issues"]))

        duplicates = (
            session.query(RuntimeConfigEntry)
            .filter(
                RuntimeConfigEntry.namespace == namespace,
                RuntimeConfigEntry.key == key,
                RuntimeConfigEntry.version == version,
            )
            .limit(1)
            .all()
        )
        if any(_matches_config_version(row, namespace, key, version) for row in duplicates):
            raise ApiException.conflict("运行时配置版本已存在")

        row = RuntimeConfigEntry(
            id=str(uuid.uuid4()),
            namespace=namespace,
            key=key,
            version=version,
            payload=copy.deepcopy(payload),
            is_active=False,
            description=description,
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            _rollback_if_possible(session)
            raise ApiException.conflict("运行时配置版本已存在") from exc

        session.refresh(row)
        defaults = get_runtime_config_defaults()
        return _serialize_runtime_config_entry(row, defaults.get((row.namespace, row.key)))
    finally:
        if session is not None:
            session.close()


def activate_runtime_config_entry(
    entry_id: str,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """安全激活某个 runtime config 版本，并关闭同一配置项的其它 active 版本。"""

    session: Session | None = None
    try:
        session = session_factory()
        row = session.query(RuntimeConfigEntry).filter(RuntimeConfigEntry.id == entry_id).first()
        if row is None:
            raise ApiException.not_found("运行时配置不存在")

        defaults = get_runtime_config_defaults()
        serialized = _serialize_runtime_config_entry(row, defaults.get((row.namespace, row.key)))
        if not serialized["valid"]:
            raise ApiException.bad_request(_format_validation_error(serialized["issues"]))

        peers = (
            session.query(RuntimeConfigEntry)
            .filter(
                RuntimeConfigEntry.namespace == row.namespace,
                RuntimeConfigEntry.key == row.key,
            )
            .all()
        )
        target_seen = False
        for peer in peers:
            if not _matches_config_key(peer, row.namespace, row.key):
                continue
            peer.is_active = peer.id == row.id
            target_seen = target_seen or peer.id == row.id
        if not target_seen:
            row.is_active = True

        session.commit()
        session.refresh(row)
        clear_runtime_config_cache()
        return _serialize_runtime_config_entry(row, defaults.get((row.namespace, row.key)))
    finally:
        if session is not None:
            session.close()


def set_runtime_config_entry_active(
    entry_id: str,
    is_active: bool,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """切换某个 runtime config 版本的 active 状态。"""

    if is_active:
        return activate_runtime_config_entry(entry_id, session_factory=session_factory)

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


def _matches_config_version(row: Any, namespace: str, key: str, version: str) -> bool:
    return (
        getattr(row, "namespace", None) == namespace
        and getattr(row, "key", None) == key
        and getattr(row, "version", None) == version
    )


def _matches_config_key(row: Any, namespace: str, key: str) -> bool:
    return getattr(row, "namespace", None) == namespace and getattr(row, "key", None) == key


def _format_validation_error(issues: list[str]) -> str:
    return "运行时配置校验失败：" + "；".join(issues)


def _rollback_if_possible(session: Session) -> None:
    rollback = getattr(session, "rollback", None)
    if callable(rollback):
        rollback()
