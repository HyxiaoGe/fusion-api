"""运行时配置读取服务。

DB 配置只作为可运营覆盖层；读取失败时必须回退到代码默认值，不能影响聊天主链路。
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.core.runtime_config_schema import validate_runtime_config_payload
from app.db.database import SessionLocal
from app.db.models import RuntimeConfigEntry

SessionFactory = Callable[[], Session]

_CACHE_TTL_SECONDS = 60.0
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any], dict[str, Any]]] = {}


def clear_runtime_config_cache() -> None:
    """清空配置缓存，供测试和配置刷新使用。"""

    _CACHE.clear()


def deep_merge_config(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并 dict；非 dict 覆盖值直接替换。"""

    merged = copy.deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_runtime_config_payload(
    namespace: str,
    key: str,
    default_payload: dict[str, Any],
    *,
    session_factory: SessionFactory = SessionLocal,
    use_cache: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 active runtime config，并返回 `(payload, meta)`。

    `payload` 始终是 dict。DB 无记录、异常或 payload 非 dict 时返回默认值。
    """

    cache_key = (namespace, key)
    now = time.monotonic()
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
            return copy.deepcopy(cached[1]), copy.deepcopy(cached[2])

    default_copy = copy.deepcopy(default_payload)
    default_meta = {
        "namespace": namespace,
        "key": key,
        "source": "default",
        "version": "code-default",
        "skipped_versions": [],
        "validation_warnings": {},
    }
    default_validation = validate_runtime_config_payload(namespace, key, default_copy)
    should_validate_candidates = default_validation.valid

    session: Session | None = None
    rows: list[RuntimeConfigEntry] = []
    try:
        session = session_factory()
        rows = (
            session.query(RuntimeConfigEntry)
            .filter(
                RuntimeConfigEntry.namespace == namespace,
                RuntimeConfigEntry.key == key,
                RuntimeConfigEntry.is_active.is_(True),
            )
            .order_by(RuntimeConfigEntry.updated_at.desc(), RuntimeConfigEntry.created_at.desc())
            .limit(10)
            .all()
        )
    except Exception as exc:
        logger.warning(f"runtime_config: load {namespace}/{key} failed: {exc}")
        return default_copy, default_meta
    finally:
        if session is not None:
            session.close()

    skipped_versions: list[str] = []
    validation_warnings: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row.payload, dict):
            skipped_versions.append(row.version)
            validation_warnings[row.version] = ["payload 必须是对象"]
            continue

        candidate_payload = deep_merge_config(default_copy, row.payload)
        validation = validate_runtime_config_payload(namespace, key, candidate_payload)
        if should_validate_candidates and not validation.valid:
            skipped_versions.append(row.version)
            validation_warnings[row.version] = validation.issues
            logger.warning(
                "runtime_config: skip invalid %s/%s@%s: %s",
                namespace,
                key,
                row.version,
                "; ".join(validation.issues),
            )
            continue

        meta = {
            "namespace": namespace,
            "key": key,
            "source": "db",
            "version": row.version,
            "skipped_versions": skipped_versions,
            "validation_warnings": validation_warnings,
        }
        if use_cache:
            _CACHE[cache_key] = (now, copy.deepcopy(candidate_payload), copy.deepcopy(meta))
        return candidate_payload, meta

    if skipped_versions:
        default_meta["skipped_versions"] = skipped_versions
        default_meta["validation_warnings"] = validation_warnings
    return default_copy, default_meta
