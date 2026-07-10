"""PromptHub bundle 的后台同步、LKG 持久化与只读诊断。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.core.prompt_bundle import (
    PromptBundleValidationError,
    clear_prompt_bundle_cache,
    validate_published_bundle,
    validate_stored_bundle_payload,
)
from app.core.prompt_catalog import PROMPT_SPECS
from app.core.runtime_config import SessionFactory, clear_runtime_config_cache, get_runtime_config_payload
from app.db.database import SessionLocal
from app.db.models import RuntimeConfigEntry
from app.services.external.prompthub_client import (
    PromptHubClientError,
    PromptHubPublishedBundleClient,
)
from app.services.runtime_config_defaults import DEFAULT_PROMPT_TEMPLATES

SyncMode = Literal["disabled", "shadow", "apply"]

_ADVISORY_LOCK_ID = 0x465553494F4E5048
_DIAGNOSTICS: dict[str, Any] = {
    "mode": settings.PROMPTHUB_SYNC_MODE,
    "status": "never_run",
    "last_attempt_at": None,
    "last_success_at": None,
    "revision": None,
    "changed_prompt_keys": [],
    "last_error": None,
}


async def sync_prompthub_bundle(
    *,
    mode: str | None = None,
    client: PromptHubPublishedBundleClient | Any | None = None,
    session_factory: SessionFactory = SessionLocal,
) -> dict[str, Any]:
    """同步完整 bundle；错误只更新诊断，不改变已激活 LKG。"""

    effective_mode = mode or settings.PROMPTHUB_SYNC_MODE
    attempted_at = _utc_now()
    if effective_mode == "disabled":
        result = _base_result(effective_mode, "disabled", attempted_at)
        _update_diagnostics(result)
        return result
    if effective_mode not in {"shadow", "apply"}:
        return _record_error(effective_mode, attempted_at, "PROMPTHUB_SYNC_MODE 无效")

    try:
        effective_client = client or _build_client()
        bundle = await effective_client.fetch_published_bundle()
        payload = validate_published_bundle(bundle)
        changed_keys = await asyncio.to_thread(
            _build_shadow_diff,
            payload,
            session_factory=session_factory,
        )
        persist_result = await asyncio.to_thread(
            _persist_bundle,
            payload,
            mode=effective_mode,
            session_factory=session_factory,
        )
    except (PromptHubClientError, PromptBundleValidationError, ValueError) as exc:
        return _record_error(effective_mode, attempted_at, str(exc))
    except Exception:
        logger.exception("PromptHub bundle 同步发生未预期错误")
        return _record_error(effective_mode, attempted_at, "同步发生未预期错误")

    result = {
        **_base_result(effective_mode, "success", attempted_at),
        "last_success_at": _utc_now(),
        "revision": payload["revision"],
        "changed_prompt_keys": changed_keys,
        "idempotent": persist_result["idempotent"],
        "active": persist_result["active"],
        "last_error": None,
    }
    _update_diagnostics(result)
    logger.info(
        "PromptHub bundle 同步成功: mode=%s revision=%s changed=%s idempotent=%s",
        effective_mode,
        payload["revision"],
        len(changed_keys),
        persist_result["idempotent"],
    )
    return copy.deepcopy(result)


async def run_prompthub_sync_best_effort() -> dict[str, Any]:
    """供 startup/scheduler 调用，任何失败都不影响应用可用性。"""

    try:
        return await sync_prompthub_bundle()
    except Exception:
        logger.exception("PromptHub best-effort 同步失败")
        return _record_error(settings.PROMPTHUB_SYNC_MODE, _utc_now(), "同步发生未预期错误")


def get_prompthub_sync_diagnostics() -> dict[str, Any]:
    """返回不含凭证和 Prompt 内容的 admin 只读诊断。"""

    return copy.deepcopy(_DIAGNOSTICS)


def _build_client() -> PromptHubPublishedBundleClient:
    if not settings.PROMPTHUB_BASE_URL or not settings.PROMPTHUB_API_KEY:
        raise ValueError("PromptHub 地址或服务凭证未配置")
    return PromptHubPublishedBundleClient(
        base_url=settings.PROMPTHUB_BASE_URL,
        api_key=settings.PROMPTHUB_API_KEY,
        project_slug=settings.PROMPTHUB_PROJECT_SLUG,
        timeout_seconds=settings.PROMPTHUB_REQUEST_TIMEOUT_SECONDS,
    )


def _build_shadow_diff(
    payload: dict[str, Any],
    *,
    session_factory: SessionFactory,
) -> list[str]:
    changed_keys: list[str] = []
    for spec in PROMPT_SPECS:
        legacy, _meta = get_runtime_config_payload(
            "prompt_template",
            spec.key,
            {"template": DEFAULT_PROMPT_TEMPLATES[spec.key]},
            session_factory=session_factory,
            use_cache=False,
        )
        current = legacy.get("template")
        remote_checksum = payload["prompts"][spec.key]["content_sha256"]
        current_checksum = _sha256(current) if isinstance(current, str) else None
        if current_checksum != remote_checksum:
            changed_keys.append(spec.key)
    return changed_keys


def _persist_bundle(
    payload: dict[str, Any],
    *,
    mode: str,
    session_factory: SessionFactory,
) -> dict[str, bool]:
    session: Session | None = None
    try:
        session = session_factory()
        _acquire_advisory_lock(session)
        rows = _load_bundle_rows(session)
        existing = next((row for row in rows if row.version == payload["revision"]), None)
        if existing is not None:
            if not validate_stored_bundle_payload(existing.payload) or existing.payload != payload:
                raise ValueError("同 revision 的本地 Prompt bundle 已损坏或内容不一致")
            if mode == "shadow":
                if not existing.is_active:
                    return {"idempotent": True, "active": False}
                existing.is_active = False
                session.commit()
                _clear_prompt_caches()
                return {"idempotent": False, "active": False}
            if existing.is_active:
                return {"idempotent": True, "active": True}
            _activate_row(rows, existing)
            session.commit()
            _clear_prompt_caches()
            return {"idempotent": False, "active": True}

        row = RuntimeConfigEntry(
            id=str(uuid.uuid4()),
            namespace="prompt_bundle",
            key="fusion",
            version=payload["revision"],
            payload=copy.deepcopy(payload),
            is_active=mode == "apply",
            description="PromptHub fusion published bundle LKG",
        )
        if mode == "apply":
            for peer in rows:
                peer.is_active = False
        session.add(row)
        session.commit()
        _clear_prompt_caches()
        return {"idempotent": False, "active": bool(row.is_active)}
    except IntegrityError:
        if session is not None:
            session.rollback()
        raise ValueError("Prompt bundle revision 并发写入冲突") from None
    except Exception:
        if session is not None:
            session.rollback()
        raise
    finally:
        if session is not None:
            session.close()


def _load_bundle_rows(session: Session) -> list[RuntimeConfigEntry]:
    rows = (
        session.query(RuntimeConfigEntry)
        .filter(
            RuntimeConfigEntry.namespace == "prompt_bundle",
            RuntimeConfigEntry.key == "fusion",
        )
        .order_by(RuntimeConfigEntry.updated_at.desc(), RuntimeConfigEntry.created_at.desc())
        .all()
    )
    return [
        row
        for row in rows
        if getattr(row, "namespace", None) == "prompt_bundle" and getattr(row, "key", None) == "fusion"
    ]


def _activate_row(rows: list[RuntimeConfigEntry], target: RuntimeConfigEntry) -> None:
    for row in rows:
        row.is_active = row.id == target.id


def _acquire_advisory_lock(session: Session) -> None:
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _ADVISORY_LOCK_ID},
        )


def _clear_prompt_caches() -> None:
    clear_runtime_config_cache()
    clear_prompt_bundle_cache()


def _record_error(mode: str, attempted_at: str, message: str) -> dict[str, Any]:
    result = {
        **_base_result(mode, "error", attempted_at),
        "last_success_at": _DIAGNOSTICS.get("last_success_at"),
        "revision": _DIAGNOSTICS.get("revision"),
        "changed_prompt_keys": copy.deepcopy(_DIAGNOSTICS.get("changed_prompt_keys", [])),
        "last_error": message,
    }
    _update_diagnostics(result)
    logger.warning("PromptHub bundle 同步失败: mode=%s error=%s", mode, message)
    return copy.deepcopy(result)


def _base_result(mode: str, status: str, attempted_at: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": status,
        "last_attempt_at": attempted_at,
        "last_success_at": _DIAGNOSTICS.get("last_success_at"),
        "revision": _DIAGNOSTICS.get("revision"),
        "changed_prompt_keys": [],
        "last_error": None,
    }


def _update_diagnostics(result: dict[str, Any]) -> None:
    _DIAGNOSTICS.update(copy.deepcopy(result))


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
