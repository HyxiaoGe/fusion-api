from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

from sqlalchemy.exc import IntegrityError

from app.ai import litellm_catalog, litellm_health
from app.db.admin_audit_repository import AdminAuditRepository
from app.db.model_admission_operation_repository import ModelAdmissionOperationRepository
from app.db.model_catalog_control_repository import ModelCatalogControlRepository
from app.db.models import ModelAdmissionOperation, ModelCatalogControl, User
from app.schemas.response import ApiException
from app.services.admin_audit_service import AdminAuditService
from app.utils.time import utc_now
from scripts.execute_litellm_candidate_admission import VerifiedPlanError, load_verified_admission_plan
from scripts.verify_litellm_governance_snapshot import (
    GovernanceSnapshotError,
    VerifiedGovernanceSnapshot,
    load_verified_governance_snapshot,
)


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class ModelManagementConfig:
    governance_root: Path | None
    management_enabled: bool
    governance_max_age_seconds: int
    worker_enabled: bool
    worker_token: str
    lease_seconds: int

    @property
    def admission_ready(self) -> bool:
        return bool(self.management_enabled and self.governance_root and self.worker_enabled and self.worker_token)


class ModelManagementService:
    def __init__(
        self,
        *,
        control_repository: ModelCatalogControlRepository,
        operation_repository: ModelAdmissionOperationRepository,
        audit_service: AdminAuditService,
        config: ModelManagementConfig,
        catalog: Any = litellm_catalog,
        plan_loader: Callable[..., Mapping[str, Any]] = load_verified_admission_plan,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.control_repository = control_repository
        self.operation_repository = operation_repository
        self.audit_service = audit_service
        self.config = config
        self.catalog = catalog
        self.plan_loader = plan_loader
        self.clock = clock

    @property
    def db(self):
        return self.control_repository.db

    def get_snapshot(self) -> dict[str, Any]:
        verified = self._try_verified_snapshot()
        governance: dict[str, Any]
        candidates: list[dict[str, Any]] = []
        operations = [self.operation_payload(operation) for operation in self.operation_repository.list_recent(100)]
        if verified is None:
            governance = {
                "available": False,
                "status": "unavailable",
                "reason": "success_snapshot_unavailable",
                "message": "治理候选暂时不可用",
            }
        elif verified.degraded_reason is not None:
            governance = {
                "available": True,
                "status": "degraded",
                "run_id": verified.run_id,
                "reason": verified.degraded_reason,
                "message": self._degraded_message(verified.degraded_reason),
            }
            candidates = [self._candidate_item(item) for item in verified.queue]
        else:
            governance = {
                "available": True,
                "status": "available",
                "run_id": verified.run_id,
            }
            candidates = [self._candidate_item(item) for item in verified.queue]
        return {
            "generated_at": self._as_utc(self.clock()).isoformat(),
            "governance": governance,
            "capabilities": {
                "admission_enabled": bool(
                    self.config.admission_ready and verified is not None and verified.degraded_reason is None
                ),
                "hard_delete_enabled": False,
            },
            "models": self.list_registered_models(),
            "candidates": candidates,
            "operations": operations,
        }

    def list_registered_models(self) -> list[dict[str, Any]]:
        catalog = self.catalog.list_aliases()
        aliases = sorted(alias for alias, entry in catalog.items() if entry.get("db_model"))
        controls = self.control_repository.get_by_model_ids(aliases)
        return [self._registered_model(alias, catalog[alias], controls.get(alias)) for alias in aliases]

    def set_visibility(
        self,
        *,
        model_id: str,
        selectable: bool,
        expected_revision: int | None,
        reason: str,
        admin: User,
        request_id: str,
    ) -> ModelCatalogControl:
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ApiException.bad_request("变更原因不能为空")
        if len(normalized_reason) > 300:
            raise ApiException.bad_request("变更原因不能超过 300 个字符")
        if not self.config.management_enabled:
            raise ApiException.service_unavailable("模型管理写操作未启用")
        entry = self.catalog.get_model_entry(model_id)
        if not entry or not entry.get("db_model"):
            raise ApiException.not_found("模型不存在或尚未注册")
        try:
            row = self._write_visibility(
                model_id=model_id,
                selectable=selectable,
                expected_revision=expected_revision,
                reason=normalized_reason,
                updated_by=str(admin.id),
            )
            self.audit_service._record(
                admin=admin,
                action="model_visibility_changed",
                resource_type="model_catalog_control",
                resource_id=model_id,
                request_id=request_id,
                reason=normalized_reason,
                metadata={
                    "selectable": selectable,
                    "routable": True,
                    "revision": row.revision,
                    "expected_revision": expected_revision,
                },
                commit=False,
            )
            self.db.commit()
            self.db.refresh(row)
            return row
        except ApiException:
            self.db.rollback()
            raise
        except IntegrityError as exc:
            self.db.rollback()
            raise ApiException.conflict("模型可见性版本已变化") from exc
        except Exception:
            self.db.rollback()
            raise

    def request_admission(
        self,
        *,
        fingerprint: str,
        model_id: str,
        expected_run_id: str,
        reason: str,
        admin: User,
        request_id: str,
    ) -> ModelAdmissionOperation:
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ApiException.bad_request("准入原因不能为空")
        if len(normalized_reason) > 300:
            raise ApiException.bad_request("准入原因不能超过 300 个字符")
        if not self.config.management_enabled:
            raise ApiException.service_unavailable("模型准入写操作未启用")
        if not self.config.admission_ready:
            raise ApiException.service_unavailable("模型准入 Worker 未就绪")
        snapshot = self._require_admission_snapshot()
        if snapshot.run_id != expected_run_id:
            raise ApiException.conflict("治理运行版本已变化")
        candidate = self._require_admission_candidate(snapshot, fingerprint=fingerprint, model_id=model_id)
        if candidate.get("state") == "admission_ready":
            self._require_verified_plan(fingerprint=fingerprint, model_id=model_id, run_id=expected_run_id)
        existing = self.operation_repository.get_by_candidate_run(fingerprint, expected_run_id)
        if existing is not None:
            if existing.model_id != model_id:
                raise ApiException.conflict("候选准入请求与既有操作不一致")
            if existing.status == "failed":
                return self._reset_failed_operation(
                    existing,
                    admin=admin,
                    request_id=request_id,
                    reason=normalized_reason,
                )
            return existing
        row = ModelAdmissionOperation(
            id=str(uuid.uuid4()),
            model_id=model_id,
            candidate_fingerprint=fingerprint,
            governance_run_id=expected_run_id,
            status="pending",
            requested_by=str(admin.id),
            request_id=request_id,
            reason=normalized_reason,
            attempts=0,
            result={},
        )
        try:
            self.operation_repository.add(row)
            self.audit_service._record(
                admin=admin,
                action="model_admission_requested",
                resource_type="litellm_model_admission",
                resource_id=model_id,
                request_id=request_id,
                reason=normalized_reason,
                metadata={
                    "operation_id": row.id,
                    "candidate_fingerprint": fingerprint,
                    "expected_run_id": expected_run_id,
                },
                commit=False,
            )
            self.db.commit()
            self.db.refresh(row)
            return row
        except IntegrityError:
            self.db.rollback()
            concurrent = self.operation_repository.get_by_candidate_run(fingerprint, expected_run_id)
            if concurrent is not None and concurrent.model_id == model_id:
                return concurrent
            raise ApiException.conflict("候选准入操作已存在") from None
        except Exception:
            self.db.rollback()
            raise

    def claim_operation(self) -> tuple[ModelAdmissionOperation, str] | None:
        if not self.config.admission_ready:
            raise ApiException.service_unavailable("模型准入 Worker 未就绪")
        self._require_admission_snapshot()
        now = self.clock()
        raw_lease = secrets.token_urlsafe(32)
        try:
            if self.operation_repository.has_active_running(now):
                self.db.rollback()
                return None
            row = self.operation_repository.lock_claimable(now)
            if row is None:
                self.db.rollback()
                return None
            row.status = "running"
            row.attempts = int(row.attempts or 0) + 1
            row.lease_token_hash = _token_hash(raw_lease)
            row.lease_expires_at = now + timedelta(seconds=max(self.config.lease_seconds, 600))
            row.catalog_invalidated_at = None
            row.updated_at = now
            self.db.commit()
            self.db.refresh(row)
            return row, raw_lease
        except IntegrityError:
            self.db.rollback()
            return None
        except Exception:
            self.db.rollback()
            raise

    def _reset_failed_operation(
        self,
        operation: ModelAdmissionOperation,
        *,
        admin: User,
        request_id: str,
        reason: str,
    ) -> ModelAdmissionOperation:
        row = self.operation_repository.lock(operation.id)
        if row is None:
            self.db.rollback()
            raise ApiException.conflict("候选准入操作已变化")
        if row.status != "failed":
            self.db.rollback()
            return row
        row.status = "pending"
        row.requested_by = str(admin.id)
        row.request_id = request_id
        row.reason = reason
        row.lease_token_hash = None
        row.lease_expires_at = None
        row.catalog_invalidated_at = None
        row.result = {}
        row.terminal_at = None
        row.updated_at = self.clock()
        try:
            self.audit_service._record(
                admin=admin,
                action="model_admission_requested",
                resource_type="litellm_model_admission",
                resource_id=row.model_id,
                request_id=request_id,
                reason=reason,
                metadata={
                    "operation_id": row.id,
                    "candidate_fingerprint": row.candidate_fingerprint,
                    "expected_run_id": row.governance_run_id,
                    "retry": True,
                },
                commit=False,
            )
            self.db.commit()
            self.db.refresh(row)
            return row
        except Exception:
            self.db.rollback()
            raise

    def invalidate_catalog(self, *, operation_id: str, lease_token: str) -> Any:
        if not self.config.management_enabled:
            raise ApiException.service_unavailable("模型准入写操作未启用")
        try:
            row = self._lock_running_operation(operation_id, lease_token)
            generation = str(self.catalog.bump_generation())
            row.catalog_invalidated_at = self.clock()
            row.updated_at = self.clock()
            self.db.commit()
            return generation
        except Exception:
            self.db.rollback()
            raise

    def complete_operation(
        self,
        *,
        operation_id: str,
        lease_token: str,
        status: str,
        phase: str,
        error_code: str | None,
        completed_phases: list[str],
        writes_performed: bool,
        compensation: Mapping[str, Any],
    ) -> ModelAdmissionOperation:
        if status not in {"succeeded", "failed"}:
            raise ApiException.bad_request("终态只支持 succeeded 或 failed")
        row = self.operation_repository.lock(operation_id)
        if row is None:
            self.db.rollback()
            raise ApiException.not_found("模型准入操作不存在")
        self._require_lease(row, lease_token, allow_terminal=True, allow_expired=True)
        if row.status in {"succeeded", "failed"}:
            if row.status != status:
                self.db.rollback()
                raise ApiException.conflict("模型准入操作已进入不同终态")
            self.db.commit()
            return row
        if status == "succeeded" and row.catalog_invalidated_at is None:
            self.db.rollback()
            raise ApiException.conflict("成功终态前必须完成目录失效")
        safe_result = self._safe_terminal_result(
            status=status,
            phase=phase,
            error_code=error_code,
            completed_phases=completed_phases,
            writes_performed=writes_performed,
            compensation=compensation,
        )
        now = self.clock()
        row.status = status
        row.result = safe_result
        row.terminal_at = now
        row.updated_at = now
        try:
            AdminAuditRepository(self.db).create_audit_event(
                commit=False,
                id=str(uuid.uuid4()),
                admin_user_id=row.requested_by,
                admin_snapshot={"id": row.requested_by, "source": "model_admission_operation"},
                action=f"model_admission_{status}",
                resource_type="litellm_model_admission",
                resource_id=row.model_id,
                request_id=row.request_id,
                reason=row.reason,
                extra_metadata={
                    "operation_id": row.id,
                    "candidate_fingerprint": row.candidate_fingerprint,
                    "expected_run_id": row.governance_run_id,
                    **safe_result,
                },
            )
            self.db.commit()
            self.db.refresh(row)
            return row
        except Exception:
            self.db.rollback()
            raise

    def operation_payload(self, row: ModelAdmissionOperation) -> dict[str, Any]:
        result = row.result if isinstance(row.result, Mapping) else {}
        return {
            "operation_id": row.id,
            "candidate_fingerprint": row.candidate_fingerprint,
            "model_id": row.model_id,
            "status": row.status,
            "phase": result.get("phase"),
            "error_code": result.get("error_code"),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _try_verified_snapshot(self) -> VerifiedGovernanceSnapshot | None:
        try:
            return self._load_verified_snapshot()
        except GovernanceSnapshotError:
            return None

    def _require_admission_snapshot(self) -> VerifiedGovernanceSnapshot:
        snapshot = self._try_verified_snapshot()
        if snapshot is None:
            raise ApiException.service_unavailable("治理候选暂时不可用")
        if snapshot.degraded_reason is not None:
            raise ApiException.service_unavailable("治理候选处于只读降级状态，模型准入已暂停")
        return snapshot

    def _load_verified_snapshot(self) -> VerifiedGovernanceSnapshot:
        root = self.config.governance_root
        if root is None:
            raise GovernanceSnapshotError("governance_root_missing")
        return load_verified_governance_snapshot(
            governance_root=root,
            max_age_seconds=self.config.governance_max_age_seconds,
            now=self._as_utc(self.clock()),
            include_queue=True,
        )

    @staticmethod
    def _degraded_message(reason: str) -> str:
        if reason == "latest_run_failed":
            return "最新治理运行失败，当前展示上一次有效快照；模型准入已暂停。"
        return "最新治理失败状态无法校验，当前展示上一次有效快照；模型准入已暂停。"

    def _require_admission_candidate(
        self,
        snapshot: VerifiedGovernanceSnapshot,
        *,
        fingerprint: str,
        model_id: str,
    ) -> Mapping[str, Any]:
        matches = [
            item
            for item in snapshot.queue
            if item.get("candidate_fingerprint") == fingerprint and item.get("model_id") == model_id
        ]
        if len(matches) != 1 or matches[0].get("state") not in {"preflight_required", "admission_ready"}:
            raise ApiException.conflict("候选模型尚未满足准入条件")
        return matches[0]

    def _require_verified_plan(self, *, fingerprint: str, model_id: str, run_id: str) -> None:
        try:
            plan = self.plan_loader(
                governance_root=self.config.governance_root,
                candidate_fingerprint=fingerprint,
            )
        except VerifiedPlanError as exc:
            raise ApiException.conflict("候选治理产物校验失败") from exc
        eligible = plan.get("eligible") if isinstance(plan, Mapping) else None
        if (
            not isinstance(plan, Mapping)
            or plan.get("run_id") != run_id
            or not isinstance(eligible, list)
            or len(eligible) != 1
            or not isinstance(eligible[0], Mapping)
            or eligible[0].get("model_id") != model_id
        ):
            raise ApiException.conflict("候选治理产物与请求不一致")

    def _write_visibility(
        self,
        *,
        model_id: str,
        selectable: bool,
        expected_revision: int | None,
        reason: str,
        updated_by: str,
    ) -> ModelCatalogControl:
        existing = self.control_repository.get(model_id)
        values = {
            "selectable": selectable,
            "routable": True,
            "reason": reason,
            "updated_by": updated_by,
            "updated_at": self.clock(),
        }
        if existing is None:
            if expected_revision is not None:
                raise ApiException.conflict("模型可见性版本已变化")
            return self.control_repository.add({"model_id": model_id, "revision": 1, **values})
        if expected_revision is None:
            raise ApiException.conflict("模型可见性版本已变化")
        updated = self.control_repository.update_if_revision(
            model_id=model_id,
            expected_revision=expected_revision,
            values=values,
        )
        if updated is None:
            raise ApiException.conflict("模型可见性版本已变化")
        return updated

    def _lock_running_operation(self, operation_id: str, lease_token: str) -> ModelAdmissionOperation:
        row = self.operation_repository.lock(operation_id)
        if row is None:
            self.db.rollback()
            raise ApiException.not_found("模型准入操作不存在")
        self._require_lease(row, lease_token)
        return row

    def _require_lease(
        self,
        row: ModelAdmissionOperation,
        lease_token: str,
        *,
        allow_terminal: bool = False,
        allow_expired: bool = False,
    ) -> None:
        allowed_statuses = {"running", "succeeded", "failed"} if allow_terminal else {"running"}
        if row.status not in allowed_statuses:
            self.db.rollback()
            raise ApiException.conflict("模型准入操作未被当前 Worker 领取")
        supplied = _token_hash(lease_token)
        if not row.lease_token_hash or not secrets.compare_digest(row.lease_token_hash, supplied):
            self.db.rollback()
            raise ApiException.conflict("模型准入租约无效")
        if (
            not allow_expired
            and row.status == "running"
            and (row.lease_expires_at is None or self._as_utc(row.lease_expires_at) <= self._as_utc(self.clock()))
        ):
            self.db.rollback()
            raise ApiException.conflict("模型准入租约已过期")

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @staticmethod
    def _safe_terminal_result(
        *,
        status: str,
        phase: str,
        error_code: str | None,
        completed_phases: list[str],
        writes_performed: bool,
        compensation: Mapping[str, Any],
    ) -> dict[str, Any]:
        safe_compensation = {
            key: value
            for key, value in compensation.items()
            if key
            in {
                "attempted",
                "key_restored",
                "model_deleted",
                "model_ownership_unverified",
                "manual_cleanup_required",
                "errors",
            }
        }
        if isinstance(safe_compensation.get("errors"), list):
            safe_compensation["errors"] = [str(item)[:100] for item in safe_compensation["errors"][:20]]
        return {
            "status": status,
            "phase": phase[:80],
            "error_code": error_code[:100] if error_code else None,
            "completed_phases": [str(item)[:80] for item in completed_phases[:20]],
            "writes_performed": bool(writes_performed),
            "compensation": safe_compensation,
        }

    def _candidate_item(
        self,
        item: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), Mapping) else {}
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
        capabilities = metadata.get("capabilities") if isinstance(metadata.get("capabilities"), Mapping) else {}
        reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
        fingerprint = item.get("candidate_fingerprint")
        return {
            "candidate_fingerprint": fingerprint,
            "model_id": str(item.get("model_id") or ""),
            "name": str(metadata.get("display_name") or item.get("model_id") or ""),
            "provider_key": str(item.get("provider_key") or ""),
            "provider_display": str(metadata.get("provider_display") or item.get("provider_key") or ""),
            "state": str(item.get("state") or "quarantined"),
            "reasons": [str(reason)[:200] for reason in reasons[:50]],
            "capabilities": {str(key): bool(value) for key, value in capabilities.items()},
        }

    @staticmethod
    def _registered_model(alias: str, entry: Mapping[str, Any], control: ModelCatalogControl | None) -> dict[str, Any]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        underlying = str(entry.get("underlying") or "")
        provider = str(metadata.get("provider_key") or "").strip().lower()
        if not provider:
            provider = underlying.split("/", 1)[0].lower() if "/" in underlying else "litellm"
        selectable = bool(control.selectable) if control is not None else True
        return {
            "model_id": alias,
            "name": str(metadata.get("display_name") or alias),
            "provider": provider,
            "provider_display": str(metadata.get("provider_display") or provider),
            "health": litellm_health.get_health(alias),
            "selectable": selectable,
            "routable": True,
            "state": "selectable" if selectable else "hidden",
            "revision": control.revision if control is not None else None,
            "reason": control.reason if control is not None else None,
            "updated_at": control.updated_at if control is not None else None,
        }
