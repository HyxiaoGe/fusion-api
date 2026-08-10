"""执行一个持久化模型准入任务。

该 Worker 运行在受信宿主机，持有 LiteLLM 管理凭据；公网 API 只负责排队、
租约和安全结果持久化。每次启动最多领取一个任务，适合由 systemd timer 反复触发。
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import httpx

from scripts.check_litellm_candidate_preflight import (
    candidate_contract_fingerprint,
    parse_candidate,
    run_preflight,
    serialize_report,
)
from scripts.execute_litellm_candidate_admission import (
    VerifiedPlanError,
    execute_admission,
    load_verified_admission_plan,
)
from scripts.plan_litellm_candidate_admission import (
    build_admission_plan,
    candidate_static_gate_reasons,
)
from scripts.verify_litellm_governance_snapshot import (
    GovernanceSnapshotError,
    canonical_sha256,
    load_verified_governance_snapshot,
)

WORKER_TOKEN_HEADER = "X-Fusion-Worker-Token"
LEASE_TOKEN_HEADER = "X-Operation-Lease"
COMPLETE_ATTEMPTS = 3
COMPLETE_RETRY_SECONDS = 2.0
LEASE_RENEW_INTERVAL_SECONDS = 60.0


class WorkerProtocolError(RuntimeError):
    """内部 Worker 协议不满足安全合同。"""


class StaleOperationLeaseError(WorkerProtocolError):
    """恢复记录的旧租约已不能提交，必须隔离后继续处理队列。"""


class OperationLeaseHeartbeat:
    """在长时预检与准入事务期间持续续租，并向主流程传播续租失败。"""

    def __init__(
        self,
        client: Any,
        *,
        fusion_base_url: str,
        worker_token: str,
        operation_id: str,
        lease_token: str,
        interval_seconds: float = LEASE_RENEW_INTERVAL_SECONDS,
    ) -> None:
        self.client = client
        self.endpoint = (
            f"{fusion_base_url.rstrip('/')}/api/internal/model-management/admissions/"
            f"{quote(operation_id, safe='')}/renew"
        )
        self.headers = _worker_headers(worker_token, lease_token)
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._error_lock = threading.Lock()

    def renew(self) -> None:
        try:
            response = self.client.post(self.endpoint, headers=self.headers, timeout=10.0)
            if response.status_code == 409:
                raise StaleOperationLeaseError("operation_lease_stale")
            response.raise_for_status()
        except StaleOperationLeaseError:
            raise
        except Exception as exc:
            raise WorkerProtocolError("operation_lease_renewal_failed") from exc

    def start(self) -> None:
        self.renew()
        self._thread = threading.Thread(
            target=self._run,
            name="model-admission-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.renew()
            except Exception as exc:
                with self._error_lock:
                    self._error = exc
                return

    def ensure_healthy(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def validate_governance_freshness(
    *,
    governance_root: Path,
    expected_run_id: str,
    max_age_seconds: int,
    now: datetime | None = None,
) -> None:
    """复用 API 的严格快照校验，并将任何降级态视为不可准入。"""
    try:
        snapshot = load_verified_governance_snapshot(
            governance_root=governance_root,
            max_age_seconds=max_age_seconds,
            now=now or datetime.now(UTC),
            include_queue=False,
        )
    except GovernanceSnapshotError as exc:
        raise VerifiedPlanError(exc.code) from exc
    if snapshot.run_id != expected_run_id:
        raise VerifiedPlanError("governance_run_changed")
    if snapshot.degraded_reason == "latest_run_failed":
        raise VerifiedPlanError("newer_governance_failure")
    if snapshot.degraded_reason is not None:
        raise VerifiedPlanError(snapshot.degraded_reason)


def _load_verified_candidate(
    *,
    governance_root: Path,
    expected_run_id: str,
    candidate_fingerprint: str,
    model_id: str,
    max_age_seconds: int,
    now: datetime | None = None,
) -> Mapping[str, Any]:
    """从严格验真的同一治理运行中定位唯一候选。"""
    try:
        snapshot = load_verified_governance_snapshot(
            governance_root=governance_root,
            max_age_seconds=max_age_seconds,
            now=now or datetime.now(UTC),
            include_queue=True,
        )
    except GovernanceSnapshotError as exc:
        raise VerifiedPlanError(exc.code) from exc
    if snapshot.run_id != expected_run_id:
        raise VerifiedPlanError("governance_run_changed")
    if snapshot.degraded_reason == "latest_run_failed":
        raise VerifiedPlanError("newer_governance_failure")
    if snapshot.degraded_reason is not None:
        raise VerifiedPlanError(snapshot.degraded_reason)
    matches = [
        item
        for item in snapshot.queue
        if item.get("candidate_fingerprint") == candidate_fingerprint and item.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise VerifiedPlanError("candidate_claim_mismatch")
    record = matches[0]
    if record.get("state") not in {"preflight_required", "admission_ready"}:
        raise VerifiedPlanError("candidate_state_not_admissible")
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        raise VerifiedPlanError("candidate_contract_invalid")
    if record.get("candidate_snapshot_sha256") != canonical_sha256(candidate):
        raise VerifiedPlanError("candidate_snapshot_sha256_mismatch")
    try:
        calculated_fingerprint = candidate_contract_fingerprint(candidate)
    except ValueError as exc:
        raise VerifiedPlanError("candidate_contract_invalid") from exc
    if calculated_fingerprint != candidate_fingerprint:
        raise VerifiedPlanError("candidate_fingerprint_mismatch")
    provider_key = record.get("provider_key")
    if (
        not isinstance(provider_key, str)
        or not provider_key
        or candidate.get("provider_key") != provider_key
        or candidate.get("model_id") != model_id
    ):
        raise VerifiedPlanError("candidate_contract_mismatch")
    if record.get("state") == "preflight_required":
        try:
            static_reasons = candidate_static_gate_reasons(candidate, provider_key=provider_key)
        except (KeyError, TypeError, ValueError) as exc:
            raise VerifiedPlanError("candidate_contract_invalid") from exc
        record_reasons = record.get("reasons")
        if static_reasons or not isinstance(record_reasons, list) or record_reasons:
            raise VerifiedPlanError("candidate_static_gate_failed")
    return record


def _worker_headers(worker_token: str, lease_token: str | None = None) -> dict[str, str]:
    headers = {WORKER_TOKEN_HEADER: worker_token}
    if lease_token is not None:
        headers[LEASE_TOKEN_HEADER] = lease_token
    return headers


def _claim(client: Any, fusion_base_url: str, worker_token: str) -> Mapping[str, str] | None:
    response = client.post(
        f"{fusion_base_url.rstrip('/')}/api/internal/model-management/admissions/claim",
        headers=_worker_headers(worker_token),
        timeout=10.0,
    )
    if response.status_code == 204:
        return None
    response.raise_for_status()
    payload = response.json()
    required = ("operation_id", "lease_token", "run_id", "candidate_fingerprint", "model_id")
    if not isinstance(payload, Mapping) or any(
        not isinstance(payload.get(name), str) or not payload[name] for name in required
    ):
        raise WorkerProtocolError("claim_contract_invalid")
    fingerprint = str(payload["candidate_fingerprint"])
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint.lower()):
        raise WorkerProtocolError("claim_fingerprint_invalid")
    return {name: str(payload[name]) for name in required}


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    error = result.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) and isinstance(error.get("code"), str) else None
    if error_code is None and isinstance(result.get("error_code"), str):
        error_code = str(result["error_code"])
    compensation = result.get("compensation")
    compensation = compensation if isinstance(compensation, Mapping) else {}
    compensation_errors = compensation.get("errors")
    return {
        "status": "succeeded" if result.get("status") == "succeeded" else "failed",
        "phase": str(result.get("phase") or "unknown")[:80],
        "error_code": error_code[:120] if error_code else None,
        "completed_phases": [
            str(item)[:80] for item in (result.get("completed_phases") or []) if isinstance(item, str)
        ],
        "writes_performed": bool(result.get("writes_performed")),
        "compensation": {
            "attempted": bool(compensation.get("attempted")),
            "key_restored": bool(compensation.get("key_restored")),
            "model_deleted": bool(compensation.get("model_deleted")),
            "catalog_invalidated": bool(compensation.get("catalog_invalidated")),
            "model_ownership_unverified": bool(compensation.get("model_ownership_unverified")),
            "manual_cleanup_required": bool(compensation.get("manual_cleanup_required")),
            "errors": [str(item)[:120] for item in (compensation_errors or []) if isinstance(item, str)],
        },
    }


def _complete(
    client: Any,
    *,
    fusion_base_url: str,
    worker_token: str,
    operation_id: str,
    lease_token: str,
    result: Mapping[str, Any],
) -> None:
    operation_path = quote(operation_id, safe="")
    endpoint = f"{fusion_base_url.rstrip('/')}/api/internal/model-management/admissions/{operation_path}/complete"
    last_error: Exception | None = None
    for attempt in range(COMPLETE_ATTEMPTS):
        try:
            response = client.post(
                endpoint,
                headers=_worker_headers(worker_token, lease_token),
                json=_safe_result(result),
                timeout=10.0,
            )
            if response.status_code == 409:
                raise StaleOperationLeaseError("operation_lease_stale")
            response.raise_for_status()
            return
        except StaleOperationLeaseError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < COMPLETE_ATTEMPTS:
                time.sleep(COMPLETE_RETRY_SECONDS)
    raise WorkerProtocolError("operation_complete_failed") from last_error


def _verification_failure(code: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "phase": "artifact_verification",
        "completed_phases": [],
        "writes_performed": False,
        "error": {"code": code},
        "compensation": {
            "attempted": False,
            "key_restored": False,
            "model_deleted": False,
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


def _interrupted_result() -> dict[str, Any]:
    """进程在外部事务中断时保守标记，禁止自动重跑造成二次写。"""
    result = _verification_failure("worker_execution_interrupted")
    result["phase"] = "worker_recovery"
    result["writes_performed"] = True
    result["compensation"]["attempted"] = True
    result["compensation"]["model_ownership_unverified"] = True
    result["compensation"]["manual_cleanup_required"] = True
    result["compensation"]["errors"] = ["worker_execution_interrupted"]
    return result


def _spool_path(state_dir: Path, operation_id: str) -> Path:
    import hashlib

    return state_dir / f"{hashlib.sha256(operation_id.encode()).hexdigest()}.json"


def _ensure_secure_state_dir(state_dir: Path, *, create: bool) -> bool:
    if create:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    elif not state_dir.exists():
        return False
    directory_stat = state_dir.lstat()
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) & 0o077
    ):
        raise WorkerProtocolError("worker_state_directory_insecure")
    return True


def _ensure_secure_acceptance_dir(acceptance_dir: Path) -> None:
    """验收目录必须预先创建，且仅当前 Worker 用户可访问。"""
    try:
        directory_stat = acceptance_dir.lstat()
    except OSError as exc:
        raise VerifiedPlanError("candidate_acceptance_directory_unavailable") from exc
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) & 0o077
    ):
        raise VerifiedPlanError("candidate_acceptance_directory_insecure")


def _write_acceptance_atomic(
    acceptance_dir: Path,
    *,
    candidate_fingerprint: str,
    acceptance: Mapping[str, Any],
) -> Path:
    """以 0600 权限原子落盘脱敏验收摘要。"""
    _ensure_secure_acceptance_dir(acceptance_dir)
    path = acceptance_dir / f"{candidate_fingerprint}.json"
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedPlanError("candidate_acceptance_write_failed") from exc
    if existing is not None and (
        not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.getuid() or stat.S_IMODE(existing.st_mode) & 0o177
    ):
        raise VerifiedPlanError("candidate_acceptance_file_insecure")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=acceptance_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(acceptance, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(acceptance_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return path
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedPlanError("candidate_acceptance_write_failed") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _single_candidate_report(record: Mapping[str, Any]) -> dict[str, Any]:
    provider_key = str(record["provider_key"])
    candidate = dict(record["candidate"])
    provider_display = str(candidate.get("provider_display") or provider_key)
    return {
        "mode": "read_only",
        "providers": {
            provider_key: {
                "status": "ok",
                "error": None,
                "report": {
                    "provider": {"key": provider_key, "display": provider_display},
                    "new": [candidate],
                    "existing": [],
                    "removed": [],
                    "unknown": [],
                },
            }
        },
    }


def _run_candidate_preflight_and_build_plan(
    *,
    record: Mapping[str, Any],
    expected_run_id: str,
    candidate_fingerprint: str,
    model_id: str,
    litellm_base_url: str,
    candidate_key: str,
    acceptance_dir: Path,
    client: Any,
) -> Mapping[str, Any]:
    if not candidate_key:
        raise VerifiedPlanError("candidate_preflight_credential_missing")
    try:
        candidate = parse_candidate(record["candidate"])
        report = run_preflight(
            candidate=candidate,
            base_url=litellm_base_url,
            api_key=candidate_key,
            apply=True,
            client=client,
        )
    except Exception as exc:
        raise VerifiedPlanError("candidate_preflight_request_failed") from exc
    if report.dry_run is not False or report.healthy is not True:
        raise VerifiedPlanError("candidate_preflight_failed")

    try:
        acceptance = serialize_report(
            report,
            context={
                "litellm_base_url": litellm_base_url,
                "credential_source": "LITELLM_CANDIDATE_KEY",
            },
        )
    except Exception as exc:
        raise VerifiedPlanError("candidate_acceptance_serialization_failed") from exc
    if (
        acceptance.get("healthy") is not True
        or acceptance.get("dry_run") is not False
        or acceptance.get("candidate_fingerprint") != candidate_fingerprint
    ):
        raise VerifiedPlanError("candidate_acceptance_contract_mismatch")
    _write_acceptance_atomic(
        acceptance_dir,
        candidate_fingerprint=candidate_fingerprint,
        acceptance=acceptance,
    )
    try:
        plan = build_admission_plan(
            candidate_report=_single_candidate_report(record),
            candidate_acceptance_summary=acceptance,
        )
    except Exception as exc:
        raise VerifiedPlanError("candidate_admission_plan_build_failed") from exc
    plan = {**plan, "run_id": expected_run_id}
    eligible = plan.get("eligible")
    if (
        plan.get("status") != "complete"
        or not isinstance(eligible, list)
        or len(eligible) != 1
        or not isinstance(eligible[0], Mapping)
        or eligible[0].get("model_id") != model_id
        or plan.get("isolated")
    ):
        raise VerifiedPlanError("candidate_admission_plan_invalid")
    return plan


def _write_spool(
    state_dir: Path,
    *,
    claim: Mapping[str, str],
    result: Mapping[str, Any] | None,
) -> Path:
    _ensure_secure_state_dir(state_dir, create=True)
    path = _spool_path(state_dir, claim["operation_id"])
    payload = {
        "schema_version": 1,
        "claim": dict(claim),
        "result": _safe_result(result) if result is not None else None,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_dir,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(state_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_spool(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("worker_spool_invalid") from exc
    claim = payload.get("claim") if isinstance(payload, Mapping) else None
    if not isinstance(claim, Mapping):
        raise WorkerProtocolError("worker_spool_invalid")
    required = ("operation_id", "lease_token", "run_id", "candidate_fingerprint", "model_id")
    if any(not isinstance(claim.get(name), str) or not claim[name] for name in required):
        raise WorkerProtocolError("worker_spool_invalid")
    result = payload.get("result")
    if result is None:
        result = _safe_result(_interrupted_result())
    if not isinstance(result, Mapping):
        raise WorkerProtocolError("worker_spool_invalid")
    return {name: str(claim[name]) for name in required}, dict(result)


def _quarantine_stale_spool(state_dir: Path, path: Path) -> Path:
    quarantine_dir = state_dir / "quarantine"
    _ensure_secure_state_dir(quarantine_dir, create=True)
    target = quarantine_dir / f"{path.stem}.{time.time_ns()}.json"
    os.replace(path, target)
    for directory in (state_dir, quarantine_dir):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return target


def flush_spooled_results(
    client: Any,
    *,
    state_dir: Path,
    fusion_base_url: str,
    worker_token: str,
) -> int:
    if not _ensure_secure_state_dir(state_dir, create=False):
        return 0
    completed = 0
    for path in sorted(state_dir.glob("*.json")):
        claim, result = _read_spool(path)
        try:
            _complete(
                client,
                fusion_base_url=fusion_base_url,
                worker_token=worker_token,
                operation_id=claim["operation_id"],
                lease_token=claim["lease_token"],
                result=result,
            )
        except StaleOperationLeaseError:
            quarantined_path = _quarantine_stale_spool(state_dir, path)
            print(
                json.dumps(
                    {
                        "status": "warning",
                        "error": {"code": "stale_spool_quarantined"},
                        "spool_id": quarantined_path.stem,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            continue
        path.unlink()
        completed += 1
    return completed


def process_once(
    *,
    client: Any,
    fusion_base_url: str,
    worker_token: str,
    governance_root: Path,
    governance_max_age_seconds: int,
    litellm_base_url: str,
    master_key: str,
    virtual_key: str,
    environ: Mapping[str, str],
    state_dir: Path | None = None,
    candidate_key: str = "",
    acceptance_dir: Path | None = None,
) -> dict[str, Any]:
    claim = _claim(client, fusion_base_url, worker_token)
    if claim is None:
        return {"status": "idle"}
    spool_path = _write_spool(state_dir, claim=claim, result=None) if state_dir is not None else None
    heartbeat = OperationLeaseHeartbeat(
        client,
        fusion_base_url=fusion_base_url,
        worker_token=worker_token,
        operation_id=claim["operation_id"],
        lease_token=claim["lease_token"],
    )
    heartbeat.start()
    try:
        try:
            candidate_record = _load_verified_candidate(
                governance_root=governance_root,
                expected_run_id=claim["run_id"],
                candidate_fingerprint=claim["candidate_fingerprint"],
                model_id=claim["model_id"],
                max_age_seconds=governance_max_age_seconds,
            )
            if candidate_record.get("state") == "admission_ready":
                plan = load_verified_admission_plan(
                    governance_root=governance_root,
                    candidate_fingerprint=claim["candidate_fingerprint"],
                )
            else:
                if acceptance_dir is None:
                    raise VerifiedPlanError("candidate_acceptance_directory_missing")
                plan = _run_candidate_preflight_and_build_plan(
                    record=candidate_record,
                    expected_run_id=claim["run_id"],
                    candidate_fingerprint=claim["candidate_fingerprint"],
                    model_id=claim["model_id"],
                    litellm_base_url=litellm_base_url,
                    candidate_key=candidate_key,
                    acceptance_dir=acceptance_dir,
                    client=client,
                )
        except VerifiedPlanError as exc:
            result = _verification_failure(exc.code)
        else:
            try:
                _load_verified_candidate(
                    governance_root=governance_root,
                    expected_run_id=claim["run_id"],
                    candidate_fingerprint=claim["candidate_fingerprint"],
                    model_id=claim["model_id"],
                    max_age_seconds=governance_max_age_seconds,
                )
            except VerifiedPlanError as exc:
                result = _verification_failure(exc.code)
            else:

                def invalidate_catalog() -> None:
                    heartbeat.ensure_healthy()
                    operation_path = quote(claim["operation_id"], safe="")
                    response = client.post(
                        f"{fusion_base_url.rstrip('/')}/api/internal/model-management/admissions/"
                        f"{operation_path}/invalidate-catalog",
                        headers=_worker_headers(worker_token, claim["lease_token"]),
                        timeout=10.0,
                    )
                    response.raise_for_status()

                heartbeat.ensure_healthy()
                result = execute_admission(
                    plan=plan,
                    apply=True,
                    expected_run_id=claim["run_id"],
                    confirm_model_id=claim["model_id"],
                    confirm_fingerprint=claim["candidate_fingerprint"],
                    litellm_base_url=litellm_base_url,
                    fusion_base_url=fusion_base_url,
                    master_key=master_key,
                    virtual_key=virtual_key,
                    environ=environ,
                    client=client,
                    catalog_invalidation_fn=invalidate_catalog,
                )
        heartbeat.ensure_healthy()
        heartbeat.renew()
    finally:
        heartbeat.stop()
    if state_dir is not None:
        spool_path = _write_spool(state_dir, claim=claim, result=result)
    _complete(
        client,
        fusion_base_url=fusion_base_url,
        worker_token=worker_token,
        operation_id=claim["operation_id"],
        lease_token=claim["lease_token"],
        result=result,
    )
    if spool_path is not None:
        spool_path.unlink(missing_ok=True)
    return result


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance-root", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".local/state/fusion/litellm-model-management",
    )
    parser.add_argument("--acceptance-dir", type=Path, required=True)
    parser.add_argument(
        "--governance-max-age-seconds",
        type=int,
        default=os.environ.get("LITELLM_GOVERNANCE_MAX_AGE_SECONDS"),
    )
    parser.add_argument(
        "--fusion-base-url",
        default=os.environ.get("FUSION_MODEL_MANAGEMENT_BASE_URL", "http://127.0.0.1:8002"),
    )
    parser.add_argument("--litellm-base-url", default=os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000"))
    args = parser.parse_args(argv)
    if args.governance_max_age_seconds is None:
        parser.error("必须配置 LITELLM_GOVERNANCE_MAX_AGE_SECONDS 或 --governance-max-age-seconds")
    if args.governance_max_age_seconds <= 0:
        parser.error("governance max age 必须为正整数")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    worker_token = os.environ.get("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", "")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    virtual_key = os.environ.get("LITELLM_VIRTUAL_KEY", "")
    candidate_key = os.environ.get("LITELLM_CANDIDATE_KEY", "")
    if not worker_token or not master_key or not virtual_key or not candidate_key:
        print(json.dumps({"status": "failed", "error": {"code": "required_environment_missing"}}))
        return 1
    try:
        _ensure_secure_acceptance_dir(args.acceptance_dir)
        with httpx.Client() as client:
            flush_spooled_results(
                client,
                state_dir=args.state_dir,
                fusion_base_url=args.fusion_base_url,
                worker_token=worker_token,
            )
            result = process_once(
                client=client,
                fusion_base_url=args.fusion_base_url,
                worker_token=worker_token,
                governance_root=args.governance_root,
                governance_max_age_seconds=args.governance_max_age_seconds,
                litellm_base_url=args.litellm_base_url,
                master_key=master_key,
                virtual_key=virtual_key,
                candidate_key=candidate_key,
                acceptance_dir=args.acceptance_dir,
                environ=os.environ,
                state_dir=args.state_dir,
            )
    except (WorkerProtocolError, VerifiedPlanError, httpx.HTTPError) as exc:
        code = exc.code if isinstance(exc, VerifiedPlanError) else type(exc).__name__
        print(json.dumps({"status": "failed", "error": {"code": code}}))
        return 1
    print(json.dumps({"status": result.get("status")}, sort_keys=True))
    return 0 if result.get("status") in {"idle", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
