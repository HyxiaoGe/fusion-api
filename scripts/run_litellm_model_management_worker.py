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
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import httpx

from scripts.execute_litellm_candidate_admission import (
    VerifiedPlanError,
    execute_admission,
    load_verified_admission_plan,
)

WORKER_TOKEN_HEADER = "X-Fusion-Worker-Token"
LEASE_TOKEN_HEADER = "X-Operation-Lease"
COMPLETE_ATTEMPTS = 3
COMPLETE_RETRY_SECONDS = 2.0


class WorkerProtocolError(RuntimeError):
    """内部 Worker 协议不满足安全合同。"""


def canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifiedPlanError(code) from exc
    if not isinstance(payload, Mapping):
        raise VerifiedPlanError(code)
    return payload


def _load_verified_summary(
    root: Path,
    *,
    pointer_name: str,
    expected_status: str,
) -> tuple[Mapping[str, Any], datetime] | None:
    pointer_path = root / pointer_name
    if not pointer_path.exists() and pointer_name == "latest-failure.json":
        return None
    pointer = _read_json(pointer_path, code=f"{pointer_name.removesuffix('.json').replace('-', '_')}_invalid")
    run_id = pointer.get("run_id")
    run_path = pointer.get("run_path")
    if not isinstance(run_id, str) or not run_id or not isinstance(run_path, str):
        raise VerifiedPlanError("governance_pointer_invalid")
    relative_path = Path(run_path)
    if relative_path.is_absolute() or relative_path.parts != ("runs", run_id):
        raise VerifiedPlanError("run_path_invalid")
    try:
        run_dir = (root / relative_path).resolve(strict=True)
        run_dir.relative_to(root)
    except (OSError, ValueError) as exc:
        raise VerifiedPlanError("run_path_invalid") from exc
    manifest = _read_json(run_dir / "manifest.json", code="manifest_invalid")
    summary = _read_json(run_dir / "cycle-summary.json", code="cycle_summary_invalid")
    if pointer.get("manifest_sha256") != canonical_sha256(manifest):
        raise VerifiedPlanError("manifest_sha256_mismatch")
    if pointer.get("summary_sha256") != canonical_sha256(summary):
        raise VerifiedPlanError("cycle_summary_sha256_mismatch")
    artifacts = manifest.get("artifacts")
    summary_artifact = artifacts.get("cycle-summary.json") if isinstance(artifacts, Mapping) else None
    if not isinstance(summary_artifact, Mapping) or summary_artifact.get("sha256") != canonical_sha256(summary):
        raise VerifiedPlanError("cycle_summary_manifest_mismatch")
    if summary.get("run_id") != run_id or summary.get("status") != expected_status:
        raise VerifiedPlanError("cycle_summary_contract_invalid")
    started_at = summary.get("started_at")
    if not isinstance(started_at, str):
        raise VerifiedPlanError("cycle_started_at_invalid")
    try:
        parsed_started_at = datetime.fromisoformat(started_at)
    except ValueError as exc:
        raise VerifiedPlanError("cycle_started_at_invalid") from exc
    if parsed_started_at.tzinfo is None:
        raise VerifiedPlanError("cycle_started_at_invalid")
    return summary, parsed_started_at.astimezone(UTC)


def validate_governance_freshness(
    *,
    governance_root: Path,
    expected_run_id: str,
    max_age_seconds: int,
    now: datetime | None = None,
) -> None:
    """再次校验治理周期的新鲜度，并拒绝成功周期之后出现的失败周期。"""
    if max_age_seconds <= 0:
        raise VerifiedPlanError("governance_max_age_invalid")
    try:
        root = governance_root.resolve(strict=True)
    except OSError as exc:
        raise VerifiedPlanError("governance_root_invalid") from exc
    success = _load_verified_summary(root, pointer_name="latest-success.json", expected_status="success")
    if success is None:
        raise VerifiedPlanError("latest_success_invalid")
    summary, success_started_at = success
    if summary.get("run_id") != expected_run_id:
        raise VerifiedPlanError("governance_run_changed")
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    age_seconds = (current_time - success_started_at).total_seconds()
    if age_seconds < -300:
        raise VerifiedPlanError("governance_run_from_future")
    if age_seconds > max_age_seconds:
        raise VerifiedPlanError("governance_run_stale")
    failure = _load_verified_summary(root, pointer_name="latest-failure.json", expected_status="failed")
    if failure is not None and failure[1] > success_started_at:
        raise VerifiedPlanError("newer_governance_failure")


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
    if not isinstance(payload, Mapping) or any(not isinstance(payload.get(name), str) or not payload[name] for name in required):
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
        "completed_phases": [str(item)[:80] for item in (result.get("completed_phases") or []) if isinstance(item, str)],
        "writes_performed": bool(result.get("writes_performed")),
        "compensation": {
            "attempted": bool(compensation.get("attempted")),
            "key_restored": bool(compensation.get("key_restored")),
            "model_deleted": bool(compensation.get("model_deleted")),
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
            response.raise_for_status()
            return
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
        _complete(
            client,
            fusion_base_url=fusion_base_url,
            worker_token=worker_token,
            operation_id=claim["operation_id"],
            lease_token=claim["lease_token"],
            result=result,
        )
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
) -> dict[str, Any]:
    claim = _claim(client, fusion_base_url, worker_token)
    if claim is None:
        return {"status": "idle"}
    spool_path = _write_spool(state_dir, claim=claim, result=None) if state_dir is not None else None
    try:
        plan = load_verified_admission_plan(
            governance_root=governance_root,
            candidate_fingerprint=claim["candidate_fingerprint"],
        )
        validate_governance_freshness(
            governance_root=governance_root,
            expected_run_id=claim["run_id"],
            max_age_seconds=governance_max_age_seconds,
        )
    except VerifiedPlanError as exc:
        result = _verification_failure(exc.code)
    else:

        def invalidate_catalog() -> None:
            operation_path = quote(claim["operation_id"], safe="")
            response = client.post(
                f"{fusion_base_url.rstrip('/')}/api/internal/model-management/admissions/"
                f"{operation_path}/invalidate-catalog",
                headers=_worker_headers(worker_token, claim["lease_token"]),
                timeout=10.0,
            )
            response.raise_for_status()

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
    parser.add_argument(
        "--governance-max-age-seconds",
        type=int,
        default=int(os.environ.get("MODEL_MANAGEMENT_GOVERNANCE_MAX_AGE_SECONDS", "86400")),
    )
    parser.add_argument(
        "--fusion-base-url",
        default=os.environ.get("FUSION_MODEL_MANAGEMENT_BASE_URL", "http://127.0.0.1:8002"),
    )
    parser.add_argument("--litellm-base-url", default=os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    worker_token = os.environ.get("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", "")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    virtual_key = os.environ.get("LITELLM_VIRTUAL_KEY", "")
    if not worker_token or not master_key or not virtual_key:
        print(json.dumps({"status": "failed", "error": {"code": "required_environment_missing"}}))
        return 1
    try:
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
                environ=os.environ,
                state_dir=args.state_dir,
            )
    except (WorkerProtocolError, httpx.HTTPError) as exc:
        print(json.dumps({"status": "failed", "error": {"code": type(exc).__name__}}))
        return 1
    print(json.dumps({"status": result.get("status")}, sort_keys=True))
    return 0 if result.get("status") in {"idle", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
