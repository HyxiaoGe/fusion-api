"""为 API 与准入 Worker 提供同一套治理快照信任链校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024


class GovernanceSnapshotError(ValueError):
    """治理快照不可作为可信输入，仅暴露稳定错误码。"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class VerifiedGovernanceSnapshot:
    run_id: str
    started_at: datetime
    queue: list[Mapping[str, Any]]
    degraded_reason: str | None = None


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, code: str) -> Any:
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise GovernanceSnapshotError(f"{code}_too_large")
        return json.loads(path.read_text(encoding="utf-8"))
    except GovernanceSnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GovernanceSnapshotError(code) from exc


def _read_mapping(path: Path, *, code: str) -> Mapping[str, Any]:
    payload = _read_json(path, code=code)
    if not isinstance(payload, Mapping):
        raise GovernanceSnapshotError(code)
    return payload


def _artifact_sha256(manifest: Mapping[str, Any], artifact_name: str) -> str:
    artifacts = manifest.get("artifacts")
    metadata = artifacts.get(artifact_name) if isinstance(artifacts, Mapping) else None
    digest = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise GovernanceSnapshotError("governance_artifact_missing")
    return digest


def _run_id_time(run_id: str) -> datetime:
    try:
        parsed = datetime.strptime(run_id, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise GovernanceSnapshotError("governance_run_id_invalid") from exc
    if parsed.strftime("%Y%m%dT%H%M%S%fZ") != run_id:
        raise GovernanceSnapshotError("governance_run_id_invalid")
    return parsed


def _parse_started_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise GovernanceSnapshotError("governance_started_at_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GovernanceSnapshotError("governance_started_at_invalid") from exc
    if parsed.tzinfo is None:
        raise GovernanceSnapshotError("governance_started_at_invalid")
    return parsed.astimezone(UTC)


def _load_verified_run(
    root: Path,
    *,
    pointer_name: str,
    expected_status: str,
    include_queue: bool,
) -> VerifiedGovernanceSnapshot:
    pointer = _read_mapping(root / pointer_name, code="governance_pointer_invalid")
    if pointer.get("schema_version") != 1:
        raise GovernanceSnapshotError("governance_pointer_schema_invalid")
    run_id = pointer.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise GovernanceSnapshotError("governance_run_id_invalid")
    run_started_at = _run_id_time(run_id)
    run_path = pointer.get("run_path")
    if not isinstance(run_path, str):
        raise GovernanceSnapshotError("governance_run_path_invalid")
    relative = Path(run_path)
    if relative.is_absolute() or relative.parts != ("runs", run_id):
        raise GovernanceSnapshotError("governance_run_path_invalid")
    try:
        run_dir = (root / relative).resolve(strict=True)
        run_dir.relative_to(root)
    except (OSError, ValueError) as exc:
        raise GovernanceSnapshotError("governance_run_path_invalid") from exc

    manifest = _read_mapping(run_dir / "manifest.json", code="governance_manifest_invalid")
    if manifest.get("schema_version") != 1:
        raise GovernanceSnapshotError("governance_manifest_schema_invalid")
    if pointer.get("manifest_sha256") != canonical_sha256(manifest):
        raise GovernanceSnapshotError("governance_manifest_sha256_mismatch")

    summary = _read_mapping(run_dir / "cycle-summary.json", code="governance_summary_invalid")
    summary_sha256 = canonical_sha256(summary)
    if pointer.get("summary_sha256") != summary_sha256:
        raise GovernanceSnapshotError("governance_summary_sha256_mismatch")
    if _artifact_sha256(manifest, "cycle-summary.json") != summary_sha256:
        raise GovernanceSnapshotError("governance_summary_manifest_mismatch")
    if (
        summary.get("schema_version") != 1
        or summary.get("run_id") != run_id
        or summary.get("status") != expected_status
    ):
        raise GovernanceSnapshotError("governance_summary_contract_invalid")
    started_at = _parse_started_at(summary.get("started_at"))
    if started_at != run_started_at:
        raise GovernanceSnapshotError("governance_start_time_mismatch")

    queue: list[Mapping[str, Any]] = []
    if include_queue:
        queue_payload = _read_json(run_dir / "candidate-queue.json", code="candidate_queue_invalid")
        if not isinstance(queue_payload, list) or not all(isinstance(item, Mapping) for item in queue_payload):
            raise GovernanceSnapshotError("candidate_queue_invalid")
        if _artifact_sha256(manifest, "candidate-queue.json") != canonical_sha256(queue_payload):
            raise GovernanceSnapshotError("candidate_queue_sha256_mismatch")
        queue = list(queue_payload)
    return VerifiedGovernanceSnapshot(run_id=run_id, started_at=started_at, queue=queue)


def load_verified_governance_snapshot(
    *,
    governance_root: Path,
    max_age_seconds: int,
    now: datetime,
    include_queue: bool,
) -> VerifiedGovernanceSnapshot:
    """校验最近成功快照；最近失败状态只会令结果降级，不会丢弃只读数据。"""
    if max_age_seconds <= 0:
        raise GovernanceSnapshotError("governance_max_age_invalid")
    try:
        root = governance_root.resolve(strict=True)
    except OSError as exc:
        raise GovernanceSnapshotError("governance_root_invalid") from exc

    success = _load_verified_run(
        root,
        pointer_name="latest-success.json",
        expected_status="success",
        include_queue=include_queue,
    )
    age = (now.astimezone(UTC) - success.started_at).total_seconds()
    if age < -300:
        raise GovernanceSnapshotError("governance_run_from_future")
    if age > max_age_seconds:
        raise GovernanceSnapshotError("governance_run_stale")

    failure_path = root / "latest-failure.json"
    if not failure_path.exists():
        return success
    try:
        failure = _load_verified_run(
            root,
            pointer_name="latest-failure.json",
            expected_status="failed",
            include_queue=False,
        )
    except GovernanceSnapshotError:
        return _degraded_snapshot(success, reason="latest_failure_unverified")
    failure_age = (now.astimezone(UTC) - failure.started_at).total_seconds()
    if failure_age < -300:
        # 宿主机时钟恢复后，未来失败指针不能永久压住已验证的正常成功快照。
        return success
    if failure.run_id > success.run_id:
        return _degraded_snapshot(success, reason="latest_run_failed")
    return success


def _degraded_snapshot(
    snapshot: VerifiedGovernanceSnapshot,
    *,
    reason: str,
) -> VerifiedGovernanceSnapshot:
    return VerifiedGovernanceSnapshot(
        run_id=snapshot.run_id,
        started_at=snapshot.started_at,
        queue=snapshot.queue,
        degraded_reason=reason,
    )
