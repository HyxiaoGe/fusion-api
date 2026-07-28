"""只读检查 LiteLLM 升级门禁。

脚本只调用 GET 端点并检查本地备份文件，不修改 LiteLLM、数据库或模型目录。
任何密钥只用于请求头，不会进入输出报告。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:4000"
DEFAULT_MAX_BACKUP_AGE_SECONDS = 36 * 60 * 60


@dataclass(frozen=True)
class FileGate:
    path: str
    ok: bool
    age_seconds: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class UpgradeReadinessReport:
    ready_for_upgrade: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    liveliness_ok: bool = False
    readiness_ok: bool = False
    database_connected: bool = False
    model_count: int = 0
    primary_backup: FileGate | None = None
    secondary_backup: FileGate | None = None
    cost_sync_scheduled: bool = False


def check_file_gate(path: Path, *, max_age_seconds: int) -> FileGate:
    """检查门禁文件存在、非空且未过期。"""
    if not path.is_file():
        return FileGate(path=str(path), ok=False, reason="文件不存在")
    if path.stat().st_size <= 0:
        return FileGate(path=str(path), ok=False, reason="文件为空")
    age_seconds = max(0, int(time.time() - path.stat().st_mtime))
    if age_seconds > max_age_seconds:
        return FileGate(
            path=str(path),
            ok=False,
            age_seconds=age_seconds,
            reason=f"文件年龄超过 {max_age_seconds} 秒",
        )
    return FileGate(path=str(path), ok=True, age_seconds=age_seconds)


def check_gzip_file_gate(path: Path, *, max_age_seconds: int) -> FileGate:
    gate = check_file_gate(path, max_age_seconds=max_age_seconds)
    if not gate.ok:
        return gate
    try:
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
    except (OSError, EOFError):
        return FileGate(path=str(path), ok=False, age_seconds=gate.age_seconds, reason="gzip 完整性校验失败")
    return gate


def check_restic_marker(path: Path, *, max_age_seconds: int) -> FileGate:
    gate = check_file_gate(path, max_age_seconds=max_age_seconds)
    if not gate.ok:
        return gate
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FileGate(path=str(path), ok=False, age_seconds=gate.age_seconds, reason="restic marker 不是有效 JSON")
    if not isinstance(payload, Mapping) or payload.get("status") != "success" or not payload.get("snapshot_id"):
        return FileGate(
            path=str(path),
            ok=False,
            age_seconds=gate.age_seconds,
            reason="restic marker 缺少 status=success 或 snapshot_id",
        )
    return gate


def evaluate_readiness(
    *,
    liveliness_ok: bool,
    readiness_ok: bool,
    database_connected: bool,
    model_count: int,
    primary_backup: FileGate,
    secondary_backup: FileGate,
    cost_sync_scheduled: bool,
) -> UpgradeReadinessReport:
    blockers: list[str] = []
    if not liveliness_ok:
        blockers.append("liveliness_failed")
    if not readiness_ok:
        blockers.append("readiness_failed")
    if not database_connected:
        blockers.append("database_not_connected")
    if model_count <= 0:
        blockers.append("model_catalog_empty")
    if not primary_backup.ok:
        blockers.append("primary_backup")
    if not secondary_backup.ok:
        blockers.append("secondary_backup")

    warnings: list[str] = []
    if not cost_sync_scheduled:
        warnings.append("cost_sync_not_scheduled")

    return UpgradeReadinessReport(
        ready_for_upgrade=not blockers,
        blockers=blockers,
        warnings=warnings,
        liveliness_ok=liveliness_ok,
        readiness_ok=readiness_ok,
        database_connected=database_connected,
        model_count=model_count,
        primary_backup=primary_backup,
        secondary_backup=secondary_backup,
        cost_sync_scheduled=cost_sync_scheduled,
    )


def serialize_report(report: UpgradeReadinessReport) -> dict[str, Any]:
    return asdict(report)


def _get_json(client: httpx.Client, path: str) -> tuple[bool, Any]:
    try:
        response = client.get(path)
        response.raise_for_status()
        return True, response.json()
    except (httpx.HTTPError, ValueError):
        return False, {}


def _get_ok(client: httpx.Client, path: str) -> bool:
    try:
        response = client.get(path)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _database_connected(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized_key = str(key).lower()
            if "db" in normalized_key or "database" in normalized_key:
                normalized_value = str(value).lower()
                if normalized_value in {"connected", "healthy", "ok", "true"}:
                    return True
                if isinstance(value, bool) and value:
                    return True
            if _database_connected(value):
                return True
    elif isinstance(payload, list):
        return any(_database_connected(item) for item in payload)
    return False


def collect_remote_state(base_url: str, master_key: str) -> tuple[bool, bool, bool, int, bool]:
    headers = {"Authorization": f"Bearer {master_key}"} if master_key else {}
    with httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=20.0) as client:
        liveliness_ok = _get_ok(client, "/health/liveliness")
        readiness_ok, readiness_payload = _get_json(client, "/health/readiness")
        model_info_ok, model_info = _get_json(client, "/model/info")
        cost_status_ok, cost_status = _get_json(client, "/schedule/model_cost_map_reload/status")

    models = model_info.get("data") if isinstance(model_info, Mapping) else []
    model_count = len(models) if model_info_ok and isinstance(models, list) else 0
    cost_sync_scheduled = (
        bool(cost_status.get("scheduled")) if cost_status_ok and isinstance(cost_status, Mapping) else False
    )
    return (
        liveliness_ok,
        readiness_ok,
        readiness_ok and _database_connected(readiness_payload),
        model_count,
        cost_sync_scheduled,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查 LiteLLM 升级门禁")
    parser.add_argument("--base-url", default=os.environ.get("LITELLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--primary-backup", type=Path, required=True)
    parser.add_argument("--secondary-backup-marker", type=Path, required=True)
    parser.add_argument("--max-backup-age-seconds", type=int, default=DEFAULT_MAX_BACKUP_AGE_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    state = collect_remote_state(args.base_url, os.environ.get("LITELLM_MASTER_KEY", ""))
    report = evaluate_readiness(
        liveliness_ok=state[0],
        readiness_ok=state[1],
        database_connected=state[2],
        model_count=state[3],
        primary_backup=check_gzip_file_gate(
            args.primary_backup,
            max_age_seconds=args.max_backup_age_seconds,
        ),
        secondary_backup=check_restic_marker(
            args.secondary_backup_marker,
            max_age_seconds=args.max_backup_age_seconds,
        ),
        cost_sync_scheduled=state[4],
    )
    print(json.dumps(serialize_report(report), ensure_ascii=False, indent=2))
    return 0 if report.ready_for_upgrade else 1


if __name__ == "__main__":
    raise SystemExit(main())
