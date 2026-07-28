"""从 restic 最新快照 JSON 原子写入成功 marker。

输入应来自 ``restic snapshots --latest 1 --json``。本脚本不执行 restic，
不读取仓库凭据，也不运行 forget/prune。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

DEFAULT_MAX_AGE_SECONDS = 36 * 60 * 60
MAX_FUTURE_SKEW_SECONDS = 5 * 60


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("snapshot time 缺失")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot time 缺少时区")
    return parsed.astimezone(UTC)


def build_success_marker(
    snapshots: Any,
    *,
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    required_tag: str = "daily",
) -> dict[str, str]:
    """验证最新快照并生成不含凭据的 marker payload。"""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds 必须大于 0")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("restic snapshots JSON 必须是非空列表")

    candidates: list[tuple[datetime, Mapping[str, Any]]] = []
    for item in snapshots:
        if not isinstance(item, Mapping):
            raise ValueError("snapshot 条目必须是 JSON 对象")
        candidates.append((_parse_timestamp(item.get("time")), item))

    completed_at, latest = max(candidates, key=lambda item: item[0])
    snapshot_id = latest.get("id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot id 缺失")
    tags = latest.get("tags")
    if required_tag and (not isinstance(tags, list) or required_tag not in tags):
        raise ValueError(f"snapshot 缺少 required tag: {required_tag}")

    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    age_seconds = (checked_at - completed_at).total_seconds()
    if age_seconds < -MAX_FUTURE_SKEW_SECONDS:
        raise ValueError("snapshot time 位于未来")
    if age_seconds > max_age_seconds:
        raise ValueError("snapshot 已过期")

    return {
        "status": "success",
        "snapshot_id": snapshot_id.strip(),
        "completed_at": completed_at.isoformat(),
        "tag": required_tag,
    }


def write_marker_atomic(path: Path, payload: Mapping[str, str]) -> None:
    """同目录落临时文件并原子替换；替换前失败不会覆盖旧 marker。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _read_snapshots(path: str, *, stdin: TextIO) -> Any:
    raw = stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots-json",
        default="-",
        help="restic snapshots JSON 文件；默认 -，表示标准输入",
    )
    parser.add_argument("--output", type=Path, required=True, help="成功 marker 输出路径")
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=DEFAULT_MAX_AGE_SECONDS,
        help="允许的最新快照最大年龄，默认 36 小时",
    )
    parser.add_argument("--required-tag", default="daily", help="快照必须包含的 tag，默认 daily")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    now: datetime | None = None,
) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        snapshots = _read_snapshots(args.snapshots_json, stdin=stdin or sys.stdin)
        payload = build_success_marker(
            snapshots,
            now=now,
            max_age_seconds=args.max_age_seconds,
            required_tag=args.required_tag,
        )
        write_marker_atomic(args.output, payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        print(
            json.dumps({"status": "error", "code": "restic_snapshot_invalid"}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
