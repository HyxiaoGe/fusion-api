"""原子写入模型准入 Worker 的受管配置，不输出凭据值。"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

MANAGED_KEYS = {
    "FUSION_MODEL_MANAGEMENT_BASE_URL",
    "LITELLM_GOVERNANCE_MAX_AGE_SECONDS",
    "LITELLM_MODEL_ADMISSION_WORKER_TOKEN",
    "LITELLM_VIRTUAL_KEY",
}


def _validate_secret_value(name: str, value: str) -> str:
    if not value or any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{name} 无效")
    return value


def _read_secure_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("治理配置不是普通文件")
        if file_stat.st_uid != os.getuid():
            raise ValueError("治理配置 owner 不匹配")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise ValueError("治理配置权限过宽")
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _render_updated_content(content: str, values: Mapping[str, str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        assignment = stripped.removeprefix("export ").lstrip() if stripped.startswith("export ") else stripped
        name = assignment.split("=", 1)[0].strip() if "=" in assignment else ""
        if name not in MANAGED_KEYS:
            lines.append(raw_line)
            continue
        if name in seen:
            raise ValueError(f"治理配置重复定义 {name}")
        seen.add(name)
        lines.append(f"{name}={values[name]}")
    if lines and lines[-1] != "":
        lines.append("")
    for name in sorted(MANAGED_KEYS - seen):
        lines.append(f"{name}={values[name]}")
    return "\n".join(lines).rstrip("\n") + "\n"


def update_worker_environment(
    path: Path,
    *,
    virtual_key: str,
    worker_token: str,
    fusion_base_url: str,
    governance_max_age_seconds: int,
) -> set[str]:
    if governance_max_age_seconds <= 0:
        raise ValueError("LITELLM_GOVERNANCE_MAX_AGE_SECONDS 无效")
    values = {
        "FUSION_MODEL_MANAGEMENT_BASE_URL": _validate_secret_value(
            "FUSION_MODEL_MANAGEMENT_BASE_URL", fusion_base_url.rstrip("/")
        ),
        "LITELLM_MODEL_ADMISSION_WORKER_TOKEN": _validate_secret_value(
            "LITELLM_MODEL_ADMISSION_WORKER_TOKEN", worker_token
        ),
        "LITELLM_GOVERNANCE_MAX_AGE_SECONDS": str(governance_max_age_seconds),
        "LITELLM_VIRTUAL_KEY": _validate_secret_value("LITELLM_VIRTUAL_KEY", virtual_key),
    }
    content = _read_secure_file(path)
    rendered = _render_updated_content(content, values)
    parent = path.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = rendered.encode()
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_path, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
    return set(MANAGED_KEYS)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        updated = update_worker_environment(
            args.env_file,
            virtual_key=os.environ.get("LITELLM_VIRTUAL_KEY", ""),
            worker_token=os.environ.get("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", ""),
            fusion_base_url=os.environ.get("FUSION_MODEL_MANAGEMENT_BASE_URL", ""),
            governance_max_age_seconds=int(os.environ.get("LITELLM_GOVERNANCE_MAX_AGE_SECONDS", "")),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(
            json.dumps(
                {"healthy": False, "issues": [f"worker_env_update_failed:{type(exc).__name__}"]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"healthy": True, "updated": sorted(updated)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
