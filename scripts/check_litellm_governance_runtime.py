"""检查宿主 LiteLLM 治理任务的 Python 与最小依赖版本。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Sequence

MIN_PYTHON = (3, 11)
EXPECTED_HTTPX_VERSION = "0.28.1"


def runtime_status(
    *,
    secret_files: Sequence[Path] = (),
) -> dict[str, object]:
    issues: list[str] = []
    if sys.version_info < MIN_PYTHON:
        issues.append("python_version_unsupported")
    try:
        httpx = importlib.import_module("httpx")
        httpx_version = str(getattr(httpx, "__version__", ""))
    except ImportError:
        httpx_version = ""
        issues.append("httpx_missing")
    else:
        if httpx_version != EXPECTED_HTTPX_VERSION:
            issues.append("httpx_version_mismatch")
    for path in secret_files:
        try:
            file_stat = path.lstat()
        except OSError:
            issues.append("secret_file_missing")
            continue
        if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            issues.append("secret_file_not_regular")
        if file_stat.st_uid != os.getuid():
            issues.append("secret_file_wrong_owner")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            issues.append("secret_file_permissions_too_open")
    return {
        "healthy": not issues,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "httpx": httpx_version or None,
        "secret_files_checked": len(secret_files),
        "issues": list(dict.fromkeys(issues)),
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--secret-file",
        action="append",
        default=[],
        type=Path,
        help="只检查 owner、文件类型和权限，不读取内容；可重复",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    status = runtime_status(secret_files=args.secret_file)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    return 0 if status["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
