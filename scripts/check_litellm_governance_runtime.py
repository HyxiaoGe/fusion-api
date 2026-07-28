"""检查宿主 LiteLLM 治理任务的 Python 与最小依赖版本。"""

from __future__ import annotations

import importlib
import json
import sys

MIN_PYTHON = (3, 11)
EXPECTED_HTTPX_VERSION = "0.28.1"


def runtime_status() -> dict[str, object]:
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
    return {
        "healthy": not issues,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "httpx": httpx_version or None,
        "issues": issues,
    }


def main() -> int:
    status = runtime_status()
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    return 0 if status["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
