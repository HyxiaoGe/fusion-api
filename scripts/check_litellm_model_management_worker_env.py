"""校验模型准入 Worker 与 fusion-api 的非敏感部署契约。"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from typing import Mapping, Sequence

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_worker_environment(
    environment: Mapping[str, str],
    *,
    expected_token_sha256: str,
    expected_base_url: str,
    expected_governance_max_age_seconds: int,
) -> list[str]:
    """只返回稳定错误码，不输出任何凭据内容。"""
    issues: list[str] = []
    token = environment.get("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", "")
    virtual_key = environment.get("LITELLM_VIRTUAL_KEY", "")
    base_url = environment.get("FUSION_MODEL_MANAGEMENT_BASE_URL", "").rstrip("/")
    governance_max_age = environment.get("LITELLM_GOVERNANCE_MAX_AGE_SECONDS", "")

    if not SHA256_PATTERN.fullmatch(expected_token_sha256):
        issues.append("expected_token_sha256_invalid")
    elif not token:
        issues.append("worker_token_missing")
    else:
        actual_sha256 = hashlib.sha256(token.encode()).hexdigest()
        if not hmac.compare_digest(actual_sha256, expected_token_sha256):
            issues.append("worker_token_mismatch")
    if not virtual_key:
        issues.append("virtual_key_missing")
    if base_url != expected_base_url.rstrip("/"):
        issues.append("fusion_base_url_mismatch")
    if governance_max_age != str(expected_governance_max_age_seconds):
        issues.append("governance_max_age_mismatch")
    return issues


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-token-sha256", required=True)
    parser.add_argument("--expected-base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--expected-governance-max-age-seconds", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    issues = validate_worker_environment(
        os.environ,
        expected_token_sha256=args.expected_token_sha256,
        expected_base_url=args.expected_base_url,
        expected_governance_max_age_seconds=args.expected_governance_max_age_seconds,
    )
    print(json.dumps({"healthy": not issues, "issues": issues}, ensure_ascii=False, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
