"""幂等确保 LiteLLM 运行时成本表按期刷新，并输出脱敏状态。

默认只读；只有显式 ``--apply`` 且当前未调度或周期不符时，才调用一次
LiteLLM schedule API。陈旧、fallback 或响应异常只告警，不通过反复重排掩盖。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping, Protocol

import httpx

from scripts.check_litellm_cost_map_sync_status import (
    DEFAULT_EXPECTED_INTERVAL_HOURS,
    DEFAULT_LITELLM_BASE_URL,
    DEFAULT_MAX_STALENESS_HOURS,
    LITELLM_BASE_URL_ENV,
    LITELLM_MASTER_KEY_ENV,
    LITELLM_PROXY_URL_ENV,
    evaluate_status,
    fetch_status,
    serialize_report,
)


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


def _repairable(payload: Mapping[str, Any], *, expected_interval_hours: float) -> bool:
    interval = payload.get("interval_hours")
    return payload.get("scheduled") is not True or (
        isinstance(interval, (int, float))
        and not isinstance(interval, bool)
        and float(interval) != expected_interval_hours
    )


def ensure_sync(
    *,
    base_url: str,
    master_key: str,
    expected_interval_hours: float = DEFAULT_EXPECTED_INTERVAL_HOURS,
    max_staleness_hours: float = DEFAULT_MAX_STALENESS_HOURS,
    timeout_seconds: float = 10.0,
    apply: bool = False,
    client: HttpClient = httpx,
) -> dict[str, Any]:
    before_payload = fetch_status(
        base_url=base_url,
        master_key=master_key,
        timeout_seconds=timeout_seconds,
        client=client,
    )
    before = evaluate_status(
        before_payload,
        expected_interval_hours=expected_interval_hours,
        max_staleness_hours=max_staleness_hours,
    )
    action = "none"
    after = before
    if _repairable(before_payload, expected_interval_hours=expected_interval_hours):
        action = "schedule_required"
        if apply:
            response = client.post(
                f"{base_url.rstrip('/')}/schedule/model_cost_map_reload",
                params={"hours": expected_interval_hours},
                headers={"Authorization": f"Bearer {master_key}"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            action = "scheduled"
            after_payload = fetch_status(
                base_url=base_url,
                master_key=master_key,
                timeout_seconds=timeout_seconds,
                client=client,
            )
            after = evaluate_status(
                after_payload,
                expected_interval_hours=expected_interval_hours,
                max_staleness_hours=max_staleness_hours,
            )
    return {
        "mode": "apply" if apply else "dry_run",
        "action": action,
        "healthy": after.healthy,
        "before": serialize_report(
            before,
            context={
                "litellm_base_url": base_url,
                "expected_interval_hours": expected_interval_hours,
                "max_staleness_hours": max_staleness_hours,
            },
        ),
        "after": serialize_report(
            after,
            context={
                "litellm_base_url": base_url,
                "expected_interval_hours": expected_interval_hours,
                "max_staleness_hours": max_staleness_hours,
            },
        ),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--expected-interval-hours", type=float, default=DEFAULT_EXPECTED_INTERVAL_HOURS)
    parser.add_argument("--max-staleness-hours", type=float, default=DEFAULT_MAX_STALENESS_HOURS)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    base_url = (
        args.base_url or os.getenv(LITELLM_BASE_URL_ENV) or os.getenv(LITELLM_PROXY_URL_ENV) or DEFAULT_LITELLM_BASE_URL
    )
    master_key = os.getenv(LITELLM_MASTER_KEY_ENV)
    if not master_key:
        print(json.dumps({"healthy": False, "error": {"code": "missing_master_key"}}))
        return 2
    try:
        result = ensure_sync(
            base_url=base_url,
            master_key=master_key,
            expected_interval_hours=args.expected_interval_hours,
            max_staleness_hours=args.max_staleness_hours,
            timeout_seconds=args.timeout_seconds,
            apply=args.apply,
        )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        print(
            json.dumps({"healthy": False, "error": {"code": "cost_map_sync_ensure_failed", "type": type(exc).__name__}})
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
