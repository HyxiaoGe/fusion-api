"""只读检查 LiteLLM 模型成本表自动同步状态。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol

import httpx

LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_PROXY_URL_ENV = "LITELLM_PROXY_URL"
LITELLM_MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
DEFAULT_LITELLM_BASE_URL = "http://localhost:4000"
DEFAULT_EXPECTED_INTERVAL_HOURS = 6.0
DEFAULT_MAX_STALENESS_HOURS = 12.0


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SyncStatus:
    scheduled: bool
    interval_hours: float | None
    last_run: str | None
    next_run: str | None
    source: str | None
    fallback_reason: str | None


@dataclass(frozen=True)
class StatusIssue:
    code: str
    message: str


@dataclass(frozen=True)
class StatusReport:
    healthy: bool
    checked_at: str
    status: SyncStatus
    issues: list[StatusIssue]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_interval(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _evaluate_timestamp(
    *,
    field_name: str,
    value: str | None,
    issues: list[StatusIssue],
) -> datetime | None:
    if value is None:
        issues.append(StatusIssue(code=f"missing_{field_name}", message=f"{field_name} 缺失"))
        return None
    try:
        return _parse_timestamp(value)
    except (TypeError, ValueError):
        issues.append(StatusIssue(code=f"invalid_{field_name}", message=f"{field_name} 不是有效时间"))
        return None


def evaluate_status(
    payload: Mapping[str, Any],
    *,
    expected_interval_hours: float,
    max_staleness_hours: float,
    now: datetime | None = None,
) -> StatusReport:
    """校验调度、周期和时间新鲜度。"""
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    interval_hours = _parse_interval(payload.get("interval_hours"))
    status = SyncStatus(
        scheduled=payload.get("scheduled") is True,
        interval_hours=interval_hours,
        last_run=payload.get("last_run") if isinstance(payload.get("last_run"), str) else None,
        next_run=payload.get("next_run") if isinstance(payload.get("next_run"), str) else None,
        source=payload.get("source") if isinstance(payload.get("source"), str) else None,
        fallback_reason=payload.get("fallback_reason") if isinstance(payload.get("fallback_reason"), str) else None,
    )
    issues: list[StatusIssue] = []

    if not status.scheduled:
        issues.append(StatusIssue(code="not_scheduled", message="成本表自动同步未调度"))
    if interval_hours is None:
        issues.append(StatusIssue(code="invalid_interval", message="interval_hours 缺失或不是数字"))
    elif interval_hours != expected_interval_hours:
        issues.append(
            StatusIssue(
                code="unexpected_interval",
                message=f"同步周期为 {interval_hours:g} 小时，期望 {expected_interval_hours:g} 小时",
            )
        )

    last_run = _evaluate_timestamp(field_name="last_run", value=status.last_run, issues=issues)
    next_run = _evaluate_timestamp(field_name="next_run", value=status.next_run, issues=issues)
    if last_run is not None:
        staleness_hours = (checked_at - last_run).total_seconds() / 3600
        if staleness_hours < -0.1:
            issues.append(StatusIssue(code="future_last_run", message="最后同步时间位于未来"))
        elif staleness_hours > max_staleness_hours:
            issues.append(
                StatusIssue(
                    code="stale_last_run",
                    message=f"最后同步距今 {staleness_hours:.2f} 小时，超过 {max_staleness_hours:g} 小时",
                )
            )
    if next_run is not None:
        next_delay_hours = (next_run - checked_at).total_seconds() / 3600
        if next_delay_hours < 0:
            issues.append(StatusIssue(code="overdue_next_run", message="下一次同步时间已经逾期"))
        elif next_delay_hours > expected_interval_hours + 1:
            issues.append(StatusIssue(code="next_run_too_far", message="下一次同步时间超出预期周期"))
    if status.fallback_reason or status.source == "bundled":
        issues.append(StatusIssue(code="cost_map_fallback", message="成本表刷新报告了 bundled fallback"))

    return StatusReport(
        healthy=not issues,
        checked_at=checked_at.isoformat(),
        status=status,
        issues=issues,
    )


def fetch_status(
    *,
    base_url: str,
    master_key: str,
    timeout_seconds: float,
    client: HttpClient = httpx,
) -> Mapping[str, Any]:
    """只通过 GET 读取 LiteLLM 成本表同步状态。"""
    response = client.get(
        f"{base_url.rstrip('/')}/schedule/model_cost_map_reload/status",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("LiteLLM 状态响应不是 JSON 对象")
    return payload


def serialize_report(
    report: StatusReport,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """序列化只读报告，并仅保留明确允许的非敏感上下文。"""
    safe_context = {
        key: context[key]
        for key in ("litellm_base_url", "expected_interval_hours", "max_staleness_hours")
        if key in context
    }
    return {
        "healthy": report.healthy,
        "checked_at": report.checked_at,
        "context": safe_context,
        "status": asdict(report.status),
        "issues": [asdict(issue) for issue in report.issues],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="LiteLLM Proxy 地址")
    parser.add_argument(
        "--expected-interval-hours",
        type=float,
        default=DEFAULT_EXPECTED_INTERVAL_HOURS,
        help="期望的自动同步周期，默认 6 小时",
    )
    parser.add_argument(
        "--max-staleness-hours",
        type=float,
        default=DEFAULT_MAX_STALENESS_HOURS,
        help="最后成功同步允许的最大陈旧时间，默认 12 小时",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="HTTP 超时秒数")
    return parser


def _print_error(code: str, error_type: str) -> None:
    print(
        json.dumps(
            {
                "healthy": False,
                "error": {"code": code, "type": error_type},
            },
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base_url = (
        args.base_url or os.getenv(LITELLM_BASE_URL_ENV) or os.getenv(LITELLM_PROXY_URL_ENV) or DEFAULT_LITELLM_BASE_URL
    )
    master_key = os.getenv(LITELLM_MASTER_KEY_ENV)
    if not master_key:
        _print_error("missing_master_key", "ConfigurationError")
        return 2

    try:
        payload = fetch_status(
            base_url=base_url,
            master_key=master_key,
            timeout_seconds=args.timeout_seconds,
        )
        report = evaluate_status(
            payload,
            expected_interval_hours=args.expected_interval_hours,
            max_staleness_hours=args.max_staleness_hours,
        )
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        _print_error("status_check_failed", type(exc).__name__)
        return 1

    output = serialize_report(
        report,
        context={
            "litellm_base_url": base_url,
            "expected_interval_hours": args.expected_interval_hours,
            "max_staleness_hours": args.max_staleness_hours,
        },
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
