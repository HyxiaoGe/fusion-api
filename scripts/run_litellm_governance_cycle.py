"""执行一次只读 LiteLLM 治理周期并保存可审计状态产物。

周期只执行 GET 和本地原子文件写入，不包含收费请求、模型注册、allowlist
修改或 LiteLLM 管理 API 写操作。真实候选验收由单独的显式 ``--apply``
命令完成，验收摘要按候选契约指纹回流到下一次治理周期。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from scripts.check_litellm_candidate_preflight import candidate_contract_fingerprint
from scripts.enrich_litellm_model_candidates import enrich_candidate_report
from scripts.fetch_litellm_cost_map import (
    DEFAULT_COST_MAP_URL,
    build_status,
    fetch_cost_map,
    write_json_atomic,
)
from scripts.orchestrate_litellm_model_candidates import coordinate_candidates
from scripts.plan_litellm_candidate_admission import (
    build_admission_plan,
    candidate_static_gate_reasons,
)

CostMapFetcher = Callable[..., tuple[Mapping[str, Any], Any]]
CandidateCoordinator = Callable[..., dict[str, Any]]
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z$")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _is_mapping(payload):
        raise ValueError(f"JSON 文件顶层必须是对象: {path}")
    return payload


def _single_candidate_report(
    report: Mapping[str, Any],
    *,
    provider_key: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    provider_result = report["providers"][provider_key]
    single = {
        "mode": report.get("mode"),
        "providers": {
            provider_key: {
                **provider_result,
                "report": {
                    **provider_result["report"],
                    "new": [dict(candidate)],
                },
            }
        },
    }
    return single


def _candidate_records(report: Mapping[str, Any]):
    providers = report.get("providers")
    if not _is_mapping(providers):
        return
    for provider_key, provider_result in providers.items():
        if not _is_mapping(provider_result) or provider_result.get("status") != "ok":
            continue
        provider_report = provider_result.get("report")
        candidates = provider_report.get("new") if _is_mapping(provider_report) else None
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if _is_mapping(candidate):
                yield str(provider_key), candidate


def _retirement_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    providers = report.get("providers")
    if not _is_mapping(providers):
        return records
    for provider_key, provider_result in providers.items():
        provider_report = provider_result.get("report") if _is_mapping(provider_result) else None
        removed = provider_report.get("removed") if _is_mapping(provider_report) else None
        if not isinstance(removed, list):
            continue
        for candidate in removed:
            if not _is_mapping(candidate):
                continue
            records.append(
                {
                    "provider_key": str(provider_key),
                    "model_id": str(candidate.get("model_id") or ""),
                    "state": "retirement_review",
                    "automatic_deletion_allowed": False,
                    "candidate_snapshot_sha256": _json_sha256(candidate),
                    "candidate": dict(candidate),
                }
            )
    records.sort(key=lambda item: (item["provider_key"], item["model_id"]))
    return records


def _acceptance_path(acceptance_dir: Path | None, fingerprint: str) -> Path | None:
    if acceptance_dir is None or not fingerprint:
        return None
    return acceptance_dir / f"{fingerprint}.json"


def _load_acceptance(path: Path | None) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        return _read_object(path), None
    except (OSError, json.JSONDecodeError, ValueError):
        return None, "candidate_acceptance_invalid"


def _build_candidate_queue(
    *,
    enriched_report: Mapping[str, Any],
    acceptance_dir: Path | None,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    queue: list[dict[str, Any]] = []
    admission_plans: dict[str, Any] = {}
    cycle_issues: list[dict[str, str]] = []
    preflight_contract = enriched_report.get("candidate_preflight")
    preflight_ready = _is_mapping(preflight_contract) and preflight_contract.get("status") == "ready"
    preflight_reasons = (
        list(preflight_contract.get("reasons") or [])
        if _is_mapping(preflight_contract)
        else ["candidate_preflight_contract_missing"]
    )
    for provider_key, candidate in _candidate_records(enriched_report):
        snapshot_sha256 = _json_sha256(candidate)
        static_reasons = candidate_static_gate_reasons(candidate, provider_key=provider_key)
        fingerprint = ""
        if not static_reasons:
            try:
                fingerprint = candidate_contract_fingerprint(candidate)
            except ValueError:
                static_reasons.append("candidate_contract_invalid")
        if not preflight_ready:
            static_reasons.extend(preflight_reasons or ["candidate_preflight_route_unready"])

        acceptance, acceptance_error = _load_acceptance(_acceptance_path(acceptance_dir, fingerprint))
        if acceptance_error:
            cycle_issues.append(
                {
                    "provider_key": provider_key,
                    "model_id": str(candidate.get("model_id") or ""),
                    "code": acceptance_error,
                }
            )

        state = "quarantined"
        reasons = list(static_reasons)
        if not reasons and acceptance is None and acceptance_error is None:
            state = "preflight_required"
        elif not reasons and acceptance is not None:
            plan = build_admission_plan(
                candidate_report=_single_candidate_report(
                    enriched_report,
                    provider_key=provider_key,
                    candidate=candidate,
                ),
                candidate_acceptance_summary=acceptance,
            )
            plan = {**plan, "run_id": run_id}
            admission_plans[fingerprint] = plan
            if plan["status"] == "complete" and plan["eligible"]:
                state = "admission_ready"
            else:
                reasons = list(plan["isolated"][0]["reasons"]) if plan["isolated"] else ["admission_plan_failed"]
        elif acceptance_error:
            reasons.append(acceptance_error)

        queue.append(
            {
                "provider_key": provider_key,
                "model_id": str(candidate.get("model_id") or ""),
                "state": state,
                "reasons": list(dict.fromkeys(reasons)),
                "candidate_snapshot_sha256": snapshot_sha256,
                "candidate_fingerprint": fingerprint or None,
                "acceptance_file": f"{fingerprint}.json" if fingerprint else None,
                "candidate": dict(candidate),
            }
        )
    queue.sort(key=lambda item: (item["provider_key"], item["model_id"]))
    return queue, admission_plans, cycle_issues


def _manifest(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": {
            name: {
                "sha256": _json_sha256(payload),
                "record_count": len(payload) if isinstance(payload, (list, dict)) else None,
            }
            for name, payload in artifacts.items()
        },
    }


def write_pointer_monotonic(path: Path, pointer: Mapping[str, Any]) -> bool:
    """在进程锁内只推进 run_id，避免慢周期覆盖较新的指针。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".governance-pointer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            next_run_id = str(pointer.get("run_id") or "")
            if not RUN_ID_PATTERN.fullmatch(next_run_id):
                raise ValueError("治理指针包含无效 run_id")
            if path.exists():
                current = _read_object(path)
                current_run_id = str(current.get("run_id") or "")
                if not RUN_ID_PATTERN.fullmatch(current_run_id):
                    raise ValueError("现有治理指针包含无效 run_id")
                if current_run_id >= next_run_id:
                    return False
            write_json_atomic(path, pointer)
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _provider_issues(candidate_report: Mapping[str, Any]) -> list[dict[str, str]]:
    providers = candidate_report.get("providers")
    if not _is_mapping(providers) or not providers:
        return [{"provider_key": "", "code": "candidate_providers_missing"}]
    issues: list[dict[str, str]] = []
    for provider_key, result in providers.items():
        if not _is_mapping(result) or result.get("status") != "ok":
            error = result.get("error") if _is_mapping(result) else None
            code = error.get("code") if _is_mapping(error) else "provider_status_not_ok"
            issues.append({"provider_key": str(provider_key), "code": str(code)})
    return issues


def run_governance_cycle(
    *,
    registry: Mapping[str, Any],
    environ: Mapping[str, str],
    output_dir: Path,
    overrides: Mapping[str, Any] | None = None,
    acceptance_dir: Path | None = None,
    cost_map_url: str = DEFAULT_COST_MAP_URL,
    timeout_seconds: float = 30.0,
    now: datetime | None = None,
    cost_map_fetcher: CostMapFetcher = fetch_cost_map,
    candidate_coordinator: CandidateCoordinator = coordinate_candidates,
) -> dict[str, Any]:
    """执行一次周期；失败周期保留 run 证据但不推进 latest-success。"""
    started_at = (now or datetime.now(UTC)).astimezone(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    runs_dir = output_dir / "runs"
    staging_dir = runs_dir / f".{run_id}.tmp"
    final_dir = runs_dir / run_id
    if staging_dir.exists() or final_dir.exists():
        raise ValueError(f"治理周期 run_id 冲突: {run_id}")
    staging_dir.mkdir(parents=True)

    try:
        cost_map, response = cost_map_fetcher(url=cost_map_url, timeout_seconds=timeout_seconds)
        cost_status = build_status(payload=cost_map, response=response, source_url=cost_map_url)
        candidate_report = candidate_coordinator(
            registry=registry,
            environ=environ,
            timeout_seconds=timeout_seconds,
        )
        enriched_report = enrich_candidate_report(
            candidate_report=candidate_report,
            registry=registry,
            cost_map=cost_map,
            cost_map_status=cost_status,
            overrides=overrides,
        )
        queue, admission_plans, cycle_issues = _build_candidate_queue(
            enriched_report=enriched_report,
            acceptance_dir=acceptance_dir,
            run_id=run_id,
        )
        retirement_review = _retirement_records(enriched_report)
        provider_issues = _provider_issues(candidate_report)
        issues = [*provider_issues, *cycle_issues]
        status = "success" if not issues else "failed"
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "status": status,
            "external_writes_performed": False,
            "paid_requests_performed": False,
            "local_artifacts_written": True,
            "providers_total": candidate_report.get("summary", {}).get("providers_total", 0),
            "providers_failed": candidate_report.get("summary", {}).get("providers_failed", 0),
            "candidates_total": len(queue),
            "quarantined": sum(item["state"] == "quarantined" for item in queue),
            "preflight_required": sum(item["state"] == "preflight_required" for item in queue),
            "admission_ready": sum(item["state"] == "admission_ready" for item in queue),
            "retirement_review": len(retirement_review),
            "candidate_preflight_status": (enriched_report.get("candidate_preflight") or {}).get("status"),
            "issues": issues,
        }
        artifacts = {
            "cost-map.json": cost_map,
            "cost-map-status.json": cost_status,
            "candidate-report.json": candidate_report,
            "enriched-candidate-report.json": enriched_report,
            "candidate-queue.json": queue,
            "admission-plans.json": admission_plans,
            "retirement-review.json": retirement_review,
            "cycle-summary.json": summary,
        }
        manifest = _manifest(artifacts)
        for name, payload in artifacts.items():
            write_json_atomic(staging_dir / name, payload)
        write_json_atomic(staging_dir / "manifest.json", manifest)
        os.replace(staging_dir, final_dir)

        pointer = {
            "schema_version": 1,
            "run_id": run_id,
            "run_path": f"runs/{run_id}",
            "summary_sha256": _json_sha256(summary),
            "manifest_sha256": _json_sha256(manifest),
        }
        if status == "success":
            write_pointer_monotonic(output_dir / "latest-success.json", pointer)
        else:
            write_pointer_monotonic(output_dir / "latest-failure.json", pointer)
        return summary
    except Exception as exc:
        # 网络、上游协议或配置失败也必须留下可审计 run；只记录稳定错误码和
        # 异常类型，不落原始响应、URL query 或凭证。
        if final_dir.exists():
            # run 已原子发布后只可能是 pointer 写入失败；不得用同一 run_id
            # 覆盖已发布证据，交给调用方报告本地存储故障。
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "status": "failed",
            "external_writes_performed": False,
            "paid_requests_performed": False,
            "local_artifacts_written": True,
            "providers_total": 0,
            "providers_failed": 0,
            "candidates_total": 0,
            "quarantined": 0,
            "preflight_required": 0,
            "admission_ready": 0,
            "retirement_review": 0,
            "candidate_preflight_status": None,
            "issues": [
                {
                    "provider_key": "",
                    "code": "governance_cycle_exception",
                    "type": type(exc).__name__,
                }
            ],
        }
        manifest = _manifest({"cycle-summary.json": summary})
        write_json_atomic(staging_dir / "cycle-summary.json", summary)
        write_json_atomic(staging_dir / "manifest.json", manifest)
        os.replace(staging_dir, final_dir)
        pointer = {
            "schema_version": 1,
            "run_id": run_id,
            "run_path": f"runs/{run_id}",
            "summary_sha256": _json_sha256(summary),
            "manifest_sha256": _json_sha256(manifest),
        }
        write_pointer_monotonic(output_dir / "latest-failure.json", pointer)
        return summary


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--acceptance-dir", type=Path)
    parser.add_argument("--cost-map-url", default=DEFAULT_COST_MAP_URL)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        summary = run_governance_cycle(
            registry=_read_object(args.registry),
            environ=os.environ,
            output_dir=args.output_dir,
            overrides=_read_object(args.overrides) if args.overrides else None,
            acceptance_dir=args.acceptance_dir,
            cost_map_url=args.cost_map_url,
            timeout_seconds=args.timeout_seconds,
        )
    except (httpx.HTTPError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": {"code": "governance_cycle_failed", "type": type(exc).__name__}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
