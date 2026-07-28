"""基于候选发现与验收证据生成 LiteLLM 准入 dry-run 计划。

首次准入只接受 direct LiteLLM wildcard 的候选验收。Fusion 全模型验收属于
注册后的发布门禁，不能替代候选预准入证据。本脚本不发 HTTP，也没有 apply 模式。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scripts.check_litellm_candidate_preflight import candidate_contract_fingerprint

REQUIRED_CANDIDATE_CHECKS = ("text", "stream", "tool", "usage", "cost")
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
SAFE_METADATA_KEYS = (
    "display_name",
    "description",
    "provider_key",
    "provider_display",
    "capabilities",
    "pricing",
    "knowledge_cutoff",
    "recommended_for",
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _nonempty_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _contains_raw_api_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() == "api_key":
                return True
            if _contains_raw_api_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_api_key(item) for item in value)
    return False


def _metadata_reasons(candidate: Mapping[str, Any]) -> list[str]:
    metadata = candidate.get("metadata")
    if not _is_mapping(metadata):
        return ["metadata_missing"]
    reasons: list[str] = []
    for key in ("display_name", "provider_key", "provider_display"):
        if not _nonempty_string(metadata.get(key)):
            reasons.append(f"metadata_{key}_missing")
    capabilities = metadata.get("capabilities")
    if not _is_mapping(capabilities) or not capabilities:
        reasons.append("metadata_capabilities_incomplete")
    pricing = metadata.get("pricing")
    if not _pricing_complete(pricing):
        reasons.append("metadata_pricing_incomplete")
    evidence = candidate.get("metadata_evidence")
    if not _is_mapping(evidence) or not (
        evidence.get("cost_map_matched") is True or evidence.get("reviewed_override_applied") is True
    ):
        reasons.append("metadata_evidence_missing")
    if metadata.get("provider_key") != candidate.get("provider_key"):
        reasons.append("metadata_provider_mismatch")
    return reasons


def _pricing_complete(pricing: Any) -> bool:
    if not _is_mapping(pricing) or not _nonempty_string(pricing.get("unit")):
        return False
    for key in ("input", "output"):
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return False
    return True


def _registration_reasons(candidate: Mapping[str, Any]) -> list[str]:
    if _contains_raw_api_key(candidate):
        return ["raw_api_key_forbidden"]
    registration = candidate.get("registration")
    if not _is_mapping(registration):
        return ["registration_missing"]
    reasons: list[str] = []
    api_base = _nonempty_string(registration.get("api_base"))
    parsed = urlparse(api_base)
    if registration.get("endpoint_status") != "verified" or parsed.scheme != "https" or not parsed.netloc:
        reasons.append("registration_endpoint_unverified")
    api_key_env = _nonempty_string(registration.get("api_key_env"))
    if not ENV_NAME_PATTERN.fullmatch(api_key_env):
        reasons.append("registration_api_key_env_invalid")
    if not _nonempty_string(candidate.get("litellm_model")):
        reasons.append("registration_litellm_model_missing")
    return reasons


def _acceptance_reasons(
    model_id: str,
    candidate: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if summary.get("dry_run") is not False or summary.get("healthy") is not True:
        reasons.append("candidate_acceptance_not_executed")
    if summary.get("acceptance_stage") != "candidate_pre_registration":
        reasons.append("candidate_acceptance_stage_invalid")
    if summary.get("transport") != "litellm_wildcard":
        reasons.append("candidate_acceptance_transport_invalid")
    try:
        expected_fingerprint = candidate_contract_fingerprint(candidate)
    except ValueError:
        expected_fingerprint = ""
    if not expected_fingerprint or summary.get("candidate_fingerprint") != expected_fingerprint:
        reasons.append("candidate_acceptance_contract_mismatch")
    model_result = (summary.get("by_model") or {}).get(model_id)
    if not _is_mapping(model_result):
        return [*reasons, "candidate_acceptance_model_missing"]
    if not _model_result_passed(model_result):
        reasons.append("candidate_acceptance_failed")
    checks = (summary.get("candidate_checks_by_model") or {}).get(model_id)
    if not _is_mapping(checks) or any(checks.get(check) is not True for check in REQUIRED_CANDIDATE_CHECKS):
        reasons.append("candidate_acceptance_checks_incomplete")
    if _has_high_quality_risk(model_id, summary):
        reasons.append("high_quality_risk")
    mismatch_by_model = summary.get("tool_expectation_mismatch_by_model")
    mismatch = (
        mismatch_by_model.get(model_id, 0)
        if _is_mapping(mismatch_by_model)
        else summary.get("tool_expectation_mismatch_count", 0)
    )
    if mismatch:
        reasons.append("tool_expectation_mismatch")
    return reasons


def _model_result_passed(result: Mapping[str, Any]) -> bool:
    total = result.get("total")
    success = result.get("success_count")
    failure = result.get("failure_count")
    skipped = result.get("skipped_count", 0)
    return isinstance(total, int) and total > 0 and success == total and failure == 0 and skipped == 0


def _has_high_quality_risk(model_id: str, summary: Mapping[str, Any]) -> bool:
    issues = summary.get("quality_issues")
    if not isinstance(issues, list):
        return True
    return any(
        _is_mapping(issue) and issue.get("model_id") == model_id and issue.get("severity") == "high" for issue in issues
    )


def _safe_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metadata = candidate["metadata"]
    return {key: metadata[key] for key in SAFE_METADATA_KEYS if key in metadata}


def _eligible_entry(candidate: Mapping[str, Any]) -> dict[str, Any]:
    registration = candidate["registration"]
    model_id = _nonempty_string(candidate.get("model_id"))
    payload = {
        "model_name": model_id,
        "litellm_params": {
            "model": candidate["litellm_model"],
            "api_base": registration["api_base"],
            "api_key": f"os.environ/{registration['api_key_env']}",
        },
        "model_info": {
            "metadata": _safe_metadata(candidate),
        },
    }
    return {
        "model_id": model_id,
        "provider_key": candidate["provider_key"],
        "model_new_plan": {
            "mode": "dry_run",
            "endpoint": "/model/new",
            "minimum_litellm_version": "1.93.0",
            "payload": payload,
        },
    }


def _isolated_entry(candidate: Mapping[str, Any], provider_key: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "model_id": _nonempty_string(candidate.get("model_id")),
        "provider_key": provider_key,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _iter_new_candidates(candidate_report: Mapping[str, Any]):
    providers = candidate_report.get("providers")
    if not _is_mapping(providers):
        return
    for provider_key, provider_result in providers.items():
        if not _is_mapping(provider_result):
            continue
        report = provider_result.get("report")
        new_candidates = report.get("new") if _is_mapping(report) else []
        if not isinstance(new_candidates, list):
            continue
        for candidate in new_candidates:
            if _is_mapping(candidate):
                yield str(provider_key), provider_result, candidate


def _pipeline_issues(candidate_report: Mapping[str, Any]) -> list[dict[str, str]]:
    providers = candidate_report.get("providers")
    if not _is_mapping(providers) or not providers:
        return [{"provider_key": "", "code": "candidate_providers_missing"}]
    issues: list[dict[str, str]] = []
    for provider_key, provider_result in providers.items():
        if not _is_mapping(provider_result):
            issues.append({"provider_key": str(provider_key), "code": "provider_result_invalid"})
            continue
        if provider_result.get("status") != "ok":
            error = provider_result.get("error")
            code = error.get("code") if _is_mapping(error) else "provider_status_not_ok"
            issues.append({"provider_key": str(provider_key), "code": str(code)})
            continue
        report = provider_result.get("report")
        if not _is_mapping(report) or not isinstance(report.get("new"), list):
            issues.append({"provider_key": str(provider_key), "code": "provider_report_invalid"})
    return issues


def _fusion_gate(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {
            "status": "required",
            "stage": "fusion_post_registration",
            "affects_pre_registration_eligibility": False,
        }
    return {
        "status": "observed",
        "stage": "fusion_post_registration",
        "affects_pre_registration_eligibility": False,
        "model_ids": sorted((summary.get("by_model") or {}).keys()) if _is_mapping(summary.get("by_model")) else [],
    }


def build_admission_plan(
    *,
    candidate_report: Mapping[str, Any],
    candidate_acceptance_summary: Mapping[str, Any],
    fusion_acceptance_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """生成不执行任何写操作的候选准入计划。"""
    eligible: list[dict[str, Any]] = []
    isolated: list[dict[str, Any]] = []
    pipeline_issues = _pipeline_issues(candidate_report)
    for provider_key, provider_result, candidate in _iter_new_candidates(candidate_report):
        reasons: list[str] = []
        if provider_result.get("status") != "ok":
            reasons.append("provider_status_not_ok")
        if candidate.get("isolation_status") != "candidate":
            reasons.append("candidate_category_not_new")
        reasons.extend(_metadata_reasons(candidate))
        reasons.extend(_registration_reasons(candidate))
        model_id = _nonempty_string(candidate.get("model_id"))
        reasons.extend(_acceptance_reasons(model_id, candidate, candidate_acceptance_summary))
        if reasons:
            isolated.append(_isolated_entry(candidate, provider_key, reasons))
        else:
            eligible.append(_eligible_entry(candidate))
    return {
        "status": "failed" if pipeline_issues else "complete",
        "mode": "dry_run",
        "writes_performed": False,
        "http_requests_performed": False,
        "eligible": eligible,
        "isolated": isolated,
        "pipeline_issues": pipeline_issues,
        "candidates_evaluated": len(eligible) + len(isolated),
        "allowlist_plan": {
            "add": sorted(item["model_id"] for item in eligible),
            "remove": [],
        },
        "post_registration_fusion_gate": _fusion_gate(fusion_acceptance_summary),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 文件 {path}") from exc
    if not _is_mapping(payload):
        raise ValueError(f"JSON 文件顶层必须是对象: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 LiteLLM 候选准入 dry-run 计划")
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-acceptance-summary", type=Path, required=True)
    parser.add_argument("--fusion-acceptance-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    plan = build_admission_plan(
        candidate_report=_read_json(args.candidate_report),
        candidate_acceptance_summary=_read_json(args.candidate_acceptance_summary),
        fusion_acceptance_summary=_read_json(args.fusion_acceptance_summary)
        if args.fusion_acceptance_summary
        else None,
    )
    _write_json_atomic(args.output, plan)
    return 0 if plan["status"] == "complete" and not plan["isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
