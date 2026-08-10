"""受控执行单个 LiteLLM candidate admission plan。

默认只返回 dry-run transaction；只有显式 apply 且 run/model/fingerprint 三重确认
全部匹配时才进入带 CAS 和补偿的写入状态机。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode

import httpx

from scripts.audit_litellm_model_catalog import audit_catalog
from scripts.check_litellm_candidate_preflight import candidate_contract_fingerprint
from scripts.plan_litellm_candidate_admission import build_eligible_entry

TRANSACTION_SCHEMA_VERSION = 1
FUSION_READBACK_ATTEMPTS = 35
FUSION_READBACK_INTERVAL_SECONDS = 2.0
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


class TransactionFailure(RuntimeError):
    def __init__(self, code: str, phase: str):
        super().__init__(code)
        self.code = code
        self.phase = phase


class VerifiedPlanError(ValueError):
    """治理产物信任链验证失败，仅暴露稳定错误码。"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json_artifact(path: Path, *, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifiedPlanError(code) from exc


def _artifact_sha256(manifest: Mapping[str, Any], name: str) -> str:
    artifacts = manifest.get("artifacts")
    metadata = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    digest = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise VerifiedPlanError(f"{name.removesuffix('.json').replace('-', '_')}_manifest_missing")
    return digest


def load_verified_admission_plan(
    *,
    governance_root: Path,
    candidate_fingerprint: str,
) -> Mapping[str, Any]:
    """从 latest-success 的完整哈希链加载并重建单候选准入计划。"""
    if len(candidate_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_fingerprint.lower()
    ):
        raise VerifiedPlanError("candidate_fingerprint_invalid")
    try:
        root = governance_root.resolve(strict=True)
    except OSError as exc:
        raise VerifiedPlanError("governance_root_invalid") from exc
    pointer = _read_json_artifact(root / "latest-success.json", code="latest_success_invalid")
    if not isinstance(pointer, Mapping):
        raise VerifiedPlanError("latest_success_invalid")
    run_id = pointer.get("run_id")
    run_path = pointer.get("run_path")
    if not isinstance(run_id, str) or not run_id or not isinstance(run_path, str):
        raise VerifiedPlanError("latest_success_invalid")
    relative_run_path = Path(run_path)
    if relative_run_path.is_absolute() or relative_run_path.parts != ("runs", run_id):
        raise VerifiedPlanError("run_path_invalid")
    try:
        run_dir = (root / relative_run_path).resolve(strict=True)
        run_dir.relative_to(root)
    except (OSError, ValueError) as exc:
        raise VerifiedPlanError("run_path_invalid") from exc

    manifest = _read_json_artifact(run_dir / "manifest.json", code="manifest_invalid")
    if not isinstance(manifest, Mapping):
        raise VerifiedPlanError("manifest_invalid")
    if pointer.get("manifest_sha256") != _canonical_sha256(manifest):
        raise VerifiedPlanError("manifest_sha256_mismatch")

    queue = _read_json_artifact(run_dir / "candidate-queue.json", code="candidate_queue_invalid")
    plans = _read_json_artifact(run_dir / "admission-plans.json", code="admission_plans_invalid")
    if not isinstance(queue, list):
        raise VerifiedPlanError("candidate_queue_invalid")
    if not isinstance(plans, Mapping):
        raise VerifiedPlanError("admission_plans_invalid")
    if _canonical_sha256(queue) != _artifact_sha256(manifest, "candidate-queue.json"):
        raise VerifiedPlanError("candidate_queue_sha256_mismatch")
    if _canonical_sha256(plans) != _artifact_sha256(manifest, "admission-plans.json"):
        raise VerifiedPlanError("admission_plans_sha256_mismatch")

    records = [
        item
        for item in queue
        if isinstance(item, Mapping) and item.get("candidate_fingerprint") == candidate_fingerprint
    ]
    if len(records) != 1:
        raise VerifiedPlanError("candidate_queue_record_invalid")
    record = records[0]
    if record.get("state") != "admission_ready":
        raise VerifiedPlanError("candidate_not_admission_ready")
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping):
        raise VerifiedPlanError("candidate_contract_invalid")
    if record.get("candidate_snapshot_sha256") != _canonical_sha256(candidate):
        raise VerifiedPlanError("candidate_snapshot_sha256_mismatch")
    try:
        recomputed_fingerprint = candidate_contract_fingerprint(candidate)
        rebuilt_entry = build_eligible_entry(candidate)
    except (KeyError, TypeError, ValueError) as exc:
        raise VerifiedPlanError("candidate_contract_invalid") from exc
    if recomputed_fingerprint != candidate_fingerprint:
        raise VerifiedPlanError("candidate_fingerprint_mismatch")

    plan = plans.get(candidate_fingerprint)
    if not isinstance(plan, Mapping):
        raise VerifiedPlanError("admission_plan_missing")
    if plan.get("run_id") != run_id:
        raise VerifiedPlanError("admission_plan_run_id_mismatch")
    if plan.get("eligible") != [rebuilt_entry]:
        raise VerifiedPlanError("eligible_entry_mismatch")
    expected_plan = {
        "run_id": run_id,
        "status": "complete",
        "mode": "dry_run",
        "writes_performed": False,
        "http_requests_performed": False,
        "eligible": [rebuilt_entry],
        "isolated": [],
        "pipeline_issues": [],
        "candidates_evaluated": 1,
        "allowlist_plan": {
            "add": [rebuilt_entry["model_id"]],
            "remove": [],
        },
        "post_registration_fusion_gate": {
            "status": "required",
            "stage": "fusion_post_registration",
            "affects_pre_registration_eligibility": False,
        },
    }
    if dict(plan) != expected_plan:
        raise VerifiedPlanError("admission_plan_mismatch")
    try:
        _eligible_candidate(plan)
    except ValueError as exc:
        raise VerifiedPlanError("admission_plan_invalid") from exc
    return plan


def _eligible_candidate(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    eligible = plan.get("eligible")
    if not isinstance(eligible, list) or len(eligible) != 1 or not isinstance(eligible[0], Mapping):
        raise ValueError("admission plan 必须只包含一个 eligible candidate")
    candidate = eligible[0]
    model_id = candidate.get("model_id")
    model_new_plan = candidate.get("model_new_plan")
    if not isinstance(model_id, str) or not model_id or not isinstance(model_new_plan, Mapping):
        raise ValueError("eligible candidate 缺少 model_id 或 model_new_plan")
    if model_new_plan.get("mode") != "dry_run" or model_new_plan.get("endpoint") != "/model/new":
        raise ValueError("model_new_plan 不是受控 dry-run /model/new 计划")
    payload = model_new_plan.get("payload")
    if not isinstance(payload, Mapping) or payload.get("model_name") != model_id:
        raise ValueError("model_new_plan payload 与候选模型不一致")
    params = payload.get("litellm_params")
    if not isinstance(params, Mapping) or not isinstance(params.get("model"), str) or not params["model"]:
        raise ValueError("model_new_plan 缺少 LiteLLM 模型标识")
    api_key_reference = params.get("api_key")
    if not isinstance(api_key_reference, str) or not api_key_reference.startswith("os.environ/"):
        raise ValueError("model_new_plan 只能引用环境变量中的 API key")
    if not ENV_NAME_PATTERN.fullmatch(api_key_reference.removeprefix("os.environ/")):
        raise ValueError("model_new_plan API key 环境变量名无效")
    model_info = payload.get("model_info")
    metadata = model_info.get("metadata") if isinstance(model_info, Mapping) else None
    governance = metadata.get("governance") if isinstance(metadata, Mapping) else None
    fingerprint = governance.get("candidate_fingerprint") if isinstance(governance, Mapping) else None
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint.lower())
    ):
        raise ValueError("eligible candidate 缺少有效的候选指纹")
    allowlist = plan.get("allowlist_plan")
    if (
        not isinstance(allowlist, Mapping)
        or allowlist.get("add") != [model_id]
        or allowlist.get("remove") not in ([], None)
    ):
        raise ValueError("allowlist_plan 与单模型候选不一致")
    return candidate


def candidate_fingerprint(candidate: Mapping[str, Any]) -> str:
    payload = candidate["model_new_plan"]["payload"]
    metadata = (payload.get("model_info") or {}).get("metadata") or {}
    governance = metadata.get("governance") or {}
    return str(governance["candidate_fingerprint"])


def _result_base(plan: Mapping[str, Any], candidate: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "run_id": str(plan.get("run_id") or ""),
        "mode": mode,
        "model_id": candidate["model_id"],
        "candidate_fingerprint": candidate_fingerprint(candidate),
        "status": "planned",
        "phase": "authorization",
        "completed_phases": [],
        "writes_performed": False,
        "owned_model_uuid": "",
        "before_state": None,
        "error": None,
        "compensation": {
            "attempted": False,
            "key_restored": False,
            "model_deleted": False,
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


def _authorization_error(
    *,
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_run_id: str,
    confirm_model_id: str,
    confirm_fingerprint: str,
) -> str | None:
    plan_run_id = str(plan.get("run_id") or "")
    if not plan_run_id:
        return "run_id_missing"
    if expected_run_id != plan_run_id:
        return "run_id_mismatch"
    if confirm_model_id != candidate["model_id"]:
        return "model_id_mismatch"
    if confirm_fingerprint != candidate_fingerprint(candidate):
        return "fingerprint_mismatch"
    return None


def _headers(master_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}


def _request_json(response: Any, code: str, phase: str) -> Mapping[str, Any]:
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise TransactionFailure(code, phase) from exc
    if not isinstance(payload, Mapping):
        raise TransactionFailure(code, phase)
    return payload


def _fetch_model_info(client: HttpClient, base_url: str, master_key: str, *, code: str, phase: str) -> list[dict]:
    try:
        response = client.get(
            f"{base_url.rstrip('/')}/model/info",
            headers=_headers(master_key),
            timeout=20.0,
        )
    except Exception as exc:
        raise TransactionFailure(code, phase) from exc
    payload = _request_json(response, code, phase)
    data = payload.get("data")
    if not isinstance(data, list):
        raise TransactionFailure(code, phase)
    return [item for item in data if isinstance(item, dict)]


def _fetch_key_models(
    client: HttpClient,
    base_url: str,
    master_key: str,
    virtual_key: str,
    *,
    code: str,
    phase: str,
) -> list[str]:
    try:
        response = client.get(
            f"{base_url.rstrip('/')}/key/info?{urlencode({'key': virtual_key})}",
            headers=_headers(master_key),
            timeout=20.0,
        )
    except Exception as exc:
        raise TransactionFailure(code, phase) from exc
    payload = _request_json(response, code, phase)
    info = payload.get("info") if isinstance(payload.get("info"), Mapping) else payload
    models = info.get("models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise TransactionFailure(code, phase)
    return list(models)


def _entry_identity(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    params = entry.get("litellm_params") or {}
    info = entry.get("model_info") or {}
    return (
        str(entry.get("model_name") or ""),
        str(params.get("model") or "") if isinstance(params, Mapping) else "",
        str(info.get("id") or "") if isinstance(info, Mapping) else "",
    )


def _state_fingerprint(entries: Sequence[Mapping[str, Any]], key_models: Sequence[str]) -> str:
    catalog = []
    for entry in entries:
        params = entry.get("litellm_params") or {}
        info = entry.get("model_info") or {}
        catalog.append(
            {
                "model_name": entry.get("model_name"),
                "litellm_model": params.get("model") if isinstance(params, Mapping) else None,
                "api_base": params.get("api_base") if isinstance(params, Mapping) else None,
                "id": info.get("id") if isinstance(info, Mapping) else None,
                "metadata": info.get("metadata") if isinstance(info, Mapping) else None,
                "input_cost_per_token": info.get("input_cost_per_token") if isinstance(info, Mapping) else None,
                "output_cost_per_token": info.get("output_cost_per_token") if isinstance(info, Mapping) else None,
                "cache_read_input_token_cost": (
                    info.get("cache_read_input_token_cost") if isinstance(info, Mapping) else None
                ),
                "max_input_tokens": info.get("max_input_tokens") if isinstance(info, Mapping) else None,
            }
        )
    catalog.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    state = {
        "models": catalog,
        "key_models": list(key_models),
    }
    return _canonical_sha256(state)


def _find_candidate_entry(
    entries: Sequence[Mapping[str, Any]],
    model_id: str,
    underlying_model: str,
) -> Mapping[str, Any] | None:
    for entry in entries:
        alias, underlying, _ = _entry_identity(entry)
        if alias == model_id and underlying == underlying_model:
            return entry
    return None


def _created_uuid(payload: Mapping[str, Any]) -> str:
    info = payload.get("model_info") or {}
    if isinstance(info, Mapping) and info.get("id"):
        return str(info["id"])
    if payload.get("id"):
        return str(payload["id"])
    return ""


def _resolved_model_payload(candidate: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    payload = copy.deepcopy(candidate["model_new_plan"]["payload"])
    params = payload.get("litellm_params")
    if not isinstance(params, dict):
        raise TransactionFailure("model_payload_invalid", "model_new")
    reference = params.get("api_key")
    if not isinstance(reference, str) or not reference.startswith("os.environ/"):
        raise TransactionFailure("api_key_reference_invalid", "model_new")
    env_name = reference.removeprefix("os.environ/")
    api_key = environ.get(env_name, "")
    if not api_key:
        raise TransactionFailure("provider_api_key_missing", "model_new")
    return payload


def _post_json(
    client: HttpClient,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    code: str,
    phase: str,
) -> Mapping[str, Any]:
    try:
        response = client.post(url, headers=dict(headers), json=dict(payload), timeout=20.0)
    except Exception as exc:
        raise TransactionFailure(code, phase) from exc
    return _request_json(response, code, phase)


def _fetch_fusion_models(client: HttpClient, fusion_base_url: str) -> list[dict]:
    try:
        response = client.get(f"{fusion_base_url.rstrip('/')}/api/models/", timeout=20.0)
    except Exception as exc:
        raise TransactionFailure("fusion_readback_failed", "fusion_readback") from exc
    payload = _request_json(response, "fusion_readback_failed", "fusion_readback")
    data = payload.get("data") or {}
    models = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(models, list):
        raise TransactionFailure("fusion_readback_failed", "fusion_readback")
    return [item for item in models if isinstance(item, dict)]


def _wait_for_fusion_model(
    client: HttpClient,
    fusion_base_url: str,
    model_id: str,
) -> list[dict]:
    """等待 Fusion 最长约 68 秒的目录缓存刷新。"""
    for attempt in range(FUSION_READBACK_ATTEMPTS):
        models = _fetch_fusion_models(client, fusion_base_url)
        if model_id in {item.get("modelId") for item in models}:
            return models
        if attempt + 1 < FUSION_READBACK_ATTEMPTS:
            time.sleep(FUSION_READBACK_INTERVAL_SECONDS)
    raise TransactionFailure("fusion_model_missing", "fusion_readback")


def _default_audit(
    *,
    litellm_entries: Sequence[Mapping[str, Any]],
    fusion_models: Sequence[Mapping[str, Any]],
    key_models: Sequence[str],
) -> bool:
    report = audit_catalog(
        litellm_entries=litellm_entries,
        fusion_models=fusion_models,
        key_models=key_models,
    )
    return int(report.summary.get("error_count", 0)) == 0


def _compensate(
    *,
    result: dict[str, Any],
    client: HttpClient,
    litellm_base_url: str,
    master_key: str,
    virtual_key: str,
    before_key_models: Sequence[str],
    candidate: Mapping[str, Any],
    created_uuid: str,
) -> None:
    compensation = result["compensation"]
    compensation["attempted"] = True
    try:
        current_key_models = _fetch_key_models(
            client,
            litellm_base_url,
            master_key,
            virtual_key,
            code="rollback_key_read_failed",
            phase="compensation",
        )
    except TransactionFailure:
        compensation["errors"].append("rollback_key_read_failed")
    else:
        expected_key_models = _expected_key_models(before_key_models, candidate["model_id"])
        if current_key_models == list(before_key_models):
            compensation["key_restored"] = True
        elif current_key_models == expected_key_models:
            try:
                _post_json(
                    client,
                    f"{litellm_base_url.rstrip('/')}/key/update",
                    headers=_headers(master_key),
                    payload={"key": virtual_key, "models": list(before_key_models)},
                    code="rollback_key_failed",
                    phase="compensation",
                )
                compensation["key_restored"] = True
            except TransactionFailure:
                compensation["errors"].append("rollback_key_failed")
        else:
            compensation["errors"].append("rollback_key_cas_conflict")
    if not created_uuid:
        compensation["model_ownership_unverified"] = True
        compensation["manual_cleanup_required"] = True
        compensation["errors"].append("rollback_model_ownership_unverified")
        return
    try:
        _post_json(
            client,
            f"{litellm_base_url.rstrip('/')}/model/delete",
            headers=_headers(master_key),
            payload={"id": created_uuid},
            code="rollback_model_delete_failed",
            phase="compensation",
        )
        compensation["model_deleted"] = True
    except TransactionFailure:
        compensation["errors"].append("rollback_model_delete_failed")


def _expected_key_models(before_key_models: Sequence[str], model_id: str) -> list[str]:
    if model_id in before_key_models:
        return list(before_key_models)
    return [*before_key_models, model_id]


def execute_admission(
    *,
    plan: Mapping[str, Any],
    apply: bool = False,
    expected_run_id: str = "",
    confirm_model_id: str = "",
    confirm_fingerprint: str = "",
    litellm_base_url: str = "",
    fusion_base_url: str = "",
    master_key: str = "",
    virtual_key: str = "",
    environ: Mapping[str, str] | None = None,
    client: HttpClient | None = None,
    audit_fn: Callable[..., bool] = _default_audit,
    catalog_invalidation_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """执行单候选事务；dry-run 路径不触发任何 HTTP。"""
    try:
        candidate = _eligible_candidate(plan)
    except ValueError:
        return _invalid_plan_result(plan, mode="apply" if apply else "dry_run")
    result = _result_base(plan, candidate, mode="apply" if apply else "dry_run")
    if not apply:
        return result
    auth_error = _authorization_error(
        plan=plan,
        candidate=candidate,
        expected_run_id=expected_run_id,
        confirm_model_id=confirm_model_id,
        confirm_fingerprint=confirm_fingerprint,
    )
    if auth_error:
        return _fail_without_writes(result, "authorization_failed", auth_error)
    if not client or not litellm_base_url or not fusion_base_url or not master_key or not virtual_key:
        return _fail_without_writes(result, "authorization_failed", "execution_context_missing")
    return _execute_apply(
        result=result,
        candidate=candidate,
        client=client,
        litellm_base_url=litellm_base_url,
        fusion_base_url=fusion_base_url,
        master_key=master_key,
        virtual_key=virtual_key,
        environ=environ or {},
        audit_fn=audit_fn,
        catalog_invalidation_fn=catalog_invalidation_fn,
    )


def _invalid_plan_result(plan: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "run_id": str(plan.get("run_id") or ""),
        "mode": mode,
        "model_id": "",
        "candidate_fingerprint": "",
        "status": "invalid_plan",
        "phase": "validation",
        "completed_phases": [],
        "writes_performed": False,
        "owned_model_uuid": "",
        "before_state": None,
        "error": {"code": "invalid_plan"},
        "compensation": {
            "attempted": False,
            "key_restored": False,
            "model_deleted": False,
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


def _fail_without_writes(result: dict[str, Any], status: str, code: str) -> dict[str, Any]:
    result["status"] = status
    result["error"] = {"code": code}
    return result


def _execute_apply(
    *,
    result: dict[str, Any],
    candidate: Mapping[str, Any],
    client: HttpClient,
    litellm_base_url: str,
    fusion_base_url: str,
    master_key: str,
    virtual_key: str,
    environ: Mapping[str, str],
    audit_fn: Callable[..., bool],
    catalog_invalidation_fn: Callable[[], None] | None,
) -> dict[str, Any]:
    created_uuid = ""
    before_key_models: list[str] = []
    mutation_started = False
    try:
        before_entries = _fetch_model_info(
            client, litellm_base_url, master_key, code="before_state_read_failed", phase="before_state"
        )
        before_key_models = _fetch_key_models(
            client,
            litellm_base_url,
            master_key,
            virtual_key,
            code="before_state_read_failed",
            phase="before_state",
        )
        payload = candidate["model_new_plan"]["payload"]
        underlying = payload["litellm_params"]["model"]
        if _find_candidate_entry(before_entries, candidate["model_id"], underlying):
            return _fail_without_writes(result, "cas_conflict", "candidate_already_exists")
        before_fingerprint = _state_fingerprint(before_entries, before_key_models)
        result["before_state"] = {
            "fingerprint": before_fingerprint,
            "model_count": len(before_entries),
            "allowlist_count": len(before_key_models),
        }
        _assert_before_state_unchanged(
            client=client,
            base_url=litellm_base_url,
            master_key=master_key,
            virtual_key=virtual_key,
            expected_fingerprint=before_fingerprint,
        )
        resolved_payload = _resolved_model_payload(candidate, environ)
        mutation_started = True
        result["writes_performed"] = True
        model_response = _post_json(
            client,
            f"{litellm_base_url.rstrip('/')}/model/new",
            headers=_headers(master_key),
            payload=resolved_payload,
            code="model_new_failed",
            phase="model_new",
        )
        created_uuid = _created_uuid(model_response)
        result["owned_model_uuid"] = created_uuid
        result["completed_phases"].append("model_new")
        verified_entry = _verify_created_model(
            client=client,
            base_url=litellm_base_url,
            master_key=master_key,
            candidate=candidate,
            created_uuid=created_uuid,
        )
        created_uuid = created_uuid or _entry_identity(verified_entry)[2]
        result["owned_model_uuid"] = created_uuid
        result["completed_phases"].append("verify")
        current_key_models = _fetch_key_models(
            client,
            litellm_base_url,
            master_key,
            virtual_key,
            code="key_cas_read_failed",
            phase="key_update",
        )
        if current_key_models != before_key_models:
            raise TransactionFailure("key_before_state_changed", "key_update")
        _post_json(
            client,
            f"{litellm_base_url.rstrip('/')}/key/update",
            headers=_headers(master_key),
            payload={
                "key": virtual_key,
                "models": _expected_key_models(before_key_models, candidate["model_id"]),
            },
            code="key_update_failed",
            phase="key_update",
        )
        result["completed_phases"].append("key_update")
        if catalog_invalidation_fn is not None:
            try:
                catalog_invalidation_fn()
            except Exception as exc:
                raise TransactionFailure(
                    "fusion_catalog_invalidation_failed",
                    "catalog_invalidation",
                ) from exc
            result["completed_phases"].append("catalog_invalidation")
        fusion_models = _wait_for_fusion_model(
            client,
            fusion_base_url,
            candidate["model_id"],
        )
        result["completed_phases"].append("fusion_readback")
        _verify_created_model(
            client=client,
            base_url=litellm_base_url,
            master_key=master_key,
            candidate=candidate,
            created_uuid=created_uuid,
            phase="audit",
        )
        _run_audit(
            client=client,
            base_url=litellm_base_url,
            master_key=master_key,
            virtual_key=virtual_key,
            fusion_models=fusion_models,
            audit_fn=audit_fn,
        )
        result["completed_phases"].append("audit")
        result["phase"] = "complete"
        result["status"] = "succeeded"
        return result
    except TransactionFailure as exc:
        result["phase"] = exc.phase
        result["status"] = "cas_conflict" if exc.phase == "cas" else "failed"
        result["error"] = {"code": exc.code}
        if mutation_started:
            _compensate(
                result=result,
                client=client,
                litellm_base_url=litellm_base_url,
                master_key=master_key,
                virtual_key=virtual_key,
                before_key_models=before_key_models,
                candidate=candidate,
                created_uuid=created_uuid,
            )
        return result


def _assert_before_state_unchanged(
    *,
    client: HttpClient,
    base_url: str,
    master_key: str,
    virtual_key: str,
    expected_fingerprint: str,
) -> None:
    entries = _fetch_model_info(
        client,
        base_url,
        master_key,
        code="cas_read_failed",
        phase="cas",
    )
    key_models = _fetch_key_models(
        client,
        base_url,
        master_key,
        virtual_key,
        code="cas_read_failed",
        phase="cas",
    )
    if _state_fingerprint(entries, key_models) != expected_fingerprint:
        raise TransactionFailure("before_state_changed", "cas")


def _verify_created_model(
    *,
    client: HttpClient,
    base_url: str,
    master_key: str,
    candidate: Mapping[str, Any],
    created_uuid: str,
    phase: str = "verify",
) -> Mapping[str, Any]:
    if not created_uuid:
        raise TransactionFailure("model_ownership_unverified", phase)
    entries = _fetch_model_info(
        client,
        base_url,
        master_key,
        code="model_verify_failed",
        phase=phase,
    )
    payload = candidate["model_new_plan"]["payload"]
    exact_entries = [entry for entry in entries if _entry_matches_expected(entry, payload)]
    owned_entries = [entry for entry in entries if _entry_identity(entry)[2] == created_uuid]
    if len(owned_entries) != 1 or owned_entries[0] not in exact_entries or len(exact_entries) != 1:
        raise TransactionFailure("model_verify_failed", phase)
    return owned_entries[0]


def _entry_matches_expected(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    params = entry.get("litellm_params")
    info = entry.get("model_info")
    expected_params = payload.get("litellm_params")
    expected_info = payload.get("model_info")
    if (
        not isinstance(params, Mapping)
        or not isinstance(info, Mapping)
        or not isinstance(expected_params, Mapping)
        or not isinstance(expected_info, Mapping)
    ):
        return False
    expected_metadata = expected_info.get("metadata")
    actual_metadata = info.get("metadata")
    expected_costs = {
        key: value
        for key, value in expected_info.items()
        if key.endswith("_cost_per_token") or key == "cache_read_input_token_cost"
    }
    actual_costs = {key: info.get(key) for key in expected_costs}
    expected_max_input_tokens = expected_info.get("max_input_tokens")
    max_input_tokens_match = (
        "max_input_tokens" not in expected_info or info.get("max_input_tokens") == expected_max_input_tokens
    )
    return (
        entry.get("model_name") == payload.get("model_name")
        and params.get("model") == expected_params.get("model")
        and params.get("api_base") == expected_params.get("api_base")
        and actual_metadata == expected_metadata
        and actual_costs == expected_costs
        and max_input_tokens_match
    )


def _run_audit(
    *,
    client: HttpClient,
    base_url: str,
    master_key: str,
    virtual_key: str,
    fusion_models: Sequence[Mapping[str, Any]],
    audit_fn: Callable[..., bool],
) -> None:
    entries = _fetch_model_info(
        client,
        base_url,
        master_key,
        code="audit_read_failed",
        phase="audit",
    )
    key_models = _fetch_key_models(
        client,
        base_url,
        master_key,
        virtual_key,
        code="audit_read_failed",
        phase="audit",
    )
    try:
        passed = audit_fn(
            litellm_entries=entries,
            fusion_models=fusion_models,
            key_models=key_models,
        )
    except Exception as exc:
        raise TransactionFailure("audit_failed", "audit") from exc
    if not passed:
        raise TransactionFailure("audit_failed", "audit")


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
    parser = argparse.ArgumentParser(description="受控执行单个 LiteLLM candidate admission plan")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--governance-root", type=Path)
    parser.add_argument("--candidate-fingerprint", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--litellm-base-url", default=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--fusion-base-url", default=os.environ.get("FUSION_BASE_URL", "https://fusion.seanfield.org"))
    parser.add_argument("--master-key-env", default="LITELLM_MASTER_KEY")
    parser.add_argument("--virtual-key-env", default="LITELLM_VIRTUAL_KEY")
    parser.add_argument("--expected-run-id", default="")
    parser.add_argument("--confirm-model-id", default="")
    parser.add_argument("--confirm-fingerprint", default="")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _verification_error_result(*, candidate_fingerprint: str, code: str) -> dict[str, Any]:
    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "run_id": "",
        "mode": "apply",
        "model_id": "",
        "candidate_fingerprint": candidate_fingerprint,
        "status": "verification_failed",
        "phase": "artifact_verification",
        "completed_phases": [],
        "writes_performed": False,
        "owned_model_uuid": "",
        "before_state": None,
        "error": {"code": code},
        "compensation": {
            "attempted": False,
            "key_restored": False,
            "model_deleted": False,
            "model_ownership_unverified": False,
            "manual_cleanup_required": False,
            "errors": [],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.apply:
        verification_error = ""
        if args.plan is not None:
            verification_error = "apply_plan_argument_forbidden"
        elif args.governance_root is None or not args.candidate_fingerprint:
            verification_error = "verified_governance_source_required"
        if verification_error:
            result = _verification_error_result(
                candidate_fingerprint=args.candidate_fingerprint,
                code=verification_error,
            )
            _write_json_atomic(args.output, result)
            return 1
        try:
            plan = load_verified_admission_plan(
                governance_root=args.governance_root,
                candidate_fingerprint=args.candidate_fingerprint,
            )
        except VerifiedPlanError as exc:
            result = _verification_error_result(
                candidate_fingerprint=args.candidate_fingerprint,
                code=exc.code,
            )
            _write_json_atomic(args.output, result)
            return 1
        with httpx.Client() as client:
            result = execute_admission(
                plan=plan,
                apply=True,
                expected_run_id=args.expected_run_id,
                confirm_model_id=args.confirm_model_id,
                confirm_fingerprint=args.confirm_fingerprint,
                litellm_base_url=args.litellm_base_url,
                fusion_base_url=args.fusion_base_url,
                master_key=os.environ.get(args.master_key_env, ""),
                virtual_key=os.environ.get(args.virtual_key_env, ""),
                environ=os.environ,
                client=client,
            )
    else:
        if args.plan is None:
            result = _verification_error_result(candidate_fingerprint="", code="dry_run_plan_required")
            result["mode"] = "dry_run"
            _write_json_atomic(args.output, result)
            return 1
        try:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = _verification_error_result(candidate_fingerprint="", code="dry_run_plan_invalid")
            result["mode"] = "dry_run"
            _write_json_atomic(args.output, result)
            return 1
        if not isinstance(plan, Mapping):
            result = _verification_error_result(candidate_fingerprint="", code="dry_run_plan_invalid")
            result["mode"] = "dry_run"
            _write_json_atomic(args.output, result)
            return 1
        result = execute_admission(plan=plan)
    _write_json_atomic(args.output, result)
    return 0 if result["status"] in {"planned", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
