"""巡检 Fusion / LiteLLM 模型目录一致性。

默认只输出 JSON 报告。显式传 `--apply` 时，只同步 Fusion LiteLLM
virtual key 的模型 allowlist，不注册或删除 LiteLLM 模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlencode

import httpx

LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_PROXY_URL_ENV = "LITELLM_PROXY_URL"
LITELLM_MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
LITELLM_API_KEY_ENV = "LITELLM_API_KEY"
LITELLM_VIRTUAL_KEY_ENV = "LITELLM_VIRTUAL_KEY"
DEFAULT_LITELLM_BASE_URL = "http://localhost:4000"
DEFAULT_FUSION_BASE_URL = "https://fusion.seanfield.org"

DEPRECATED_MODELS = {"mimo-v2-flash", "mimo-v2-pro"}
REQUIRED_METADATA_KEYS = ("provider_key", "provider_display", "capabilities", "pricing")


class HttpClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    severity: str
    model_name: str
    message: str


@dataclass(frozen=True)
class SyncPlan:
    allowlist_before: list[str] = field(default_factory=list)
    allowlist_after: list[str] = field(default_factory=list)
    add: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.add or self.remove)


@dataclass(frozen=True)
class AuditReport:
    summary: dict[str, int | bool]
    issues: list[CatalogIssue]
    sync_plan: SyncPlan


def _entry_model_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("model_name") or "")


def _entry_model_info(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get("model_info") or {}
    return value if isinstance(value, Mapping) else {}


def _entry_metadata(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _entry_model_info(entry).get("metadata") or {}
    return metadata if isinstance(metadata, Mapping) else {}


def extract_db_models(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """提取 LiteLLM 中可给 Fusion 前端展示的业务别名。"""
    models: list[dict[str, Any]] = []
    for entry in entries:
        model_name = _entry_model_name(entry)
        if not model_name:
            continue
        model_info = _entry_model_info(entry)
        if not bool(model_info.get("db_model")):
            continue
        models.append({"model_name": model_name, "metadata": dict(_entry_metadata(entry))})
    return models


def extract_fusion_model_ids(fusion_models: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for model in fusion_models:
        model_id = str(model.get("modelId") or "")
        if model_id:
            ids.append(model_id)
    return ids


def _missing_metadata_keys(metadata: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_METADATA_KEYS:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            missing.append(key)
    return missing


def _has_required_metadata(metadata: Mapping[str, Any]) -> bool:
    return not _missing_metadata_keys(metadata)


def build_allowlist_sync_plan(
    *,
    db_model_names: Sequence[str],
    key_models: Sequence[str] | None,
) -> SyncPlan:
    if key_models is None:
        return SyncPlan()

    db_set = set(db_model_names)
    before = list(key_models)
    after: list[str] = []
    seen: set[str] = set()
    remove: list[str] = []

    for model in before:
        if model in DEPRECATED_MODELS:
            remove.append(model)
            continue
        if model in seen:
            continue
        after.append(model)
        seen.add(model)

    add: list[str] = []
    for model in db_model_names:
        if model not in db_set or model in seen:
            continue
        add.append(model)
        seen.add(model)
    after.extend(add)
    return SyncPlan(allowlist_before=before, allowlist_after=after, add=add, remove=remove)


def audit_catalog(
    *,
    litellm_entries: Sequence[Mapping[str, Any]],
    fusion_models: Sequence[Mapping[str, Any]],
    key_models: Sequence[str] | None,
) -> AuditReport:
    raw_db_models = extract_db_models(litellm_entries)
    db_models_by_name: dict[str, dict[str, Any]] = {}
    duplicate_names: set[str] = set()
    for model in raw_db_models:
        model_name = model["model_name"]
        if model_name in db_models_by_name:
            duplicate_names.add(model_name)
            continue
        db_models_by_name[model_name] = model
    db_models = list(db_models_by_name.values())
    db_model_names = [model["model_name"] for model in db_models]
    db_model_set = set(db_model_names)
    eligible_db_model_names = [
        model["model_name"]
        for model in db_models
        if model["model_name"] not in DEPRECATED_MODELS and _has_required_metadata(model["metadata"])
    ]
    eligible_db_model_set = set(eligible_db_model_names)
    fusion_model_ids = extract_fusion_model_ids(fusion_models)
    fusion_model_set = set(fusion_model_ids)
    issues: list[CatalogIssue] = []

    for model_name in sorted(duplicate_names):
        issues.append(
            CatalogIssue(
                code="db_model_duplicate",
                severity="warning",
                model_name=model_name,
                message=f"LiteLLM /model/info 中存在重复业务模型别名 {model_name}，巡检按唯一别名处理",
            )
        )

    for model_id in fusion_model_ids:
        if model_id not in db_model_set:
            issues.append(
                CatalogIssue(
                    code="fusion_unknown_model",
                    severity="error",
                    model_name=model_id,
                    message=f"Fusion /api/models 展示了 LiteLLM 业务目录中不存在的模型 {model_id}",
                )
            )

    for model_name in eligible_db_model_names:
        if model_name not in fusion_model_set:
            issues.append(
                CatalogIssue(
                    code="fusion_missing_db_model",
                    severity="error",
                    model_name=model_name,
                    message=f"LiteLLM 业务模型 {model_name} 未出现在 Fusion /api/models",
                )
            )

    for model in db_models:
        missing = _missing_metadata_keys(model["metadata"])
        if missing:
            issues.append(
                CatalogIssue(
                    code="metadata_missing",
                    severity="warning",
                    model_name=model["model_name"],
                    message=f"业务模型缺少关键 metadata: {', '.join(missing)}",
                )
            )

    sync_plan = build_allowlist_sync_plan(db_model_names=eligible_db_model_names, key_models=key_models)
    if key_models is not None:
        for model_name in sync_plan.add:
            issues.append(
                CatalogIssue(
                    code="key_missing_db_model",
                    severity="error",
                    model_name=model_name,
                    message=f"Fusion virtual key 缺少业务模型 {model_name}",
                )
            )
        for model_name in sync_plan.remove:
            issues.append(
                CatalogIssue(
                    code="key_deprecated_model",
                    severity="error",
                    model_name=model_name,
                    message=f"Fusion virtual key 包含已退役模型 {model_name}",
                )
            )
        for model_name in key_models:
            if model_name not in eligible_db_model_set and model_name not in DEPRECATED_MODELS:
                issues.append(
                    CatalogIssue(
                        code="key_extra_model",
                        severity="warning",
                        model_name=model_name,
                        message=f"Fusion virtual key 包含非 Fusion 业务模型 {model_name}，v1 保守保留",
                    )
                )

    summary = {
        "litellm_db_models": len(db_models),
        "fusion_models": len(fusion_model_ids),
        "virtual_key_models": len(key_models) if key_models is not None else 0,
        "key_audit_enabled": key_models is not None,
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
    }
    return AuditReport(summary=summary, issues=issues, sync_plan=sync_plan)


def fetch_litellm_models(base_url: str, master_key: str) -> list[dict[str, Any]]:
    response = httpx.get(
        f"{base_url.rstrip('/')}/model/info",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=20.0,
    )
    response.raise_for_status()
    return list(response.json().get("data") or [])


def fetch_fusion_models(base_url: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{base_url.rstrip('/')}/api/models/", timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    return list((payload.get("data") or {}).get("models") or [])


def fetch_key_models(base_url: str, master_key: str, virtual_key: str) -> list[str]:
    query = urlencode({"key": virtual_key})
    response = httpx.get(
        f"{base_url.rstrip('/')}/key/info?{query}",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=20.0,
    )
    response.raise_for_status()
    payload = response.json()
    info = payload.get("info") or payload
    return list(info.get("models") or [])


def update_key_models(
    *,
    base_url: str,
    master_key: str,
    virtual_key: str,
    models: Sequence[str],
    client: HttpClient | None = None,
) -> None:
    http_client = client or httpx
    response = http_client.post(
        f"{base_url.rstrip('/')}/key/update",
        headers={"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"},
        json={"key": virtual_key, "models": list(models)},
        timeout=20.0,
    )
    response.raise_for_status()


def apply_sync(
    *,
    base_url: str,
    master_key: str,
    virtual_key: str,
    sync_plan: SyncPlan,
    client: HttpClient | None = None,
) -> None:
    if not sync_plan.has_changes:
        return
    update_key_models(
        base_url=base_url,
        master_key=master_key,
        virtual_key=virtual_key,
        models=sync_plan.allowlist_after,
        client=client,
    )


def serialize_report(report: AuditReport, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    safe_context = {}
    for key, value in (context or {}).items():
        if key in {"master_key", "virtual_key", "api_key", "token"}:
            safe_context[f"has_{key}"] = bool(value)
        else:
            safe_context[key] = value
    return {
        "context": safe_context,
        "summary": dict(report.summary),
        "issues": [asdict(issue) for issue in report.issues],
        "sync_plan": asdict(report.sync_plan),
    }


def _default_litellm_base_url() -> str:
    return os.environ.get(LITELLM_BASE_URL_ENV) or os.environ.get(LITELLM_PROXY_URL_ENV) or DEFAULT_LITELLM_BASE_URL


def _default_master_key() -> str:
    return os.environ.get(LITELLM_MASTER_KEY_ENV) or os.environ.get(LITELLM_API_KEY_ENV) or ""


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="巡检 Fusion / LiteLLM 模型目录一致性")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只输出报告，不写 LiteLLM（默认）")
    mode.add_argument("--apply", action="store_true", help="同步 Fusion virtual key 模型 allowlist")
    parser.add_argument("--litellm-base-url", default=_default_litellm_base_url())
    parser.add_argument("--fusion-base-url", default=os.environ.get("FUSION_BASE_URL", DEFAULT_FUSION_BASE_URL))
    parser.add_argument("--master-key", default=_default_master_key())
    parser.add_argument("--virtual-key", default=os.environ.get(LITELLM_VIRTUAL_KEY_ENV, ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if not args.master_key:
        raise RuntimeError(f"缺少 {LITELLM_MASTER_KEY_ENV} 或 {LITELLM_API_KEY_ENV}")
    if args.apply and not args.virtual_key:
        raise RuntimeError(f"--apply 需要 --virtual-key 或 {LITELLM_VIRTUAL_KEY_ENV}")

    litellm_entries = fetch_litellm_models(args.litellm_base_url, args.master_key)
    fusion_models = fetch_fusion_models(args.fusion_base_url)
    key_models = (
        fetch_key_models(args.litellm_base_url, args.master_key, args.virtual_key) if args.virtual_key else None
    )
    report = audit_catalog(
        litellm_entries=litellm_entries,
        fusion_models=fusion_models,
        key_models=key_models,
    )
    print(
        json.dumps(
            serialize_report(
                report,
                context={
                    "litellm_base_url": args.litellm_base_url,
                    "fusion_base_url": args.fusion_base_url,
                    "master_key": args.master_key,
                    "virtual_key": args.virtual_key,
                    "mode": "apply" if args.apply else "dry-run",
                },
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.apply:
        apply_sync(
            base_url=args.litellm_base_url,
            master_key=args.master_key,
            virtual_key=args.virtual_key,
            sync_plan=report.sync_plan,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
