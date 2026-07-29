"""用 LiteLLM 成本表和受审策略只读富化模型候选。

本脚本不访问网络、不读取真实 API key，也不注册模型。成本或能力证据不完整时，
候选会保留缺失字段，交给后续准入门禁继续隔离。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from scripts.fetch_litellm_cost_map import cost_map_sha256

FUSION_CAPABILITY_FIELDS = {
    "imageGen": ("supports_image_generation",),
    "deepThinking": ("supports_reasoning",),
    "fileSupport": ("supports_pdf_input",),
    "functionCalling": ("supports_function_calling",),
    "vision": ("supports_vision",),
    "webSearch": ("supports_web_search",),
}
PRICING_RATE_FIELDS = ("input", "output", "cache_read_input")
REGIONAL_PRICE_RATIO_RELATIVE_TOLERANCE = 0.02


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _provider_configs(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    providers = registry.get("providers")
    if not isinstance(providers, list):
        raise ValueError("registry.providers 必须是列表")
    configs: dict[str, Mapping[str, Any]] = {}
    for provider in providers:
        if not _is_mapping(provider):
            continue
        key = "moonshot" if provider.get("adapter") == "moonshot" else str(provider.get("provider_key") or "")
        if key:
            configs[key] = provider
    return configs


def verify_cost_map_status(
    *,
    cost_map: Mapping[str, Any],
    status: Mapping[str, Any],
    max_age_seconds: int = 86400,
    now: datetime | None = None,
) -> None:
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds 必须大于 0")
    if status.get("status") != "success":
        raise ValueError("成本表状态不是 success")
    source_url = urlparse(str(status.get("source_url") or ""))
    if source_url.scheme != "https" or not source_url.netloc:
        raise ValueError("成本表 source_url 必须是 HTTPS 地址")
    if status.get("model_count") != len(cost_map) or status.get("sha256") != cost_map_sha256(cost_map):
        raise ValueError("成本表状态与数据文件不匹配")
    try:
        fetched_at = datetime.fromisoformat(str(status.get("fetched_at") or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("成本表 fetched_at 无效") from exc
    if fetched_at.tzinfo is None:
        raise ValueError("成本表 fetched_at 必须包含时区")
    reference = now or datetime.now(timezone.utc)
    age = (reference.astimezone(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds()
    if age < -300 or age > max_age_seconds:
        raise ValueError("成本表快照时间不在允许范围内")


def _cost_map_keys(candidate: Mapping[str, Any], provider: Mapping[str, Any]) -> list[str]:
    model_id = str(candidate.get("model_id") or "")
    underlying = str(candidate.get("litellm_model") or "")
    prefix = str(provider.get("cost_map_prefix") or "").strip().strip("/")
    keys = (f"{prefix}/{model_id}", model_id) if prefix else (underlying, model_id)
    return list(dict.fromkeys(key for key in keys if key))


def _cost_entry_matches_provider(
    *,
    key: str,
    entry: Mapping[str, Any],
    provider_key: str,
    provider: Mapping[str, Any],
) -> bool:
    litellm_prefix = str(provider.get("litellm_prefix") or "").strip().strip("/").lower()
    expected = {
        provider_key.lower(),
        str(provider.get("cost_map_prefix") or "").strip().strip("/").lower(),
    }
    if provider.get("adapter") != "openai-compatible" or litellm_prefix != "openai":
        expected.add(litellm_prefix)
    expected.discard("")
    declared = str(entry.get("litellm_provider") or entry.get("provider") or "").strip().lower()
    if declared:
        return declared in expected
    namespace = key.split("/", 1)[0].lower() if "/" in key else ""
    return bool(namespace and namespace in expected)


def _cost_entry(
    candidate: Mapping[str, Any],
    provider: Mapping[str, Any],
    cost_map: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    provider_key = str(candidate.get("provider_key") or "")
    for key in _cost_map_keys(candidate, provider):
        value = cost_map.get(key)
        if _is_mapping(value) and _cost_entry_matches_provider(
            key=key,
            entry=value,
            provider_key=provider_key,
            provider=provider,
        ):
            return key, value
    return "", None


def _capabilities(entry: Mapping[str, Any]) -> dict[str, bool]:
    evidence_present = any(source in entry for sources in FUSION_CAPABILITY_FIELDS.values() for source in sources)
    if not evidence_present:
        return {}
    return {
        target: any(entry.get(source) is True for source in sources)
        for target, sources in FUSION_CAPABILITY_FIELDS.items()
    }


def _pricing(entry: Mapping[str, Any]) -> dict[str, Any]:
    input_cost = entry.get("input_cost_per_token")
    output_cost = entry.get("output_cost_per_token")
    if (
        isinstance(input_cost, bool)
        or not isinstance(input_cost, (int, float))
        or input_cost < 0
        or isinstance(output_cost, bool)
        or not isinstance(output_cost, (int, float))
        or output_cost < 0
    ):
        return {}
    return {
        "input": input_cost * 1_000_000,
        "output": output_cost * 1_000_000,
        "unit": "USD/1M tokens",
    }


def _override_for(
    overrides: Mapping[str, Any] | None,
    provider_key: str,
    model_id: str,
) -> Mapping[str, Any]:
    if not _is_mapping(overrides):
        return {}
    providers = overrides.get("providers")
    provider = providers.get(provider_key) if _is_mapping(providers) else None
    models = provider.get("models") if _is_mapping(provider) else None
    value = models.get(model_id) if _is_mapping(models) else None
    return value if _is_mapping(value) else {}


def validate_override_approval(overrides: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = overrides.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("override.schema_version 必须为 1 或 2")
    providers = overrides.get("providers")
    approval = overrides.get("approval")
    if not _is_mapping(providers) or not _is_mapping(approval):
        raise ValueError("override 缺少 providers 或 approval")
    required = ("policy_version", "reviewed_by", "reviewed_at", "providers_sha256")
    if any(not str(approval.get(key) or "").strip() for key in required):
        raise ValueError("override approval 字段不完整")
    try:
        reviewed_at = datetime.fromisoformat(str(approval["reviewed_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("override reviewed_at 无效") from exc
    if reviewed_at.tzinfo is None:
        raise ValueError("override reviewed_at 必须包含时区")
    source_urls = approval.get("source_urls")
    if not isinstance(source_urls, list) or not source_urls:
        raise ValueError("override approval 缺少 source_urls")
    for source_url in source_urls:
        parsed = urlparse(str(source_url))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("override source_urls 必须全部为 HTTPS")
    expected_sha256 = cost_map_sha256(providers)
    if approval.get("providers_sha256") != expected_sha256:
        raise ValueError("override providers_sha256 不匹配")
    normalized = {
        "schema_version": schema_version,
        "policy_version": str(approval["policy_version"]),
        "reviewed_by": str(approval["reviewed_by"]),
        "reviewed_at": reviewed_at.astimezone(timezone.utc).isoformat(),
        "source_urls": [str(item) for item in source_urls],
        "providers_sha256": expected_sha256,
    }
    if schema_version == 2:
        valid_until_text = str(approval.get("valid_until") or "")
        try:
            valid_until = datetime.fromisoformat(valid_until_text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("override valid_until 无效") from exc
        if valid_until.tzinfo is None:
            raise ValueError("override valid_until 必须包含时区")
        now = datetime.now(timezone.utc)
        if reviewed_at.astimezone(timezone.utc) > now.replace(microsecond=0):
            raise ValueError("override reviewed_at 来自未来")
        if valid_until.astimezone(timezone.utc) <= now:
            raise ValueError("override 已过期")
        document_sha256 = str(approval.get("document_sha256") or "")
        document = {
            "schema_version": schema_version,
            "approval": {key: copy.deepcopy(value) for key, value in approval.items() if key != "document_sha256"},
            "providers": providers,
        }
        expected_document_sha256 = cost_map_sha256(document)
        if document_sha256 != expected_document_sha256:
            raise ValueError("override document_sha256 不匹配")
        normalized["document_sha256"] = expected_document_sha256
        normalized["valid_until"] = valid_until.astimezone(timezone.utc).isoformat()
    _validate_override_models(providers, schema_version=schema_version)
    if schema_version == 2:
        approved_sources = set(normalized["source_urls"])
        for provider in providers.values():
            for model in provider["models"].values():
                provenance = model.get("pricing_provenance")
                if not _is_mapping(provenance):
                    continue
                evidence_urls = {
                    str(item.get("url")) for item in provenance.get("source_evidence") or [] if _is_mapping(item)
                }
                if not evidence_urls.issubset(approved_sources):
                    raise ValueError("override pricing_provenance 来源未纳入审批")
    return normalized


def _valid_nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _validate_pricing_rates(rates: Any, *, label: str) -> Mapping[str, Any]:
    if not _is_mapping(rates) or not str(rates.get("unit") or "").strip():
        raise ValueError(f"override {label} 不完整")
    for key in ("input", "output"):
        if not _valid_nonnegative_number(rates.get(key)):
            raise ValueError(f"override {label}.{key} 无效")
    if "cache_read_input" in rates and not _valid_nonnegative_number(rates.get("cache_read_input")):
        raise ValueError(f"override {label}.cache_read_input 无效")
    return rates


def _validate_override_models(providers: Mapping[str, Any], *, schema_version: int) -> None:
    for provider in providers.values():
        if not _is_mapping(provider):
            raise ValueError("override provider 必须是对象")
        models = provider.get("models")
        if not _is_mapping(models):
            raise ValueError("override provider.models 必须是对象")
        for model in models.values():
            if not _is_mapping(model):
                raise ValueError("override model 必须是对象")
            pricing = model.get("pricing")
            if pricing is not None:
                pricing = _validate_pricing_rates(pricing, label="pricing")
            provenance = model.get("pricing_provenance")
            if provenance is None:
                continue
            if not _is_mapping(provenance):
                raise ValueError("override pricing_provenance 必须是对象")
            required_provenance_keys = [
                "billing_region",
                "billing_currency",
                "canonical_currency",
                "canonicalization_method",
            ]
            if schema_version == 2:
                required_provenance_keys.extend(("api_base", "input_basis"))
            for key in required_provenance_keys:
                if not str(provenance.get(key) or "").strip():
                    raise ValueError(f"override pricing_provenance.{key} 缺失")
            billing_rates = _validate_pricing_rates(
                provenance.get("billing_rates"),
                label="pricing_provenance.billing_rates",
            )
            canonical_rates = _validate_pricing_rates(
                provenance.get("canonical_rates"),
                label="pricing_provenance.canonical_rates",
            )
            if pricing is None or any(
                pricing.get(key) != canonical_rates.get(key)
                for key in (*PRICING_RATE_FIELDS, "unit")
                if key in pricing or key in canonical_rates
            ):
                raise ValueError("override pricing_provenance.canonical_rates 与 pricing 不一致")
            if str(billing_rates["unit"]).split("/", 1)[0] != str(provenance["billing_currency"]):
                raise ValueError("override pricing_provenance.billing_rates 币种不一致")
            if str(canonical_rates["unit"]).split("/", 1)[0] != str(provenance["canonical_currency"]):
                raise ValueError("override pricing_provenance.canonical_rates 币种不一致")
            if (
                schema_version == 2
                and provenance["canonicalization_method"] == "provider_published_regional_price_pair"
            ):
                ratios = [
                    billing_rates[key] / canonical_rates[key]
                    for key in PRICING_RATE_FIELDS
                    if key in billing_rates and key in canonical_rates and canonical_rates[key] > 0
                ]
                ratio_midpoint = sum(ratios) / len(ratios) if ratios else 0
                relative_spread = (
                    (max(ratios) - min(ratios)) / ratio_midpoint
                    if ratio_midpoint > 0
                    else math.inf
                )
                if (
                    len(ratios) < 2
                    or relative_spread > REGIONAL_PRICE_RATIO_RELATIVE_TOLERANCE
                ):
                    raise ValueError("override pricing_provenance 区域价格比例不一致")
            source_evidence = provenance.get("source_evidence")
            if schema_version == 1 and source_evidence is None:
                continue
            if not isinstance(source_evidence, list) or len(source_evidence) < 2:
                raise ValueError("override pricing_provenance.source_evidence 不完整")
            for evidence in source_evidence:
                if not _is_mapping(evidence) or not str(evidence.get("region") or "").strip():
                    raise ValueError("override pricing_provenance.source_evidence 无效")
                parsed = urlparse(str(evidence.get("url") or ""))
                if parsed.scheme != "https" or not parsed.netloc:
                    raise ValueError("override pricing_provenance.source_evidence URL 无效")


def _metadata(
    *,
    candidate: Mapping[str, Any],
    provider_key: str,
    provider_display: str,
    cost_entry: Mapping[str, Any] | None,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "display_name": str(override.get("display_name") or candidate.get("model_id") or ""),
        "provider_key": provider_key,
        "provider_display": provider_display,
    }
    if cost_entry is not None:
        capabilities = _capabilities(cost_entry)
        pricing = _pricing(cost_entry)
        if capabilities:
            metadata["capabilities"] = capabilities
        if pricing:
            metadata["pricing"] = pricing
    for key in (
        "description",
        "capabilities",
        "pricing",
        "pricing_provenance",
        "knowledge_cutoff",
        "recommended_for",
    ):
        if key in override:
            metadata[key] = copy.deepcopy(override[key])
    return metadata


def enrich_candidate_report(
    *,
    candidate_report: Mapping[str, Any],
    registry: Mapping[str, Any],
    cost_map: Mapping[str, Any],
    cost_map_status: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    max_cost_map_age_seconds: int = 86400,
) -> dict[str, Any]:
    """返回新的富化报告；输入对象保持不变。"""
    verify_cost_map_status(
        cost_map=cost_map,
        status=cost_map_status,
        max_age_seconds=max_cost_map_age_seconds,
    )
    report = copy.deepcopy(dict(candidate_report))
    providers = report.get("providers")
    if not _is_mapping(providers):
        raise ValueError("candidate_report.providers 必须是对象")
    configs = _provider_configs(registry)
    preflight_contract = report.get("candidate_preflight")
    preflight_providers = preflight_contract.get("providers") if _is_mapping(preflight_contract) else {}
    override_approval = validate_override_approval(overrides) if overrides is not None else None
    for provider_key, provider_result in providers.items():
        if not _is_mapping(provider_result):
            continue
        provider_report = provider_result.get("report")
        new_candidates = provider_report.get("new") if _is_mapping(provider_report) else None
        provider_config = configs.get(str(provider_key))
        if not isinstance(new_candidates, list) or provider_config is None:
            continue
        provider_display = str((provider_report.get("provider") or {}).get("display") or provider_key)
        for candidate in new_candidates:
            if not _is_mapping(candidate):
                continue
            key, entry = _cost_entry(candidate, provider_config, cost_map)
            override = _override_for(overrides, str(provider_key), str(candidate.get("model_id") or ""))
            provenance = override.get("pricing_provenance") if _is_mapping(override) else None
            if _is_mapping(provenance) and provenance.get("api_base") not in (None, provider_config.get("base_url")):
                raise ValueError("override pricing_provenance.api_base 与 provider registry 不一致")
            cost_pricing = _pricing(entry) if entry is not None else {}
            override_pricing = override.get("pricing") if _is_mapping(override) else None
            override_cost_map_conflict = bool(
                cost_pricing
                and _is_mapping(override_pricing)
                and any(cost_pricing.get(field) != override_pricing.get(field) for field in ("input", "output", "unit"))
            )
            candidate["registration"] = {
                "api_base": provider_config.get("base_url"),
                "api_key_env": provider_config.get("api_key_env"),
                "endpoint_status": "verified" if provider_result.get("status") == "ok" else "unknown",
            }
            route = preflight_providers.get(str(provider_key)) if _is_mapping(preflight_providers) else None
            if _is_mapping(route):
                route_pattern = str(route.get("route_model_name") or "")
                model_id = str(candidate.get("model_id") or "")
                candidate["preflight_route"] = copy.deepcopy(dict(route))
                candidate["preflight_model"] = (
                    route_pattern[:-1] + model_id if route_pattern.endswith("*") and model_id else ""
                )
            candidate["metadata"] = _metadata(
                candidate=candidate,
                provider_key=str(provider_key),
                provider_display=provider_display,
                cost_entry=entry,
                override=override,
            )
            candidate["metadata_evidence"] = {
                "cost_map_key": key or None,
                "cost_map_matched": entry is not None,
                "cost_map_sha256": cost_map_status["sha256"],
                "reviewed_override_applied": bool(override),
                "override_approval": copy.deepcopy(override_approval) if override else None,
                "override_cost_map_conflict": override_cost_map_conflict,
            }
    report["enrichment"] = {
        "mode": "read_only",
        "writes_performed": False,
        "cost_map_source": "local_file",
        "cost_map_sha256": cost_map_status["sha256"],
        "cost_map_fetched_at": cost_map_status["fetched_at"],
        "overrides_applied": overrides is not None,
        "override_approval": override_approval,
    }
    return report


def _read_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cost-map", type=Path, required=True)
    parser.add_argument("--cost-map-status", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cost-map-age-seconds", type=int, default=86400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    payload = enrich_candidate_report(
        candidate_report=_read_object(args.candidate_report),
        registry=_read_object(args.registry),
        cost_map=_read_object(args.cost_map),
        cost_map_status=_read_object(args.cost_map_status),
        overrides=_read_object(args.overrides) if args.overrides else None,
        max_cost_map_age_seconds=args.max_cost_map_age_seconds,
    )
    _write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
