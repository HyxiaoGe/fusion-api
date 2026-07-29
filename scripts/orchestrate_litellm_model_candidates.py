"""只读协调多个厂商的 LiteLLM 模型候选发现。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import httpx

from scripts.discover_litellm_model_candidates import (
    MoonshotProviderAdapter,
    OpenAICompatibleProviderAdapter,
    ProviderAdapter,
    discover_candidates,
    serialize_report,
)


def _provider_key(config: Mapping[str, Any]) -> str:
    if config.get("adapter") == "moonshot":
        return "moonshot"
    return str(config.get("provider_key") or "").strip().lower() or "unknown"


def _empty_report(provider_key: str) -> dict[str, Any]:
    return {
        "mode": "read_only",
        "writes_performed": False,
        "write_actions": [],
        "provider": {"key": provider_key},
        "summary": {"new": 0, "existing": 0, "removed": 0, "unknown": 0},
        "new": [],
        "existing": [],
        "removed": [],
        "unknown": [],
    }


def _error_result(provider_key: str, code: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": {"code": code},
        "report": _empty_report(provider_key),
    }


def _build_adapter(config: Mapping[str, Any]) -> ProviderAdapter:
    adapter = config.get("adapter")
    if adapter == "moonshot":
        return MoonshotProviderAdapter()
    if adapter != "openai-compatible":
        raise ValueError("不支持的 provider adapter")
    return OpenAICompatibleProviderAdapter(
        provider_key=str(config.get("provider_key") or ""),
        provider_display=str(config.get("provider_display") or ""),
        litellm_prefix=str(config.get("litellm_prefix") or ""),
        api_model_prefix=str(config.get("api_model_prefix") or ""),
    )


def _models_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"


def _read_json_response(response: Any) -> Any:
    response.raise_for_status()
    return response.json()


def _candidate_preflight_status(
    *,
    litellm_config: Mapping[str, Any],
    providers: Sequence[Mapping[str, Any]],
    litellm_snapshot: Any,
    candidate_key_models: Sequence[str] | None,
    candidate_key_lookup_failed: bool,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    config = litellm_config.get("candidate_preflight")
    if not isinstance(config, Mapping):
        return {
            "status": "blocked",
            "transport": None,
            "route_model_name": None,
            "credential_source": None,
            "reasons": ["candidate_preflight_config_missing"],
        }
    transport = str(config.get("transport") or "")
    credential_env = str(config.get("api_key_env") or "")
    master_key_env = str(litellm_config.get("api_key_env") or "")
    reasons: list[str] = []
    if transport != "litellm_provider_wildcard":
        reasons.append("candidate_preflight_transport_invalid")
    entries = litellm_snapshot.get("data") if isinstance(litellm_snapshot, Mapping) else None
    provider_statuses: dict[str, Any] = {}
    expected_routes: set[str] = set()
    for provider in providers:
        provider_key = _provider_key(provider)
        route_config = provider.get("candidate_preflight")
        provider_reasons: list[str] = []
        if not isinstance(route_config, Mapping):
            route_config = {}
            provider_reasons.append("candidate_preflight_provider_config_missing")
        route_model_name = str(route_config.get("route_model_name") or "")
        route_litellm_model = str(route_config.get("route_litellm_model") or "")
        credential_generation = str(provider.get("credential_generation") or "").strip()
        if (
            not route_model_name.startswith(f"candidate/{provider_key}/")
            or route_model_name.count("*") != 1
            or not route_model_name.endswith("*")
            or route_litellm_model.count("*") != 1
            or not route_litellm_model.endswith("*")
        ):
            provider_reasons.append("candidate_preflight_route_invalid")
        if route_model_name:
            expected_routes.add(route_model_name)
        if not credential_generation:
            provider_reasons.append("candidate_preflight_credential_generation_missing")
        matches = [
            entry
            for entry in entries or []
            if isinstance(entry, Mapping) and entry.get("model_name") == route_model_name
        ]
        if len(matches) != 1:
            provider_reasons.append(
                "candidate_preflight_route_missing" if not matches else "candidate_preflight_route_ambiguous"
            )
        else:
            entry = matches[0]
            params = entry.get("litellm_params")
            model_info = entry.get("model_info")
            model_info = model_info if isinstance(model_info, Mapping) else {}
            metadata = model_info.get("metadata")
            if model_info.get("db_model") is True:
                provider_reasons.append("candidate_preflight_route_db_backed")
            if not isinstance(params, Mapping) or params.get("model") != route_litellm_model:
                provider_reasons.append("candidate_preflight_route_model_mismatch")
            if str((params or {}).get("api_base") or "").rstrip("/") != str(provider.get("base_url") or "").rstrip("/"):
                provider_reasons.append("candidate_preflight_route_api_base_mismatch")
            if not isinstance(metadata, Mapping) or (
                metadata.get("provider_key") != provider_key or metadata.get("purpose") != "candidate_preflight"
            ):
                provider_reasons.append("candidate_preflight_route_metadata_mismatch")
        provider_statuses[provider_key] = {
            "status": "ready" if not provider_reasons else "blocked",
            "route_model_name": route_model_name or None,
            "route_litellm_model": route_litellm_model or None,
            "api_base": provider.get("base_url"),
            "api_key_env": provider.get("api_key_env"),
            "credential_generation": credential_generation or None,
            "reasons": provider_reasons,
        }
    if not credential_env or not environ.get(credential_env):
        reasons.append("candidate_preflight_key_missing")
    elif credential_env == master_key_env or environ.get(credential_env) == environ.get(master_key_env):
        reasons.append("candidate_preflight_key_not_dedicated")
    elif candidate_key_lookup_failed:
        reasons.append("candidate_preflight_key_allowlist_unverifiable")
    elif candidate_key_models is None:
        reasons.append("candidate_preflight_key_route_missing")
    elif set(candidate_key_models) != expected_routes:
        reasons.append("candidate_preflight_key_allowlist_mismatch")
    if reasons:
        for status in provider_statuses.values():
            status["status"] = "blocked"
            status["reasons"] = list(dict.fromkeys([*status["reasons"], *reasons]))
    ready_count = sum(status["status"] == "ready" for status in provider_statuses.values())
    return {
        "status": "ready"
        if provider_statuses and ready_count == len(provider_statuses)
        else ("partial" if ready_count else "blocked"),
        "transport": transport or None,
        "credential_source": credential_env or None,
        "reasons": reasons,
        "providers": provider_statuses,
    }


def _resolve_raw_candidate_routes(
    *,
    providers: Sequence[Mapping[str, Any]],
    litellm_snapshot: Mapping[str, Any],
    litellm_base_url: str,
    litellm_key: str,
    client: Any,
    timeout_seconds: float,
) -> dict[str, Mapping[str, Any]]:
    """兼容 v1.93：从展开目录定位 deployment id，再读取原始 wildcard route。"""
    entries = litellm_snapshot.get("data")
    entries = entries if isinstance(entries, list) else []
    resolved: dict[str, Mapping[str, Any]] = {}
    for provider in providers:
        provider_key = _provider_key(provider)
        route_config = provider.get("candidate_preflight")
        if not isinstance(route_config, Mapping):
            continue
        route_name = str(route_config.get("route_model_name") or "")
        direct = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("model_name") == route_name]
        if len(direct) == 1:
            resolved[provider_key] = direct[0]
            continue
        prefix = route_name.removesuffix("*")
        deployment_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            model_info = entry.get("model_info")
            model_info = model_info if isinstance(model_info, Mapping) else {}
            metadata = model_info.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            purpose = str(metadata.get("purpose") or "")
            matches_expanded_route = str(entry.get("model_name") or "").startswith(prefix)
            matches_route_metadata = (
                metadata.get("provider_key") == provider_key and metadata.get("purpose") == "candidate_preflight"
            )
            if matches_route_metadata or (matches_expanded_route and not purpose.startswith("candidate_")):
                deployment_ids.add(str(model_info.get("id") or ""))
        deployment_ids.discard("")
        raw_matches: list[Mapping[str, Any]] = []
        for deployment_id in sorted(deployment_ids):
            try:
                payload = _read_json_response(
                    client.get(
                        f"{litellm_base_url}/model/info?{urlencode({'litellm_model_id': deployment_id})}",
                        headers={"Authorization": f"Bearer {litellm_key}"},
                        timeout=timeout_seconds,
                    )
                )
            except Exception:
                continue
            raw_entries = payload.get("data") if isinstance(payload, Mapping) else None
            raw_matches.extend(
                entry
                for entry in raw_entries or []
                if isinstance(entry, Mapping) and entry.get("model_name") == route_name
            )
        if len(raw_matches) == 1:
            resolved[provider_key] = raw_matches[0]
    return resolved


def coordinate_candidates(
    *,
    registry: Mapping[str, Any],
    environ: Mapping[str, str],
    client: Any = httpx,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """读取 LiteLLM 和多个厂商目录，返回完全只读的候选报告。"""
    providers = registry.get("providers")
    if not isinstance(providers, list):
        raise ValueError("registry.providers 必须是列表")
    litellm = registry.get("litellm")
    if not isinstance(litellm, Mapping):
        raise ValueError("registry.litellm 必须是对象")

    litellm_key_env = str(litellm.get("api_key_env") or "")
    litellm_key = environ.get(litellm_key_env, "")
    results: dict[str, Any] = {}
    if not litellm_key:
        for provider in providers:
            if isinstance(provider, Mapping):
                key = _provider_key(provider)
                results[key] = _error_result(key, "missing_litellm_api_key")
        return _report(results, candidate_preflight=None)

    litellm_base_url = str(litellm.get("base_url") or "").rstrip("/")
    try:
        litellm_snapshot = _read_json_response(
            client.get(
                f"{litellm_base_url}/model/info",
                headers={"Authorization": f"Bearer {litellm_key}"},
                timeout=timeout_seconds,
            )
        )
    except Exception:
        for provider in providers:
            if isinstance(provider, Mapping):
                key = _provider_key(provider)
                results[key] = _error_result(key, "litellm_request_failed")
        return _report(results, candidate_preflight=None)

    provider_configs = [provider for provider in providers if isinstance(provider, Mapping)]
    raw_routes = _resolve_raw_candidate_routes(
        providers=provider_configs,
        litellm_snapshot=litellm_snapshot,
        litellm_base_url=litellm_base_url,
        litellm_key=litellm_key,
        client=client,
        timeout_seconds=timeout_seconds,
    )
    raw_snapshot = {"data": list(raw_routes.values())}

    preflight_config = litellm.get("candidate_preflight")
    candidate_key_env = str(preflight_config.get("api_key_env") or "") if isinstance(preflight_config, Mapping) else ""
    candidate_key = environ.get(candidate_key_env, "")
    candidate_key_models: list[str] | None = None
    candidate_key_lookup_failed = False
    if candidate_key and candidate_key_env != litellm_key_env and candidate_key != litellm_key:
        try:
            key_payload = _read_json_response(
                client.get(
                    f"{litellm_base_url}/key/info?{urlencode({'key': candidate_key})}",
                    headers={"Authorization": f"Bearer {litellm_key}"},
                    timeout=timeout_seconds,
                )
            )
            key_info = key_payload.get("info") if isinstance(key_payload, Mapping) else None
            key_models = key_info.get("models") if isinstance(key_info, Mapping) else None
            if not isinstance(key_models, list) or not all(isinstance(model, str) for model in key_models):
                raise ValueError("candidate key allowlist 无效")
            candidate_key_models = list(key_models)
        except Exception:
            candidate_key_lookup_failed = True

    candidate_preflight = _candidate_preflight_status(
        litellm_config=litellm,
        providers=provider_configs,
        litellm_snapshot=raw_snapshot,
        candidate_key_models=candidate_key_models,
        candidate_key_lookup_failed=candidate_key_lookup_failed,
        environ=environ,
    )

    for provider in providers:
        if not isinstance(provider, Mapping):
            continue
        key = _provider_key(provider)
        api_key_env = str(provider.get("api_key_env") or "")
        api_key = environ.get(api_key_env, "")
        if not api_key:
            results[key] = _error_result(key, "missing_api_key")
            continue
        try:
            adapter = _build_adapter(provider)
            upstream = _read_json_response(
                client.get(
                    _models_url(str(provider.get("base_url") or "")),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout_seconds,
                )
            )
            raw_models = upstream.get("data") if isinstance(upstream, Mapping) else upstream
            if not isinstance(raw_models, list) or not raw_models:
                results[key] = _error_result(key, "upstream_empty")
                continue
            candidate_report = discover_candidates(
                adapter=adapter,
                upstream_snapshot=upstream,
                litellm_snapshot=litellm_snapshot,
            )
            results[key] = {"status": "ok", "error": None, "report": serialize_report(candidate_report)}
        except Exception:
            # 错误类型足够用于告警；第三方异常正文可能包含 token，禁止输出。
            results[key] = _error_result(key, "upstream_request_failed")
    return _report(results, candidate_preflight=candidate_preflight)


def _report(
    results: Mapping[str, Any],
    *,
    candidate_preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ok_count = sum(1 for value in results.values() if value.get("status") == "ok")
    return {
        "mode": "read_only",
        "writes_performed": False,
        "write_actions": [],
        "candidate_preflight": dict(candidate_preflight)
        if candidate_preflight is not None
        else {
            "status": "blocked",
            "transport": None,
            "credential_source": None,
            "reasons": ["litellm_snapshot_unavailable"],
            "providers": {},
        },
        "summary": {
            "providers_total": len(results),
            "providers_ok": ok_count,
            "providers_failed": len(results) - ok_count,
        },
        "providers": dict(results),
    }


def write_report_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    report = coordinate_candidates(
        registry=registry,
        environ=os.environ,
        timeout_seconds=args.timeout_seconds,
    )
    write_report_atomic(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if report["summary"]["providers_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
