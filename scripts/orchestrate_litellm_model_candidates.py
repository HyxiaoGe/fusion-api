"""只读协调多个厂商的 LiteLLM 模型候选发现。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        return _report(results)

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
        return _report(results)

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
    return _report(results)


def _report(results: Mapping[str, Any]) -> dict[str, Any]:
    ok_count = sum(1 for value in results.values() if value.get("status") == "ok")
    return {
        "mode": "read_only",
        "writes_performed": False,
        "write_actions": [],
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
