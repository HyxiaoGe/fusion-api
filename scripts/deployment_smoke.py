#!/usr/bin/env python3
"""部署后 smoke：校验 API 基础可用面，不触发真实 LLM 调用。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8002"


class DeploymentSmokeError(RuntimeError):
    """部署 smoke 失败。"""


def build_url(base_url: str, path: str) -> str:
    normalized_base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{normalized_base}{normalized_path}"


def fetch_json(url: str, timeout: float = 5) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status < 200 or response.status >= 300:
                raise DeploymentSmokeError(f"HTTP {response.status} from {url}: {body[:500]}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise DeploymentSmokeError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise DeploymentSmokeError(f"request failed for {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DeploymentSmokeError(f"invalid JSON from {url}: {body[:500]}") from exc

    if not isinstance(payload, dict):
        raise DeploymentSmokeError(f"expected JSON object from {url}: {body[:500]}")
    return payload


def validate_health_payload(payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if status != "healthy":
        raise DeploymentSmokeError(f"health status is not healthy: {status!r}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentSmokeError(f"{label} must be an object")
    return value


def _require_non_empty_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise DeploymentSmokeError(f"{label} must be a non-empty list")
    return value


def validate_models_payload(payload: dict[str, Any]) -> None:
    code = payload.get("code")
    if code is not None and code != "SUCCESS":
        raise DeploymentSmokeError(f"/api/models/ code is not SUCCESS: {code!r}")

    data = _require_mapping(payload.get("data"), "/api/models/ data")
    models = _require_non_empty_list(data.get("models"), "/api/models/ data.models")
    providers = _require_non_empty_list(data.get("providers"), "/api/models/ data.providers")

    for index, provider in enumerate(providers):
        provider_data = _require_mapping(provider, f"providers[{index}]")
        for key in ("id", "name"):
            if not provider_data.get(key):
                raise DeploymentSmokeError(f"providers[{index}].{key} is required")

    for index, model in enumerate(models):
        model_data = _require_mapping(model, f"models[{index}]")
        for key in ("modelId", "name", "provider", "enabled"):
            if key not in model_data or model_data.get(key) in ("", None):
                raise DeploymentSmokeError(f"models[{index}].{key} is required")

        capabilities = model_data.get("capabilities")
        if not isinstance(capabilities, dict):
            raise DeploymentSmokeError(f"models[{index}].capabilities must be an object")


def run_smoke(base_url: str) -> dict[str, int | str]:
    health_payload = fetch_json(build_url(base_url, "/health"))
    validate_health_payload(health_payload)

    models_payload = fetch_json(build_url(base_url, "/api/models/"))
    validate_models_payload(models_payload)
    data = models_payload["data"]

    return {
        "health": str(health_payload["status"]),
        "models": len(data["models"]),
        "providers": len(data["providers"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fusion-api deployment smoke")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_BASE_URL", DEFAULT_BASE_URL),
        help="部署后的 API 地址，例如 http://127.0.0.1:8002",
    )
    args = parser.parse_args(argv)

    try:
        result = run_smoke(args.base_url)
    except DeploymentSmokeError as exc:
        print(f"deployment smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        "deployment smoke ok: "
        f"health={result['health']} models={result['models']} providers={result['providers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
