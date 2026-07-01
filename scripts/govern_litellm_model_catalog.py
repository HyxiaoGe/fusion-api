"""治理 LiteLLM 中 Fusion 暴露给前端的业务模型目录。

默认只做 dry-run，显式传 `--apply` 才会调用 LiteLLM 管理 API。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import httpx

LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
XIAOMI_API_KEY_ENV = "XIAOMI_API_KEY"
LITELLM_VIRTUAL_KEY_ENV = "LITELLM_VIRTUAL_KEY"
XIAOMI_API_BASE = "https://api.xiaomimimo.com/v1"

DEPRECATED_MODELS = {"mimo-v2-flash", "mimo-v2-pro"}

TARGET_MODELS: tuple[dict[str, Any], ...] = (
    {
        "model_name": "mimo-v2.5-pro",
        "display_name": "MiMo V2.5 Pro",
        "description": "小米 MiMo V2.5 旗舰文本模型，适合 Agent、代码和长任务场景",
        "cost_tier": "mid",
        "pricing": {"input": 1.0, "output": 3.0, "unit": "USD"},
        "knowledge_cutoff": "2026-05",
        "recommended_for": ["agent", "coding", "long_context"],
    },
    {
        "model_name": "mimo-v2.5-pro-ultraspeed",
        "display_name": "MiMo V2.5 Pro UltraSpeed",
        "description": "小米 MiMo V2.5 Pro 高速版，适合低延迟对话和 Agent 调用",
        "cost_tier": "mid",
        "pricing": {"input": 1.0, "output": 3.0, "unit": "USD"},
        "knowledge_cutoff": "2026-05",
        "recommended_for": ["fast_response", "agent"],
    },
)

XIAOMI_TEXT_CAPABILITIES = {
    "imageGen": False,
    "deepThinking": True,
    "fileSupport": False,
    "functionCalling": True,
    "vision": False,
    "webSearch": False,
}


@dataclass(frozen=True)
class CatalogAction:
    action: str
    model_name: str
    reason: str
    model_uuid: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class CatalogPlan:
    actions: list[CatalogAction] = field(default_factory=list)

    @property
    def has_writes(self) -> bool:
        return any(action.action in {"create", "delete"} for action in self.actions)


def _entry_model_uuid(entry: Mapping[str, Any]) -> str | None:
    model_info = entry.get("model_info") or {}
    value = model_info.get("id")
    return str(value) if value else None


def _build_xiaomi_payload(target: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    model_name = str(target["model_name"])
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/{model_name}",
            "api_base": XIAOMI_API_BASE,
            "api_key": api_key,
        },
        "model_info": {
            "metadata": {
                "display_name": target["display_name"],
                "description": target["description"],
                "provider_key": "xiaomi",
                "provider_display": "小米 MiMo",
                "cost_tier": target["cost_tier"],
                "capabilities": dict(XIAOMI_TEXT_CAPABILITIES),
                "pricing": target["pricing"],
                "knowledge_cutoff": target["knowledge_cutoff"],
                "recommended_for": list(target["recommended_for"]),
                "source": "fusion-governance",
            }
        },
    }


def build_governance_plan(entries: Sequence[Mapping[str, Any]], env: Mapping[str, str]) -> CatalogPlan:
    existing_by_name = {str(entry.get("model_name")): entry for entry in entries if entry.get("model_name")}
    actions: list[CatalogAction] = []

    for model_name in sorted(DEPRECATED_MODELS):
        entry = existing_by_name.get(model_name)
        if not entry:
            continue
        model_uuid = _entry_model_uuid(entry)
        if not model_uuid:
            actions.append(
                CatalogAction(
                    action="skip",
                    model_name=model_name,
                    reason="旧小米模型缺少 LiteLLM model UUID，无法安全删除",
                )
            )
            continue
        actions.append(
            CatalogAction(
                action="delete",
                model_name=model_name,
                model_uuid=model_uuid,
                reason="小米 V2 系列已退役，从 Fusion 可选模型目录移除",
            )
        )

    missing_targets = [target for target in TARGET_MODELS if target["model_name"] not in existing_by_name]
    if missing_targets and not env.get(XIAOMI_API_KEY_ENV):
        raise RuntimeError(f"缺少 {XIAOMI_API_KEY_ENV}，无法注册小米 V2.5 模型")

    api_key = env.get(XIAOMI_API_KEY_ENV, "")
    for target in missing_targets:
        model_name = str(target["model_name"])
        actions.append(
            CatalogAction(
                action="create",
                model_name=model_name,
                reason="注册小米 MiMo V2.5 文本模型到 Fusion 可选模型目录",
                payload=_build_xiaomi_payload(target, api_key),
            )
        )

    return CatalogPlan(actions=actions)


def replace_deprecated_models_in_allowlist(models: Sequence[str]) -> list[str]:
    updated: list[str] = []
    seen: set[str] = set()
    target_names = [str(target["model_name"]) for target in TARGET_MODELS]

    for model in models:
        if model in DEPRECATED_MODELS or model in target_names or model in seen:
            continue
        updated.append(model)
        seen.add(model)

    for model in target_names:
        if model not in seen:
            updated.append(model)
            seen.add(model)

    return updated


def fetch_existing_models(base_url: str, master_key: str) -> list[dict[str, Any]]:
    response = httpx.get(
        f"{base_url.rstrip('/')}/model/info",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=15.0,
    )
    response.raise_for_status()
    return list(response.json().get("data", []))


def fetch_key_models(base_url: str, master_key: str, virtual_key: str) -> list[str]:
    query = urlencode({"key": virtual_key})
    response = httpx.get(
        f"{base_url.rstrip('/')}/key/info?{query}",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    info = payload.get("info") or payload
    return list(info.get("models") or [])


def update_key_models(base_url: str, master_key: str, virtual_key: str, models: Sequence[str]) -> None:
    response = httpx.post(
        f"{base_url.rstrip('/')}/key/update",
        headers={
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        },
        json={"key": virtual_key, "models": list(models)},
        timeout=15.0,
    )
    response.raise_for_status()


def apply_plan(base_url: str, master_key: str, plan: CatalogPlan) -> None:
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
    }
    root = base_url.rstrip("/")
    with httpx.Client(timeout=15.0) as client:
        for action in plan.actions:
            if action.action == "delete":
                response = client.post(
                    f"{root}/model/delete",
                    headers=headers,
                    json={"id": action.model_uuid},
                )
                response.raise_for_status()
            elif action.action == "create":
                response = client.post(
                    f"{root}/model/new",
                    headers=headers,
                    json=action.payload,
                )
                response.raise_for_status()


def _redact_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    redacted = copy.deepcopy(payload)
    litellm_params = redacted.get("litellm_params")
    if isinstance(litellm_params, dict) and litellm_params.get("api_key"):
        litellm_params["api_key"] = "***"
    return redacted


def serialize_action(action: CatalogAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "model_name": action.model_name,
        "reason": action.reason,
        "model_uuid": action.model_uuid,
        "payload": _redact_payload(action.payload),
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="治理 Fusion LiteLLM 模型目录")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只输出计划，不写入 LiteLLM（默认）")
    mode.add_argument("--apply", action="store_true", help="执行模型删除/注册")
    parser.add_argument("--base-url", default=os.environ.get(LITELLM_BASE_URL_ENV, "http://localhost:4000"))
    parser.add_argument("--master-key", default=os.environ.get(LITELLM_MASTER_KEY_ENV, ""))
    parser.add_argument(
        "--virtual-key",
        default=os.environ.get(LITELLM_VIRTUAL_KEY_ENV, ""),
        help="需要同步模型白名单的 LiteLLM virtual key；为空则只治理模型目录",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if not args.master_key:
        raise RuntimeError(f"缺少 {LITELLM_MASTER_KEY_ENV}")

    entries = fetch_existing_models(args.base_url, args.master_key)
    plan = build_governance_plan(entries, os.environ)
    print(json.dumps([serialize_action(action) for action in plan.actions], ensure_ascii=False, indent=2))

    key_models: list[str] | None = None
    updated_key_models: list[str] | None = None
    if args.virtual_key:
        key_models = fetch_key_models(args.base_url, args.master_key, args.virtual_key)
        updated_key_models = replace_deprecated_models_in_allowlist(key_models)
        print(
            json.dumps(
                {
                    "action": "update_key_models",
                    "removed": [model for model in key_models if model in DEPRECATED_MODELS],
                    "added": [model for model in updated_key_models if model not in key_models],
                    "models_count_before": len(key_models),
                    "models_count_after": len(updated_key_models),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.apply:
        apply_plan(args.base_url, args.master_key, plan)
        if args.virtual_key and updated_key_models is not None:
            update_key_models(args.base_url, args.master_key, args.virtual_key, updated_key_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
