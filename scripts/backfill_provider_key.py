"""一次性脚本：给 LiteLLM 里 source=fusion-migration 的模型补 metadata.provider_key。

迁移脚本一开始没写 provider_key，导致前端拿到中文 provider key（"通义千问" 等）。
此脚本读 LiteLLM `/model/info`，按 underlying / api_base 推断 provider_key
（qwen/doubao/xiaomi/deepseek/...），调 `/model/update` 写回。

跑完之后 metadata 才符合 _normalize_provider_key 的期待。

用法：
    LITELLM_BASE_URL=http://localhost:4000 \
    LITELLM_MASTER_KEY=sk-litellm-master-... \
    python scripts/backfill_provider_key.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
LITELLM_MASTER_KEY = os.environ["LITELLM_MASTER_KEY"]
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# api_base 子串 → provider_key（openai 兼容路由的反查表）
_API_BASE_TO_KEY = [
    ("dashscope.aliyuncs.com", "qwen"),
    ("ark.cn-beijing.volces.com", "doubao"),
    ("api.xiaomimimo.com", "xiaomi"),
]

# openrouter 中段的 vendor 名 → fusion 里的 provider_id（旧前端用这个 key 分组）
_OPENROUTER_VENDOR_NORMALIZE = {
    "x-ai": "xai",
}


def _infer_provider_key(underlying: str, api_base: str) -> str | None:
    """按 underlying / api_base 推断稳定 ASCII provider key。

    优先级：
    1. api_base 命中 dashscope/ark/xiaomimimo → 对应自建 provider
    2. underlying 形如 openrouter/{vendor}/{id} → vendor（x-ai → xai 归一化）
    3. underlying 普通前缀（deepseek/minimax/moonshot 等）直接取前缀
    """
    if api_base:
        for needle, key in _API_BASE_TO_KEY:
            if needle in api_base:
                return key
    if underlying and "/" in underlying:
        parts = underlying.split("/")
        prefix = parts[0].lower()
        # openrouter/{vendor}/{id} 形式——真实 provider 是中段，不能用 "openrouter"
        if prefix == "openrouter" and len(parts) >= 3:
            vendor = parts[1].lower()
            return _OPENROUTER_VENDOR_NORMALIZE.get(vendor, vendor)
        # openai/ 前缀的 wildcard 模型必须靠 api_base，前缀给不了真实 provider
        if prefix != "openai":
            return prefix
    return None


def main() -> int:
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            f"{LITELLM_BASE_URL}/model/info",
            headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
        )
        resp.raise_for_status()
        entries = resp.json().get("data", [])

        patched = 0
        skipped = 0
        failed = 0
        for entry in entries:
            info = entry.get("model_info") or {}
            metadata = info.get("metadata") or {}
            if metadata.get("source") != "fusion-migration":
                continue
            if metadata.get("provider_key"):
                skipped += 1
                continue

            model_uuid = info.get("id")
            model_name = entry.get("model_name")
            litellm_params = entry.get("litellm_params") or {}
            underlying = litellm_params.get("model") or ""
            api_base = litellm_params.get("api_base") or ""

            provider_key = _infer_provider_key(underlying, api_base)
            if not provider_key:
                print(f"  [skip-noinfer] {model_name}: underlying={underlying} api_base={api_base}")
                skipped += 1
                continue

            updated_metadata = {**metadata, "provider_key": provider_key}
            # /model/update 要求带上 litellm_params（LiteLLM 校验它存在），
            # 否则报 "Authentication Error, litellm_params not provided"
            payload = {
                "model_id": model_uuid,
                "litellm_params": litellm_params,
                "model_info": {**info, "metadata": updated_metadata},
            }
            if DRY_RUN:
                print(f"  [dry-run] {model_name} → provider_key={provider_key}")
                patched += 1
                continue

            update_resp = client.post(
                f"{LITELLM_BASE_URL}/model/update",
                headers={
                    "Authorization": f"Bearer {LITELLM_MASTER_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if update_resp.status_code in (200, 201):
                print(f"  [ok] {model_name} → provider_key={provider_key}")
                patched += 1
            else:
                print(f"  [FAIL] {model_name}: HTTP {update_resp.status_code} {update_resp.text[:200]}")
                failed += 1

    print()
    print(f"完成：补 {patched}，跳过 {skipped}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
