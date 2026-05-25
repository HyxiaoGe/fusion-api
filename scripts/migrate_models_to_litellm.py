"""把 fusion 数据库里 `model_sources` + `providers` 的启用模型注册到 LiteLLM Proxy。

一次性迁移脚本——执行完之后 fusion-api 改成薄代理 LiteLLM `/model/info`，
本地的 `model_sources` / `providers` / `user_credentials` 三张表会被删掉。

用法：

    DRY_RUN=1 python scripts/migrate_models_to_litellm.py   # 只打印
    python scripts/migrate_models_to_litellm.py             # 真写（增量）
    RESET=1 python scripts/migrate_models_to_litellm.py     # 先删旧 fusion-migration 再写

需要的 env：
    FUSION_DATABASE_URL    fusion 库连接串（psycopg2 风格）
    LITELLM_BASE_URL       LiteLLM Proxy 根 URL，默认 http://localhost:4000
    LITELLM_MASTER_KEY     LiteLLM 的 master key（sk-litellm-master-...）

行为：
- 每个 enabled=true 的 model_sources 行，POST LiteLLM `/model/new`：
  * model_name = 模型 model_id（保持稳定，不破坏前端 selectedModelId）
  * litellm_params.model = `{providers.litellm_prefix}/{model_id}`（命中 wildcard 路由）
  * model_info.metadata = {display_name, description, provider_display, cost_tier,
                           capabilities, pricing, knowledge_cutoff,
                           recommended_for, source}
- RESET 模式：先 `/model/delete` 所有 source=fusion-migration 的别名，再统一重建
- 普通模式：LiteLLM 已存在同名 model_name 时跳过（幂等）

不动 fusion 数据库——只读 + 写 LiteLLM。删表的事在 alembic migration 里做。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
import psycopg2
import psycopg2.extras

DRY_RUN = os.environ.get("DRY_RUN") == "1"
RESET = os.environ.get("RESET") == "1"
FUSION_DB_URL = os.environ["FUSION_DATABASE_URL"]
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
LITELLM_MASTER_KEY = os.environ["LITELLM_MASTER_KEY"]

# cost_tier 启发式分桶：根据 input/output 单价定档
# 单位：USD per 1k tokens（fusion DB 里 pricing 字段就是这个）
_COST_TIER_THRESHOLDS = [
    ("low", 0.5),   # 输入价 <= 0.5/1k tokens
    ("mid", 3.0),   # 输入价 <= 3/1k tokens
    ("high", 1e9),  # 其他
]

# 自定义 OpenAI 兼容 provider：LiteLLM 没有原生 routing 前缀，必须把 underlying
# 改写成 `openai/{model_id}` + 显式 api_base/api_key，否则 router 会拒绝路由。
# (moonshot / minimax 是 LiteLLM 原生 provider 前缀，不在此列。)
_OPENAI_COMPATIBLE_PROVIDERS = {
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
    },
    "doubao": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "DOUBAO_API_KEY",
    },
    "xiaomi": {
        "api_base": "https://api.xiaomimimo.com/v1",
        "api_key_env": "XIAOMI_API_KEY",
    },
}


def _classify_cost_tier(pricing: dict | None) -> str:
    if not pricing:
        return "mid"
    input_price = float(pricing.get("input") or 0)
    for tier, threshold in _COST_TIER_THRESHOLDS:
        if input_price <= threshold:
            return tier
    return "high"


def fetch_models() -> list[dict[str, Any]]:
    """读 fusion 库里 enabled=true 的模型 + provider 信息（含 capabilities/pricing）。"""
    conn = psycopg2.connect(FUSION_DB_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    ms.model_id,
                    ms.name AS model_name_cn,
                    ms.description,
                    ms.capabilities,
                    ms.pricing,
                    ms.knowledge_cutoff,
                    ms.provider AS provider_id,
                    p.name AS provider_name_cn,
                    p.litellm_prefix,
                    p.status AS provider_status
                FROM model_sources ms
                JOIN providers p ON ms.provider = p.id
                WHERE ms.enabled = TRUE
                ORDER BY p.id, ms.model_id
                """
            )
            return list(cur.fetchall())
    finally:
        conn.close()


def fetch_existing(client: httpx.Client) -> list[dict[str, Any]]:
    """拉 LiteLLM 当前已注册的模型（含 db uuid + metadata.source 用于区分 fusion-migration）。"""
    resp = client.get(
        f"{LITELLM_BASE_URL}/model/info",
        headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def delete_fusion_migration_models(client: httpx.Client, existing: list[dict[str, Any]]) -> int:
    """删掉所有 metadata.source=fusion-migration 的别名（清旧版 + 去重）。

    注意：LiteLLM 自身的 /model/info 不一定能 list 出所有 DB 行
    （重复的 model_name 会被它去重），因此 RESET 之后建议直接到 DB
    检查是否存在残留的 fusion-migration 行（model_info->metadata->source）。
    """
    removed = 0
    for entry in existing:
        info = entry.get("model_info") or {}
        metadata = info.get("metadata") or {}
        if metadata.get("source") != "fusion-migration":
            continue
        model_uuid = info.get("id")
        if not model_uuid:
            continue
        if DRY_RUN:
            print(f"  [dry-run delete] {entry['model_name']} ({model_uuid})")
            removed += 1
            continue
        resp = client.post(
            f"{LITELLM_BASE_URL}/model/delete",
            headers={
                "Authorization": f"Bearer {LITELLM_MASTER_KEY}",
                "Content-Type": "application/json",
            },
            json={"id": model_uuid},
            timeout=15.0,
        )
        if resp.status_code in (200, 201):
            print(f"  [del]  {entry['model_name']}")
            removed += 1
        else:
            print(f"  [FAIL delete] {entry['model_name']}: HTTP {resp.status_code} {resp.text[:200]}")
    return removed


def build_payload(row: dict[str, Any]) -> dict[str, Any]:
    """构造 LiteLLM /model/new 请求体（含 capabilities + pricing 等扩展元数据）。

    对 qwen/doubao/xiaomi 这种非 LiteLLM 原生 provider，必须把 underlying
    重写成 openai/{model_id} + 显式 api_base/api_key（否则 LiteLLM 会注册
    成功但 router 拒绝路由，访问时报 'no healthy deployments'）。
    """
    prefix = row["litellm_prefix"]
    capabilities = row.get("capabilities") or {}
    pricing = row.get("pricing") or {}

    if prefix in _OPENAI_COMPATIBLE_PROVIDERS:
        compat = _OPENAI_COMPATIBLE_PROVIDERS[prefix]
        # LiteLLM 的 `os.environ/XXX` 占位只在 YAML 加载时解析；通过 /model/new 注册的
        # 模型存进 DB 时会原样加密占位串，运行时不会再解析，导致 Aliyun/Ark 等返回 401。
        # 所以这里必须在迁移脚本里直接读取脚本运行环境的真实 key 值写过去。
        api_key_value = os.environ.get(compat["api_key_env"])
        if not api_key_value:
            raise RuntimeError(
                f"环境变量 {compat['api_key_env']} 未设置，无法注册 {row['model_id']}（"
                f"provider={prefix}）。请在迁移脚本运行时把对应 key 注入。"
            )
        litellm_params = {
            "model": f"openai/{row['model_id']}",
            "api_base": compat["api_base"],
            "api_key": api_key_value,
        }
    else:
        # 原生 provider（deepseek / openrouter / moonshot / minimax / gemini）
        litellm_params = {"model": f"{prefix}/{row['model_id']}"}

    return {
        "model_name": row["model_id"],
        "litellm_params": litellm_params,
        "model_info": {
            "metadata": {
                "display_name": row["model_name_cn"],
                "description": row["description"] or "",
                # provider_key 是稳定的 ASCII id（"qwen"/"doubao"/...），前端按这个分组
                "provider_key": row["provider_id"],
                # provider_display 是给人看的中文名（"通义千问" 等）
                "provider_display": row["provider_name_cn"],
                "cost_tier": _classify_cost_tier(pricing),
                "capabilities": capabilities,
                "pricing": pricing,
                "knowledge_cutoff": row.get("knowledge_cutoff") or "",
                "recommended_for": [],
                # source 标记便于将来识别 / RESET
                "source": "fusion-migration",
            }
        },
    }


def register_model(client: httpx.Client, payload: dict[str, Any]) -> tuple[bool, str]:
    """调 LiteLLM /model/new。返回 (success, message)。"""
    resp = client.post(
        f"{LITELLM_BASE_URL}/model/new",
        headers={
            "Authorization": f"Bearer {LITELLM_MASTER_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15.0,
    )
    if resp.status_code in (200, 201):
        return True, "ok"
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def main() -> int:
    rows = fetch_models()
    print(f"读到 fusion 启用模型 {len(rows)} 条")

    with httpx.Client() as client:
        existing = fetch_existing(client)
        existing_aliases = {e["model_name"] for e in existing if e.get("model_name")}
        print(f"LiteLLM 已存在的 alias: {len(existing_aliases)}")

        if RESET:
            print("\n--- RESET 模式：先删除所有 source=fusion-migration 的别名 ---")
            removed = delete_fusion_migration_models(client, existing)
            print(f"已删除 {removed} 条 fusion-migration 别名\n")
            # 重新拉一遍，避免拿到已删的旧数据
            if not DRY_RUN:
                existing = fetch_existing(client)
                existing_aliases = {e["model_name"] for e in existing if e.get("model_name")}

        registered = 0
        skipped = 0
        failed = 0
        for row in rows:
            alias = row["model_id"]
            if alias in existing_aliases:
                print(f"  [skip] {alias}（已存在）")
                skipped += 1
                continue

            payload = build_payload(row)
            if DRY_RUN:
                print(f"  [dry-run] {alias} → {payload['litellm_params']['model']}")
                print(f"           metadata: {json.dumps(payload['model_info']['metadata'], ensure_ascii=False)}")
                registered += 1
                continue

            ok, msg = register_model(client, payload)
            if ok:
                print(f"  [ok]   {alias} → {payload['litellm_params']['model']}")
                registered += 1
            else:
                print(f"  [FAIL] {alias}: {msg}")
                failed += 1

    print()
    print(f"完成：注册 {registered}，跳过 {skipped}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
