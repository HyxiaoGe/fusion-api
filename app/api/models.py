"""模型目录 API（薄代理 LiteLLM `/model/info`）。

设计：fusion-api 不再维护本地 model_sources / providers 表，
所有模型清单 + 元数据都来自 LiteLLM Proxy。本路由把 LiteLLM 的
`/model/info` 转成前端选择器需要的形状（保留 `modelId` / `capabilities`
/ `provider` 等老字段，兼容前端旧代码）。

只读端点，CRUD 已删——增删改模型直接到 LiteLLM Proxy 后台（或重跑
`scripts/migrate_models_to_litellm.py`）。
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from app.ai import litellm_catalog, litellm_health
from app.schemas.response import success
from app.services.agent_strategy_config import get_agent_tools_disabled_aliases
from app.services.model_presentation import build_model_capability_presentation

router = APIRouter()

# cost_tier 用于排序：low > mid > high
_COST_TIER_ORDER = {"low": 0, "mid": 1, "high": 2}


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_provider_key(metadata: Dict[str, Any], underlying: str) -> str:
    """归一化 provider key（稳定 ASCII，给前端做分组用）。

    优先级：metadata.provider_key（迁移脚本写入的稳定 id）→ 底层 model 前缀
    （deepseek/openai/gemini/...）→ 兜底 "litellm"。绝不用 provider_display
    做 key，那里面可能是中文显示名（"通义千问" 等），前端代码炸。
    """
    pk = (metadata or {}).get("provider_key")
    if pk:
        return str(pk).strip().lower()
    if underlying and "/" in underlying:
        return underlying.split("/", 1)[0].lower()
    return "litellm"


def _entry_to_card(alias: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """LiteLLM 单条 model_info → 前端模型卡片。"""
    metadata = entry.get("metadata") or {}
    underlying = entry.get("underlying") or ""
    capabilities = litellm_catalog.normalize_capabilities(
        alias,
        metadata.get("capabilities") or {},
        agent_tools_disabled_aliases=get_agent_tools_disabled_aliases(),
    )
    pricing = metadata.get("pricing") or {}

    provider_key = _normalize_provider_key(metadata, underlying)
    card = {
        "modelId": alias,
        "name": metadata.get("display_name") or alias,
        "provider": provider_key,
        "provider_display": metadata.get("provider_display") or provider_key,
        "knowledgeCutoff": metadata.get("knowledge_cutoff") or None,
        "contextWindowTokens": _positive_int_or_none(entry.get("max_input_tokens")),
        "maxOutputTokens": _positive_int_or_none(entry.get("max_output_tokens")),
        "capabilities": {
            "imageGen": bool(capabilities.get("imageGen", False)),
            "deepThinking": bool(capabilities.get("deepThinking", False)),
            "fileSupport": bool(capabilities.get("fileSupport", False)),
            "functionCalling": bool(capabilities.get("functionCalling", False)),
            "agentTools": bool(capabilities.get("agentTools", False)),
            "searchCapable": bool(capabilities.get("searchCapable", False)),
            "vision": bool(capabilities.get("vision", False)),
            "webSearch": bool(capabilities.get("webSearch", False)),
        },
        "pricing": {
            "input": float(pricing.get("input") or 0),
            "output": float(pricing.get("output") or 0),
            "unit": pricing.get("unit") or "USD",
        },
        "enabled": True,  # LiteLLM 里能查到就算注册成功；可不可调由 health 决定
        # health 由后台轮询 LiteLLM /health 得到，FE 用来决定是否灰显
        "health": litellm_health.get_health(alias),
        "description": metadata.get("description") or "",
        "cost_tier": metadata.get("cost_tier") or "mid",
        "recommended_for": list(metadata.get("recommended_for") or []),
    }
    card["capabilityPresentation"] = build_model_capability_presentation(card)
    return card


def _collect_providers(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 cards 里反向归纳 providers 列表（供前端做筛选下拉）。"""
    seen: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        key = card["provider"]
        if key in seen:
            continue
        seen[key] = {
            "id": key,
            "name": card["provider_display"] or key,
            "order": len(seen) + 1,
            # 状态由 LiteLLM 自己管，fusion 不再追踪 provider health
            "status": "ok",
            "offline_reason": None,
            "offline_message": None,
            "last_failure_at": None,
        }
    return list(seen.values())


@router.get("/")
async def get_models(
    request: Request,
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
    capability: Optional[str] = None,
):
    """返回 LiteLLM 注册的所有模型（前端用，兼容旧字段）。

    Query params:
        provider: 按 provider key 过滤（'qwen' / 'openai' / ...）
        enabled: 兼容旧参数，恒视为 True（LiteLLM 里查到的都算启用）
        capability: 只返回支持指定能力的模型（'vision' / 'functionCalling' / ...）
    """
    catalog = litellm_catalog.list_aliases()
    # 只展示 db_model=true 的别名，避免把 LiteLLM 自身的 wildcard 路由暴露给前端
    cards = [_entry_to_card(alias, entry) for alias, entry in catalog.items() if entry.get("db_model")]

    if provider:
        cards = [c for c in cards if c["provider"] == provider]
    if enabled is False:
        # 没有 disabled 概念了，明确 enabled=false 时返回空
        cards = []
    if capability:
        cards = [c for c in cards if c["capabilities"].get(capability)]

    # 默认按 (cost_tier, modelId) 排序，picker 看着稳定
    cards.sort(key=lambda c: (_COST_TIER_ORDER.get(c["cost_tier"], 5), c["modelId"]))

    return success(
        data={"models": cards, "providers": _collect_providers(cards)},
        request_id=request.state.request_id,
    )


@router.get("/{model_id}")
async def get_model(model_id: str, request: Request):
    """按 alias 查单个模型详情。"""
    entry = litellm_catalog.get_model_entry(model_id)
    if not entry or not entry.get("db_model"):
        from app.schemas.response import ApiException

        raise ApiException.not_found(f"模型 {model_id} 不存在")
    return success(data=_entry_to_card(model_id, entry), request_id=request.state.request_id)
