"""LiteLLM 模型目录的薄缓存层。

fusion-api 现在不维护本地 model_sources 表，所有模型元数据都来自
LiteLLM Proxy 的 `/model/info`。chat 流程需要在调用前查 capabilities
（vision / functionCalling 决定是否走多模态/工具）和 provider_id
（决定是否走 reasoning 模式），按需走这里的缓存。

设计：
- TTL 60s 内复用，避免高频请求把 LiteLLM 拖垮
- 拉清单失败时返回空，由调用方决定 fallback（默认能力全 False / provider 用 "litellm"）
- 同步接口（litellm 调用都在 chat_service / stream runner 的同步上下文）
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import httpx
import redis

from app.core.config import settings
from app.core.logger import app_logger as logger

_LITELLM_BASE_URL = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000").rstrip("/")
_LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
_DEFAULT_AGENT_TOOLS_DISABLED_ALIASES = {"qwen-vl-max"}

# 缓存生效时间——LiteLLM 模型变更频次低，60s 足够
_CACHE_TTL_SECONDS = 60.0
_FAILED_FETCH_BACKOFF_SECONDS = float(os.environ.get("LITELLM_CATALOG_FAILURE_BACKOFF_SECONDS", "30"))
_GENERATION_CHECK_INTERVAL_SECONDS = float(os.environ.get("LITELLM_CATALOG_GENERATION_CHECK_INTERVAL_SECONDS", "1"))
_CATALOG_GENERATION_KEY = "fusion:litellm:catalog:generation"

_cache_lock = threading.Lock()
_cache_payload: Optional[Dict[str, Dict[str, Any]]] = None
_cache_loaded_at: float = 0.0
_cache_last_attempt_at: float | None = None
_cache_last_attempt_failed = False
_cache_generation: str | None = None
_last_generation_check_at = 0.0
_generation_redis: redis.Redis | None = None
_generation_error_logged = False
_generation_check_in_flight = False
_generation_check_epoch = 0
_generation_check_thread: threading.Thread | None = None


def _get_generation_redis() -> redis.Redis:
    global _generation_redis
    if _generation_redis is None:
        _generation_redis = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
    return _generation_redis


def _read_catalog_generation() -> str | None:
    global _generation_error_logged
    try:
        value = _get_generation_redis().get(_CATALOG_GENERATION_KEY)
    except Exception:
        if not _generation_error_logged:
            logger.warning("litellm_catalog: Redis generation unavailable")
            _generation_error_logged = True
        return None
    _generation_error_logged = False
    return str(value or "0")


def _sync_distributed_generation(now: float) -> None:
    """后台检查跨进程目录代次，避免同步 Redis I/O 阻塞请求事件循环。"""
    global _generation_check_in_flight, _generation_check_thread, _last_generation_check_at
    with _cache_lock:
        if (
            _generation_check_in_flight
            or now - _last_generation_check_at < _GENERATION_CHECK_INTERVAL_SECONDS
        ):
            return
        _last_generation_check_at = now
        _generation_check_in_flight = True
        epoch = _generation_check_epoch
    thread = threading.Thread(
        target=_refresh_distributed_generation,
        args=(epoch,),
        name="litellm-catalog-generation",
        daemon=True,
    )
    _generation_check_thread = thread
    try:
        thread.start()
    except Exception as exc:
        with _cache_lock:
            _generation_check_in_flight = False
            _generation_check_thread = None
        logger.warning(f"litellm_catalog: generation background check failed to start: {exc}")


def _refresh_distributed_generation(epoch: int) -> None:
    global _cache_generation, _cache_last_attempt_at, _cache_last_attempt_failed
    global _cache_loaded_at, _cache_payload, _generation_check_in_flight
    try:
        generation = _read_catalog_generation()
        if generation is None:
            return
        with _cache_lock:
            if epoch != _generation_check_epoch:
                return
            if _cache_generation is None:
                if _cache_payload is None:
                    _cache_generation = generation
                    return
            elif generation == _cache_generation:
                return
            _cache_payload = None
            _cache_loaded_at = 0.0
            _cache_last_attempt_at = None
            _cache_last_attempt_failed = False
            _cache_generation = generation
    finally:
        with _cache_lock:
            _generation_check_in_flight = False


def _fetch_catalog() -> Dict[str, Dict[str, Any]]:
    """同步拉 LiteLLM `/model/info`，按 model_name (alias) 聚合。"""
    headers = {"Authorization": f"Bearer {_LITELLM_API_KEY}"} if _LITELLM_API_KEY else {}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_LITELLM_BASE_URL}/model/info", headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", [])
    except Exception as exc:  # 网络故障/超时不应让 chat 链路炸
        logger.warning(f"litellm_catalog: fetch /model/info failed: {exc}")
        return {}

    catalog: Dict[str, Dict[str, Any]] = {}
    for entry in data:
        alias = entry.get("model_name")
        if not alias:
            continue
        info = entry.get("model_info") or {}
        metadata = info.get("metadata") or {}
        underlying = (entry.get("litellm_params") or {}).get("model") or ""
        # 只覆盖一次，先到先得：wildcard 路由（"qwen/*"）和具体别名（"qwq-plus-latest"）共存时
        # 后者优先（具体别名会带 metadata）
        if alias in catalog and not metadata:
            continue
        catalog[alias] = {
            "underlying": underlying,
            "metadata": metadata,
            "litellm_provider": info.get("litellm_provider"),
            "max_input_tokens": info.get("max_input_tokens"),
            "max_output_tokens": info.get("max_output_tokens"),
            "db_model": bool(info.get("db_model")),
        }
    return catalog


def _ensure_loaded() -> Dict[str, Dict[str, Any]]:
    """返回当前缓存内容，过期时同步刷新。"""
    global _cache_payload, _cache_loaded_at, _cache_last_attempt_at, _cache_last_attempt_failed
    now = time.monotonic()
    _sync_distributed_generation(now)
    with _cache_lock:
        if _cache_payload is not None and now - _cache_loaded_at < _CACHE_TTL_SECONDS:
            return _cache_payload
        if _cache_last_attempt_at is not None and now - _cache_last_attempt_at < _FAILED_FETCH_BACKOFF_SECONDS:
            return _cache_payload or {}
        # 缓存过期/未加载——拉一次
        _cache_last_attempt_at = now
        payload = _fetch_catalog()
        # 拉空时不覆盖旧缓存（避免短暂网络抖动把 chat 链路打瘸）
        if payload:
            _cache_payload = payload
            _cache_loaded_at = now
            _cache_last_attempt_failed = False
        elif _cache_payload is None:
            _cache_payload = {}
            _cache_last_attempt_failed = True
        else:
            _cache_last_attempt_failed = True
        return _cache_payload


def get_model_entry(alias: str) -> Optional[Dict[str, Any]]:
    """按 LiteLLM alias 查模型元数据。alias 不存在时返回 None。"""
    return _ensure_loaded().get(alias)


def get_cached_model_entry(alias: str) -> tuple[Optional[Dict[str, Any]], str]:
    """只读现有缓存，不触发网络刷新，供请求热路径的旁路观测使用。"""
    if not _cache_lock.acquire(blocking=False):
        return None, "busy"
    try:
        if _cache_payload is None or (_cache_last_attempt_failed and not _cache_payload):
            return None, "unavailable"
        entry = _cache_payload.get(alias)
        if entry is None:
            return None, "missing"
        is_stale = _cache_last_attempt_failed or time.monotonic() - _cache_loaded_at >= _CACHE_TTL_SECONDS
        return dict(entry), "stale" if is_stale else "known"
    finally:
        _cache_lock.release()


def get_capabilities(
    alias: str,
    *,
    agent_tools_disabled_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """读 capabilities (vision/functionCalling/...)，不存在时返回空 dict。"""
    entry = get_model_entry(alias)
    if not entry:
        return {}
    return normalize_capabilities(
        alias,
        entry["metadata"].get("capabilities") or {},
        agent_tools_disabled_aliases=agent_tools_disabled_aliases,
    )


def normalize_capabilities(
    alias: str,
    capabilities: Dict[str, Any] | None,
    *,
    agent_tools_disabled_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    """补齐 Fusion 运行时需要的派生能力位。"""
    normalized = dict(capabilities or {})
    function_calling = bool(normalized.get("functionCalling", False))
    normalized["functionCalling"] = function_calling
    normalized["vision"] = bool(normalized.get("vision", False))
    normalized["deepThinking"] = bool(normalized.get("deepThinking", False))
    normalized["fileSupport"] = bool(normalized.get("fileSupport", False))
    normalized["imageGen"] = bool(normalized.get("imageGen", False))

    if "agentTools" in normalized:
        requested_agent_tools = bool(normalized.get("agentTools"))
    else:
        disabled_aliases = set(agent_tools_disabled_aliases or _DEFAULT_AGENT_TOOLS_DISABLED_ALIASES)
        requested_agent_tools = alias not in disabled_aliases

    agent_tools = function_calling and requested_agent_tools
    normalized["agentTools"] = agent_tools
    # searchCapable 是 Fusion 产品语义：后端会实际下发 web_search/url_read 工具。
    normalized["searchCapable"] = agent_tools
    # webSearch 保留给旧前端/脚本，但不再直接信任 metadata 旧字段。
    normalized["webSearch"] = agent_tools
    return normalized


def get_underlying_provider(alias: str, fallback: str = "litellm") -> str:
    """提取底层 provider key（"deepseek" / "qwen" / "openrouter" 等），
    供 stream runner 判断是否启用 reasoning 模式。

    优先用 LiteLLM 自身的 litellm_provider 字段，没有就从 litellm_params.model 取前缀。
    """
    entry = get_model_entry(alias)
    if not entry:
        return fallback
    provider = entry.get("litellm_provider")
    if provider:
        return str(provider)
    underlying: str = entry.get("underlying") or ""
    if "/" in underlying:
        return underlying.split("/", 1)[0]
    return fallback


def list_aliases() -> Dict[str, Dict[str, Any]]:
    """返回所有已缓存的模型条目（供 /api/models 端点使用）。"""
    return dict(_ensure_loaded())


def get_cache_status() -> dict[str, Any]:
    """返回不含错误正文的目录可用状态。"""
    with _cache_lock:
        return {
            "availability": "degraded" if _cache_last_attempt_failed else "available",
            "has_cache": bool(_cache_payload),
        }


def _clear_local_cache(*, generation: str | None = None) -> None:
    global _cache_payload, _cache_loaded_at, _cache_last_attempt_at, _cache_last_attempt_failed, _cache_generation
    global _generation_check_epoch
    with _cache_lock:
        _generation_check_epoch += 1
        _cache_payload = None
        _cache_loaded_at = 0.0
        _cache_last_attempt_at = None
        _cache_last_attempt_failed = False
        if generation is not None:
            _cache_generation = generation


def invalidate() -> None:
    """主动清除当前进程缓存；不推进跨进程目录代次。"""
    global _cache_generation, _last_generation_check_at, _generation_error_logged
    _clear_local_cache()
    _cache_generation = None
    _last_generation_check_at = 0.0
    _generation_error_logged = False


def bump_generation() -> str:
    """推进 Redis 目录代次并立即清除当前进程缓存。"""
    generation = str(_get_generation_redis().incr(_CATALOG_GENERATION_KEY))
    _clear_local_cache(generation=generation)
    return generation
