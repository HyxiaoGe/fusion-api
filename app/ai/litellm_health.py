"""LiteLLM `/health` 探测的后台缓存层。

为什么单独搞一个模块：
- `/health` 会真的对每个 model 打一次 completion，单次 5~30s 不止；不能放
  在 `/api/models` 请求路径上同步触发。
- 但 fusion-ui 又需要知道 "OpenAI/Anthropic 这条目前是不是真的能调用"，
  让选择器把不可用的项目灰显出来。

设计：
- 启动后跑一个后台 asyncio 任务，固定间隔（默认 5min）拉一次 `/health`
- 把结果按 LiteLLM 内部 `model_id` (UUID) 索引，并通过 `/model/info` 的
  UUID → alias 映射，反推每个 fusion 别名当前是 healthy / unhealthy / unknown
- 首次未拉到时，所有别名返回 status="unknown"，FE 当 healthy 处理（不要
  在还没探测出来时就误报"全挂了"）
- 探测失败（网络问题、proxy 重启）不会清空上一次的结果——保留 stale 数据
  比突然全清空更稳

对外接口：get_health(alias) -> {status, error, checked_at}。
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.logger import app_logger as logger

_LITELLM_BASE_URL = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000").rstrip("/")
_LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")

# 探测间隔。默认 5min；可被 LITELLM_HEALTH_INTERVAL_SECONDS 覆盖。
_REFRESH_INTERVAL_SECONDS = float(os.environ.get("LITELLM_HEALTH_INTERVAL_SECONDS", "300"))
# 单次 `/health` 调用超时——LiteLLM 会并发探测所有端点，但慢的 provider 可能拖到 1min+
_HEALTH_REQUEST_TIMEOUT = float(os.environ.get("LITELLM_HEALTH_REQUEST_TIMEOUT", "90"))

_lock = threading.Lock()
# alias -> {"status": "healthy"|"unhealthy", "error": str|None}
_by_alias: Dict[str, Dict[str, Any]] = {}
_last_checked_at: float = 0.0
_refresh_task: Optional[asyncio.Task] = None


def _build_alias_index(model_info: List[Dict[str, Any]]) -> Dict[str, str]:
    """从 /model/info 的 data 列表里抽 alias → model_id (UUID) 映射。

    LiteLLM 给每条 model 配置都会分配一个 UUID（model_info.id），它也是
    `/health` 返回 entries 里的 `model_id` 字段。两个端点拿不同字段，必须
    自己拼起来。
    """
    index: Dict[str, str] = {}
    for entry in model_info:
        alias = entry.get("model_name")
        uuid = (entry.get("model_info") or {}).get("id")
        if alias and uuid:
            # 同一 alias 可能有多条配置（fallback / 多 region），后写优先；这里
            # 反向取，方便用 alias 查任一条 UUID 的健康
            index[alias] = uuid
    return index


def _classify_error(raw_error: str) -> str:
    """把 LiteLLM 抛出的 stack trace 翻成给用户看的中文一句话。

    分类思路：先看异常类型（AuthenticationError / NotFoundError / BadRequest），
    再看消息体里的关键词（invalid api key / not activated / Terms Of Service /
    only support stream / quota / rate limit / timeout）。识别不到时 fallback
    到 "服务商暂时不可用" — 反正用户能看出 unhealthy，原始 trace 已经在
    fusion-api 日志里，要排查走那边。
    """
    if not raw_error:
        return "服务商暂时不可用"

    head = raw_error.split("\n", 1)[0]
    lower = head.lower()

    # 1) 认证失败：401 / invalid api key / Invalid Authentication / authorized_error
    if (
        "authenticationerror" in lower
        or "invalid api key" in lower
        or "invalid authentication" in lower
        or "authorized_error" in lower
        or '"http_code":"401"' in head
        or " 401 " in head
    ):
        return "服务商认证失败：API key 无效或已过期，请联系管理员补全密钥"

    # 2) 账号未开通模型 / Doubao 的 "has not activated the model"
    if "has not activated the model" in head or "activate the model service" in lower:
        return "服务商账号未开通此模型，请到服务商控制台启用后再用"

    # 3) 服务商策略拒绝 / OpenRouter ToS
    if (
        "terms of service" in lower
        or "prohibited" in lower
        or '"code":403' in head
        or " 403 " in head
    ):
        return "请求被服务商拒绝（额度/合规策略），暂不可用"

    # 4) 调用参数不兼容
    if "only support stream" in lower or "stream parameter" in lower:
        return "调用参数不兼容（此模型仅支持流式调用），已在排查"
    if "model_not_found" in lower or "does not exist" in head.lower() or "permission denied" in lower:
        return "模型不存在或当前账号无权访问"

    # 5) 额度/限流
    if (
        "rate limit" in lower
        or "ratelimit" in lower
        or "quota" in lower
        or "insufficient" in lower
        or " 429 " in head
    ):
        return "服务商额度不足或被限流，稍后再试"

    # 6) 网络/超时
    if "timeout" in lower or "connectionerror" in lower:
        return "连接服务商超时，稍后再试"

    return "服务商暂时不可用"


async def _fetch_once() -> None:
    """跑一次完整的探测，更新 _by_alias。失败时保留旧数据。"""
    headers = {"Authorization": f"Bearer {_LITELLM_API_KEY}"} if _LITELLM_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_REQUEST_TIMEOUT, headers=headers) as client:
            # /model/info 拿 alias → UUID 映射，/health 拿 UUID → 健康
            info_resp, health_resp = await asyncio.gather(
                client.get(f"{_LITELLM_BASE_URL}/model/info"),
                client.get(f"{_LITELLM_BASE_URL}/health"),
            )
            info_resp.raise_for_status()
            health_resp.raise_for_status()
            info_data = info_resp.json().get("data", []) or []
            health_data = health_resp.json() or {}
    except Exception as exc:
        logger.warning(f"litellm_health: probe failed (keeping stale data): {exc}")
        return

    alias_to_uuid = _build_alias_index(info_data)
    healthy_uuids = {e.get("model_id") for e in (health_data.get("healthy_endpoints") or []) if e.get("model_id")}
    unhealthy_by_uuid: Dict[str, str] = {}
    for e in (health_data.get("unhealthy_endpoints") or []):
        uuid = e.get("model_id")
        if uuid:
            unhealthy_by_uuid[uuid] = _classify_error(e.get("error") or "")

    new_state: Dict[str, Dict[str, Any]] = {}
    for alias, uuid in alias_to_uuid.items():
        if uuid in healthy_uuids:
            new_state[alias] = {"status": "healthy", "error": None}
        elif uuid in unhealthy_by_uuid:
            new_state[alias] = {"status": "unhealthy", "error": unhealthy_by_uuid[uuid] or "探测失败"}
        # uuid 既不在 healthy 也不在 unhealthy（极少见，可能是探测中或被跳过）：
        # 不写入 new_state，下面 get_health 兜底返回 "unknown"

    with _lock:
        global _last_checked_at
        _by_alias.clear()
        _by_alias.update(new_state)
        _last_checked_at = time.time()
    logger.info(
        f"litellm_health: probe done, healthy={sum(1 for v in new_state.values() if v['status'] == 'healthy')}, "
        f"unhealthy={sum(1 for v in new_state.values() if v['status'] == 'unhealthy')}"
    )


async def _refresh_loop() -> None:
    """后台任务循环：启动后立即跑一次，然后按间隔重复。"""
    try:
        while True:
            await _fetch_once()
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        logger.info("litellm_health: refresh loop cancelled")
        raise


async def start() -> None:
    """在 lifespan startup 阶段调用。"""
    global _refresh_task
    if _refresh_task is None or _refresh_task.done():
        _refresh_task = asyncio.create_task(_refresh_loop(), name="litellm_health_refresh")
        logger.info(
            f"litellm_health: background refresh started, interval={_REFRESH_INTERVAL_SECONDS}s"
        )


async def stop() -> None:
    """在 lifespan shutdown 阶段调用。"""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
        _refresh_task = None


def get_health(alias: str) -> Dict[str, Any]:
    """返回某个 alias 的当前健康。未探测过的返回 status=unknown。"""
    with _lock:
        entry = _by_alias.get(alias)
        if entry is None:
            return {"status": "unknown", "error": None, "checked_at": _last_checked_at or None}
        return {**entry, "checked_at": _last_checked_at or None}
