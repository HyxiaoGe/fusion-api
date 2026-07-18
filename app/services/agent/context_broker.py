"""Agent 等待客户端运行上下文的短期 Redis broker。"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.redis import (
    LUA_SUBMIT_AGENT_CONTEXT,
    agent_context_notify_key,
    agent_context_request_key,
    get_redis_pool,
    stream_meta_key,
)
from app.services.stream_state_service import check_lock_owner

ContextType = Literal["geolocation"]
ContextPurpose = Literal["nearby_search", "route_origin", "route_destination", "local_weather"]
ContextResultStatus = Literal["provided", "denied", "timeout", "unavailable"]
ContextSubmissionOutcome = Literal[
    "accepted",
    "idempotent",
    "conflict",
    "expired",
    "forbidden",
    "not_found",
    "stale",
]

_REQUEST_TTL_GRACE_SECONDS = 60
_NOTIFY_TTL_SECONDS = 60


class Geolocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(ge=0, le=50_000)
    acquired_at: float = Field(ge=0)


class ContextSubmission(BaseModel):
    """提交结果；故意不包含 location，避免 API 回显精确坐标。"""

    outcome: ContextSubmissionOutcome
    request_id: str
    context_type: ContextType = "geolocation"
    status: ContextResultStatus


class ResolvedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    context_type: ContextType = "geolocation"
    status: ContextResultStatus
    location: Geolocation | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PendingContextRequest:
    request_id: str
    context_type: ContextType
    purpose: ContextPurpose
    reason: str
    user_id: str
    conversation_id: str
    message_id: str
    run_id: str
    task_id: str
    expires_at: float


def _canonical_result_json(*, status: str, location: dict[str, Any] | None, reason: str | None) -> str:
    normalized_location = Geolocation.model_validate(location).model_dump(mode="json") if status == "provided" else None
    payload = {
        "status": status,
        "location": normalized_location,
        "reason": reason if status != "provided" else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_resolution(request_id: str, raw_json: str) -> ResolvedContext:
    payload = json.loads(raw_json)
    return ResolvedContext(
        request_id=request_id,
        status=payload.get("status", "unavailable"),
        location=payload.get("location"),
        reason=payload.get("reason"),
    )


async def create_context_request(
    *,
    request_id: str,
    context_type: ContextType,
    purpose: ContextPurpose,
    reason: str,
    user_id: str,
    conversation_id: str,
    message_id: str,
    run_id: str,
    task_id: str,
    expires_at: float,
) -> PendingContextRequest:
    redis = get_redis_pool()
    if redis is None:
        raise RuntimeError("Redis 不可用，无法等待运行上下文")
    pending = PendingContextRequest(
        request_id=request_id,
        context_type=context_type,
        purpose=purpose,
        reason=reason,
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        run_id=run_id,
        task_id=task_id,
        expires_at=expires_at,
    )
    ttl = max(1, math.ceil(expires_at - time.time()) + _REQUEST_TTL_GRACE_SECONDS)
    key = agent_context_request_key(request_id)
    await redis.hset(
        key,
        mapping={
            "status": "pending",
            "context_type": context_type,
            "purpose": purpose,
            "reason": reason,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "run_id": run_id,
            "task_id": task_id,
            "expires_at": str(expires_at),
        },
    )
    await redis.expire(key, ttl)
    return pending


async def submit_context_result(
    *,
    request_id: str,
    user_id: str,
    conversation_id: str,
    run_id: str,
    status: ContextResultStatus,
    location: dict[str, Any] | None,
    reason: str | None,
    now: float | None = None,
) -> ContextSubmission:
    if status == "provided" and location is None:
        raise ValueError("provided 必须携带 location")
    if status != "provided" and location is not None:
        raise ValueError("非 provided 不得携带 location")
    redis = get_redis_pool()
    if redis is None:
        raise RuntimeError("Redis 不可用，无法提交运行上下文")
    result_json = _canonical_result_json(status=status, location=location, reason=reason)
    outcome = await redis.eval(
        LUA_SUBMIT_AGENT_CONTEXT,
        3,
        agent_context_request_key(request_id),
        agent_context_notify_key(request_id),
        stream_meta_key(conversation_id),
        user_id,
        conversation_id,
        run_id,
        str(time.time() if now is None else now),
        result_json,
        str(_NOTIFY_TTL_SECONDS),
    )
    return ContextSubmission(
        outcome=str(outcome),
        request_id=request_id,
        status=status,
    )


async def wait_for_context_result(
    pending: PendingContextRequest,
    *,
    clock: Callable[[], float] = time.time,
    ownership_check: Callable[[str, str], Awaitable[bool]] = check_lock_owner,
    poll_interval_seconds: float = 1.0,
) -> ResolvedContext:
    redis = get_redis_pool()
    if redis is None:
        return ResolvedContext(
            request_id=pending.request_id,
            status="unavailable",
            reason="redis_unavailable",
        )
    request_key = agent_context_request_key(pending.request_id)
    notify_key = agent_context_notify_key(pending.request_id)
    while True:
        fields = await redis.hgetall(request_key)
        if fields.get("status") == "resolved" and fields.get("result_json"):
            return _parse_resolution(pending.request_id, fields["result_json"])
        if clock() >= pending.expires_at or fields.get("status") == "expired":
            await redis.hset(request_key, "status", "expired")
            return ResolvedContext(request_id=pending.request_id, status="timeout", reason="context_timeout")
        if not await ownership_check(pending.conversation_id, pending.task_id):
            return ResolvedContext(request_id=pending.request_id, status="unavailable", reason="stream_replaced")
        try:
            await asyncio.wait_for(
                redis.blpop(notify_key, timeout=max(1, math.ceil(poll_interval_seconds))),
                timeout=poll_interval_seconds,
            )
        except TimeoutError:
            continue
