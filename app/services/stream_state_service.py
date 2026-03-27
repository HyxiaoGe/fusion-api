"""
流式状态缓存服务

负责管理 SSE 流的生命周期状态，存储于 Redis。
key 格式：stream:{conversation_id}
lock key 格式：stream:lock:{conversation_id}
TTL：由 settings.REDIS_STREAM_TTL 控制（默认 300 秒）

Redis 不可用时所有操作静默降级（不影响主流程）。
"""
import json
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.core.redis import get_redis_pool
from app.core.logger import app_logger as logger


def _stream_key(conversation_id: str) -> str:
    return f"stream:{conversation_id}"


def _lock_key(conversation_id: str) -> str:
    return f"stream:lock:{conversation_id}"


async def acquire_stream_lock(conversation_id: str) -> str:
    """
    获取流互斥锁，防止同一会话并发开流。
    返回本次请求的 request_id（锁持有者标识）。
    后来者强制覆盖前者。
    """
    redis = get_redis_pool()
    if not redis:
        return str(uuid.uuid4())
    try:
        request_id = str(uuid.uuid4())
        await redis.set(_lock_key(conversation_id), request_id, ex=settings.REDIS_STREAM_TTL)
        return request_id
    except Exception as e:
        logger.warning(f"获取流锁失败: {e}")
        return str(uuid.uuid4())


async def is_lock_owner(conversation_id: str, request_id: str) -> bool:
    """检查当前 request_id 是否仍是锁持有者"""
    redis = get_redis_pool()
    if not redis:
        return True  # Redis 不可用时默认不踢
    try:
        current = await redis.get(_lock_key(conversation_id))
        return current == request_id
    except Exception:
        return True


async def set_stream_start(
    conversation_id: str,
    user_id: str,
    model: str,
) -> None:
    """流开始时写入初始状态"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        state = {
            "status": "streaming",
            "user_id": user_id,
            "model": model,
            "started_at": int(time.time()),
            "content_blocks": [],
            "last_chunk_at": int(time.time()),
        }
        await redis.set(
            _stream_key(conversation_id),
            json.dumps(state, ensure_ascii=False),
            ex=settings.REDIS_STREAM_TTL,
        )
    except Exception as e:
        logger.warning(f"写入流状态失败: {e}")


async def append_stream_chunk(
    conversation_id: str,
    block_type: str,
    content_delta: str,
) -> None:
    """
    追加 chunk 到 content_blocks。
    简单 GET → 修改 → SET（单写者场景，无需乐观锁）。
    """
    redis = get_redis_pool()
    if not redis:
        return
    try:
        key = _stream_key(conversation_id)
        raw = await redis.get(key)
        if not raw:
            return

        state = json.loads(raw)
        blocks = state.get("content_blocks", [])

        # 同类型 block 合并，不同类型新建
        if blocks and blocks[-1]["type"] == block_type:
            blocks[-1]["content"] += content_delta
        else:
            blocks.append({"type": block_type, "content": content_delta})

        state["content_blocks"] = blocks
        state["last_chunk_at"] = int(time.time())

        await redis.set(key, json.dumps(state, ensure_ascii=False), ex=settings.REDIS_STREAM_TTL)
    except Exception as e:
        logger.warning(f"追加流 chunk 失败: {e}")


async def set_stream_complete(conversation_id: str) -> None:
    """流正常结束，清除 Redis 状态"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        await redis.delete(_stream_key(conversation_id))
        await redis.delete(_lock_key(conversation_id))
    except Exception as e:
        logger.warning(f"清除流状态失败: {e}")


async def set_stream_error(conversation_id: str, error_msg: str = "") -> None:
    """流异常结束，更新状态为 error，保留 content_blocks 供前端恢复"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        key = _stream_key(conversation_id)
        raw = await redis.get(key)
        if raw:
            state = json.loads(raw)
            state["status"] = "error"
            state["error"] = error_msg
            # 保留较短 TTL（60 秒），让前端有时间读取后自动过期
            await redis.set(key, json.dumps(state, ensure_ascii=False), ex=60)
        await redis.delete(_lock_key(conversation_id))
    except Exception as e:
        logger.warning(f"更新流错误状态失败: {e}")


async def get_stream_status(conversation_id: str) -> Optional[dict]:
    """
    查询流状态，供 /stream-status 端点和前端重连使用。
    返回 None 表示无进行中的流。
    """
    redis = get_redis_pool()
    if not redis:
        return None
    try:
        raw = await redis.get(_stream_key(conversation_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"查询流状态失败: {e}")
        return None
