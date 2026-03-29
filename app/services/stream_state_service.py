"""
流状态服务（基于 Redis Stream）

职责：
1. 写端：后台任务调用，把 LLM chunk 写入 Redis Stream
2. 读端：SSE 端点调用，从 Redis Stream 消费 chunk 推送给客户端
3. 元数据：记录流状态供 /stream-status 查询

Redis 不可用时所有写操作静默降级。
"""
import json
import time
from typing import AsyncIterator, Optional

from app.core.redis import (
    get_redis_pool,
    stream_chunks_key, stream_meta_key, stream_lock_key,
    STREAM_CHUNK_TTL, STREAM_DONE_TTL, LOCK_TTL,
)
from app.core.logger import app_logger as logger


# ──────────────────────────────────────────────
# 写端（后台任务调用）
# ──────────────────────────────────────────────

async def init_stream(
    conversation_id: str,
    user_id: str,
    model: str,
    message_id: str,
    task_id: str,
) -> None:
    """流开始时初始化 Redis Stream 和 Meta"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        # 写 meta
        await redis.hset(stream_meta_key(conversation_id), mapping={
            "status": "streaming",
            "user_id": user_id,
            "model": model,
            "started_at": str(int(time.time())),
            "message_id": message_id,
            "conversation_id": conversation_id,
        })
        await redis.expire(stream_meta_key(conversation_id), STREAM_CHUNK_TTL)

        # 写 lock
        await redis.set(stream_lock_key(conversation_id), task_id, ex=LOCK_TTL)

        # 清除上一轮的 Stream 数据，避免新轮次读到旧内容
        await redis.delete(stream_chunks_key(conversation_id))

        # 初始化 Stream（写一条 start 标记）
        await redis.xadd(
            stream_chunks_key(conversation_id),
            {"type": "start", "content": ""},
        )
        await redis.expire(stream_chunks_key(conversation_id), STREAM_CHUNK_TTL)

        logger.debug(f"Stream 初始化: conv_id={conversation_id}, msg_id={message_id}")
    except Exception as e:
        logger.warning(f"Stream 初始化失败: {e}")


async def append_chunk(
    conversation_id: str,
    chunk_type: str,
    content: str,
    block_id: str,
) -> Optional[str]:
    """
    追加一个 chunk 到 Redis Stream。
    返回 Redis 分配的 entry ID。
    """
    redis = get_redis_pool()
    if not redis:
        return None
    try:
        entry_id = await redis.xadd(
            stream_chunks_key(conversation_id),
            {"type": chunk_type, "content": content, "block_id": block_id},
        )
        # 刷新 TTL
        await redis.expire(stream_chunks_key(conversation_id), STREAM_CHUNK_TTL)
        return entry_id
    except Exception as e:
        logger.warning(f"追加 chunk 失败: {e}")
        return None


async def finalize_stream(conversation_id: str, success: bool, error_msg: str = "", task_id: str = "") -> None:
    """流结束时调用，写 done/error 标记，更新 meta，缩短 TTL。"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        # 如果提供了 task_id，检查是否还是当前锁持有者
        # 被新任务接管后不应该往 Stream 里写，否则会污染新任务的数据
        if task_id:
            current_lock = await redis.get(stream_lock_key(conversation_id))
            if current_lock and current_lock != task_id:
                logger.debug(f"finalize 跳过：锁已转移给新任务 conv_id={conversation_id}")
                return
        if success:
            await redis.xadd(
                stream_chunks_key(conversation_id),
                {"type": "done", "content": ""},
            )
            await redis.hset(stream_meta_key(conversation_id), "status", "done")
        else:
            await redis.xadd(
                stream_chunks_key(conversation_id),
                {"type": "error", "content": error_msg},
            )
            await redis.hset(stream_meta_key(conversation_id), "status", "error")

        # 缩短 TTL，给断线重连留 60 秒窗口
        await redis.expire(stream_chunks_key(conversation_id), STREAM_DONE_TTL)
        await redis.expire(stream_meta_key(conversation_id), STREAM_DONE_TTL)
        await redis.delete(stream_lock_key(conversation_id))
    except Exception as e:
        logger.warning(f"finalize stream 失败: {e}")


async def check_lock_owner(conversation_id: str, task_id: str) -> bool:
    """检查当前 task_id 是否仍是锁持有者"""
    redis = get_redis_pool()
    if not redis:
        return True
    try:
        current = await redis.get(stream_lock_key(conversation_id))
        return current == task_id
    except Exception:
        return True


async def get_stream_meta(conversation_id: str) -> Optional[dict]:
    """查询流元数据（供 /stream-status 端点）"""
    redis = get_redis_pool()
    if not redis:
        return None
    try:
        meta = await redis.hgetall(stream_meta_key(conversation_id))
        return meta if meta else None
    except Exception as e:
        logger.warning(f"查询 stream meta 失败: {e}")
        return None


# ──────────────────────────────────────────────
# 读端（SSE 端点调用）
# ──────────────────────────────────────────────

async def read_stream_chunks(
    conversation_id: str,
    last_entry_id: str = "0",
) -> AsyncIterator[dict]:
    """
    异步生成器，从 Redis Stream 读取 chunk 并 yield。
    阻塞等待新 chunk（XREAD BLOCK），直到读到 done/error 为止。

    last_entry_id:
      "0"        → 从头读（新连接）
      "xxx-xxx"  → 从该 ID 之后续读（断线重连）
    """
    redis = get_redis_pool()
    if not redis:
        return

    key = stream_chunks_key(conversation_id)
    current_id = last_entry_id

    # 最多等 30 分钟
    deadline = time.time() + 1800

    while time.time() < deadline:
        try:
            results = await redis.xread(
                {key: current_id},
                block=5000,
                count=50,
            )
        except Exception as e:
            logger.warning(f"XREAD 失败: {e}")
            return

        if not results:
            # 超时没有新数据，检查流是否已结束
            meta = await get_stream_meta(conversation_id)
            if meta and meta.get("status") in ("done", "error"):
                # 把剩余 entry 全部读完
                try:
                    remaining = await redis.xrange(key, min=current_id, count=100)
                    for entry_id, fields in remaining:
                        if entry_id == current_id:
                            continue
                        current_id = entry_id
                        yield {"entry_id": entry_id, **fields}
                except Exception:
                    pass
                return
            continue

        for _stream_key, entries in results:
            for entry_id, fields in entries:
                current_id = entry_id
                yield {"entry_id": entry_id, **fields}
                if fields.get("type") in ("done", "error"):
                    return
