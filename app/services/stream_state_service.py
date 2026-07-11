"""
流状态服务（基于 Redis Stream）

职责：
1. 写端：后台任务调用，把 LLM chunk 写入 Redis Stream
2. 读端：SSE 端点调用，从 Redis Stream 消费 chunk 推送给客户端
3. 元数据：记录流状态供 /stream-status 查询

初始化失败会显式返回，流式追加连续失败会中止生成，避免继续消耗模型资源。
"""

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

from app.core.logger import app_logger as logger
from app.core.redis import (
    LOCK_TTL,
    LUA_APPEND_STREAM,
    LUA_CANCEL_STREAM,
    LUA_CLAIM_STREAM_STOP,
    LUA_CLEANUP_STREAM_INIT,
    LUA_FINALIZE_STREAM,
    LUA_INIT_STREAM,
    LUA_INSPECT_STREAM,
    LUA_RELEASE_STREAM_STOP_GUARD,
    STREAM_CHUNK_TTL,
    STREAM_DONE_TTL,
    STREAM_STOP_GUARD_TTL,
    get_redis_pool,
    stream_chunks_key,
    stream_lock_key,
    stream_meta_key,
    stream_stop_guard_key,
)

# ──────────────────────────────────────────────
# 写端（后台任务调用）
# ──────────────────────────────────────────────

STREAM_APPEND_MAX_CONSECUTIVE_FAILURES = 3
_append_failure_counts: dict[tuple[str, str], int] = {}


@dataclass(frozen=True)
class StreamInitResult:
    ok: bool
    error_code: str | None = None
    message: str | None = None


class StreamWriteTerminalError(RuntimeError):
    """当前生成必须立即终止的 Stream 写入错误。"""


class StreamWriteUnavailableError(StreamWriteTerminalError):
    """Redis Stream 连续写入失败，生成侧应立即终止。"""


class StreamOwnershipLostError(StreamWriteTerminalError):
    """当前 task 已被替代或终结，不再拥有 Stream 写入权。"""


def _append_failure_key(conversation_id: str, task_id: str) -> tuple[str, str]:
    return conversation_id, task_id


def _clear_append_failures(conversation_id: str, task_id: str) -> None:
    _append_failure_counts.pop(_append_failure_key(conversation_id, task_id), None)


def _record_append_failure(conversation_id: str, task_id: str, message: str) -> None:
    failure_key = _append_failure_key(conversation_id, task_id)
    failure_count = _append_failure_counts.get(failure_key, 0) + 1
    _append_failure_counts[failure_key] = failure_count
    logger.warning(
        "追加 chunk 失败: conv_id=%s, consecutive_failures=%s, error=%s",
        conversation_id,
        failure_count,
        message,
    )
    if failure_count >= STREAM_APPEND_MAX_CONSECUTIVE_FAILURES:
        _append_failure_counts.pop(failure_key, None)
        raise StreamWriteUnavailableError(f"Redis Stream 连续写入失败 {failure_count} 次，已终止生成")


async def init_stream(
    conversation_id: str,
    user_id: str,
    model: str,
    message_id: str,
    task_id: str,
    *,
    stream_mode: Literal["initial", "continuation"] = "initial",
) -> StreamInitResult:
    """流开始时初始化 Redis Stream 和 Meta"""
    redis = get_redis_pool()
    if not redis:
        return StreamInitResult(
            ok=False,
            error_code="redis_unavailable",
            message="Redis 不可用",
        )
    try:
        result = await redis.eval(
            LUA_INIT_STREAM,
            4,
            stream_lock_key(conversation_id),
            stream_chunks_key(conversation_id),
            stream_meta_key(conversation_id),
            stream_stop_guard_key(conversation_id),
            task_id,
            user_id,
            model,
            message_id,
            conversation_id,
            str(int(time.time())),
            str(LOCK_TTL),
            str(STREAM_CHUNK_TTL),
            stream_mode,
        )

        if not result:
            logger.info("Stream 初始化被 stop guard 拒绝: conv_id=%s", conversation_id)
            return StreamInitResult(
                ok=False,
                error_code="stream_stop_in_progress",
                message="当前生成正在停止，请稍后重试",
            )

        logger.debug(f"Stream 初始化: conv_id={conversation_id}, msg_id={message_id}")
        _clear_append_failures(conversation_id, task_id)
        return StreamInitResult(ok=True)
    except Exception as e:
        logger.warning(f"Stream 初始化失败: {e}")
        try:
            await redis.eval(
                LUA_CLEANUP_STREAM_INIT,
                3,
                stream_lock_key(conversation_id),
                stream_chunks_key(conversation_id),
                stream_meta_key(conversation_id),
                task_id,
            )
        except Exception as cleanup_error:
            logger.warning(f"清理失败的 Stream 初始化状态失败: {cleanup_error}")
        return StreamInitResult(
            ok=False,
            error_code="stream_init_failed",
            message="Redis Stream 初始化失败",
        )


async def append_chunk(
    conversation_id: str,
    chunk_type: str,
    content: str,
    block_id: str,
    *,
    task_id: str,
    **extras: Any,
) -> Optional[str]:
    """追加一个 chunk 到 Redis Stream。

    extras 用于附加 SSE envelope 字段（如 run_id / step_id），
    会作为额外的 hash field 写入 Redis Stream entry。
    None 值跳过（避免污染 hash）；非 str 值转 str（Redis hash 字段都是 str）。
    返回 Redis 分配的 entry ID。
    """
    redis = get_redis_pool()
    if not redis:
        _record_append_failure(conversation_id, task_id, "Redis 不可用")
        return None
    try:
        fields: dict[str, str] = {"type": chunk_type, "content": content, "block_id": block_id}
        for k, v in extras.items():
            if v is None:
                continue
            fields[k] = v if isinstance(v, str) else str(v)
        field_args = [item for pair in fields.items() for item in pair]
        result = await redis.eval(
            LUA_APPEND_STREAM,
            3,
            stream_lock_key(conversation_id),
            stream_chunks_key(conversation_id),
            stream_meta_key(conversation_id),
            task_id,
            str(STREAM_CHUNK_TTL),
            *field_args,
        )
        if not result or int(result[0]) != 1:
            _clear_append_failures(conversation_id, task_id)
            raise StreamOwnershipLostError(f"Stream 写入权已失效: conv_id={conversation_id}, task_id={task_id}")
        entry_id = str(result[1])
        _clear_append_failures(conversation_id, task_id)
        return entry_id
    except StreamOwnershipLostError:
        raise
    except Exception as e:
        _record_append_failure(conversation_id, task_id, str(e))
        return None


async def finalize_stream(
    conversation_id: str,
    success: bool,
    error_msg: str = "",
    task_id: str = "",
    error_code: str = "",
    error_data: Optional[dict[str, Any]] = None,
) -> bool:
    """
    流结束时调用，写 done/error 标记，更新 meta，缩短 TTL，释放锁。

    使用 Lua 脚本保证原子性：锁检查 + 写标记 + 释放锁在同一个 Redis 命令内完成，
    彻底避免并发任务之间的竞态条件。

    BYOK 协议扩展：当 success=False 且传入 error_code 时，把 {code, message, data}
    JSON 编码到 entry_content，供 stream_redis_as_sse 解析后挂到 SSE chunk 的
    顶级 error 字段，前端 chat.ts 据此显示结构化错误卡片 + CTA。
    """
    _clear_append_failures(conversation_id, task_id)
    redis = get_redis_pool()
    if not redis:
        return False
    try:
        entry_type = "done" if success else "error"
        if success:
            entry_content = ""
        elif error_code:
            entry_content = json.dumps(
                {"code": error_code, "message": error_msg or "", "data": error_data or {}},
                ensure_ascii=False,
            )
        else:
            entry_content = error_msg

        result = await redis.eval(
            LUA_FINALIZE_STREAM,
            3,  # KEYS 数量
            stream_lock_key(conversation_id),
            stream_chunks_key(conversation_id),
            stream_meta_key(conversation_id),
            task_id,
            entry_type,
            entry_content,
            str(STREAM_DONE_TTL),
        )

        if not result:
            logger.debug(f"finalize 跳过（Lua 原子检查）：锁不匹配 conv_id={conversation_id}")
        return bool(result)
    except Exception as e:
        logger.warning(f"finalize stream 失败: {e}")
        return False


async def cancel_stream(
    conversation_id: str,
    message_id: str = "",
    expected_task_id: str = "",
) -> bool:
    """
    跨 worker 取消流：删除 lock + 写 error entry + 更新 meta。

    使用 Lua 脚本原子执行：
    - 仅当 meta 状态为 streaming 时才取消
    - 如果传了 message_id，还要校验匹配
    - 如果传了 expected_task_id，还要校验当前任务归属，防止误杀复用同一消息的新 continuation
    """
    redis = get_redis_pool()
    if not redis:
        return False
    try:
        result = await redis.eval(
            LUA_CANCEL_STREAM,
            3,
            stream_lock_key(conversation_id),
            stream_chunks_key(conversation_id),
            stream_meta_key(conversation_id),
            str(STREAM_DONE_TTL),
            message_id or "",
            expected_task_id or "",
        )
        if result:
            logger.info(f"流已通过 Redis 取消: conv_id={conversation_id}")
        else:
            logger.debug(f"cancel_stream 跳过（CAS 不匹配或流已结束）: conv_id={conversation_id}")
        return bool(result)
    except Exception as e:
        logger.warning(f"取消流失败: {e}")
        return False


async def claim_stream_stop(conversation_id: str, message_id: str, expected_task_id: str) -> bool:
    """原子占有当前 task 的 stop 权，并阻止新流初始化。"""
    redis = get_redis_pool()
    if not redis or not message_id or not expected_task_id:
        return False
    try:
        result = await redis.eval(
            LUA_CLAIM_STREAM_STOP,
            3,
            stream_lock_key(conversation_id),
            stream_meta_key(conversation_id),
            stream_stop_guard_key(conversation_id),
            message_id,
            expected_task_id,
            str(STREAM_STOP_GUARD_TTL),
        )
        return bool(result)
    except Exception as error:
        logger.warning("占有 stop guard 失败: conv_id=%s, error=%s", conversation_id, error)
        return False


async def release_stream_stop_guard(conversation_id: str, expected_task_id: str) -> bool:
    """按 task_id CAS 释放 stop guard。"""
    redis = get_redis_pool()
    if not redis or not expected_task_id:
        return False
    try:
        result = await redis.eval(
            LUA_RELEASE_STREAM_STOP_GUARD,
            1,
            stream_stop_guard_key(conversation_id),
            expected_task_id,
        )
        return bool(result)
    except Exception as error:
        logger.warning("释放 stop guard 失败: conv_id=%s, error=%s", conversation_id, error)
        return False


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
        if not meta:
            return None
        meta.setdefault("stream_mode", "initial")
        return meta
    except Exception as e:
        logger.warning(f"查询 stream meta 失败: {e}")
        return None


# ──────────────────────────────────────────────
# 读端（SSE 端点调用）
# ──────────────────────────────────────────────


def _terminal_error_fields(*, code: str, message: str, reason: str) -> dict[str, str]:
    return {
        "type": "error",
        "content": json.dumps(
            {
                "code": code,
                "message": message,
                "data": {"reason": reason},
            },
            ensure_ascii=False,
        ),
        "block_id": "",
    }


async def _inspect_stream_state(
    redis: Any,
    conversation_id: str,
    *,
    expected_message_id: str,
    expected_task_id: str,
) -> tuple[str, str]:
    fields = _terminal_error_fields(
        code="stream_interrupted",
        message="生成连接已中断，请重试",
        reason="orphaned_stream",
    )
    result = await redis.eval(
        LUA_INSPECT_STREAM,
        3,
        stream_lock_key(conversation_id),
        stream_chunks_key(conversation_id),
        stream_meta_key(conversation_id),
        expected_message_id,
        expected_task_id,
        fields["content"],
        str(int(time.time())),
        str(STREAM_DONE_TTL),
    )
    return str(result[0]), str(result[1])


async def read_stream_chunks(
    conversation_id: str,
    last_entry_id: str = "0",
    *,
    expected_message_id: str,
    expected_task_id: str,
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
            state, terminal_entry_id = await _inspect_stream_state(
                redis,
                conversation_id,
                expected_message_id=expected_message_id,
                expected_task_id=expected_task_id,
            )
        except Exception as e:
            logger.warning(f"检查流状态失败: {e}")
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="redis_read_failed",
                    message="生成连接暂时中断，请重试",
                    reason="redis_liveness_check_failed",
                ),
            }
            return

        if state == "replaced":
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="stream_interrupted",
                    message="当前生成已被新请求取代，请重试",
                    reason="stream_replaced",
                ),
            }
            return
        if state == "missing":
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="stream_interrupted",
                    message="生成连接已中断，请重试",
                    reason="orphaned_stream",
                ),
            }
            return
        if state == "orphaned":
            yield {
                "entry_id": terminal_entry_id or "0-0",
                **_terminal_error_fields(
                    code="stream_interrupted",
                    message="生成连接已中断，请重试",
                    reason="orphaned_stream",
                ),
            }
            return
        if state == "terminal":
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

        try:
            results = await redis.xread(
                {key: current_id},
                block=5000,
                count=50,
            )
        except Exception as e:
            logger.warning(f"XREAD 失败: {e}")
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="redis_read_failed",
                    message="生成连接暂时中断，请重试",
                    reason="redis_read_failed",
                ),
            }
            return

        if not results:
            continue

        try:
            post_read_state, _ = await _inspect_stream_state(
                redis,
                conversation_id,
                expected_message_id=expected_message_id,
                expected_task_id=expected_task_id,
            )
        except Exception as e:
            logger.warning(f"读取后校验流状态失败: {e}")
            post_read_state = "read_failed"
        if post_read_state == "replaced":
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="stream_interrupted",
                    message="当前生成已被新请求取代，请重试",
                    reason="stream_replaced",
                ),
            }
            return
        if post_read_state == "read_failed":
            yield {
                "entry_id": current_id if current_id != "0" else "0-0",
                **_terminal_error_fields(
                    code="redis_read_failed",
                    message="生成连接暂时中断，请重试",
                    reason="redis_liveness_check_failed",
                ),
            }
            return

        for _stream_key, entries in results:
            for entry_id, fields in entries:
                current_id = entry_id
                yield {"entry_id": entry_id, **fields}
                if fields.get("type") in ("done", "error"):
                    return

    yield {
        "entry_id": current_id if current_id != "0" else "0-0",
        **_terminal_error_fields(
            code="stream_interrupted",
            message="生成等待超时，请重试",
            reason="stream_deadline_exceeded",
        ),
    }
