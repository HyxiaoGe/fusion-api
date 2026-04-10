"""
Redis 连接管理模块

连接池在应用启动时初始化，shutdown 时关闭。
get_redis() 作为 FastAPI 依赖函数，供路由层注入。
Lua 脚本在模块加载时从文件读取，供 stream_state_service 调用。
"""

from pathlib import Path

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logger import app_logger as logger

# 模块级连接池，全局唯一
_redis_pool: aioredis.Redis | None = None

# Redis Stream key 和 TTL 常量
STREAM_CHUNK_TTL = 600  # 流进行中 TTL（10 分钟）
STREAM_DONE_TTL = 60  # 流结束后 TTL（60 秒，供断线重连最后窗口）
LOCK_TTL = 600  # 互斥锁 TTL


def stream_chunks_key(conversation_id: str) -> str:
    """Redis Stream key：存储流的所有 chunk"""
    return f"stream:chunks:{conversation_id}"


def stream_meta_key(conversation_id: str) -> str:
    """Redis Hash key：存储流的元信息"""
    return f"stream:meta:{conversation_id}"


def stream_lock_key(conversation_id: str) -> str:
    """Redis String key：流的互斥锁"""
    return f"stream:lock:{conversation_id}"


async def init_redis() -> None:
    """在 lifespan startup 阶段调用"""
    global _redis_pool
    _redis_pool = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    try:
        await _redis_pool.ping()
        logger.info("Redis 连接池初始化成功")
    except Exception as e:
        logger.warning(f"Redis 连接失败，流状态缓存将不可用: {e}")
        _redis_pool = None


async def close_redis() -> None:
    """在 lifespan shutdown 阶段调用"""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("Redis 连接池已关闭")


def get_redis_pool() -> aioredis.Redis | None:
    """返回模块级连接池实例。返回 None 表示 Redis 不可用。"""
    return _redis_pool


async def get_redis() -> aioredis.Redis | None:
    """FastAPI 依赖函数，供路由层注入。"""
    return _redis_pool


# ──────────────────────────────────────────────
# Lua 脚本加载
# ──────────────────────────────────────────────

_LUA_DIR = Path(__file__).parent / "lua"


def _load_lua(name: str) -> str:
    return (_LUA_DIR / f"{name}.lua").read_text()


LUA_FINALIZE_STREAM = _load_lua("finalize_stream")
LUA_RELEASE_LOCK = _load_lua("release_lock")
LUA_CANCEL_STREAM = _load_lua("cancel_stream")
