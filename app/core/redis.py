"""
Redis 连接管理模块

连接池在应用启动时初始化，shutdown 时关闭。
get_redis() 作为 FastAPI 依赖函数，供路由层注入。
"""
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logger import app_logger as logger

# 模块级连接池，全局唯一
_redis_pool: aioredis.Redis | None = None


async def init_redis() -> None:
    """在 lifespan startup 阶段调用"""
    global _redis_pool
    _redis_pool = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    # 验证连接
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
    """
    返回模块级连接池实例。
    stream_state_service 直接调用此函数。
    返回 None 表示 Redis 不可用（降级模式）。
    """
    return _redis_pool


async def get_redis() -> aioredis.Redis | None:
    """
    FastAPI 依赖函数，供路由层注入。
    用法：redis = Depends(get_redis)
    """
    return _redis_pool
