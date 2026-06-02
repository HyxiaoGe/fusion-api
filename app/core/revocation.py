"""跨应用单点登出（SLO）：基于共享 Redis 的按用户访问令牌吊销标记。

访问令牌是无状态 RS256 JWT，资源服务离线校验签名 + exp，因此登出**无法**撤销一张尚未过期
的访问令牌。auth-service 在 ``/auth/logout`` 时向**共享 Redis** 写入
``revoked_user:{user_id}`` = 登出时刻（float 墙钟秒），TTL=访问令牌寿命（自清理）。每个资源
服务在签名/类型校验通过后，对 ``iat < 标记`` 的令牌一律拒绝，使「一处退出 = 处处退出」在下一
次接口调用即生效。约定见 auth-service ``docs/AUTH_CONTRACT.md``。

与 audio-web / auth-service 同款，但用**同步** redis 客户端：fusion-api 的鉴权依赖
``get_current_user`` 是同步的（FastAPI 在线程池中运行，内部 httpx/SQLAlchemy 均为阻塞调用），
保持同步可避免把阻塞 I/O 推到事件循环。
"""

import logging

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

USER_REVOKED_PREFIX = "revoked_user:"

_redis_client: "redis.Redis | None" = None


def get_redis() -> "redis.Redis":
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def is_user_access_revoked(sub: str, token_iat: "float | int | None") -> bool:
    """sub 用户的这张访问令牌是否已被单点登出吊销。

    过度吊销是有意的：标记是 float 墙钟秒、JWT ``iat`` 是整数秒，严格 ``<`` 保证登出前所有
    令牌（含同秒内更早签发者）被吊销；重新登录因需多次 OAuth 往返，新令牌 ``iat`` 落入下一整秒
    得以存活。

    本检查处于每请求鉴权热路径：Redis 不可用必须**失败开放**（吞掉异常、视为未吊销），绝不
    500 拖垮全站；降级期吊销时延退化为令牌自身 exp（≤访问令牌寿命）。
    """
    if not sub or token_iat is None:
        return False
    try:
        raw = get_redis().get(f"{USER_REVOKED_PREFIX}{sub}")
    except Exception:
        logger.warning("SLO 吊销检查不可用（Redis），失败开放放行", exc_info=True)
        return False
    if raw is None:
        return False
    return float(token_iat) < float(raw)
