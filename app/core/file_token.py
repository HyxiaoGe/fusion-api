"""文件访问签名 token 模块

为本地存储模式生成短期签名 URL，与 MinIO presigned URL 统一架构。
token 格式：{file_id}.{expires_ts}.{signature}
签名算法：HMAC-SHA256(SECRET_KEY, "{file_id}:{expires_ts}")
"""

import hashlib
import hmac
import time
from threading import Lock
from typing import Dict, Optional, Tuple

from app.core.config import settings

_TOKEN_REFRESH_MARGIN_SECONDS = 60
_TOKEN_CACHE: Dict[Tuple[str, int], Tuple[str, int]] = {}
_TOKEN_CACHE_LOCK = Lock()


def _signature(file_id: str, expires_ts: int) -> str:
    message = f"{file_id}:{expires_ts}"
    return hmac.new(settings.SECRET_KEY.encode(), message.encode(), hashlib.sha256).hexdigest()


def _sign(file_id: str, expires_ts: int) -> str:
    signature = _signature(file_id, expires_ts)
    return f"{file_id}.{expires_ts}.{signature}"


def _purge_expired_tokens(now: int) -> None:
    expired_keys = [key for key, (_, expires_ts) in _TOKEN_CACHE.items() if expires_ts < now]
    for key in expired_keys:
        _TOKEN_CACHE.pop(key, None)


def generate_file_token(file_id: str, expires: int = 3600) -> str:
    """
    生成文件访问签名 token。

    Args:
        file_id: 文件 ID
        expires: 有效期（秒），默认 1 小时

    Returns:
        签名 token 字符串，格式: {file_id}.{expires_ts}.{signature}
    """
    now = int(time.time())
    cache_key = (file_id, expires)
    refresh_margin = min(_TOKEN_REFRESH_MARGIN_SECONDS, max(expires - 1, 0))

    with _TOKEN_CACHE_LOCK:
        cached = _TOKEN_CACHE.get(cache_key)
        if cached:
            token, expires_ts = cached
            if now <= expires_ts - refresh_margin:
                return token

        _purge_expired_tokens(now)
        expires_ts = now + expires
        token = _sign(file_id, expires_ts)
        _TOKEN_CACHE[cache_key] = (token, expires_ts)
        return token


def verify_file_token(token: str) -> Optional[str]:
    """
    验证文件访问签名 token。

    Args:
        token: 签名 token 字符串

    Returns:
        验证通过返回 file_id，失败返回 None
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None

    file_id, expires_ts_str, signature = parts

    # 检查过期时间
    try:
        expires_ts = int(expires_ts_str)
    except ValueError:
        return None

    if time.time() > expires_ts:
        return None

    # 验证签名
    expected = _signature(file_id, expires_ts)

    if not hmac.compare_digest(signature, expected):
        return None

    return file_id
