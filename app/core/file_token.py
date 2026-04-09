"""文件访问签名 token 模块

为本地存储模式生成短期签名 URL，与 MinIO presigned URL 统一架构。
token 格式：{file_id}.{expires_ts}.{signature}
签名算法：HMAC-SHA256(SECRET_KEY, "{file_id}:{expires_ts}")
"""

import hashlib
import hmac
import time
from typing import Optional, Tuple

from app.core.config import settings


def generate_file_token(file_id: str, expires: int = 3600) -> str:
    """
    生成文件访问签名 token。

    Args:
        file_id: 文件 ID
        expires: 有效期（秒），默认 1 小时

    Returns:
        签名 token 字符串，格式: {file_id}.{expires_ts}.{signature}
    """
    expires_ts = int(time.time()) + expires
    message = f"{file_id}:{expires_ts}"
    signature = hmac.new(
        settings.SECRET_KEY.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return f"{file_id}.{expires_ts}.{signature}"


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
    message = f"{file_id}:{expires_ts_str}"
    expected = hmac.new(
        settings.SECRET_KEY.encode(), message.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        return None

    return file_id
