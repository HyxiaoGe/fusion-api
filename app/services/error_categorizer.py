"""把 litellm 抛出的异常分类成稳定的 ErrorKind，供 health service 决策。"""

from enum import Enum
from typing import Tuple


class ErrorKind(str, Enum):
    KEY_INVALID = "key_invalid"
    QUOTA_EXCEEDED = "quota_exceeded"
    TOS_BLOCKED = "tos_blocked"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


_QUOTA_HINTS = ("insufficient credits", "insufficient_quota", "quota exceeded", "余额", "账户欠费")
_TOS_HINTS = ("terms of service", "tos", "moderation", "content policy")


def categorize(exc: Exception) -> Tuple[ErrorKind, str]:
    """从 litellm 异常解析 (ErrorKind, human_message)。

    优先看 status_code，再看消息关键字。429 / 5xx / 网络错均归 TRANSIENT。
    """
    msg = str(exc) or exc.__class__.__name__
    status = getattr(exc, "status_code", None)

    msg_lower = msg.lower()
    has_quota_hint = any(h in msg_lower for h in _QUOTA_HINTS)
    has_tos_hint = any(h in msg_lower for h in _TOS_HINTS)

    if has_quota_hint:
        return ErrorKind.QUOTA_EXCEEDED, msg

    if status == 401:
        return ErrorKind.KEY_INVALID, msg
    if status == 402:
        return ErrorKind.QUOTA_EXCEEDED, msg
    if status == 403:
        if has_tos_hint:
            return ErrorKind.TOS_BLOCKED, msg
        return ErrorKind.KEY_INVALID, msg
    if status == 429:
        return ErrorKind.TRANSIENT, msg
    if status is not None and 500 <= status < 600:
        return ErrorKind.TRANSIENT, msg

    return ErrorKind.UNKNOWN, msg
