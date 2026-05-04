"""LiteLLM 调用辅助：extra_body merge + Provider 离线异常 + 错误分类"""

from enum import Enum
from typing import Dict, Optional, Tuple


def merge_extra_body(kwargs: Dict, extra: Dict) -> None:
    """把 extra 浅合并进 kwargs['extra_body']，保留两边字段。

    用法：
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})

    注意：不要直接给 kwargs['extra_body'] 赋值，否则会覆盖前置层（如 LLMManager）
    塞进去的 user api_key。
    """
    existing = kwargs.get("extra_body") or {}
    kwargs["extra_body"] = {**existing, **extra}


class ErrorKind(str, Enum):
    """LiteLLM 异常分类枚举（AI 层专属）。"""

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


class ProviderOfflineError(Exception):
    """provider 整体下线时由 LLMManager.resolve_model 抛出，service 层捕获转 503。"""

    def __init__(self, provider_id: str, reason: Optional[str], message: Optional[str]):
        self.provider_id = provider_id
        self.reason = reason or "unknown"
        self.message = message or ""
        super().__init__(f"Provider {provider_id} offline (reason={self.reason}): {self.message}")
