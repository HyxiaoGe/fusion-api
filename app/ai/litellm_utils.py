"""LiteLLM 调用辅助：extra_body merge + Provider 离线异常"""

from typing import Dict, Optional


def merge_extra_body(kwargs: Dict, extra: Dict) -> None:
    """把 extra 浅合并进 kwargs['extra_body']，保留两边字段。

    用法：
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})

    注意：不要直接给 kwargs['extra_body'] 赋值，否则会覆盖前置层（如 LLMManager）
    塞进去的 user api_key。
    """
    existing = kwargs.get("extra_body") or {}
    kwargs["extra_body"] = {**existing, **extra}


class ProviderOfflineError(Exception):
    """provider 整体下线时由 LLMManager.resolve_model 抛出，service 层捕获转 503。"""

    def __init__(self, provider_id: str, reason: Optional[str], message: Optional[str]):
        self.provider_id = provider_id
        self.reason = reason or "unknown"
        self.message = message or ""
        super().__init__(f"Provider {provider_id} offline (reason={self.reason}): {self.message}")
