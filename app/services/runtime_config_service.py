"""运行时配置服务兼容入口。

实现位于 app.core.runtime_config，避免 app.ai 反向依赖 app.services。
"""

from app.core.runtime_config import clear_runtime_config_cache, deep_merge_config, get_runtime_config_payload

__all__ = [
    "clear_runtime_config_cache",
    "deep_merge_config",
    "get_runtime_config_payload",
]
