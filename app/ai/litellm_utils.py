"""LiteLLM 调用辅助：extra_body merge。

provider 健康追踪 / 错误分类已迁到 LiteLLM Proxy 内部，本模块只剩
extra_body 浅合并这一个工具。
"""

from typing import Dict


def merge_extra_body(kwargs: Dict, extra: Dict) -> None:
    """把 extra 浅合并进 kwargs['extra_body']，保留两边字段。

    用法：
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})
    """
    existing = kwargs.get("extra_body") or {}
    kwargs["extra_body"] = {**existing, **extra}
