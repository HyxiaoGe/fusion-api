"""面向 UI 的提示词模板目录。"""

from __future__ import annotations

import copy
from typing import Any

from app.core.runtime_config import get_runtime_config_payload
from app.services.runtime_config_defaults import DEFAULT_HOME_PROMPT_CATALOG


def get_home_prompt_catalog() -> dict[str, Any]:
    """读取首页任务卡和系统模板，配置异常时自动回退代码默认值。"""

    payload, meta = get_runtime_config_payload(
        "ui_prompt_catalog",
        "home",
        DEFAULT_HOME_PROMPT_CATALOG,
    )
    items = [
        copy.deepcopy(item)
        for item in payload.get("items", [])
        if isinstance(item, dict) and item.get("enabled") is True
    ]
    items.sort(key=lambda item: (item.get("sort_order", 0), item.get("id", "")))
    return {
        "items": items,
        "source": meta.get("source", "default"),
        "version": meta.get("version", "code-default"),
    }
