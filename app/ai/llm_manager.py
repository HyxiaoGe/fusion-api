# app/ai/llm_manager.py
"""统一的 LLM 调用解析器（薄代理 LiteLLM Proxy）。

fusion-api 不再维护本地 provider / model / user_credential 表。所有模型路由
都交给 LiteLLM Proxy 的业务别名（alias）：调用时 model=alias，proxy 自己
解析到底层 provider + key + base_url。

resolve_model 只负责：
- 校验 alias 在 LiteLLM 目录里存在
- 返回 LiteLLM 调用参数（model_name + proxy api_key/base_url）

provider 健康追踪、BYOK、provider 离线状态全部由 LiteLLM Proxy 内部管。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import litellm

from app.ai import litellm_catalog

litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMManager:
    """LiteLLM 调用参数解析器（不读 DB，纯薄代理）。"""

    def resolve_model(
        self,
        model_id: str,
        db: Any = None,  # 保留参数兼容老 caller，不再读 DB
        user_id: Optional[str] = None,  # 同上，BYOK 已删
    ) -> Tuple[str, str, Dict[str, Any]]:
        """解析 LiteLLM 调用参数。

        Returns:
            (litellm_model, provider, kwargs)
            - litellm_model: alias，直接传给 litellm.acompletion(model=...)
            - provider: 底层 provider key（"deepseek" / "qwen" / "openrouter"...），
              stream runner 用来判断是否开启 reasoning 模式
            - kwargs: 含 api_key/api_base，让 litellm 走 fusion 的 LiteLLM Proxy

        Raises:
            ValueError: alias 不在 LiteLLM 目录里
        """
        entry = litellm_catalog.get_model_entry(model_id)
        if not entry:
            raise ValueError(f"未找到模型: {model_id}（不在 LiteLLM Proxy 注册表里）")

        provider = litellm_catalog.get_underlying_provider(model_id, fallback="litellm")

        proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
        proxy_key = os.environ.get("LITELLM_API_KEY", "")

        kwargs: Dict[str, Any] = {
            "api_key": proxy_key,
            "api_base": proxy_url,
        }
        return model_id, provider, kwargs


# 全局单例
llm_manager = LLMManager()
