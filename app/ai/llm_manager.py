# app/ai/llm_manager.py
import logging
import os
from typing import Any, Dict, Tuple

import litellm
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.db.repositories import ModelSourceRepository

# 关闭 LiteLLM 的冗余日志
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMManager:
    """
    统一的 LLM 调用管理器，基于 LiteLLM。
    职责：根据 model_id 查询模型信息，通过 LiteLLM Proxy 路由所有请求。
    """

    def resolve_model(self, model_id: str, db: Session) -> Tuple[str, str, Dict[str, Any]]:
        """
        根据 model_id 解析出 LiteLLM 调用所需的完整参数。
        通过 LiteLLM Proxy 路由，不再直接读取 API 凭证。
        返回：(litellm_model_string, provider, litellm_kwargs)
        """

        model_source = ModelSourceRepository(db).get_by_id(model_id)
        if not model_source:
            raise ValueError(f"未找到模型配置: {model_id}")

        provider_rel = model_source.provider_rel
        if not provider_rel:
            raise ValueError(f"模型 {model_id} 的提供商 {model_source.provider} 未配置")

        # 通过 LiteLLM Proxy 路由。
        # 本地 LiteLLM SDK 需要 openai/ 前缀才能按 OpenAI-compatible endpoint 转发到 proxy。
        # 真正的路由前缀来自 provider 表里的 litellm_prefix（如 deepseek、openrouter/x-ai 等），
        # 必须与 proxy 端 model_list 注册的 model_name 模式对得上。
        litellm_model = f"openai/{provider_rel.litellm_prefix}/{model_id}"

        proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
        proxy_key = os.environ.get("LITELLM_API_KEY", "")

        litellm_kwargs: Dict[str, Any] = {
            "api_key": proxy_key,
            "api_base": proxy_url,
        }

        return litellm_model, model_source.provider, litellm_kwargs

    async def test_credentials(
        self,
        provider: str,
        model_id: str,
        credentials: Dict[str, Any],
        db: Session = None,
    ) -> bool:
        """通过 LiteLLM Proxy 测试模型是否可用"""
        try:
            if db is None:
                raise ValueError("test_credentials 需要 db 会话以解析 provider 路由前缀")
            from app.db.repositories import ProviderRepository

            provider_obj = ProviderRepository(db).get_by_id(provider)
            if not provider_obj:
                raise ValueError(f"未找到 provider: {provider}")
            litellm_model = f"openai/{provider_obj.litellm_prefix}/{model_id}"

            proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
            proxy_key = os.environ.get("LITELLM_API_KEY", "")

            await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                api_key=proxy_key,
                api_base=proxy_url,
            )
            return True
        except Exception as e:
            logger.error(f"凭证测试失败 [{provider}/{model_id}]: {e}")
            return False


# 全局单例
llm_manager = LLMManager()
