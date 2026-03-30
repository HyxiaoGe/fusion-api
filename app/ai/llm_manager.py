# app/ai/llm_manager.py
import logging
from typing import Any, Dict, Optional, Tuple

import litellm
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger

# 关闭 LiteLLM 的冗余日志
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# provider 内部标识 → LiteLLM 模型前缀映射
# 部分 provider 使用 OpenAI 兼容接口，需要通过 api_base 指定，前缀统一用 openai/
PROVIDER_LITELLM_PREFIX = {
    "openai":      "openrouter/openai",       # 通过 OpenRouter 路由
    "anthropic":   "openrouter/anthropic",    # 通过 OpenRouter 路由
    "google":      "openrouter/google",       # 通过 OpenRouter 路由
    "xai":         "openrouter/x-ai",         # 通过 OpenRouter 路由（OpenRouter 上为 x-ai）
    "deepseek":    "deepseek",                # 直连
    "qwen":        "openai",                  # 通义千问使用 OpenAI 兼容接口
    "volcengine":  "openai",                  # 火山引擎使用 OpenAI 兼容接口
}

# 需要自定义 api_base 的 provider（从凭证里读取 base_url）
CUSTOM_BASE_URL_PROVIDERS = {"qwen", "volcengine", "wenxin", "hunyuan"}


class LLMManager:
    """
    统一的 LLM 调用管理器，基于 LiteLLM。
    职责：根据 model_id 查询凭证，构造 LiteLLM 调用参数。
    """

    def resolve_model(self, model_id: str, db: Session) -> Tuple[str, str, Dict[str, Any]]:
        """
        根据 model_id 解析出 LiteLLM 调用所需的完整参数。
        返回：(litellm_model_string, provider, litellm_kwargs)

        litellm_model_string 格式示例：
          "openai/gpt-4o"
          "anthropic/claude-3-5-sonnet-20241022"
          "deepseek/deepseek-chat"
          "gemini/gemini-2.0-flash"
          "openai/qwen-max"（通义千问走 openai 兼容接口）
        """
        from app.db.repositories import ModelSourceRepository, ModelCredentialRepository

        # 查询模型来源，获取 provider 和真实 model 名称
        model_source = ModelSourceRepository(db).get_by_id(model_id)
        if not model_source:
            raise ValueError(f"未找到模型配置: {model_id}")

        provider = model_source.provider
        model_name = model_id  # model_id 即为传给 provider 的模型名称

        # 获取默认凭证
        credential = ModelCredentialRepository(db).get_default(model_id)
        if not credential:
            raise ValueError(f"未找到模型 {model_id} 的凭证配置")

        creds = credential.credentials

        # 构造 LiteLLM 前缀
        prefix = PROVIDER_LITELLM_PREFIX.get(provider, provider)
        litellm_model = f"{prefix}/{model_name}"

        # 构造 LiteLLM 调用参数
        litellm_kwargs: Dict[str, Any] = {
            "api_key": creds.get("api_key"),
        }

        # 需要自定义 base_url 的 provider
        if provider in CUSTOM_BASE_URL_PROVIDERS and creds.get("base_url"):
            litellm_kwargs["api_base"] = creds["base_url"]

        return litellm_model, provider, litellm_kwargs

    async def test_credentials(
        self,
        provider: str,
        model_id: str,
        credentials: Dict[str, Any],
    ) -> bool:
        """测试凭证是否有效，发送最小测试请求"""
        try:
            prefix = PROVIDER_LITELLM_PREFIX.get(provider, provider)
            litellm_model = f"{prefix}/{model_id}"

            kwargs: Dict[str, Any] = {"api_key": credentials.get("api_key")}
            if provider in CUSTOM_BASE_URL_PROVIDERS and credentials.get("base_url"):
                kwargs["api_base"] = credentials["base_url"]

            await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                **kwargs,
            )
            return True
        except Exception as e:
            logger.error(f"凭证测试失败 [{provider}/{model_id}]: {e}")
            return False


# 全局单例
llm_manager = LLMManager()


# 保留展示名称映射，其他地方有引用
MODEL_DISPLAY_NAMES = {
    "anthropic":  "Anthropic",
    "deepseek":   "DeepSeek",
    "google":     "Google",
    "openai":     "OpenAI",
    "qwen":       "通义千问",
    "volcengine": "火山引擎",
    "xai":        "xAI",
}


def get_model_display_name(provider: str) -> str:
    return MODEL_DISPLAY_NAMES.get(provider, provider)
