# app/ai/llm_manager.py
import logging
from typing import Any, Dict, Tuple

import litellm
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger

# 关闭 LiteLLM 的冗余日志
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMManager:
    """
    统一的 LLM 调用管理器，基于 LiteLLM。
    职责：根据 model_id 查询凭证，构造 LiteLLM 调用参数。
    """

    def resolve_model(self, model_id: str, db: Session) -> Tuple[str, str, Dict[str, Any]]:
        """
        根据 model_id 解析出 LiteLLM 调用所需的完整参数。
        返回：(litellm_model_string, provider, litellm_kwargs)
        """
        from app.db.repositories import ModelCredentialRepository, ModelSourceRepository

        model_source = ModelSourceRepository(db).get_by_id(model_id)
        if not model_source:
            raise ValueError(f"未找到模型配置: {model_id}")

        provider_rel = model_source.provider_rel
        if not provider_rel:
            raise ValueError(f"模型 {model_id} 的提供商 {model_source.provider} 未配置")

        credential = ModelCredentialRepository(db).get_default(model_id)
        if not credential:
            raise ValueError(f"未找到模型 {model_id} 的凭证配置")

        creds = credential.credentials

        # 从 provider 表读取 LiteLLM 前缀
        litellm_model = f"{provider_rel.litellm_prefix}/{model_id}"

        litellm_kwargs: Dict[str, Any] = {
            "api_key": creds.get("api_key"),
        }

        # 从 provider 表读取是否需要自定义 base_url
        if provider_rel.custom_base_url and creds.get("base_url"):
            litellm_kwargs["api_base"] = creds["base_url"]

        return litellm_model, model_source.provider, litellm_kwargs

    async def test_credentials(
        self,
        provider: str,
        model_id: str,
        credentials: Dict[str, Any],
        db: Session = None,
    ) -> bool:
        """测试凭证是否有效，发送最小测试请求"""
        try:
            from app.db.repositories import ProviderRepository

            if not db:
                raise ValueError("需要数据库会话来查询 provider 配置")

            provider_obj = ProviderRepository(db).get_by_id(provider)
            if not provider_obj:
                raise ValueError(f"未找到提供商配置: {provider}")

            litellm_model = f"{provider_obj.litellm_prefix}/{model_id}"

            kwargs: Dict[str, Any] = {"api_key": credentials.get("api_key")}
            if provider_obj.custom_base_url and credentials.get("base_url"):
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
