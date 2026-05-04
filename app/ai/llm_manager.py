# app/ai/llm_manager.py
import logging
import os
from typing import Any, Dict, Optional, Tuple

import litellm
from sqlalchemy.orm import Session

from app.ai.litellm_utils import ProviderOfflineError
from app.core.logger import app_logger as logger
from app.db.repositories import ModelSourceRepository, UserCredentialRepository

litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLMManager:
    """统一的 LLM 调用管理器，基于 LiteLLM Proxy 路由。"""

    def resolve_model(
        self,
        model_id: str,
        db: Session,
        user_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """解析出 LiteLLM 调用所需参数。

        - 用户有 active credential → 通过 extra_body['api_key'] 透传上游 key
        - 否则 → 不传 extra_body，由 proxy 用 .env 系统 key
        - provider.status == 'offline' → 抛 ProviderOfflineError，根本不发请求
        - 兼容：user_id=None 等同 system 路径
        """
        model_source = ModelSourceRepository(db).get_by_id(model_id)
        if not model_source:
            raise ValueError(f"未找到模型配置: {model_id}")

        provider_rel = model_source.provider_rel
        if not provider_rel:
            raise ValueError(f"模型 {model_id} 的提供商 {model_source.provider} 未配置")

        if provider_rel.status == "offline":
            raise ProviderOfflineError(
                provider_id=provider_rel.id,
                reason=provider_rel.offline_reason,
                message=provider_rel.offline_message,
            )

        upstream_key: Optional[str] = None
        source = "system"
        if user_id:
            try:
                upstream_key, source = UserCredentialRepository(db).resolve(user_id, provider_rel.id)
            except Exception as e:
                logger.warning(f"凭证解析失败 fallback 系统 key: user={user_id} provider={provider_rel.id} err={e}")

        litellm_model = f"openai/{provider_rel.litellm_prefix}/{model_id}"
        proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
        proxy_key = os.environ.get("LITELLM_API_KEY", "")

        kwargs: Dict[str, Any] = {
            "api_key": proxy_key,
            "api_base": proxy_url,
            "metadata": {"credential_source": source, "provider_id": provider_rel.id},
        }
        if upstream_key:
            kwargs["extra_body"] = {"api_key": upstream_key}

        return litellm_model, provider_rel.id, kwargs

    async def test_credentials(
        self,
        provider: str,
        model_id: str,
        credentials: Optional[Dict[str, Any]],
        db: Session,
    ) -> Dict[str, Any]:
        """验证一个 key 是否有效。
        - credentials.api_key 有 → 通过 extra_body 透传到 upstream
        - 否则 → 测系统默认 key（无 extra_body）

        返回 {"valid": bool, "reason"?: str, "message"?: str}。
        本方法不写任何 health 状态。
        """
        from app.ai.litellm_utils import categorize
        from app.db.repositories import ProviderRepository

        provider_obj = ProviderRepository(db).get_by_id(provider)
        if not provider_obj:
            return {"valid": False, "reason": "unknown", "message": f"未知 provider: {provider}"}

        litellm_model = f"openai/{provider_obj.litellm_prefix}/{model_id}"
        proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
        proxy_key = os.environ.get("LITELLM_API_KEY", "")

        kwargs: Dict[str, Any] = {
            "api_key": proxy_key,
            "api_base": proxy_url,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        if credentials and credentials.get("api_key"):
            kwargs["extra_body"] = {"api_key": credentials["api_key"]}

        try:
            await litellm.acompletion(model=litellm_model, **kwargs)
            return {"valid": True}
        except Exception as exc:
            kind, msg = categorize(exc)
            return {"valid": False, "reason": kind.value, "message": msg}


# 全局单例
llm_manager = LLMManager()
