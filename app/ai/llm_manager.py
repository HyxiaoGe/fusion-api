from typing import Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM

from app.core.config import settings
from app.core.logger import llm_logger

logger = llm_logger


class LLMManager:
    """管理不同LLM模型的工厂类"""

    def __init__(self):
        self.models = {}
        self._default_model = None

    def get_model(self, provider: str = None, model: str = None) -> Union[LLM, BaseChatModel]:
        """获取指定的LLM模型实例"""
        logger.info(f"获取模型: provider={provider}, model={model}")

        # 如果提供了provider和model，动态创建模型实例
        if provider and model:
            try:
                if provider == "qwen":
                    from langchain_community.chat_models.tongyi import ChatTongyi
                    return ChatTongyi(
                        model=model,
                        api_key=settings.QWEN_API_KEY,
                        streaming=True,
                    )
                elif provider == "wenxin":
                    from langchain_community.chat_models import QianfanChatEndpoint
                    return QianfanChatEndpoint(
                        api_key=settings.WENXIN_API_KEY,
                        secret_key=settings.WENXIN_SECRET_KEY,
                        model=model,
                        streaming=True,
                        timeout=60,
                    )
                elif provider == "openai":
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(
                        api_key=settings.OPENAI_API_KEY,
                        model=model,
                        temperature=0.7,
                        streaming=True,
                    )
                elif provider == "deepseek":
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(
                        api_key=settings.DEEPSEEK_API_KEY,
                        base_url="https://api.deepseek.com/v1",
                        model=model,
                        temperature=0.7,
                        streaming=True,
                    )

            except Exception as e:
                logger.error(f"创建 {provider}/{model} 模型失败: {e}")
                raise

        return self.get_default_model()

    def get_default_model(self) -> Union[LLM, BaseChatModel]:
        """获取默认模型实例，用于标题生成等通用场景"""
        # 如果已经缓存了默认模型，直接返回
        if self._default_model is not None:
            return self._default_model

        if settings.QWEN_API_KEY:
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                self._default_model = ChatTongyi(
                    model="qwen-max-0125",
                    api_key=settings.QWEN_API_KEY,
                    streaming=True,
                )
                return self._default_model
            except Exception as e:
                logger.error(f"默认通义千问模型初始化失败: {e}")

        # 如果所有模型都不可用，抛出异常
        raise ValueError("无法创建默认模型。请检查API密钥配置。")


# 创建一个全局的LLM管理器实例
llm_manager = LLMManager()

MODEL_DISPLAY_NAMES = {
    "qwen": "通义千问",
    "wenxin": "文心一言",
    "claude": "Claude",
    "deepseek": "Deepseek",
    "openai": "OpenAI"
}


def get_model_display_name(model_code):
    return MODEL_DISPLAY_NAMES.get(model_code, model_code)
