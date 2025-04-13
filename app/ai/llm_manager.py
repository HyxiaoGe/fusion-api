from typing import Union, Dict, Type, Protocol
from abc import ABC, abstractmethod
import os

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM

from app.core.config import settings
from app.core.logger import llm_logger

logger = llm_logger

class ModelFactory(Protocol):
    """模型工厂接口"""
    @abstractmethod
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        """创建模型实例"""
        pass

class AnthropicFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            base_url=os.getenv("ANTHROPIC_API_BASE"),
        )

class DeepseekFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True
        )

class GoogleFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True
        )

class HunyuanFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models.hunyuan import ChatHunyuan
        return ChatHunyuan(
            model=model,
            streaming=True
        )

class OpenAIFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True,
            base_url=os.getenv("OPENAI_BASE_URL")
        ) 

class QwenFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        if "qwq" in model.lower():
            from langchain_qwq import ChatQwQ
            return ChatQwQ(
                model=model,
                max_tokens=3_000,
                timeout=None,
                max_retries=2,
                streaming=True,
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            from langchain_community.chat_models.tongyi import ChatTongyi
            return ChatTongyi(
                model=model,
                streaming=True,
            )

class VolcengineFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=4000,
            timeout=30,
            max_retries=2,
            streaming=True,
            api_key=os.getenv("VOLCENGINE_API_KEY"),
            base_url=os.getenv("VOLCENGINE_API_BASE")
        )

class WenxinFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models import QianfanChatEndpoint
        return QianfanChatEndpoint(
            model=model,
            streaming=True,
            timeout=60,
        )
 
class XAIFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_xai import ChatXAI

        return ChatXAI(
            model="grok-beta",
            temperature=0,
            max_tokens=None,
            max_retries=2,
            xai_api_base=os.getenv("XAI_API_BASE")
        )

class LLMManager:
    """管理不同LLM模型的工厂类"""

    def __init__(self):
        self.models = {}
        self._default_model = None
        self._factories: Dict[str, ModelFactory] = {
            "anthropic": AnthropicFactory(),
            "deepseek": DeepseekFactory(),
            "google": GoogleFactory(),
            "hunyuan": HunyuanFactory(),
            "openai": OpenAIFactory(),
            "qwen": QwenFactory(),
            "volcengine": VolcengineFactory(),
            "wenxin": WenxinFactory(),
            "xai": XAIFactory(),
        }

    def get_model(self, provider: str = None, model: str = None) -> Union[LLM, BaseChatModel]:
        """获取指定的LLM模型实例"""
        logger.info(f"获取模型: provider={provider}, model={model}")

        if provider and model:
            try:
                factory = self._factories.get(provider)
                if factory:
                    return factory.create_model(model)
                else:
                    raise ValueError(f"不支持的模型提供者: {provider}")
            except Exception as e:
                logger.error(f"创建 {provider}/{model} 模型失败: {e}")
                raise

        return self.get_default_model()

    def get_default_model(self) -> Union[LLM, BaseChatModel]:
        """获取默认模型实例，用于标题生成等通用场景"""
        if self._default_model is not None:
            return self._default_model

        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
            self._default_model = ChatTongyi(
                model="qwen-max-0125",
                streaming=True,
            )
            return self._default_model
        except Exception as e:
            logger.error(f"默认通义千问模型初始化失败: {e}")

        raise ValueError("无法创建默认模型。请检查API密钥配置。")

# 创建一个全局的LLM管理器实例
llm_manager = LLMManager()

MODEL_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "deepseek": "Deepseek",
    "google": "Google",
    "hunyuan": "混元",
    "openai": "OpenAI",
    "qwen": "通义千问",
    "volcengine": "火山引擎",
    "weixin": "文心一言",
    "xai": "XAI"
}

def get_model_display_name(model_code):
    return MODEL_DISPLAY_NAMES.get(model_code, model_code)
