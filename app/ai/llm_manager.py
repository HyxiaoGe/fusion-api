from typing import Union, Dict, Type, Protocol
from abc import ABC, abstractmethod

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

class WenxinFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models import QianfanChatEndpoint
        return QianfanChatEndpoint(
            model=model,
            streaming=True,
            timeout=60,
        )

class OpenAIFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True,
            base_url=settings.OPENAI_PROXY_URL
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

class OllamaFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=0.8,
            num_predict=256,
        )

class GeminiFactory:
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

class AnthropicFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=settings.ANTHROPIC_API_KEY,
            base_url="https://api.anthropic.com/v1",
        )

class GroqFactory:
    def create_model(self, model: str) -> Union[LLM, BaseChatModel]:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

class LLMManager:
    """管理不同LLM模型的工厂类"""

    def __init__(self):
        self.models = {}
        self._default_model = None
        self._factories: Dict[str, ModelFactory] = {
            "qwen": QwenFactory(),
            "wenxin": WenxinFactory(),
            "openai": OpenAIFactory(),
            "deepseek": DeepseekFactory(),
            "ollama": OllamaFactory(),
            "gemini": GeminiFactory(),
            "anthropic": AnthropicFactory(),
            "groq": GroqFactory(),
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

        if settings.DASHSCOPE_API_KEY:
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                self._default_model = ChatTongyi(
                    model="qwen-max-0125",
                    api_key=settings.DASHSCOPE_API_KEY,
                    streaming=True,
                )
                return self._default_model
            except Exception as e:
                logger.error(f"默认通义千问模型初始化失败: {e}")

        raise ValueError("无法创建默认模型。请检查API密钥配置。")

# 创建一个全局的LLM管理器实例
llm_manager = LLMManager()

MODEL_DISPLAY_NAMES = {
    "qwen": "通义千问",
    "weixin": "文心一言",
    "claude": "Claude",
    "deepseek": "Deepseek",
    "openai": "OpenAI"
}

def get_model_display_name(model_code):
    return MODEL_DISPLAY_NAMES.get(model_code, model_code)
