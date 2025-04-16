from typing import Union, Dict, Type, Protocol, Any
from abc import ABC, abstractmethod
import os

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM

from app.core.logger import llm_logger

logger = llm_logger

class ModelFactory(Protocol):
    """模型工厂接口"""
    @abstractmethod
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        """创建模型实例"""
        pass

class AnthropicFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_anthropic import ChatAnthropic
        
        
        return ChatAnthropic(
            model=model,
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=credentials.get("api_key"),
            base_url=credentials.get("base_url"),
        )

class DeepseekFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
            api_key=credentials.get("api_key"),
        )

class GoogleFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
            google_api_key=credentials.get("api_key"),
            google_api_base=credentials.get("base_url"),
        )

class HunyuanFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models.hunyuan import ChatHunyuan
        return ChatHunyuan(
            model=model,
            streaming=True,
            hunyuan_app_id=credentials.get("hunyuan_app_id"),
            hunyuan_secret_id=credentials.get("hunyuan_secret_id"),
            hunyuan_secret_key=credentials.get("hunyuan_secret_key"),
        )

class OpenAIFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True,
            base_url=credentials.get("base_url"),
            api_key=credentials.get("api_key"),
        ) 

class QwenFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        if "qwq" in model.lower():
            from langchain_qwq import ChatQwQ
            return ChatQwQ(
                model=model,
                max_tokens=3_000,
                timeout=None,
                max_retries=2,
                streaming=True,
                api_base=credentials.get("base_url"),
                api_key=credentials.get("api_key"),
            )
        else:
            from langchain_community.chat_models.tongyi import ChatTongyi
            return ChatTongyi(
                model=model,
                streaming=True,
            )

class VolcengineFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=4000,
            timeout=30,
            max_retries=2,
            streaming=True,
            api_key=credentials.get("api_key"),
            base_url=credentials.get("base_url"),
        )

class WenxinFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models import QianfanChatEndpoint
        return QianfanChatEndpoint(
            model=model,
            streaming=True,
            timeout=60,
            api_key=credentials.get("api_key"),
            secret_key=credentials.get("secret_key"),
        )
 
class XAIFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_xai import ChatXAI

        return ChatXAI(
            model="grok-beta",
            temperature=0,
            max_tokens=None,
            max_retries=2,
            xai_api_key=credentials.get("api_key"),
            xai_api_base=credentials.get("base_url"),
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
        self.db = None

    def get_model(self, provider: str = None, model: str = None) -> Union[LLM, BaseChatModel]:
        """获取指定的LLM模型实例"""
        logger.info(f"获取模型: provider={provider}, model={model}")

        if provider and model:
            try:
                factory = self._factories.get(provider)
                if factory:
                    # 获取模型凭证
                    credentials = self._get_model_credentials(provider, model)
                    return factory.create_model(model, credentials)
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
    
    def _get_model_credentials(self, provider: str, model: str) -> Dict[str, Any]:
        """从数据库获取模型凭证"""
        if not self.db:
            # 如果未设置数据库，返回环境变量中的凭证（向后兼容）
            return self._get_env_credentials(provider)
            
        try:
            # 从数据库获取凭证
            from app.db.repositories import ModelSourceRepository, ModelCredentialRepository
            
            # 获取模型信息
            model_repo = ModelSourceRepository(self.db)
            model_source = model_repo.get_by_id(model)
            
            if not model_source:
                # 模型不存在，返回环境变量中的凭证
                return self._get_env_credentials(provider)
                
            # 获取模型凭证
            cred_repo = ModelCredentialRepository(self.db)
            credential = cred_repo.get_default(model)
            
            if not credential:
                # 凭证不存在，返回环境变量中的凭证
                return self._get_env_credentials(provider)
                
            return credential.credentials
        except Exception as e:
            logger.error(f"获取模型凭证失败: {e}")
            # 出错时返回环境变量中的凭证
            return self._get_env_credentials(provider)

    def _get_env_credentials(self, provider: str) -> Dict[str, Any]:
        """从环境变量获取凭证（向后兼容）"""
        credentials = {}
        
        if provider == "anthropic":
            credentials = {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": os.getenv("ANTHROPIC_API_BASE")
            }
        elif provider == "openai":
            credentials = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL")
            }
        # ... 其他模型提供商的凭证
            
        return credentials

    async def test_credentials(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """测试凭证是否有效"""
        try:
            # 创建临时模型实例来测试凭证
            factory = self._factories.get(provider)
            if not factory:
                raise ValueError(f"不支持的模型提供者: {provider}")
                
            # 获取一个通用的模型ID
            model_id = self._get_default_model_id(provider)
            
            # 创建模型实例
            model = factory.create_model(model_id, credentials)
            
            # 执行一个简单测试查询
            model.invoke("hello")
            
            # 如果没有错误，则认为凭证有效
            return True
        except Exception as e:
            # 记录错误并返回失败
            logger.error(f"凭证测试失败: {e}")
            return False
            
    def _get_default_model_id(self, provider: str) -> str:
        """获取提供商默认模型ID"""
        provider_models = {
            "anthropic": "claude-3-haiku-20240307",
            "deepseek": "deepseek-chat",
            "google": "gemini-1.5-flash",
            "hunyuan": "hunyuan-turbos-latest",
            "openai": "gpt-3.5-turbo",
            "qwen": "qwen-max-latest",
            "volcengine": "doubao-1-5-lite-32k-250115",
            "wenxin": "ERNIE-4.0-8K-Latest",
            "xai": "grok-2-image-1212"
        }
        
        return provider_models.get(provider, "default-model")

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
