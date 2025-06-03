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
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        """创建模型实例"""
        pass

class AnthropicFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
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
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
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
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
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
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models.hunyuan import ChatHunyuan
        return ChatHunyuan(
            model=model,
            streaming=True,
            hunyuan_app_id=credentials.get("hunyuan_app_id"),
            hunyuan_secret_id=credentials.get("hunyuan_secret_id"),
            hunyuan_secret_key=credentials.get("hunyuan_secret_key"),
        )

class OpenAIFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True,
            base_url=credentials.get("base_url"),
            api_key=credentials.get("api_key"),
        ) 

class QwenFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        if options is None:
            options = {}
            
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
        elif "qwen3" in model.lower():
            # QWen3模型需要特殊的enable_thinking配置
            from langchain_community.chat_models.tongyi import ChatTongyi
            
            # 根据options中的use_reasoning参数控制enable_thinking
            # 只有当use_reasoning未明确设置时才默认启用思考模式
            use_reasoning = options.get("use_reasoning")
            if use_reasoning is None:
                use_reasoning = True  # QWen3默认启用思考模式
            
            model_kwargs = {}
            if use_reasoning:
                model_kwargs["enable_thinking"] = True
                
            return ChatTongyi(
                model=model,
                streaming=True,
                model_kwargs=model_kwargs,
                dashscope_api_key=credentials.get("api_key") if credentials else None,
            )
        else:
            from langchain_community.chat_models.tongyi import ChatTongyi
            return ChatTongyi(
                model=model,
                streaming=True,
            )

class VolcengineFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
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
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_community.chat_models import QianfanChatEndpoint
        return QianfanChatEndpoint(
            model=model,
            streaming=True,
            timeout=60,
            api_key=credentials.get("api_key"),
            secret_key=credentials.get("secret_key"),
        )
 
class XAIFactory:
    def create_model(self, model: str, credentials: Dict[str, Any] = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        from langchain_xai import ChatXAI

        return ChatXAI(
            model=model,
            temperature=0,
            max_tokens=None,
            max_retries=2,
            streaming=True,
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

    def get_model(self, provider: str = None, model: str = None, options: Dict[str, Any] = None) -> Union[LLM, BaseChatModel]:
        """获取指定的LLM模型实例"""
        logger.info(f"获取模型: provider={provider}, model={model}")

        if provider and model:
            try:
                factory = self._factories.get(provider)
                if factory:
                    # 获取模型凭证
                    credentials = self._get_model_credentials(provider, model)
                    return factory.create_model(model, credentials, options)
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
            
        try:
            # 使用独立的数据库Session，避免全局连接事务问题
            from app.db.database import SessionLocal
            from app.db.repositories import ModelSourceRepository, ModelCredentialRepository
            
            db = SessionLocal()
            try:
                # 获取模型信息
                model_repo = ModelSourceRepository(db)
                model_source = model_repo.get_by_id(model)
                    
                # 获取模型凭证
                cred_repo = ModelCredentialRepository(db)
                credential = cred_repo.get_default(model)
                
                if not credential:
                    raise ValueError(f"未找到模型 {model} 的凭证配置")
                    
                return credential.credentials
            finally:
                db.close()
        except Exception as e:
            logger.error(f"获取模型凭证失败: {e}")
            raise

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
