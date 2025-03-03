import logging
from typing import Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """管理不同LLM模型的工厂类"""

    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """初始化所有配置的模型"""

        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                self.models["openai"] = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model="gpt-3.5-turbo"
                )
                logger.info("OpenAI模型初始化成功")
            except Exception as e:
                logger.error(f"OpenAI模型初始化失败: {e}")

        # 文心一言
        if settings.WENXIN_API_KEY and settings.WENXIN_SECRET_KEY:
            try:
                from langchain_community.chat_models import QianfanChatEndpoint
                self.models["wenxin"] = QianfanChatEndpoint(
                    api_key=settings.WENXIN_API_KEY,
                    secret_key=settings.WENXIN_SECRET_KEY,
                    model_name="ERNIE-Bot-4"
                )
                logger.info("文心一言模型初始化成功")
            except Exception as e:
                logger.error(f"文心一言模型初始化失败: {e}")

        # 通义千问
        if settings.QIANWEN_API_KEY:
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                self.models["qianwen"] = ChatTongyi(
                    model="Qianfan-Chinese-Llama-2-7B",
                    api_key=settings.QIANWEN_API_KEY
                )
                logger.info("通义千问模型初始化成功")
            except Exception as e:
                logger.error(f"通义千问模型初始化失败: {e}")

        # Claude
        if settings.CLAUDE_API_KEY:
            try:
                self.models["claude"] = ChatAnthropic(
                    anthropic_api_key=settings.CLAUDE_API_KEY,
                    model="claude-3-sonnet-20240229"
                )
                logger.info("Claude模型初始化成功")
            except Exception as e:
                logger.error(f"Claude模型初始化失败: {e}")

        # Deepseek
        if settings.DEEPSEEK_API_KEY:
            try:
                self.models["deepseek"] = ChatOpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-chat"
                )
                logger.info("Deepseek模型初始化成功")
            except Exception as e:
                logger.error(f"Deepseek模型初始化失败: {e}")

    def get_model(self, model_name: str = None) -> Union[LLM, BaseChatModel]:
        """获取指定的LLM模型实例"""
        if not model_name:
            model_name = settings.DEFAULT_MODEL

        if model_name not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                raise ValueError("没有可用的LLM模型。请检查API密钥配置。")

            logger.warning(f"请求的模型 '{model_name}' 不可用，使用 '{available_models[0]}' 替代")
            model_name = available_models[0]

        return self.models[model_name]

    def list_available_models(self):
        """列出所有可用的模型名称"""
        return list(self.models.keys())


# 创建一个全局的LLM管理器实例
llm_manager = LLMManager()
