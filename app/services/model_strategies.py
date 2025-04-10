import logging
from abc import ABC, abstractmethod

from app.ai.llm_manager import llm_manager
from app.schemas.chat import Message


class ModelStrategy(ABC):
    """模型处理策略的抽象基类"""
    
    @abstractmethod
    async def process(self, provider, model, messages, conversation_id, memory_service):
        """处理请求并返回响应"""
        pass


class NormalModelStrategy(ModelStrategy):
    """普通模型处理策略"""
    
    async def process(self, provider, model, messages, conversation_id, memory_service):
        try:
            # 获取AI模型
            llm = llm_manager.get_model(provider=provider, model=model)

            # 调用模型
            response = llm.invoke(messages)
            
            # 获取最终答案
            ai_content = response.content if hasattr(response, 'content') else response

            # 记录最终答案
            ai_message = Message(
                role="assistant",
                content=ai_content
            )
            
            return ai_message, None
        except Exception as e:
            logging.error(f"普通模型处理失败: {e}")
            raise


class ReasoningModelStrategy(ModelStrategy):
    """推理模型处理策略"""
    
    async def process(self, provider, model, messages, conversation_id, memory_service):
        try:
            # 获取AI模型
            llm = llm_manager.get_model(provider=provider, model=model)

            # 调用模型
            response = llm.invoke(messages)
            
            # 从响应中提取 reasoning_content 和 content
            reasoning_content = ''
            if hasattr(response, 'reasoning_content'):
                reasoning_content = response.reasoning_content
            elif hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
                reasoning_content = response.additional_kwargs['reasoning_content']
            
            # 获取最终答案
            ai_content = response.content if hasattr(response, 'content') else response

            # 记录推理过程
            reasoning_message = None
            if reasoning_content:
                reasoning_message = Message(
                    role="reasoning",
                    content=reasoning_content
                )

            # 记录最终答案
            ai_message = Message(
                role="assistant",
                content=ai_content
            )
            
            return ai_message, reasoning_message
        except Exception as e:
            logging.error(f"推理模型处理失败: {e}")
            raise


class ModelStrategyFactory:
    """策略工厂，用于创建适合特定模型的处理策略"""
    
    @staticmethod
    def get_strategy(provider, model):
        """根据提供商和模型选择合适的策略"""
        if provider == "deepseek" and model == "deepseek-reasoner":
            return ReasoningModelStrategy()
        elif provider == "qwen" and "qwq" in model.lower():
            return ReasoningModelStrategy()
        else:
            return NormalModelStrategy() 