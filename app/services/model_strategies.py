import logging
from abc import ABC, abstractmethod

from app.ai.llm_manager import llm_manager
from app.schemas.chat import Message

logger = logging.getLogger(__name__)
class ModelStrategy(ABC):
    """模型处理策略的抽象基类"""
    
    @abstractmethod
    async def process(self, provider, model, messages, conversation_id, memory_service, options=None):
        """处理请求并返回响应"""
        pass


class NormalModelStrategy(ModelStrategy):
    """普通模型处理策略"""
    
    async def process(self, provider, model, messages, conversation_id, memory_service, options=None):
        if options is None:
            options = {}
            
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
    
    async def process(self, provider, model, messages, conversation_id, memory_service, options=None):
        if options is None:
            options = {}
            
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
    def get_strategy(provider, model, options=None):
        """根据提供商、模型和选项选择合适的策略
        
        Args:
            provider (str): 模型提供商
            model (str): 模型名称
            options (dict, optional): 其他选项，包含use_reasoning表示是否使用推理
            
        Returns:
            ModelStrategy: 合适的模型处理策略
        """
        if options is None:
            options = {}
            
        # 获取是否使用推理模式的标志
        use_reasoning = options.get("use_reasoning", False)
        
        # 优先根据options中的use_reasoning判断
        if use_reasoning:
            return ReasoningModelStrategy()
        
        # 火山引擎模型特殊处理
        if provider == "volcengine" and ("thinking" in model.lower() or "deepseek-r1" in model.lower()):
            return ReasoningModelStrategy()
            
        # 根据模型名称判断（兼容旧代码）
        if provider == "deepseek" and model == "deepseek-reasoner":
            return ReasoningModelStrategy()
        elif provider == "qwen" and "qwq" in model.lower():
            return ReasoningModelStrategy()
        else:
            return NormalModelStrategy() 