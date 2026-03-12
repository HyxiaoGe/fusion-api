import json
from typing import Dict, List, Any, Optional

from app.core.function_registry import FunctionRegistry
from app.core.logger import app_logger as logger


TOOL_PROVIDER_SET = {"openai", "anthropic", "deepseek", "qwen", "volcengine", "google", "xai"}


class FunctionCallAdapter:
    """处理不同提供商的Function Call适配"""
    
    def __init__(self, function_registry: FunctionRegistry):
        """
        初始化适配器
        
        参数:
            function_registry: 函数注册表实例
        """
        self.function_registry = function_registry
    
    def prepare_functions_for_model(self, provider: str, model: str, function_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        准备函数定义为模型可接受的格式
        
        参数:
            provider: 模型提供商
            model: 模型名称
            function_names: 要使用的函数名称列表, 如果为None则使用所有函数
            
        返回:
            适用于特定模型的函数调用参数
        """
        functions = self.function_registry.get_functions_for_provider(
            provider, function_names
        )

        if provider in TOOL_PROVIDER_SET and (provider != "qwen" or "qwq" in model.lower()):
            return {"tools": functions}

        # 默认格式
        return {"functions": functions}
    
    async def process_function_call(self, provider: str, function_call: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理函数调用并返回结果
        
        参数:
            provider: 模型提供商
            function_call: 函数调用信息
            context: 上下文信息
            
        返回:
            函数执行结果
        """
        try:
            function_name = function_call.get("name")
            arguments = self._parse_function_arguments(provider, function_call)
                
            logger.info(f"正在调用函数: {function_name}, 参数: {arguments}")
                
            # 调用函数并返回结果
            result = await self.function_registry.call_function(
                function_name, arguments, context
            )
            logger.info(f"函数调用结果: {result}")
            return result
        except Exception as e:
            logger.error(f"处理函数调用时出错: {e}")
            logger.exception("函数调用适配器异常详情")
            return {"error": f"函数调用出错: {str(e)}"}

    def _parse_function_arguments(self, provider: str, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """归一化不同 provider 的函数参数格式。"""
        arguments = function_call.get("arguments", "{}" if provider in TOOL_PROVIDER_SET else {})
        if isinstance(arguments, str):
            try:
                return json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return {}
        return arguments or {}

    def detect_function_call_in_stream(self, chunk):
        """
        从流式响应中检测函数调用
        
        参数:
            chunk: 流式响应的一个数据块
            
        返回:
            (已检测到函数调用, 函数调用数据)
        """
        # 默认未检测到
        function_call_detected = False
        function_call_data = {}
        
        # 1. 首先检查additional_kwargs中的tool_calls格式
        if hasattr(chunk, "additional_kwargs") and "tool_calls" in chunk.additional_kwargs:
            tool_calls = chunk.additional_kwargs["tool_calls"]
            if tool_calls and len(tool_calls) > 0:
                # 完整性检查
                function_name = tool_calls[0].get("function", {}).get("name")
                if function_name:
                    return True, {
                        "function": tool_calls[0].get("function", {}),
                        "tool_call_id": tool_calls[0].get("id")
                    }
        
        # 2. 检查additional_kwargs中的function_call格式 (如Google等)
        if hasattr(chunk, "additional_kwargs") and "function_call" in chunk.additional_kwargs:
            function_call = chunk.additional_kwargs["function_call"]
            if isinstance(function_call, dict) and function_call.get("name"):
                return True, {
                    "function": function_call,
                    "tool_call_id": None
                }
        
        # 3. 检查标准tool_calls格式 (适用于Google等模型)
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                # 检查是否有name属性（直接在tool_call上）
                if hasattr(tool_call, 'name') and tool_call.name:
                    return True, {
                        "function": {
                            "name": tool_call.name,
                            "arguments": getattr(tool_call, 'args', getattr(tool_call, 'arguments', '{}'))
                        },
                        "tool_call_id": getattr(tool_call, 'id', None)
                    }
                
                # 检查是否有function属性
                if hasattr(tool_call, 'function'):
                    function_data = tool_call.function
                    if hasattr(function_data, 'name') and function_data.name:
                        return True, {
                            "function": {
                                "name": function_data.name,
                                "arguments": getattr(function_data, 'arguments', '{}')
                            },
                            "tool_call_id": getattr(tool_call, 'id', None)
                        }
                
                # 如果tool_call是字典格式
                if isinstance(tool_call, dict):
                    if tool_call.get('name'):
                        return True, {
                            "function": {
                                "name": tool_call.get('name'),
                                "arguments": tool_call.get('args', tool_call.get('arguments', '{}'))
                            },
                            "tool_call_id": tool_call.get('id')
                        }
        
        return function_call_detected, function_call_data
