import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from app.core.function_registry import FunctionRegistry
from app.core.logger import app_logger as logger

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
        
        if provider == "openai":
            return {"tools": functions}
        elif provider == "anthropic":
            return {"tools": functions}
        elif provider == "deepseek":
            return {"tools": functions}
        elif provider == "qwen" and "qwq" in model.lower():
            return {"tools": functions}
        elif provider == "volcengine":
            return {"tools": functions}
        else:
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
            # 根据不同提供商提取函数名和参数
            if provider == "openai":
                function_name = function_call.get("name")
                arguments_str = function_call.get("arguments", "{}")
                arguments = json.loads(arguments_str) if arguments_str.strip() else {}
            elif provider == "anthropic":
                function_name = function_call.get("name")
                arguments_str = function_call.get("arguments", "{}")
                arguments = json.loads(arguments_str) if arguments_str.strip() else {}
            elif provider == "deepseek":
                function_name = function_call.get("name")
                arguments_str = function_call.get("arguments", "{}")
                arguments = json.loads(arguments_str) if arguments_str.strip() else {}
            else:
                # 通用格式
                function_name = function_call.get("name")
                arguments = function_call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except:
                        arguments = {}
                
            logger.info(f"正在调用函数: {function_name}, 参数: {arguments}")
                
            # 调用函数并返回结果
            result = await self.function_registry.call_function(
                function_name, arguments, context
            )
            logger.info(f"函数调用结果: {result}")
            return result
        except Exception as e:
            logger.error(f"处理函数调用时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"函数调用出错: {str(e)}"}
            
    def extract_function_call(self, provider: str, response: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        从模型响应中提取函数调用信息
        
        参数:
            provider: 模型提供商
            response: 模型响应
            
        返回:
            (函数调用信息, 工具调用ID) 如果没有函数调用则返回 (None, None)
        """
        function_call = None
        tool_call_id = None
        
        if provider == "openai":
            # OpenAI格式
            if hasattr(response, "tool_calls") and response.tool_calls:
                function_call = response.tool_calls[0].function
                tool_call_id = response.tool_calls[0].id
        elif provider == "anthropic":
            # Anthropic格式
            if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs and response.additional_kwargs["tool_calls"]:
                function_call = response.additional_kwargs["tool_calls"][0].get("function")
                tool_call_id = response.additional_kwargs["tool_calls"][0].get("id")
        elif provider == "deepseek":
            # DeepSeek格式
            if hasattr(response, "additional_kwargs") and "function_call" in response.additional_kwargs:
                function_call = response.additional_kwargs["function_call"]
        elif provider == "qwen":
            # 通义千问格式 (QwQ)
            if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
                tools = response.additional_kwargs["tool_calls"]
                if tools and len(tools) > 0:
                    function_call = tools[0].get("function")
                    tool_call_id = tools[0].get("id")
        elif provider == "volcengine":
            # 火山引擎格式
            if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
                tools = response.additional_kwargs["tool_calls"]
                if tools and len(tools) > 0:
                    function_call = tools[0].get("function")
                    tool_call_id = tools[0].get("id")
                    
        return function_call, tool_call_id
    
    def detect_function_call_in_stream(self, chunk, model_type=None):
        """
        从流式响应中检测函数调用
        
        参数:
            chunk: 流式响应的一个数据块
            model_type: 模型类型（可选，用于特殊处理）
            
        返回:
            (已检测到函数调用, 函数调用数据)
        """
        # 默认未检测到
        function_call_detected = False
        function_call_data = {}
        
        # print(f"chunk: {chunk}")
        
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
        
        try:
            # 基于OpenAI格式的模型（包括OpenAI, QwQ, 火山引擎等）
            # if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            #     function_call_detected = True
            #     function_call_data = {
            #         "function": chunk.tool_calls[0].function,
            #         "tool_call_id": chunk.tool_calls[0].id
            #     }
            
            # # 基于additional_kwargs格式的模型
            # elif hasattr(chunk, "additional_kwargs"):
            #     # Anthropic格式
            #     if "tool_calls" in chunk.additional_kwargs and chunk.additional_kwargs["tool_calls"]:
            #         function_call_detected = True
            #         tool_call = chunk.additional_kwargs["tool_calls"][0]
            #         function_call_data = {
            #             "function": tool_call.get("function", {}),
            #             "tool_call_id": tool_call.get("id")
            #         }
                
            #     # DeepSeek格式
            #     elif "function_call" in chunk.additional_kwargs:
            #         function_call_detected = True
            #         function_call_data = {
            #             "function": chunk.additional_kwargs["function_call"],
            #             "tool_call_id": None
            #         }
            
            return function_call_detected, function_call_data
        
        except Exception as e:
            logger.error(f"流式函数调用检测出错: {e}")
            return False, {}
    
    def prepare_tool_message(self, provider: str, function_name: str, function_result: Dict[str, Any], tool_call_id: Optional[str] = None) -> Dict[str, Any]:
        """
        准备工具/函数响应消息
        
        参数:
            provider: 模型提供商
            function_name: 函数名称
            function_result: 函数执行结果
            tool_call_id: 工具调用ID (如果有)
            
        返回:
            格式化的工具/函数响应消息
        """
        content = json.dumps(function_result, ensure_ascii=False)
        
        if provider in ["openai", "anthropic", "qwen", "volcengine", "deepseek"] and tool_call_id:
            return {
                "role": "tool", 
                "content": content,
                "tool_call_id": tool_call_id
            }
        else:
            # 通用格式
            return {
                "role": "function", 
                "name": function_name,
                "content": content
            }