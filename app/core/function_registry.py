import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable

Function = Dict[str, Any]
FunctionHandler = Callable[[Dict[str, Any], Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]

class FunctionRegistry:
    """函数注册表，管理可供AI调用的函数"""
    
    def __init__(self):
        # 存储函数定义和处理器
        self._functions: Dict[str, Dict[str, Any]] = {}
        # 缓存格式化后的函数定义
        self._formatted_functions: Dict[str, Dict[str, Any]] = {
            "openai": {},
            "anthropic": {},
            "deepseek": {},
            "qwen": {}
        }
        
    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 handler: FunctionHandler, categories: Optional[list[str]] = None) -> None:
        """
        注册一个新函数
        
        参数:
            name: 函数名称
            description: 函数描述
            parameters: 函数参数定义(JSON Schema格式)
            handler: 函数处理器
            categories: 函数分类标签列表
        """
        function_def = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        
        self._functions[name] = {
            "definition": function_def,
            "handler": handler,
            "categories": categories or []
        }
        
        # 清除缓存的格式化函数
        for provider in self._formatted_functions:
            if name in self._formatted_functions[provider]:
                del self._formatted_functions[provider][name]
                
    def get_functions_for_provider(self, provider: str, function_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        获取指定提供商格式的函数定义
        
        参数:
            provider: 模型提供商名称 (openai, anthropic, deepseek, qwen 等)
            function_names: 要获取的函数名称列表, 如果为None则获取所有函数
            
        返回:
            格式化后的函数定义列表
        """
        if function_names is None:
            function_names = list(self._functions.keys())
            
        result = []
        for name in function_names:
            if name not in self._functions:
                continue
                
            # 检查缓存
            if name in self._formatted_functions.get(provider, {}):
                result.append(self._formatted_functions[provider][name])
                continue
                
            # 格式化为特定提供商的格式
            func_def = self._format_for_provider(self._functions[name]["definition"], provider)
            
            # 缓存结果
            if provider not in self._formatted_functions:
                self._formatted_functions[provider] = {}
            self._formatted_functions[provider][name] = func_def
            result.append(func_def)
            
        return result
    
    def _format_for_provider(self, function_def: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        将通用函数定义格式化为特定提供商的格式
        
        参数:
            function_def: 通用函数定义
            provider: 模型提供商名称
            
        返回:
            格式化后的函数定义
        """
        if provider in ["openai", "deepseek", "qwen"]:
            return {
                "type": "function",
                "function": function_def
            }
        elif provider == "anthropic":
            return {
                "name": function_def["name"],
                "description": function_def["description"],
                "input_schema": function_def["parameters"]
            }
        else:
            # 默认返回原始定义
            return function_def
        
    async def call_function(self, function_name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        调用注册的函数
        
        参数:
            function_name: 要调用的函数名称
            args: 函数参数
            context: 上下文信息(如数据库连接)
        
        返回:
            函数执行结果
        """
        if function_name not in self._functions:
            raise ValueError(f"未注册的函数: {function_name}")
            
        handler = self._functions[function_name]["handler"]
        
        # 支持同步和异步处理器
        if asyncio.iscoroutinefunction(handler):
            return await handler(args, context or {})
        else:
            return handler(args, context or {})
        
    def get_functions_by_category(self, category: str) -> List[str]:
        """
        获取特定分类的所有函数名称
        
        参数:
            category: 分类名称
            
        返回:
            属于该分类的函数名称列表
        """
        return [
            name for name, func in self._functions.items() 
            if category in func["categories"]
        ]
        
    def get_all_function_names(self) -> List[str]:
        """
        获取所有注册的函数名称
        
        返回:
            所有注册函数的名称列表
        """
        return list(self._functions.keys())
    
    def get_handler(self, function_name: str) -> Optional[FunctionHandler]:
        """
        获取函数处理器
        
        参数:
            function_name: 函数名称
            
        返回:
            函数处理器，如果不存在则返回None
        """
        if function_name not in self._functions:
            return None
        return self._functions[function_name]["handler"]