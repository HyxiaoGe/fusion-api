"""
函数调用管理器
负责初始化和管理全局函数注册表
"""
import asyncio
from typing import Dict, List, Any, Optional

from app.core.function_registry import FunctionRegistry
from app.ai.function_call_adapter import FunctionCallAdapter
from app.services.function_handlers import web_search_handler, analyze_file_handler, hot_topics_handler
from app.core.logger import app_logger as logger

# 创建全局函数注册表实例
function_registry = FunctionRegistry()

# 创建全局适配器实例
function_adapter = FunctionCallAdapter(function_registry)

def init_function_registry():
    """初始化函数注册表，注册所有可用函数"""
    try:
        # 注册Web搜索函数
        function_registry.register(
            name="web_search",
            description="在互联网上搜索最新信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询文本"},
                    "limit": {"type": "integer", "description": "返回结果数量", "default": 5}
                },
                "required": ["query"]
            },
            handler=web_search_handler,
            categories=["web", "search"]
        )

        # 注册文件分析函数
        function_registry.register(
            name="analyze_file",
            description="分析上传的文件内容",
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "要分析的文件ID"},
                    "analysis_type": {
                        "type": "string", 
                        "enum": ["summary", "extract_data", "answer_questions"],
                        "description": "分析类型，summary=生成摘要，extract_data=提取数据，answer_questions=回答关于文件的问题"
                    },
                    "query": {"type": "string", "description": "如果类型是answer_questions，这里是要提问的问题"}
                },
                "required": ["file_id"]
            },
            handler=analyze_file_handler,
            categories=["file", "analysis"]
        )
        
        # 注册热点话题函数
        function_registry.register(
            name="hot_topics",
            description="获取当前热点话题信息",
            parameters={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "话题类别，如科技、财经等(可选)"},
                    "limit": {"type": "integer", "description": "返回结果数量", "default": 5},
                    "topic_id": {"type": "string", "description": "如果要获取特定话题详情，提供话题ID"}
                }
            },
            handler=hot_topics_handler,
            categories=["news", "topics"]
        )

        logger.info("函数注册表初始化完成")
        return True
    except Exception as e:
        logger.error(f"初始化函数注册表失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False