from app.services.function_handlers.web_search import web_search_handler
from app.services.function_handlers.file_analysis import analyze_file_handler
from app.services.function_handlers.hot_topics import hot_topics_handler

# 导出处理器供其他模块使用
__all__ = [
    "web_search_handler",
    "analyze_file_handler",
    "hot_topics_handler"
]