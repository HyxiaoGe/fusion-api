"""
Tool Handler 调度器 — 根据 tool_call_name 分发到对应 handler
"""

from app.services.tool_handlers.base import BaseToolHandler, ToolContext, ToolResult

# 延迟注册，避免循环导入
_registry: dict[str, BaseToolHandler] = {}


def register_handler(handler: BaseToolHandler) -> None:
    """注册 tool handler"""
    _registry[handler.tool_name] = handler


def get_handler(tool_name: str) -> BaseToolHandler | None:
    """获取已注册的 handler"""
    # 首次访问时触发注册
    if not _registry:
        _register_all()
    return _registry.get(tool_name)


def get_all_handlers() -> dict[str, BaseToolHandler]:
    """获取所有已注册的 handler"""
    if not _registry:
        _register_all()
    return _registry


def _register_all() -> None:
    """注册所有 handler（延迟加载）"""
    from app.services.tool_handlers.url_read import UrlReadHandler
    from app.services.tool_handlers.web_search import WebSearchHandler

    register_handler(WebSearchHandler())
    register_handler(UrlReadHandler())


__all__ = [
    "BaseToolHandler",
    "ToolContext",
    "ToolResult",
    "get_handler",
    "get_all_handlers",
    "register_handler",
]
