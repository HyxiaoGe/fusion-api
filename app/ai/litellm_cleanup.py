"""LiteLLM 异步客户端清理。

LiteLLM 会在内部缓存 aiohttp/httpx 异步客户端；应用 reload/shutdown 时如果不显式
关闭，asyncio 会记录 "Unclosed client session"。
"""

from app.core.logger import app_logger


async def close_async_clients() -> None:
    """关闭 LiteLLM 内部缓存的异步客户端。

    这是 best-effort shutdown 清理：LiteLLM 版本变更或清理异常不应阻断应用关闭。
    """
    try:
        from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
    except ImportError as exc:
        app_logger.warning(f"LiteLLM async clients 清理入口不可用: {exc}")
        return

    try:
        await close_litellm_async_clients()
    except Exception as exc:
        app_logger.warning(f"LiteLLM async clients 清理失败: {exc}")
