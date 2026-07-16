from functools import lru_cache

from app.core.config import settings
from app.services.mcp.client import McpClientManager, McpClientPolicy


@lru_cache(maxsize=1)
def get_mcp_client_manager() -> McpClientManager:
    """返回进程级 MCP Client Manager；连接本身仍按操作安全关闭。"""

    policy = McpClientPolicy(
        allowed_hosts=frozenset(settings.RESOLVED_MCP_ALLOWED_HOSTS),
        allowed_credential_refs=frozenset(settings.RESOLVED_MCP_ALLOWED_CREDENTIAL_REFS),
        connect_timeout_seconds=max(0.1, settings.MCP_CONNECT_TIMEOUT_SECONDS),
        call_timeout_seconds=max(0.1, settings.MCP_CALL_TIMEOUT_SECONDS),
        max_discovery_pages=max(1, settings.MCP_MAX_DISCOVERY_PAGES),
        max_discovered_tools=max(1, settings.MCP_MAX_DISCOVERED_TOOLS),
        max_tool_description_chars=max(1, settings.MCP_MAX_TOOL_DESCRIPTION_CHARS),
        max_tool_schema_bytes=max(1_024, settings.MCP_MAX_TOOL_SCHEMA_BYTES),
        max_response_bytes=max(4_096, settings.MCP_MAX_RESPONSE_BYTES),
    )
    return McpClientManager(policy=policy)
