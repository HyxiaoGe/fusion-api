from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

AMAP_MCP_HOST = "mcp.amap.com"
CONTEXT7_MCP_HOST = "mcp.context7.com"
AMAP_CREDENTIAL_REF = "AMAP_MCP_API_KEY"
CONTEXT7_CREDENTIAL_REF = "CONTEXT7_API_KEY"
AMAP_READ_ONLY_TOOL_ALLOWLIST = frozenset(
    {
        "maps_geo",
        "maps_regeocode",
        "maps_weather",
        "maps_direction_bicycling",
        "maps_direction_walking",
        "maps_direction_driving",
        "maps_direction_transit_integrated",
        "maps_distance",
        "maps_text_search",
        "maps_around_search",
        "maps_search_detail",
    }
)
CONTEXT7_READ_ONLY_TOOL_ALLOWLIST = frozenset(
    {
        "resolve-library-id",
        "query-docs",
    }
)

_AMAP_TOOL_GUIDANCE = {
    "maps_geo": " 当用户给出自然语言地点且后续需要坐标时先使用本工具，不得猜测经纬度。",
    "maps_regeocode": " 仅使用用户提供或可信工具返回的坐标，不得猜测经纬度。",
    "maps_text_search": " 用于按城市和关键词粗搜 POI；不得把未返回的实时排队、人均价格或空位信息当作事实。",
    "maps_around_search": " location 必须来自本轮可信工具结果，不得猜测经纬度。",
    "maps_search_detail": " id 必须来自本轮地点搜索返回的 POI ID，不得自行编造。",
    "maps_distance": " 起终点坐标必须来自用户输入或本轮可信工具结果，不得猜测经纬度。",
    "maps_direction_bicycling": " 起终点坐标必须来自用户输入或本轮可信工具结果，不得猜测经纬度。",
    "maps_direction_walking": " 起终点坐标必须来自用户输入或本轮可信工具结果，不得猜测经纬度。",
    "maps_direction_driving": " 起终点坐标必须来自用户输入或本轮可信工具结果，不得猜测经纬度。",
    "maps_direction_transit_integrated": " 起终点坐标必须来自用户输入或本轮可信工具结果，不得猜测经纬度。",
}
_CONTEXT7_TOOL_GUIDANCE = {
    "resolve-library-id": (
        " 仅用于解析公开软件库的 Context7 标识；libraryName 使用公开库名，query 只描述待查的公开文档问题。"
        "不得发送密钥、令牌、个人数据、私有源码、内部实现或完整用户内容。"
    ),
    "query-docs": (
        " 必须先调用 resolve-library-id；libraryId 只能使用本轮解析结果，不得自行编造。"
        "query 只写单行、最小化的公开文档问题，不得发送代码、密钥、令牌、个人数据或内部实现。"
    ),
}
_CONTEXT7_TOOL_SCHEMA_OVERRIDES: dict[str, dict[str, Any]] = {
    "resolve-library-id": {
        "type": "object",
        "properties": {
            "libraryName": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
        },
        "required": ["libraryName", "query"],
        "additionalProperties": False,
    },
    "query-docs": {
        "type": "object",
        "properties": {
            "libraryId": {
                "type": "string",
                "minLength": 3,
                "maxLength": 200,
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
        },
        "required": ["libraryId", "query"],
        "additionalProperties": False,
    },
}


def _endpoint_hostname(endpoint_url: str) -> str:
    try:
        return (urlsplit(endpoint_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def endpoint_tool_allowlist(endpoint_url: str) -> frozenset[str] | None:
    """返回指定官方端点的硬白名单；其他 MCP 服务继续使用管理员授权。"""

    hostname = _endpoint_hostname(endpoint_url)
    if hostname == AMAP_MCP_HOST:
        return AMAP_READ_ONLY_TOOL_ALLOWLIST
    if hostname == CONTEXT7_MCP_HOST:
        return CONTEXT7_READ_ONLY_TOOL_ALLOWLIST
    return None


def is_official_amap_endpoint(endpoint_url: str) -> bool:
    return _endpoint_hostname(endpoint_url) == AMAP_MCP_HOST


def is_official_context7_endpoint(endpoint_url: str) -> bool:
    return _endpoint_hostname(endpoint_url) == CONTEXT7_MCP_HOST


def endpoint_auth_binding_is_allowed(
    endpoint_url: str,
    *,
    auth_type: str,
    auth_name: str | None,
    credential_ref: str | None,
) -> bool:
    """将官方凭证绑定到固定主机和传输位置，阻止跨提供商误投。"""

    hostname = _endpoint_hostname(endpoint_url)
    if credential_ref == CONTEXT7_CREDENTIAL_REF or hostname == CONTEXT7_MCP_HOST:
        if auth_type == "none":
            return hostname == CONTEXT7_MCP_HOST and auth_name is None and credential_ref is None
        return (
            hostname == CONTEXT7_MCP_HOST
            and auth_type == "header"
            and auth_name == CONTEXT7_CREDENTIAL_REF
            and credential_ref == CONTEXT7_CREDENTIAL_REF
        )
    if credential_ref == AMAP_CREDENTIAL_REF or hostname == AMAP_MCP_HOST:
        return (
            hostname == AMAP_MCP_HOST
            and auth_type == "query"
            and auth_name == "key"
            and credential_ref == AMAP_CREDENTIAL_REF
        )
    return True


def tool_is_allowed_for_endpoint(endpoint_url: str, tool_name: str) -> bool:
    allowlist = endpoint_tool_allowlist(endpoint_url)
    return allowlist is None or tool_name in allowlist


def endpoint_tool_guidance(endpoint_url: str, tool_name: str) -> str:
    """返回可信的产品级工具使用约束，不采信远端描述作为策略。"""

    hostname = _endpoint_hostname(endpoint_url)
    if hostname == AMAP_MCP_HOST:
        return _AMAP_TOOL_GUIDANCE.get(tool_name, "")
    if hostname == CONTEXT7_MCP_HOST:
        return _CONTEXT7_TOOL_GUIDANCE.get(tool_name, "")
    return ""


def endpoint_tool_schema_override(endpoint_url: str, tool_name: str) -> dict[str, Any] | None:
    """返回 Fusion 维护的可信参数契约，避免官方工具依赖远端可变 schema。"""

    if _endpoint_hostname(endpoint_url) != CONTEXT7_MCP_HOST:
        return None
    schema = _CONTEXT7_TOOL_SCHEMA_OVERRIDES.get(tool_name)
    return deepcopy(schema) if schema is not None else None
