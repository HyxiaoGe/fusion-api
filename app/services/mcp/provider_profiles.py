from __future__ import annotations

from urllib.parse import urlsplit

AMAP_MCP_HOST = "mcp.amap.com"
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


def endpoint_tool_allowlist(endpoint_url: str) -> frozenset[str] | None:
    """返回指定官方端点的硬白名单；其他 MCP 服务继续使用管理员授权。"""

    try:
        hostname = (urlsplit(endpoint_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if hostname == AMAP_MCP_HOST:
        return AMAP_READ_ONLY_TOOL_ALLOWLIST
    return None


def is_official_amap_endpoint(endpoint_url: str) -> bool:
    try:
        hostname = (urlsplit(endpoint_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return hostname == AMAP_MCP_HOST


def tool_is_allowed_for_endpoint(endpoint_url: str, tool_name: str) -> bool:
    allowlist = endpoint_tool_allowlist(endpoint_url)
    return allowlist is None or tool_name in allowlist


def endpoint_tool_guidance(endpoint_url: str, tool_name: str) -> str:
    """返回可信的产品级工具使用约束，不采信远端描述作为策略。"""

    if endpoint_tool_allowlist(endpoint_url) is None:
        return ""
    return _AMAP_TOOL_GUIDANCE.get(tool_name, "")
