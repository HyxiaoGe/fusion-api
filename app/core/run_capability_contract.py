"""Run 能力路由在执行与观测协议间共享的低层安全契约。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import MappingProxyType

CAPABILITY_CONTROL_TOOL_NAMES = frozenset({"update_plan"})
CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER = (
    "web_search",
    "url_read",
    "weather_forecast",
    "local_place_search",
    "route_compare",
    "search_flights",
    "search_trains",
)
CAPABILITY_PACKAGE_EXTERNAL_TOOL_NAMES = MappingProxyType(
    {
        "direct": (),
        "transform": (),
        "date": (),
        "fresh_web": ("web_search",),
        "verified_web": ("web_search", "url_read"),
        "url_read": ("url_read",),
        "weather": ("weather_forecast",),
        "place_discovery": ("local_place_search",),
        "mobility_route": ("route_compare",),
        "flight": ("search_flights",),
        "train": ("search_trains",),
        "travel_air_rail": ("search_flights", "search_trains"),
        "mobility_intercity": ("route_compare", "search_flights", "search_trains"),
        "mixed_itinerary": ("route_compare", "search_flights", "search_trains"),
        "deep_research": ("web_search", "url_read"),
        "knowledge_grounded": (),
        "tools_unavailable": (),
        "clarification_only": (),
    }
)
CAPABILITY_AUTO_PLAN_PACKAGES = frozenset(
    {"verified_web", "mobility_route", "travel_air_rail", "mobility_intercity", "mixed_itinerary"}
)

_MCP_TOOL_ALIAS_RE = re.compile(r"mcp_[A-Za-z0-9_-]+")
_ZERO_EXTERNAL_TOOL_PACKAGES = frozenset(
    {"direct", "transform", "date", "knowledge_grounded", "tools_unavailable", "clarification_only"}
)
_FIXED_INCLUDE_CURRENT_DATE = MappingProxyType(
    {
        "direct": False,
        "transform": False,
        "date": True,
        "fresh_web": True,
        "verified_web": True,
        "url_read": False,
        "weather": True,
        "place_discovery": False,
        "flight": True,
        "train": True,
        "travel_air_rail": True,
        "mobility_intercity": True,
        "mixed_itinerary": True,
        "deep_research": True,
        "clarification_only": False,
    }
)
_FIXED_NETWORK_BOUNDARY = MappingProxyType(
    {
        **{package_id: False for package_id in CAPABILITY_PACKAGE_EXTERNAL_TOOL_NAMES},
        "mcp_explicit": False,
        "tools_unavailable": True,
    }
)
_VARIABLE_NETWORK_BOUNDARY_PACKAGES = frozenset({"knowledge_grounded"})
_PACKAGE_REASON_CODE_OPTIONS = MappingProxyType(
    {
        "direct": frozenset(
            {
                ("direct_greeting",),
                ("assistant_identity_question",),
                ("stable_knowledge_question",),
                ("simple_calculation",),
            }
        ),
        "transform": frozenset({("text_transform_request",)}),
        "date": frozenset({("current_date_question",)}),
        "fresh_web": frozenset({("fresh_external_fact",)}),
        "verified_web": frozenset({("verified_source_request",)}),
        "url_read": frozenset({("explicit_url_read",)}),
        "weather": frozenset({("explicit_weather_request",)}),
        "place_discovery": frozenset({("explicit_place_discovery",)}),
        "mobility_route": frozenset({("explicit_route_task",), ("adjacent_route_followup",)}),
        "flight": frozenset({("explicit_flight_request",)}),
        "train": frozenset({("explicit_train_request",)}),
        "travel_air_rail": frozenset({("air_rail_comparison",)}),
        "mobility_intercity": frozenset({("origin_destination_relation", "intercity_locations")}),
        "mixed_itinerary": frozenset({("mixed_itinerary_request",)}),
        "deep_research": frozenset({("deep_research_mode",)}),
        "knowledge_grounded": frozenset({("knowledge_grounded_mode",)}),
        "tools_unavailable": frozenset(
            {
                ("tools_disabled",),
                ("function_calling_unavailable",),
                ("search_capability_unavailable",),
                ("required_tools_unavailable",),
            }
        ),
        "clarification_only": frozenset({("insufficient_capability_signal",)}),
        "mcp_explicit": frozenset({("explicit_authorized_tool_alias",)}),
    }
)
CAPABILITY_REASON_CODES = frozenset(
    reason_code
    for reason_code_options in _PACKAGE_REASON_CODE_OPTIONS.values()
    for option in reason_code_options
    for reason_code in option
)
_PACKAGE_CONFIDENCE_OPTIONS = MappingProxyType(
    {
        **{package_id: frozenset({"high"}) for package_id in _PACKAGE_REASON_CODE_OPTIONS},
        "mobility_intercity": frozenset({"medium"}),
        "tools_unavailable": frozenset({"high", "medium"}),
        "clarification_only": frozenset({"low"}),
    }
)
_PACKAGE_RESOLUTION_MODE = MappingProxyType(
    {
        **{package_id: "routed" for package_id in _PACKAGE_REASON_CODE_OPTIONS},
        "tools_unavailable": "degraded",
        "clarification_only": "clarification",
    }
)


def is_authorized_mcp_tool_alias(value: object) -> bool:
    """判断工具名是否具备服务端生成的 MCP alias 形状。"""

    return isinstance(value, str) and _MCP_TOOL_ALIAS_RE.fullmatch(value) is not None


def validate_capability_resolution_semantics(
    *,
    package_id: str,
    confidence: str,
    resolution_mode: str,
    reason_codes: Sequence[str],
    external_tool_names: Sequence[str],
    effective_plan_mode: str,
    include_current_date: bool,
    network_boundary_required: bool,
) -> None:
    """拒绝无法由能力路由器产生的工具与固定包语义组合。"""

    tool_names = tuple(external_tool_names)
    if CAPABILITY_CONTROL_TOOL_NAMES.intersection(tool_names):
        raise ValueError("能力路由外部工具不得包含内部控制工具")

    if package_id == "mcp_explicit":
        if len(tool_names) != 1 or not is_authorized_mcp_tool_alias(tool_names[0]):
            raise ValueError("显式 MCP 能力包必须且只能包含一个 mcp_ 授权别名")
    else:
        allowed_tool_names = CAPABILITY_PACKAGE_EXTERNAL_TOOL_NAMES.get(package_id)
        if allowed_tool_names is None:
            raise ValueError("能力路由包含未知能力包")
        actual_tool_names = frozenset(tool_names)
        allowed_tool_name_set = frozenset(allowed_tool_names)
        if package_id in _ZERO_EXTERNAL_TOOL_PACKAGES and actual_tool_names:
            raise ValueError("零外部工具能力包不得公告工具")
        if allowed_tool_name_set and not actual_tool_names:
            raise ValueError("外部工具能力包不得缺少全部工具")
        if not actual_tool_names.issubset(allowed_tool_name_set):
            raise ValueError("能力包公告了不属于该包的外部工具")
        if package_id == "deep_research" and actual_tool_names != allowed_tool_name_set:
            raise ValueError("Deep Research 必须公告完整搜索与读取工具集合")

    if package_id == "deep_research":
        allowed_plan_modes = frozenset({"on"})
    elif package_id in {"knowledge_grounded", "tools_unavailable"}:
        allowed_plan_modes = frozenset({"off"})
    elif package_id in CAPABILITY_AUTO_PLAN_PACKAGES:
        allowed_plan_modes = frozenset({"auto", "on", "off"})
    else:
        allowed_plan_modes = frozenset({"on", "off"})
    if effective_plan_mode not in allowed_plan_modes:
        raise ValueError("能力包与有效计划模式不匹配")

    fixed_include_current_date = _FIXED_INCLUDE_CURRENT_DATE.get(package_id)
    if fixed_include_current_date is not None and include_current_date is not fixed_include_current_date:
        raise ValueError("能力包与当前日期上下文语义不匹配")

    if package_id not in _VARIABLE_NETWORK_BOUNDARY_PACKAGES:
        fixed_network_boundary = _FIXED_NETWORK_BOUNDARY.get(package_id)
        if fixed_network_boundary is None or network_boundary_required is not fixed_network_boundary:
            raise ValueError("能力包与网络边界语义不匹配")

    allowed_reason_codes = _PACKAGE_REASON_CODE_OPTIONS.get(package_id)
    if allowed_reason_codes is None or tuple(reason_codes) not in allowed_reason_codes:
        raise ValueError("能力包与路由原因码不匹配")

    allowed_confidences = _PACKAGE_CONFIDENCE_OPTIONS.get(package_id)
    if allowed_confidences is None or confidence not in allowed_confidences:
        raise ValueError("能力包与路由置信度不匹配")

    expected_resolution_mode = _PACKAGE_RESOLUTION_MODE.get(package_id)
    if expected_resolution_mode is None or resolution_mode != expected_resolution_mode:
        raise ValueError("能力包与 resolution mode 不匹配")
