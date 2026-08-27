"""在首个 LLM Round 前解析并冻结 Run 级能力包。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from app.services.agent.plan_coordinator import PlanMode
from app.services.mcp.amap_product_tools import (
    AMAP_LOCAL_PLACE_SEARCH,
    AMAP_ROUTE_COMPARE,
    AMAP_WEATHER_FORECAST,
)
from app.services.mcp.flyai_travel_tools import (
    FLYAI_SEARCH_FLIGHTS,
    FLYAI_SEARCH_TRAINS,
)
from app.services.stream.agent_plan_tool_policy import (
    INTERCITY_LOCATION_NAMES,
    resolve_product_capability_signals,
)
from app.services.stream.agent_task_policy import AgentTaskPolicy

Confidence = Literal["high", "medium", "low"]
ResolutionMode = Literal["routed", "degraded", "clarification"]

SCHEMA_VERSION = 1
ROUTER_VERSION = "2026-08-27.1"

_CANONICAL_EXTERNAL_TOOL_ORDER = (
    "web_search",
    "url_read",
    AMAP_WEATHER_FORECAST,
    AMAP_LOCAL_PLACE_SEARCH,
    AMAP_ROUTE_COMPARE,
    FLYAI_SEARCH_FLIGHTS,
    FLYAI_SEARCH_TRAINS,
)
_CONTROL_TOOL_NAMES = frozenset({"update_plan"})

_TRANSFORM_RE = re.compile(
    r"翻译|译成|改写|重写|润色|措辞|"
    r"(?:概括|摘要|总结)(?:这|以下|上述|给定|已给|后面|内容|文本|[:：])|"
    r"(?:对|将|把)(?:这|以下|上述|给定|已给|后面).{0,24}(?:概括|摘要|总结)"
)
_CURRENT_DATE_ONLY_RE = re.compile(
    r"^(?:请问|请告诉我|帮我看下|帮我看看)?(?:今天|现在)"
    r"(?:是)?(?:几月几日|几号|星期几|周几|日期|什么日子)"
    r"(?:[、，,和及](?:星期几|周几|几月几日|几号|日期))?[？?。！!]*$"
)
_RELATIVE_DATE_RE = re.compile(r"今天|今日|明天|后天|昨天|本周|下周|这个月|本月|下个月|当前|现在")
_FRESH_EXTERNAL_RE = re.compile(
    r"最新|新闻|开市|收盘|股价|汇率|比分|发布了什么|刚刚发布|公开发布|现任|目前的"
)
_VERIFIED_SOURCE_RE = re.compile(
    r"官方(?:公告|原文|资料|来源)|一手来源|可靠来源|权威来源|"
    r"(?:查证|核验|验证|交叉验证)|只依据(?:该|这个)页面"
)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_URL_READ_ACTION_RE = re.compile(r"总结|摘要|读取|阅读|分析|概括|只依据|基于")
_GREETING_RE = re.compile(
    r"^(?:(?:你?好|嗨)(?:[，,\s]*很高兴见到你)?|hi|hello|早上好|下午好|晚上好|很高兴见到你)"
    r"[呀啊！!。\s]*$",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(r"你是谁|你叫什么|介绍一下你自己|你能做什么")
_STABLE_KNOWLEDGE_RE = re.compile(r"^(?:为什么|为何|什么是|解释一下|介绍一下|讲讲|how\b|what\b)", re.IGNORECASE)
_SIMPLE_CALC_RE = re.compile(r"^(?:请)?(?:计算|算一下|算算)?\s*[\d\s()+\-*/.%]+(?:等于多少|是多少)?[？?]?$", re.IGNORECASE)
_ROUTE_TRANSFER_RE = re.compile(r"接驳|市内|机场到|车站到|落地后")
_PACKAGE_TOOLS: dict[str, tuple[str, ...]] = {
    "direct": (),
    "transform": (),
    "date": (),
    "fresh_web": ("web_search",),
    "verified_web": ("web_search", "url_read"),
    "url_read": ("url_read",),
    "weather": (AMAP_WEATHER_FORECAST,),
    "place_discovery": (AMAP_LOCAL_PLACE_SEARCH,),
    "mobility_route": (AMAP_ROUTE_COMPARE,),
    "flight": (FLYAI_SEARCH_FLIGHTS,),
    "train": (FLYAI_SEARCH_TRAINS,),
    "travel_air_rail": (FLYAI_SEARCH_FLIGHTS, FLYAI_SEARCH_TRAINS),
    "mobility_intercity": (
        AMAP_ROUTE_COMPARE,
        FLYAI_SEARCH_FLIGHTS,
        FLYAI_SEARCH_TRAINS,
    ),
    "mixed_itinerary": (
        AMAP_ROUTE_COMPARE,
        FLYAI_SEARCH_FLIGHTS,
        FLYAI_SEARCH_TRAINS,
    ),
    "deep_research": ("web_search", "url_read"),
    "knowledge_grounded": (),
    "tools_unavailable": (),
    "clarification_only": (),
}
_AUTO_PLAN_PACKAGES = frozenset(
    {"verified_web", "mobility_route", "travel_air_rail", "mobility_intercity", "mixed_itinerary"}
)
_REASON_CODES = frozenset(
    {
        "direct_greeting",
        "assistant_identity_question",
        "stable_knowledge_question",
        "simple_calculation",
        "text_transform_request",
        "current_date_question",
        "fresh_external_fact",
        "verified_source_request",
        "explicit_url_read",
        "explicit_weather_request",
        "explicit_place_discovery",
        "explicit_route_task",
        "explicit_flight_request",
        "explicit_train_request",
        "air_rail_comparison",
        "mixed_itinerary_request",
        "origin_destination_relation",
        "intercity_locations",
        "adjacent_route_followup",
        "deep_research_mode",
        "knowledge_grounded_mode",
        "tools_disabled",
        "function_calling_unavailable",
        "search_capability_unavailable",
        "required_tools_unavailable",
        "explicit_authorized_tool_alias",
        "insufficient_capability_signal",
    }
)


@dataclass(frozen=True)
class RunCapabilityResolution:
    schema_version: int
    router_version: str
    package_id: str
    confidence: Confidence
    resolution_mode: ResolutionMode
    reason_codes: tuple[str, ...]
    external_tool_names: tuple[str, ...]
    effective_plan_mode: PlanMode
    include_current_date: bool
    network_boundary_required: bool


@dataclass(frozen=True)
class _CandidateRoute:
    package_id: str
    confidence: Confidence
    reason_codes: tuple[str, ...]
    include_current_date: bool
    resolution_mode: ResolutionMode = "routed"
    explicit_tool_names: tuple[str, ...] | None = None


def resolve_run_capability_route(
    *,
    original_message: str | None,
    task_context_messages: list[object] | None,
    available_tool_names: list[str],
    requested_plan_mode: PlanMode,
    task_policy: AgentTaskPolicy,
    capabilities: dict,
    tools_disabled: bool,
    knowledge_grounded: bool,
) -> RunCapabilityResolution:
    """根据受信运行态与当前用户消息解析最小能力包。"""

    message = _normalize_message(original_message)
    function_calling = capabilities.get("functionCalling") is True
    search_capable = capabilities.get("searchCapable") is True

    if knowledge_grounded:
        blocked_candidate = (
            _CandidateRoute(
                package_id="deep_research",
                confidence="high",
                reason_codes=("deep_research_mode",),
                include_current_date=True,
            )
            if task_policy.task_mode == "deep_research"
            else _classify_standard_request(
                message=message,
                task_context_messages=task_context_messages,
                available_tool_names=available_tool_names,
            )
        )
        blocked_tool_names = blocked_candidate.explicit_tool_names or _PACKAGE_TOOLS.get(
            blocked_candidate.package_id,
            (),
        )
        return _resolution(
            candidate=_CandidateRoute(
                package_id="knowledge_grounded",
                confidence="high",
                reason_codes=("knowledge_grounded_mode",),
                include_current_date=blocked_candidate.include_current_date,
            ),
            available_tool_names=available_tool_names,
            requested_plan_mode="off",
            function_calling=function_calling,
            tools_disabled=True,
            network_boundary_required=bool(blocked_tool_names),
        )

    if task_policy.task_mode == "deep_research":
        candidate = _CandidateRoute(
            package_id="deep_research",
            confidence="high",
            reason_codes=("deep_research_mode",),
            include_current_date=True,
        )
    else:
        candidate = _classify_standard_request(
            message=message,
            task_context_messages=task_context_messages,
            available_tool_names=available_tool_names,
        )

    requested_tools = candidate.explicit_tool_names or _PACKAGE_TOOLS.get(
        candidate.package_id,
        (),
    )
    requires_search = any(name in {"web_search", "url_read"} for name in requested_tools)
    needs_external_capability = bool(requested_tools)
    degraded_reason: str | None = None
    if needs_external_capability and tools_disabled:
        degraded_reason = "tools_disabled"
    elif needs_external_capability and not function_calling:
        degraded_reason = "function_calling_unavailable"
    elif requires_search and not search_capable:
        degraded_reason = "search_capability_unavailable"

    if degraded_reason is not None:
        return _resolution(
            candidate=_CandidateRoute(
                package_id="tools_unavailable",
                confidence=candidate.confidence,
                reason_codes=(degraded_reason,),
                include_current_date=candidate.include_current_date,
                resolution_mode="degraded",
            ),
            available_tool_names=available_tool_names,
            requested_plan_mode="off",
            function_calling=function_calling,
            tools_disabled=True,
            network_boundary_required=True,
        )

    resolution = _resolution(
        candidate=candidate,
        available_tool_names=available_tool_names,
        requested_plan_mode=requested_plan_mode,
        function_calling=function_calling,
        tools_disabled=tools_disabled,
    )
    if candidate.package_id == "deep_research" and not frozenset(requested_tools).issubset(
        resolution.external_tool_names
    ):
        return _resolution(
            candidate=_CandidateRoute(
                package_id="tools_unavailable",
                confidence=candidate.confidence,
                reason_codes=("required_tools_unavailable",),
                include_current_date=candidate.include_current_date,
                resolution_mode="degraded",
            ),
            available_tool_names=available_tool_names,
            requested_plan_mode="off",
            function_calling=function_calling,
            tools_disabled=True,
            network_boundary_required=True,
        )
    if needs_external_capability and not resolution.external_tool_names:
        return _resolution(
            candidate=_CandidateRoute(
                package_id="tools_unavailable",
                confidence=candidate.confidence,
                reason_codes=("required_tools_unavailable",),
                include_current_date=candidate.include_current_date,
                resolution_mode="degraded",
            ),
            available_tool_names=available_tool_names,
            requested_plan_mode="off",
            function_calling=function_calling,
            tools_disabled=True,
            network_boundary_required=True,
        )
    return resolution


def serialize_capability_resolution(resolution: RunCapabilityResolution) -> dict:
    """转换为可持久化的安全协议，不包含原文或自由文本。"""

    payload = asdict(resolution)
    payload["reason_codes"] = list(resolution.reason_codes)
    payload["external_tool_names"] = list(resolution.external_tool_names)
    return payload


def _classify_standard_request(
    *,
    message: str,
    task_context_messages: list[object] | None,
    available_tool_names: list[str],
) -> _CandidateRoute:
    include_current_date = _needs_current_date(message)

    if _TRANSFORM_RE.search(message):
        return _CandidateRoute(
            "transform",
            "high",
            ("text_transform_request",),
            False,
        )
    if _CURRENT_DATE_ONLY_RE.search(message):
        return _CandidateRoute("date", "high", ("current_date_question",), True)

    if _URL_RE.search(message) and _URL_READ_ACTION_RE.search(message):
        return _CandidateRoute("url_read", "high", ("explicit_url_read",), False)
    if _VERIFIED_SOURCE_RE.search(message):
        return _CandidateRoute(
            "verified_web",
            "high",
            ("verified_source_request",),
            True,
        )
    if _FRESH_EXTERNAL_RE.search(message):
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )

    signals = resolve_product_capability_signals(
        original_message=message,
        task_context_messages=task_context_messages,
    )
    if signals.adjacent_route_followup:
        return _CandidateRoute(
            "mobility_route",
            "high",
            ("adjacent_route_followup",),
            include_current_date,
        )
    if signals.flight and signals.train and signals.explicit_route and _ROUTE_TRANSFER_RE.search(message):
        return _CandidateRoute(
            "mixed_itinerary",
            "high",
            ("mixed_itinerary_request",),
            True,
        )
    if signals.flight and signals.train:
        return _CandidateRoute(
            "travel_air_rail",
            "high",
            ("air_rail_comparison",),
            True,
        )
    if signals.flight:
        return _CandidateRoute("flight", "high", ("explicit_flight_request",), True)
    if signals.train:
        return _CandidateRoute("train", "high", ("explicit_train_request",), True)
    if signals.weather:
        return _CandidateRoute(
            "weather",
            "high",
            ("explicit_weather_request",),
            True,
        )
    if signals.place:
        return _CandidateRoute(
            "place_discovery",
            "high",
            ("explicit_place_discovery",),
            False,
        )
    if signals.explicit_route:
        return _CandidateRoute(
            "mobility_route",
            "high",
            ("explicit_route_task",),
            include_current_date,
        )
    if (
        signals.endpoint_relation
        and signals.intercity_mobility
        and _has_two_intercity_locations(message)
    ):
        return _CandidateRoute(
            "mobility_intercity",
            "medium",
            ("origin_destination_relation", "intercity_locations"),
            True,
        )

    if include_current_date and re.search(r"查|查询|多少|是否|吗[？?]?$", message):
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )

    explicit_alias = _resolve_explicit_authorized_alias(message, available_tool_names)
    if explicit_alias is not None:
        return _CandidateRoute(
            "mcp_explicit",
            "high",
            ("explicit_authorized_tool_alias",),
            include_current_date,
            explicit_tool_names=(explicit_alias,),
        )

    if _GREETING_RE.search(message):
        return _CandidateRoute("direct", "high", ("direct_greeting",), False)
    if _IDENTITY_RE.search(message):
        return _CandidateRoute(
            "direct",
            "high",
            ("assistant_identity_question",),
            False,
        )
    if _SIMPLE_CALC_RE.search(message):
        return _CandidateRoute("direct", "high", ("simple_calculation",), False)
    if _STABLE_KNOWLEDGE_RE.search(message):
        return _CandidateRoute(
            "direct",
            "high",
            ("stable_knowledge_question",),
            False,
        )
    return _CandidateRoute(
        "clarification_only",
        "low",
        ("insufficient_capability_signal",),
        False,
        resolution_mode="clarification",
    )


def _resolution(
    *,
    candidate: _CandidateRoute,
    available_tool_names: list[str],
    requested_plan_mode: PlanMode,
    function_calling: bool,
    tools_disabled: bool,
    network_boundary_required: bool = False,
) -> RunCapabilityResolution:
    requested_tools = candidate.explicit_tool_names or _PACKAGE_TOOLS.get(candidate.package_id, ())
    available = frozenset(name for name in available_tool_names if isinstance(name, str) and name)
    tools = tuple(
        name
        for name in _canonicalize_tool_names(requested_tools)
        if name in available and name not in _CONTROL_TOOL_NAMES
    )
    reason_codes = tuple(code for code in candidate.reason_codes if code in _REASON_CODES)
    if reason_codes != candidate.reason_codes:
        raise ValueError("能力路由包含未注册的 reason code")
    return RunCapabilityResolution(
        schema_version=SCHEMA_VERSION,
        router_version=ROUTER_VERSION,
        package_id=candidate.package_id,
        confidence=candidate.confidence,
        resolution_mode=candidate.resolution_mode,
        reason_codes=reason_codes,
        external_tool_names=tools,
        effective_plan_mode=_effective_plan_mode(
            package_id=candidate.package_id,
            requested_plan_mode=requested_plan_mode,
            function_calling=function_calling,
            tools_disabled=tools_disabled,
        ),
        include_current_date=candidate.include_current_date,
        network_boundary_required=network_boundary_required,
    )


def _effective_plan_mode(
    *,
    package_id: str,
    requested_plan_mode: PlanMode,
    function_calling: bool,
    tools_disabled: bool,
) -> PlanMode:
    if not function_calling or tools_disabled:
        return "off"
    if package_id == "deep_research":
        return "on"
    if requested_plan_mode in {"on", "off"}:
        return requested_plan_mode
    return "auto" if package_id in _AUTO_PLAN_PACKAGES else "off"


def _canonicalize_tool_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    known_order = {name: index for index, name in enumerate(_CANONICAL_EXTERNAL_TOOL_ORDER)}
    return tuple(sorted(set(tool_names), key=lambda name: (known_order.get(name, 10_000), name)))


def _resolve_explicit_authorized_alias(
    message: str,
    available_tool_names: list[str],
) -> str | None:
    product_names = frozenset(_CANONICAL_EXTERNAL_TOOL_ORDER) | _CONTROL_TOOL_NAMES
    aliases = sorted(
        {
            name
            for name in available_tool_names
            if isinstance(name, str) and name and name not in product_names
        }
    )
    matched = [
        alias
        for alias in aliases
        if re.search(rf"(?<![\w]){re.escape(alias.lower())}(?![\w])", message)
    ]
    return matched[0] if len(matched) == 1 else None


def _normalize_message(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _needs_current_date(message: str) -> bool:
    return bool(_RELATIVE_DATE_RE.search(message))


def _has_two_intercity_locations(message: str) -> bool:
    return len({location for location in INTERCITY_LOCATION_NAMES if location in message}) >= 2
