"""计划模式下显式产品任务的工具范围策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.mcp.amap_product_tools import (
    AMAP_LOCAL_PLACE_SEARCH,
    AMAP_ROUTE_COMPARE,
    AMAP_WEATHER_FORECAST,
)
from app.services.mcp.flyai_travel_tools import (
    FLYAI_SEARCH_FLIGHTS,
    FLYAI_SEARCH_TRAINS,
)
from app.services.search_budget import infer_search_intent

_COMMUTE_RE = re.compile(r"通勤")
_ROUTE_ACTION_RE = re.compile(r"路线|怎么走|如何去|如何到|导航|到达")
_ROUTE_MODE_RE = re.compile(r"驾车|开车|自驾|公交|公共交通|地铁|轨道交通|骑行|步行|摩托")
_STRUCTURED_ROUTE_ENDPOINT_RE = re.compile(
    r"(?:从.{1,80}?(?:到|去|前往).{1,80})|"
    r"(?:(?:我)?(?:现在)?在.{1,80}?(?:想|要|准备)?(?:到|去|前往).{1,80})|"
    r"(?:(?:住在|起点|出发地).{1,80}?(?:公司在|学校在|终点|目的地|到|去|前往).{1,80})"
)
_BARE_ROUTE_ENDPOINT_RE = re.compile(r"^(?:[^，,。；;]{1,40})到(?:[^，,。；;]{1,40})")
_ROUTE_FOLLOWUP_RE = re.compile(r"哪个|哪种|推荐|更合适|怎么选|如何选|选择|优先|日常通勤")
_BARE_INTERCITY_MOBILITY_RE = re.compile(
    r"想去|要去|准备去|前往|出发|哪种方式|什么方式|哪种交通|交通方式|"
    r"怎么去|如何去|怎么到|如何到|怎么走|出行|行程|通勤"
)
_NON_TRAVEL_ENDPOINT_RE = re.compile(
    r"大学.{0,24}(?:申请|选择|哪个)|"
    r"公司.{0,24}(?:发展|哪家|业务)|"
    r"文章|写作|技术路线|发展路线|业务路线"
)
_WEATHER_RE = re.compile(r"天气|气温|温度|降雨|下雨|下雪|风力")
_FLIGHT_RE = re.compile(r"航班|飞机|机场|机票")
_TRAIN_RE = re.compile(r"高铁|动车|火车|列车|车次")
_PLACE_RE = re.compile(r"附近|周边|餐厅|饭店|酒店|景点|地点|推荐.*(?:吃|玩|住)")
_VERIFIED_RESEARCH_TOOL_NAMES = frozenset({"web_search", "url_read"})
_RESEARCH_REQUEST_RE = re.compile(
    r"(?:联网|深入|全面|系统|专题)?调研|(?:深入|全面|系统|专题)研究|"
    r"\b(?:research|investigate|investigation)\b"
)
_VERIFIED_EVIDENCE_RE = re.compile(
    r"(?:可靠|权威|可信|一手).{0,12}(?:来源|资料|证据|原文|出处)|"
    r"(?:来源|资料|证据|原文|出处).{0,12}(?:可靠|权威|可信|一手)"
)
_CONTROVERSY_VERIFICATION_RE = re.compile(
    r"(?:核验|查证|验证|交叉验证).{0,16}(?:争议|说法|来源|证据|事实)|"
    r"(?:争议|说法|来源|证据|事实).{0,16}(?:核验|查证|验证|交叉验证)"
)
_NEGATED_EVIDENCE_RE = re.compile(
    r"(?:不需要|无需|不用|不要|不必)(?:再|做|进行|展开)?"
    r"(?:联网|深入|全面|系统|专题)?(?:调研|研究)|"
    r"(?:不需要|无需|不用|不要|不必)(?:提供|查找|给出|引用|附上)?(?:任何)?"
    r"(?:可靠|权威|可信|一手)?(?:来源|资料|证据|原文|出处)|"
    r"(?:不需要|无需|不用|不要|不必)(?:再|做|进行)?(?:核验|查证|验证|交叉验证)"
)


@dataclass(frozen=True)
class AgentPlanToolPolicy:
    """服务端对首个模型计划施加的最小工具契约。"""

    required_initial_tool_counts: dict[str, int] = field(default_factory=dict)
    allowed_tool_names: frozenset[str] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ProductCapabilitySignals:
    """产品能力路由与计划策略共用的代码固定信号。"""

    explicit_route: bool
    adjacent_route_followup: bool
    endpoint_relation: bool
    intercity_mobility: bool
    weather: bool
    flight: bool
    train: bool
    place: bool


def resolve_product_capability_signals(
    *,
    original_message: str | None,
    task_context_messages: list[object] | None,
) -> ProductCapabilitySignals:
    """集中解析产品意图，避免 Run 路由和计划策略维护两套正则。"""

    message = _normalize_message(original_message)
    adjacent_route_followup = _has_adjacent_route_result_context(
        task_context_messages
    ) and _is_route_followup(message)
    structured_endpoint = bool(_STRUCTURED_ROUTE_ENDPOINT_RE.search(message))
    bare_endpoint = bool(_BARE_ROUTE_ENDPOINT_RE.search(message))
    endpoint_relation = structured_endpoint or bare_endpoint
    non_travel_context = bool(_NON_TRAVEL_ENDPOINT_RE.search(message))
    return ProductCapabilitySignals(
        explicit_route=_is_explicit_route_task(message),
        adjacent_route_followup=adjacent_route_followup,
        endpoint_relation=endpoint_relation,
        intercity_mobility=(
            endpoint_relation
            and not non_travel_context
            and (
                structured_endpoint
                or (bare_endpoint and bool(_BARE_INTERCITY_MOBILITY_RE.search(message)))
            )
        ),
        weather=bool(_WEATHER_RE.search(message)),
        flight=bool(_FLIGHT_RE.search(message)),
        train=bool(_TRAIN_RE.search(message)),
        place=bool(_PLACE_RE.search(message)),
    )


def resolve_agent_plan_tool_policy(
    *,
    original_message: str | None,
    announced_tool_names: list[str],
    task_context_messages: list[object] | None = None,
) -> AgentPlanToolPolicy:
    """仅收窄语义明确的产品任务；模糊请求继续交给模型自主选择。"""

    message = _normalize_message(original_message)
    announced = frozenset(name for name in announced_tool_names if name)
    if not message:
        return AgentPlanToolPolicy()
    is_verified_research = _VERIFIED_RESEARCH_TOOL_NAMES.issubset(announced) and _is_verified_research_request(
        message
    )
    product_signals = resolve_product_capability_signals(
        original_message=message,
        task_context_messages=task_context_messages,
    )
    is_explicit_route = AMAP_ROUTE_COMPARE in announced and product_signals.explicit_route
    if is_verified_research:
        required_counts = {"web_search": 1, "url_read": 2}
        reason = "verified_research_request"
        if is_explicit_route:
            required_counts[AMAP_ROUTE_COMPARE] = 1
            reason = f"{reason}+explicit_route_task"
        return AgentPlanToolPolicy(
            required_initial_tool_counts=required_counts,
            reason=reason,
        )
    if AMAP_ROUTE_COMPARE not in announced:
        return AgentPlanToolPolicy()
    if product_signals.adjacent_route_followup:
        return AgentPlanToolPolicy(
            allowed_tool_names=frozenset({AMAP_ROUTE_COMPARE}),
            reason="adjacent_route_followup",
        )
    if not product_signals.explicit_route:
        return AgentPlanToolPolicy()

    allowed = {AMAP_ROUTE_COMPARE}
    if AMAP_WEATHER_FORECAST in announced and product_signals.weather:
        allowed.add(AMAP_WEATHER_FORECAST)
    if FLYAI_SEARCH_FLIGHTS in announced and product_signals.flight:
        allowed.add(FLYAI_SEARCH_FLIGHTS)
    if FLYAI_SEARCH_TRAINS in announced and product_signals.train:
        allowed.add(FLYAI_SEARCH_TRAINS)
    if AMAP_LOCAL_PLACE_SEARCH in announced and product_signals.place:
        allowed.add(AMAP_LOCAL_PLACE_SEARCH)
    return AgentPlanToolPolicy(
        required_initial_tool_counts={AMAP_ROUTE_COMPARE: 1},
        allowed_tool_names=frozenset(allowed),
        reason="explicit_route_task",
    )


def _normalize_message(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _is_verified_research_request(message: str) -> bool:
    """仅对高置信研究请求启用强制证据阶段，避免放大单次事实查询。"""

    search_intent = infer_search_intent(message)
    if _NEGATED_EVIDENCE_RE.search(message):
        return False
    explicit_research = bool(_RESEARCH_REQUEST_RE.search(message)) and search_intent != "quick_fact"
    if explicit_research:
        return True
    return bool(
        _VERIFIED_EVIDENCE_RE.search(message) or _CONTROVERSY_VERIFICATION_RE.search(message)
    )


def _has_adjacent_route_result_context(messages: list[object] | None) -> bool:
    """只继承紧邻上一轮的结构化路线结果，禁止从任意旧话题借地点。"""

    normalized_messages = list(messages or [])
    if not normalized_messages:
        return False
    last_user_index = next(
        (
            index
            for index in range(len(normalized_messages) - 1, -1, -1)
            if _message_role(normalized_messages[index]) == "user"
        ),
        None,
    )
    if last_user_index is None:
        return False
    previous_index = last_user_index - 1
    if previous_index < 0:
        return False
    previous_message = normalized_messages[previous_index]
    return _message_role(previous_message) == "assistant" and _content_has_block_type(
        _message_content(previous_message),
        "route_results",
    )


def _message_role(message: object) -> object:
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def _message_content(message: object) -> object:
    return message.get("content") if isinstance(message, dict) else getattr(message, "content", None)


def _content_has_block_type(content: object, block_type: str) -> bool:
    if not isinstance(content, (list, tuple)):
        return False
    return any(
        (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == block_type
        for block in content
    )


def _is_explicit_route_task(message: str) -> bool:
    has_route_goal = bool(_COMMUTE_RE.search(message) or _ROUTE_ACTION_RE.search(message))
    has_route_mode = bool(_ROUTE_MODE_RE.search(message))
    has_structured_endpoint = bool(_STRUCTURED_ROUTE_ENDPOINT_RE.search(message))
    is_non_travel_context = bool(_NON_TRAVEL_ENDPOINT_RE.search(message))
    return has_structured_endpoint and not is_non_travel_context and (has_route_goal or has_route_mode)


def _is_route_followup(message: str) -> bool:
    """已有结构化路线结果时，比较和推荐追问不强制重复查询。"""

    return bool(
        _COMMUTE_RE.search(message)
        or _ROUTE_ACTION_RE.search(message)
        or _ROUTE_MODE_RE.search(message)
        or _ROUTE_FOLLOWUP_RE.search(message)
    )
