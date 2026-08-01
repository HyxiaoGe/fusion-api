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

_COMMUTE_RE = re.compile(r"通勤")
_ROUTE_ACTION_RE = re.compile(r"路线|怎么走|如何去|如何到|导航|到达")
_ROUTE_MODE_RE = re.compile(r"驾车|开车|自驾|公交|公共交通|地铁|轨道交通|骑行|步行|摩托")
_ROUTE_ENDPOINT_RE = re.compile(
    r"(?:从.{1,80}?(?:到|去|前往).{1,80})|"
    r"(?:(?:住在|起点|出发地).{1,80}?(?:公司在|学校在|终点|目的地|到|去|前往).{1,80})"
)
_ROUTE_FOLLOWUP_RE = re.compile(r"哪个|哪种|推荐|更合适|怎么选|如何选|选择|优先|日常通勤")
_WEATHER_RE = re.compile(r"天气|气温|温度|降雨|下雨|下雪|风力")
_FLIGHT_RE = re.compile(r"航班|飞机|机场|机票")
_TRAIN_RE = re.compile(r"高铁|动车|火车|列车|车次")
_PLACE_RE = re.compile(r"附近|周边|餐厅|饭店|酒店|景点|地点|推荐.*(?:吃|玩|住)")


@dataclass(frozen=True)
class AgentPlanToolPolicy:
    """服务端对首个模型计划施加的最小工具契约。"""

    required_initial_tool_counts: dict[str, int] = field(default_factory=dict)
    allowed_tool_names: frozenset[str] | None = None
    reason: str | None = None


def resolve_agent_plan_tool_policy(
    *,
    original_message: str | None,
    announced_tool_names: list[str],
    task_context_messages: list[object] | None = None,
) -> AgentPlanToolPolicy:
    """仅收窄语义明确的产品任务；模糊请求继续交给模型自主选择。"""

    message = _normalize_message(original_message)
    announced = frozenset(name for name in announced_tool_names if name)
    has_adjacent_route_context = _has_adjacent_route_result_context(task_context_messages)
    if not message or AMAP_ROUTE_COMPARE not in announced:
        return AgentPlanToolPolicy()
    if has_adjacent_route_context and _is_route_followup(message):
        return AgentPlanToolPolicy(
            allowed_tool_names=frozenset({AMAP_ROUTE_COMPARE}),
            reason="adjacent_route_followup",
        )
    if not _is_explicit_route_task(message):
        return AgentPlanToolPolicy()

    allowed = {AMAP_ROUTE_COMPARE}
    if AMAP_WEATHER_FORECAST in announced and _WEATHER_RE.search(message):
        allowed.add(AMAP_WEATHER_FORECAST)
    if FLYAI_SEARCH_FLIGHTS in announced and _FLIGHT_RE.search(message):
        allowed.add(FLYAI_SEARCH_FLIGHTS)
    if FLYAI_SEARCH_TRAINS in announced and _TRAIN_RE.search(message):
        allowed.add(FLYAI_SEARCH_TRAINS)
    if AMAP_LOCAL_PLACE_SEARCH in announced and _PLACE_RE.search(message):
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
    has_endpoint_relation = bool(_ROUTE_ENDPOINT_RE.search(message))
    return has_endpoint_relation and (has_route_goal or has_route_mode)


def _is_route_followup(message: str) -> bool:
    """已有结构化路线结果时，比较和推荐追问不强制重复查询。"""

    return bool(
        _COMMUTE_RE.search(message)
        or _ROUTE_ACTION_RE.search(message)
        or _ROUTE_MODE_RE.search(message)
        or _ROUTE_FOLLOWUP_RE.search(message)
    )
