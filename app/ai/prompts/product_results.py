"""结构化产品结果出现后，供当前模型轮次使用的静态系统约束。"""

from __future__ import annotations

from typing import Any

_PRODUCT_RESULT_TYPES = {
    "place_results",
    "route_results",
    "weather_results",
    "flight_results",
    "train_results",
    "itinerary_results",
}

PRODUCT_RESULT_ROUND_BASE_PROMPT = (
    "【本轮产品结果综合约束】\n"
    "当前上下文已经包含本轮实际返回的结构化产品结果。最终回答只能把这些结果中的字段作为产品事实，"
    "不得用常识、训练数据或工具结果之外的信息补齐缺失字段。先直接回答用户的决策问题，再用简短正文补充结构化卡片的决策价值；"
    "不复述调用过程，不出现内部工具名或供应商名称。正文不使用 Markdown 表格。"
)

WEATHER_RESULT_ROUND_PROMPT = (
    "【天气结果】\n"
    "只能引用实际返回的日期、昼夜天气、高低温、风向和风力。不得补充实时温度、湿度、空气质量、降雨概率、"
    "预警、积水、路况或其他未返回信息，也不得推断路况、安全性或舒适度。"
    "如果用户询问是否适合某项活动，应给出基于已返回字段的条件化结论，并明确结论所依赖的用户条件；"
    "例如用户希望避雨时，可以说明返回的雨天是否满足避雨条件。"
    "如果结果只有昼夜粒度，而用户询问上午或下午等更细时段，必须明确本次结果无法确认该细分时段，不能把白天预报改写成上午预报。"
)

PLACE_RESULT_ROUND_PROMPT = (
    "【地点结果】\n"
    "只能引用实际返回的地点名称、地址、分类、评分和参考消费等字段。未返回两地距离、步行关系或路线时，"
    "不得推断相邻、顺路、就近或转场方便；用户要求这类判断时必须说明无法从本次结果确认。"
)

ROUTE_RESULT_ROUND_PROMPT = (
    "【路线结果】\n"
    "只能引用实际返回的路线类型、时长、距离、换乘和分段字段。不得自行估算票价、到达时间或实时路况；"
    "公共交通结果不得把起终点步行距离表述为全程距离。"
)

TRAVEL_RESULT_ROUND_PROMPT = (
    "【航班与高铁结果】\n"
    "航班号、车次、机场、车站、时间、时长、舱等、席别和参考价格必须逐项来自实际结果；"
    "不得补充或推断余票、准点率、延误、退改签、行李、登机口、检票口、站台、接驳便利性或实时价格。"
    "结构化卡片负责完整班次列表，正文只概括有依据的选择和差异，并说明价格与班次仅代表本次查询时刻。"
)

MIXED_TRAVEL_RESULT_ROUND_PROMPT = (
    "【航班与高铁联合比较】\n"
    "当前结果同时包含航班和高铁，最终回答必须同时覆盖航班和高铁，不得遗漏任一类型。"
    "分别使用各自实际返回的价格与时长找出本次候选中的省钱和快速选择，再给出条件化的跨类型结论；"
    "不使用 Markdown 表格，也不得把某一类型的字段拼接到另一类型的班次上。"
)


def build_product_result_round_prompt(content_blocks: list[Any]) -> str:
    """仅根据安全的结果类型选择静态约束，不读取或提升第三方结果正文。"""

    block_types = {
        block_type for block in content_blocks if (block_type := _block_type(block)) in _PRODUCT_RESULT_TYPES
    }
    if not block_types:
        return ""

    sections = [PRODUCT_RESULT_ROUND_BASE_PROMPT]
    if "place_results" in block_types:
        sections.append(PLACE_RESULT_ROUND_PROMPT)
    if "route_results" in block_types:
        sections.append(ROUTE_RESULT_ROUND_PROMPT)
    if "weather_results" in block_types:
        sections.append(WEATHER_RESULT_ROUND_PROMPT)
    if block_types.intersection({"flight_results", "train_results", "itinerary_results"}):
        sections.append(TRAVEL_RESULT_ROUND_PROMPT)
    if {"flight_results", "train_results"}.issubset(block_types):
        sections.append(MIXED_TRAVEL_RESULT_ROUND_PROMPT)
    return "\n\n".join(sections)


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        value = block.get("type")
    else:
        value = getattr(block, "type", None)
    return value if isinstance(value, str) else None
