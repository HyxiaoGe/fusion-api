"""基于结构化产品结果生成可验证的最终摘要。"""

from __future__ import annotations

from typing import Any

_PRODUCT_RESULT_TYPES = {"place_results", "route_results"}
_ROUTE_MODE_LABELS = {
    "driving": "驾车",
    "transit": "公交",
    "walking": "步行",
    "bicycling": "骑行",
}


def has_product_result_blocks(content_blocks: list[Any]) -> bool:
    return any(_value(block, "type") in _PRODUCT_RESULT_TYPES for block in content_blocks)


def build_grounded_product_answer(content_blocks: list[Any]) -> str:
    """只读取产品结果块的已校验字段，不复用模型生成的自由文本。"""
    product_blocks = [block for block in content_blocks if _value(block, "type") in _PRODUCT_RESULT_TYPES][-4:]
    paragraphs: list[str] = []
    for block in product_blocks:
        block_type = _value(block, "type")
        if block_type == "place_results":
            paragraph = _build_place_answer(block)
        else:
            paragraph = _build_route_answer(block)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _build_place_answer(block: Any) -> str:
    places = [item for item in (_value(block, "places") or []) if _value(item, "name")]
    query = _value(block, "query") or "地点搜索"
    provider = _provider_label(_value(block, "provider"))
    if not places:
        lead = f"{provider}本次没有返回可展示的「{query}」地点。"
    else:
        names = "、".join(str(_value(place, "name")) for place in places[:3])
        suffix = "等" if len(places) > 3 else ""
        lead = f"{provider}返回 {len(places)} 个「{query}」地点：{names}{suffix}。"
    limitations = _limitations_sentence(block)
    return f"{lead}{limitations}"


def _build_route_answer(block: Any) -> str:
    routes = [route for route in (_value(block, "routes") or []) if _value(route, "mode")]
    if not routes:
        return ""
    origin = _value(_value(block, "origin"), "label") or "起点"
    destination = _value(_value(block, "destination"), "label") or "终点"
    provider = _provider_label(_value(block, "provider"))
    route_summaries = [_format_route(route) for route in routes]
    route_summaries = [summary for summary in route_summaries if summary]
    if not route_summaries:
        return ""
    lead = f"{provider}返回了{origin}到{destination}的路线：{'；'.join(route_summaries)}。"
    fastest = _fastest_route_sentence(routes)
    limitations = _limitations_sentence(block)
    return f"{lead}{fastest}{limitations}"


def _format_route(route: Any) -> str:
    mode = _ROUTE_MODE_LABELS.get(str(_value(route, "mode")), "路线")
    details: list[str] = []
    duration = _value(route, "duration_s")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
        details.append(f"约 {max(1, round(duration / 60))} 分钟")
    distance = _value(route, "distance_m")
    if isinstance(distance, (int, float)) and not isinstance(distance, bool) and distance >= 0:
        details.append(_format_distance(distance))
    transfers = _value(route, "transfers")
    if isinstance(transfers, int) and not isinstance(transfers, bool) and transfers >= 0:
        details.append(f"换乘 {transfers} 次")
    return mode if not details else f"{mode}{'、'.join(details)}"


def _format_distance(distance_m: int | float) -> str:
    if distance_m < 1000:
        return f"{round(distance_m)} 米"
    decimals = 0 if distance_m >= 10_000 else 1
    return f"{distance_m / 1000:.{decimals}f} 公里"


def _fastest_route_sentence(routes: list[Any]) -> str:
    timed: list[tuple[str, int | float]] = []
    for route in routes:
        duration = _value(route, "duration_s")
        mode = str(_value(route, "mode") or "")
        if (
            mode in _ROUTE_MODE_LABELS
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 0
        ):
            timed.append((mode, duration))
    if len(timed) < 2:
        return ""
    minimum = min(duration for _, duration in timed)
    fastest_modes = [_ROUTE_MODE_LABELS[mode] for mode, duration in timed if duration == minimum]
    if len(fastest_modes) != 1:
        return ""
    return f"按本次返回的用时，{fastest_modes[0]}用时最短。"


def _limitations_sentence(block: Any) -> str:
    limitations = [
        str(item).strip() for item in (_value(block, "limitations") or []) if isinstance(item, str) and item.strip()
    ]
    if not limitations:
        return ""
    return "；".join(dict.fromkeys(limitations)) + "。"


def _provider_label(provider: Any) -> str:
    return "高德" if str(provider).lower() in {"amap", "gaode", "高德"} else "地图服务"


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
