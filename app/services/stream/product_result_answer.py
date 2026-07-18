"""基于结构化产品结果生成可验证的最终摘要。"""

from __future__ import annotations

import json
import re
from typing import Any

_PRODUCT_RESULT_TYPES = {"place_results", "route_results"}
_ROUTE_MODE_LABELS = {
    "driving": "驾车",
    "transit": "公交",
    "walking": "步行",
    "bicycling": "骑行",
}
_TRANSIT_TYPE_LABELS = {
    "subway": "地铁",
    "bus": "公交",
    "mixed": "公交与地铁",
    "public_transit": "公共交通",
}
_PROVIDER_ATTRIBUTION_REPLACEMENTS = (
    (re.compile(r"本次\s*高德(?:地图)?\s*结果"), "本次查询结果"),
    (re.compile(r"高德(?:地图)?\s*本次返回(?:的)?结果"), "本次查询结果"),
    (re.compile(r"根据\s*高德(?:地图)?\s*(?:本次)?返回的"), "根据本次查询返回的"),
    (re.compile(r"高德(?:地图)?\s*(?:当前|本次)?\s*(?:未能|没有|未)\s*返回"), "本次查询未能返回"),
    (re.compile(r"高德(?:地图)?\s*(?:本次)?返回了"), "本次查询返回了"),
    (re.compile(r"高德(?:地图)?\s*(?:本次)?返回"), "本次查询返回"),
    (re.compile(r"高德(?:地图)?\s*(?:的)?(?:查询|路线|地点)?结果"), "本次查询结果"),
    (re.compile(r"高德(?:地图)?\s*(?:预估|估算)"), "本次查询预估"),
    (re.compile(r"高德(?:地图)?\s*路线(?:服务|规划)"), "路线查询"),
    (re.compile(r"高德(?:地图)?\s*参考消费"), "参考消费"),
    (re.compile(r"高德(?:地图)?\s*(?:接口|工具|服务)"), "地图服务"),
)
_PROVIDER_NAME_RE = re.compile(r"高德(?:地图)?")


def neutralize_product_provider_mentions(answer: str, content_blocks: list[Any] | None = None) -> str:
    """中性化产品正文中的供应商归因，同时保护结构化结果里的真实实体名。"""

    neutralized = answer
    protected_terms: dict[str, str] = {}
    for index, term in enumerate(_provider_entity_terms(content_blocks or [])):
        placeholder = f"\ue000{index}\ue001"
        if term in neutralized:
            neutralized = neutralized.replace(term, placeholder)
            protected_terms[placeholder] = term
    for pattern, replacement in _PROVIDER_ATTRIBUTION_REPLACEMENTS:
        neutralized = pattern.sub(replacement, neutralized)
    neutralized = _PROVIDER_NAME_RE.sub("地图服务", neutralized)
    for placeholder, term in protected_terms.items():
        neutralized = neutralized.replace(placeholder, term)
    return neutralized


def _provider_entity_terms(content_blocks: list[Any]) -> list[str]:
    terms: set[str] = set()
    for block in content_blocks:
        block_type = _value(block, "type")
        if block_type == "place_results":
            for place in _value(block, "places") or []:
                for key in ("name", "address", "district", "business_area"):
                    value = _value(place, key)
                    if isinstance(value, str) and "高德" in value:
                        terms.add(value)
        elif block_type == "route_results":
            for endpoint_key in ("origin", "destination"):
                value = _value(_value(block, endpoint_key), "label")
                if isinstance(value, str) and "高德" in value:
                    terms.add(value)
    return sorted(terms, key=len, reverse=True)


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


def build_product_tool_failure_answer(messages: list[dict[str, Any]] | None = None) -> str:
    """产品工具未取得结构化结果时，阻止模型用训练知识补全具体事实。"""

    if _has_unavailable_geolocation_context(messages or []):
        return (
            "本次未能获取当前位置，请检查浏览器或系统定位权限后重试，也可以直接提供明确起点。"
            "由于位置没有获取成功，路线查询尚未执行。"
        )
    return (
        "本次未取得可用的地点或路线数据，因此无法可靠给出具体地点、线路、时间、距离或费用。"
        "你可以稍后重试，或补充更明确的城市、起点和终点。"
    )


def _has_unavailable_geolocation_context(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("error_code") == "context_required_not_provided"
            and payload.get("context_type") == "geolocation"
            and payload.get("context_status") in {"denied", "timeout", "unavailable"}
        ):
            return True
    return False


def _build_place_answer(block: Any) -> str:
    places = [item for item in (_value(block, "places") or []) if _value(item, "name")]
    query = _value(block, "query") or "地点搜索"
    if not places:
        lead = f"本次查询没有返回可展示的「{query}」地点。"
    else:
        names = "、".join(str(_value(place, "name")) for place in places[:3])
        suffix = "等" if len(places) > 3 else ""
        lead = f"本次查询返回 {len(places)} 个「{query}」地点：{names}{suffix}。"
    limitations = _limitations_sentence(block)
    return f"{lead}{limitations}"


def _build_route_answer(block: Any) -> str:
    routes = [route for route in (_value(block, "routes") or []) if _value(route, "mode")]
    if not routes:
        return ""
    origin = _value(_value(block, "origin"), "label") or "起点"
    destination = _value(_value(block, "destination"), "label") or "终点"
    route_summaries = [_format_route(route) for route in routes]
    route_summaries = [summary for summary in route_summaries if summary]
    if not route_summaries:
        return ""
    lead = f"本次查询返回了{origin}到{destination}的路线：{'；'.join(route_summaries)}。"
    recommendation = _route_recommendation_sentence(routes)
    limitations = _limitations_sentence(block)
    return f"{lead}{recommendation}{limitations}"


def _format_route(route: Any) -> str:
    mode = _route_mode_label(route)
    details: list[str] = []
    duration = _value(route, "duration_s")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
        details.append(f"约 {max(1, round(duration / 60))} 分钟")
    distance = _value(route, "distance_m")
    if (
        _value(route, "mode") != "transit"
        and isinstance(distance, (int, float))
        and not isinstance(distance, bool)
        and distance >= 0
    ):
        details.append(_format_distance(distance))
    transfers = _value(route, "transfers")
    if isinstance(transfers, int) and not isinstance(transfers, bool) and transfers >= 0:
        details.append(f"换乘 {transfers} 次")
    line_names = _transit_line_names(route)
    if line_names:
        details.append(f"线路 {'→'.join(line_names)}")
    return mode if not details else f"{mode}{'、'.join(details)}"


def _route_mode_label(route: Any) -> str:
    mode = str(_value(route, "mode") or "")
    if mode == "transit":
        return _TRANSIT_TYPE_LABELS.get(str(_value(route, "transit_type") or ""), "公交")
    return _ROUTE_MODE_LABELS.get(mode, "路线")


def _transit_line_names(route: Any) -> list[str]:
    if _value(route, "mode") != "transit":
        return []
    names: list[str] = []
    for leg in (_value(route, "legs") or [])[:8]:
        if _value(leg, "kind") not in {"subway", "bus", "other"}:
            continue
        name = _value(leg, "line_name")
        if not isinstance(name, str) or not name.strip():
            continue
        compact = name.strip()[:32]
        if compact not in names:
            names.append(compact)
        if len(names) >= 2:
            break
    return names


def _format_distance(distance_m: int | float) -> str:
    if distance_m < 1000:
        return f"{round(distance_m)} 米"
    decimals = 0 if distance_m >= 10_000 else 1
    return f"{distance_m / 1000:.{decimals}f} 公里"


def _route_recommendation_sentence(routes: list[Any]) -> str:
    timed: list[tuple[Any, str, int | float]] = []
    for route in routes:
        duration = _value(route, "duration_s")
        mode = str(_value(route, "mode") or "")
        if (
            mode in _ROUTE_MODE_LABELS
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 0
        ):
            timed.append((route, _route_mode_label(route), duration))
    if len(timed) < 2:
        return ""
    minimum = min(duration for _, _, duration in timed)
    fastest_modes = [(route, label) for route, label, duration in timed if duration == minimum]
    if len(fastest_modes) != 1:
        return ""
    fastest_route, fastest_label = fastest_modes[0]
    recommendation = f"如果优先考虑本次返回的用时，建议选择{fastest_label}。"
    transit_route = next((route for route, _, _ in timed if _value(route, "mode") == "transit"), None)
    if transit_route is not None and transit_route is not fastest_route:
        recommendation += f"如果更倾向公共交通，可选择{_route_mode_label(transit_route)}方案。"
    return recommendation


def _limitations_sentence(block: Any) -> str:
    limitations = [
        neutralize_product_provider_mentions(str(item).strip())
        for item in (_value(block, "limitations") or [])
        if isinstance(item, str) and item.strip()
    ]
    if not limitations:
        return ""
    return "；".join(dict.fromkeys(limitations)) + "。"


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
