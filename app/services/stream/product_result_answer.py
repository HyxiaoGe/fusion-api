"""基于结构化产品结果生成可验证的最终摘要。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.utils.user_visible_content import sanitize_internal_tool_names

_PRODUCT_RESULT_TYPES = {"place_results", "route_results", "weather_results", "flight_results", "train_results"}
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
_TRAVEL_PROVIDER_ATTRIBUTION_REPLACEMENTS = (
    (re.compile(r"(?:根据|基于)\s*(?:FlyAI|飞猪(?:旅行)?)(?:本次)?(?:返回|查询)(?:的)?"), "根据本次查询返回的"),
    (re.compile(r"(?:FlyAI|飞猪(?:旅行)?)(?:本次)?(?:返回|查询)(?:的)?结果"), "本次查询结果"),
    (re.compile(r"(?:FlyAI|飞猪(?:旅行)?)(?:本次)?(?:未能|没有|未)返回"), "本次查询未能返回"),
)
_TRAVEL_PROVIDER_NAME_RE = re.compile(r"FlyAI|飞猪(?:旅行)?", re.IGNORECASE)


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
    for pattern, replacement in _TRAVEL_PROVIDER_ATTRIBUTION_REPLACEMENTS:
        neutralized = pattern.sub(replacement, neutralized)
    neutralized = _TRAVEL_PROVIDER_NAME_RE.sub("出行查询", neutralized)
    for placeholder, term in protected_terms.items():
        neutralized = neutralized.replace(placeholder, term)
    return sanitize_internal_tool_names(neutralized, final=True)


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
        elif block_type == "weather_results":
            for key in ("query", "resolved_location"):
                value = _value(block, key)
                if isinstance(value, str) and "高德" in value:
                    terms.add(value)
        elif block_type in {"flight_results", "train_results"}:
            collection = "flights" if block_type == "flight_results" else "trains"
            for option in _value(block, collection) or []:
                for key in ("airline_name", "train_type"):
                    value = _value(option, key)
                    if isinstance(value, str) and _TRAVEL_PROVIDER_NAME_RE.search(value):
                        terms.add(value)
                for endpoint_key in ("departure", "arrival"):
                    endpoint = _value(option, endpoint_key)
                    for key in ("city", "station_name"):
                        value = _value(endpoint, key)
                        if isinstance(value, str) and _TRAVEL_PROVIDER_NAME_RE.search(value):
                            terms.add(value)
    return sorted(terms, key=len, reverse=True)


def has_product_result_blocks(content_blocks: list[Any]) -> bool:
    return any(_value(block, "type") in _PRODUCT_RESULT_TYPES for block in content_blocks)


def build_grounded_product_answer(content_blocks: list[Any]) -> str:
    """只读取产品结果块的已校验字段，不复用模型生成的自由文本。"""
    product_blocks = [block for block in content_blocks if _value(block, "type") in _PRODUCT_RESULT_TYPES][-4:]
    latest_flight = next(
        (block for block in reversed(product_blocks) if _value(block, "type") == "flight_results"), None
    )
    latest_train = next((block for block in reversed(product_blocks) if _value(block, "type") == "train_results"), None)
    if latest_flight is not None and latest_train is not None:
        comparison = _build_mixed_travel_answer(latest_flight, latest_train)
        if comparison:
            return comparison
    paragraphs: list[str] = []
    for block in product_blocks:
        block_type = _value(block, "type")
        if block_type == "place_results":
            paragraph = _build_place_answer(block)
        elif block_type == "route_results":
            paragraph = _build_route_answer(block)
        elif block_type == "weather_results":
            paragraph = _build_weather_answer(block)
        elif block_type == "flight_results":
            paragraph = _build_flight_answer(block)
        elif block_type == "train_results":
            paragraph = _build_train_answer(block)
        else:
            paragraph = ""
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def build_product_tool_failure_answer(messages: list[dict[str, Any]] | None = None) -> str:
    """产品工具未取得结构化结果时，阻止模型用训练知识补全具体事实。"""

    if _has_unavailable_geolocation_context(messages or []):
        return (
            "本次未能获取当前位置，请检查浏览器或系统定位权限后重试，也可以直接提供明确地点。"
            "由于位置没有获取成功，依赖当前位置的查询尚未执行。"
        )
    return (
        "本次未取得可用的地点或路线数据，也未取得可用的天气预报、航班或高铁数据，因此无法可靠给出具体地点、"
        "天气、线路、班次、时间、距离或费用。你可以稍后重试，或补充更明确的城市、地点、起点、终点和日期。"
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


def _build_weather_answer(block: Any) -> str:
    days = [item for item in (_value(block, "forecast_days") or []) if _value(item, "date")][:4]
    if not days:
        return ""
    location = _value(block, "resolved_location") or "该行政区"
    summaries: list[str] = []
    has_precipitation = False
    has_strong_wind = False
    weekday_labels = ("一", "二", "三", "四", "五", "六", "日")
    for day in days:
        raw_date = _value(day, "date")
        if hasattr(raw_date, "strftime"):
            date_label = f"{raw_date.month}月{raw_date.day}日"
        else:
            try:
                parsed_date = datetime.strptime(str(raw_date), "%Y-%m-%d")
                date_label = f"{parsed_date.month}月{parsed_date.day}日"
            except ValueError:
                continue
        weekday = _value(day, "weekday")
        weekday_label = (
            f"周{weekday_labels[weekday - 1]}"
            if isinstance(weekday, int) and not isinstance(weekday, bool) and 1 <= weekday <= 7
            else ""
        )
        day_weather = _value(day, "day_weather")
        night_weather = _value(day, "night_weather")
        high_c = _value(day, "high_c")
        low_c = _value(day, "low_c")
        if not all(
            (
                isinstance(day_weather, str) and day_weather,
                isinstance(night_weather, str) and night_weather,
                isinstance(high_c, (int, float)) and not isinstance(high_c, bool),
                isinstance(low_c, (int, float)) and not isinstance(low_c, bool),
            )
        ):
            continue
        detail = (
            f"{date_label}{f'（{weekday_label}）' if weekday_label else ''}"
            f"白天{day_weather}、夜间{night_weather}，{_format_temperature(low_c)}–{_format_temperature(high_c)}℃"
        )
        wind_parts: list[str] = []
        for direction_key, power_key, period in (
            ("day_wind_direction", "day_wind_power", "白天"),
            ("night_wind_direction", "night_wind_power", "夜间"),
        ):
            direction = _value(day, direction_key)
            power = _value(day, power_key)
            if isinstance(direction, str) and direction and isinstance(power, str) and power:
                wind_parts.append(f"{period}{direction}风{power}级")
        if wind_parts:
            detail += f"，{'、'.join(wind_parts)}"
        summaries.append(detail)
        has_precipitation = has_precipitation or bool(re.search(r"雨|雪|雷", f"{day_weather}{night_weather}"))
        has_strong_wind = has_strong_wind or bool(
            re.search(r"大风|台风", f"{day_weather}{night_weather}{''.join(wind_parts)}")
        )
    if not summaries:
        return ""
    lead = f"{location}天气预报：{'；'.join(summaries)}。"
    if has_precipitation:
        advice = "如需外出，建议携带雨具并根据天气减少长时间步行。"
    elif has_strong_wind:
        advice = "如需外出，建议做好一般防风措施并减少长时间步行。"
    else:
        advice = ""
    return f"{lead}{advice}{_limitations_sentence(block)}"


def _format_temperature(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(float(value), 1))


def _build_flight_answer(block: Any) -> str:
    flights = [item for item in (_value(block, "flights") or []) if _value(item, "flight_no")]
    return _build_scheduled_travel_answer(block, flights, number_key="flight_no", kind_label="直达航班")


def _build_train_answer(block: Any) -> str:
    trains = [item for item in (_value(block, "trains") or []) if _value(item, "train_no")]
    return _build_scheduled_travel_answer(block, trains, number_key="train_no", kind_label="直达车次")


def _build_mixed_travel_answer(flight_block: Any, train_block: Any) -> str:
    """为模型格式违规场景生成短小、可校验的航班与火车对比。"""

    if any(
        _value(flight_block, key) != _value(train_block, key) for key in ("origin", "destination", "departure_date")
    ):
        return ""
    flights = [item for item in (_value(flight_block, "flights") or []) if _value(item, "flight_no")]
    trains = [item for item in (_value(train_block, "trains") or []) if _value(item, "train_no")]
    if not flights or not trains:
        return ""
    cheapest_flight = _minimum_travel_option(flights, "price")
    fastest_flight = _minimum_travel_option(flights, "duration")
    fastest_train = _minimum_travel_option(trains, "duration")
    cheapest_train = _minimum_travel_option(trains, "price")
    if cheapest_flight is None or fastest_flight is None or fastest_train is None or cheapest_train is None:
        return ""

    fastest_kind, fastest_option, fastest_number_key = min(
        (
            ("航班", fastest_flight, "flight_no"),
            ("高铁", fastest_train, "train_no"),
        ),
        key=lambda candidate: _travel_metric_value(candidate[1], "duration"),
    )
    cheapest_kind, cheapest_option, cheapest_number_key = min(
        (
            ("航班", cheapest_flight, "flight_no"),
            ("高铁", cheapest_train, "train_no"),
        ),
        key=lambda candidate: _travel_metric_value(candidate[1], "price"),
    )

    origin = _value(flight_block, "origin")
    destination = _value(flight_block, "destination")
    departure_date = _value(flight_block, "departure_date")
    paragraphs = [
        f"本次查询同时返回{origin}到{destination}在{departure_date}的 {len(flights)} 个直达航班和 {len(trains)} 个直达车次。",
        (
            f"本次返回航班中参考价最低的是{_compact_travel_option(cheapest_flight, 'flight_no')}；"
            f"本次返回火车中用时最短的是{_compact_travel_option(fastest_train, 'train_no')}，"
            f"本次返回火车中参考价最低的是{_compact_travel_option(cheapest_train, 'train_no')}。"
        ),
        (
            "如果优先考虑本次返回的计划行程时长，可优先考虑"
            f"{fastest_kind}{_value(fastest_option, fastest_number_key)}；"
            f"如果预算优先，可考虑{cheapest_kind}{_value(cheapest_option, cheapest_number_key)}。"
        ),
        "卡片中的时长是班次计划行程时长，不包含前后接驳、值机或安检等额外时间。",
    ]
    limitations = _combined_limitations_sentence(flight_block, train_block)
    if limitations:
        paragraphs.append(limitations)
    return "\n\n".join(paragraphs)


def _minimum_travel_option(options: list[Any], metric: str) -> Any | None:
    candidates = [(value, option) for option in options if (value := _travel_metric_value(option, metric)) is not None]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _travel_metric_value(option: Any, metric: str) -> int | None:
    value = _value(_value(option, "price"), "amount_minor") if metric == "price" else _value(option, "duration_s")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _compact_travel_option(option: Any, number_key: str) -> str:
    number = _value(option, number_key)
    duration_s = _value(option, "duration_s")
    price_minor = _value(_value(option, "price"), "amount_minor")
    details: list[str] = []
    if isinstance(duration_s, int) and not isinstance(duration_s, bool) and duration_s >= 0:
        details.append(f"约{_format_duration(duration_s)}")
    if isinstance(price_minor, int) and not isinstance(price_minor, bool) and price_minor >= 0:
        details.append(f"参考价{_format_yuan(price_minor)}元")
    return f"{number}（{'，'.join(details)}）" if details else str(number)


def _format_duration(duration_s: int) -> str:
    minutes = round(duration_s / 60)
    hours, remaining = divmod(minutes, 60)
    if hours and remaining:
        return f"{hours}小时{remaining}分钟"
    if hours:
        return f"{hours}小时"
    return f"{remaining}分钟"


def _combined_limitations_sentence(*blocks: Any) -> str:
    limitations: list[str] = []
    for block in blocks:
        limitations.extend(
            neutralize_product_provider_mentions(str(item).strip())
            for item in (_value(block, "limitations") or [])
            if isinstance(item, str) and item.strip()
        )
    return "；".join(dict.fromkeys(limitations)) + "。" if limitations else ""


def _build_scheduled_travel_answer(
    block: Any,
    options: list[Any],
    *,
    number_key: str,
    kind_label: str,
) -> str:
    origin = _value(block, "origin") or "出发地"
    destination = _value(block, "destination") or "目的地"
    departure_date = _value(block, "departure_date") or "指定日期"
    if not options:
        lead = f"本次查询没有返回{origin}到{destination}在{departure_date}可展示的{kind_label}。"
    else:
        summaries = [_format_scheduled_option(option, number_key=number_key) for option in options[:5]]
        lead = (
            f"本次查询返回{origin}到{destination}在{departure_date}的 {len(options)} 个{kind_label}："
            f"{'；'.join(summary for summary in summaries if summary)}。"
        )
    return f"{lead}{_limitations_sentence(block)}"


def _format_scheduled_option(option: Any, *, number_key: str) -> str:
    number = str(_value(option, number_key) or "班次")
    operator = _value(option, "airline_name") or _value(option, "train_type")
    departure = _value(option, "departure")
    arrival = _value(option, "arrival")
    departure_station = _value(departure, "station_name")
    arrival_station = _value(arrival, "station_name")
    departure_time = _format_scheduled_time(_value(departure, "scheduled_at"))
    arrival_time = _format_scheduled_time(_value(arrival, "scheduled_at"))
    details = [f"{departure_time} 从{departure_station}出发", f"{arrival_time} 到达{arrival_station}"]
    duration_s = _value(option, "duration_s")
    if isinstance(duration_s, int) and not isinstance(duration_s, bool) and duration_s >= 0:
        details.append(f"约 {round(duration_s / 60)} 分钟")
    travel_class = _value(option, "cabin_class") or _value(option, "seat_class")
    if isinstance(travel_class, str) and travel_class:
        details.append(travel_class)
    price_minor = _value(_value(option, "price"), "amount_minor")
    if isinstance(price_minor, int) and not isinstance(price_minor, bool) and price_minor >= 0:
        details.append(f"参考价{_format_yuan(price_minor)}元")
    operator_text = f"（{operator}）" if isinstance(operator, str) and operator else ""
    return f"{number}{operator_text}，{'，'.join(details)}"


def _format_scheduled_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    if isinstance(value, str):
        match = re.search(r"T(\d{2}:\d{2})", value)
        if match:
            return match.group(1)
    return "时间未提供"


def _format_yuan(amount_minor: int) -> str:
    yuan = amount_minor / 100
    return str(int(yuan)) if yuan.is_integer() else f"{yuan:.2f}".rstrip("0").rstrip(".")


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
