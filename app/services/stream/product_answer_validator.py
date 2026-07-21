"""校验模型基于结构化产品结果生成的最终回答。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class ProductAnswerValidation:
    """只返回稳定原因码，避免日志或调用方持有模型原文。"""

    is_valid: bool
    reason_code: str


_PRODUCT_RESULT_TYPES = {"place_results", "route_results"}
_RISK_TERM_RE = re.compile(
    r"排队|空位|预约|停车|拥堵|堵车|路况|候车|票价|免费|实时|人均|"
    r"准点|稳定|靠谱|拥挤|安全|舒适|坡度|自行车道|共享单车|省钱|便宜|实惠|性价比|"
    r"等待|灵活|掐点|翻倍|早高峰|晚高峰|高峰期|雨天|天气"
)
_COST_TERM_RE = re.compile(r"费用|成本|过路费")
_LIMITATION_CUE_RE = re.compile(
    r"未(?:提供|返回|包含|显示)|无法(?:确认|判断)|不能(?:确认|判断)|不代表|不等于|"
    r"未按.{0,20}(?:实时)?(?:路况|班次).{0,8}(?:计算|查询)|"
    r"不(?:包含|提供|返回|显示)|"
    r"需(?:要)?(?:另行|提前|自行)?(?:确认|核实|查询)|建议.{0,12}(?:确认|核实|查询)|不确定"
)
_CLAUSE_SPLIT_RE = re.compile(r"[，,。！？!?；;\n]+")
_GENERIC_PLACE_RELATION_RE = re.compile(
    r"(?:两家|两处|二者|彼此|互相).{0,12}(?:步行|相距|距离|车程|驾车|骑行)|"
    r"(?:步行|相距|距离|车程|驾车|骑行).{0,12}(?:两家|两处|二者|彼此|互相)"
)
_PLACE_PROXIMITY_RE = re.compile(
    r"(?:地址|位置|两地|两家|二者|彼此)?(?:相邻|临近)|"
    r"(?:地址|位置).{0,6}(?:相近|接近|靠近|很近)|"
    r"(?:两家|两地|二者|彼此|两个?地点).{0,16}(?:附近|接近|靠近|很近)|"
    r"(?:两家|两地|二者|彼此).{0,6}都在.{0,12}附近|"
    r"距离(?:也)?(?:很|较|比较|不算)?近|离得(?:很|较|比较)?近|"
    r"(?:步行|走路).{0,8}(?:超近|即达|可达|很近|几步)|"
    r"(?:吃完|走|步行).{0,12}(?:走几步|几步就到|溜达过去|步行即达)|"
    r"溜达.{0,8}(?:过去|到|方便)|隔壁(?:片区|区域|街区|附近)?|"
    r"就近|(?:两个?|两处|两地|两家).{0,8}(?:点|地点)?最?近|区域重叠度.{0,3}高"
)
_PLACE_UNGROUNDED_EXPERIENCE_RE = re.compile(
    r"转场.{0,8}(?:方便|轻松)|(?:吃完|饭后).{0,12}(?:随时|顺路|方便|直接|轻松)|"
    r"顺路|好找(?:好走)?|好走|靠近|节奏.{0,8}自由"
)
_PLACE_NAME_INFERENCE_RE = re.compile(r"适合.{0,12}(?:爱吃|喜欢|偏好)")
_USER_PLACE_RELATION_REQUEST_RE = re.compile(
    r"不想.{0,8}(?:走|离).{0,6}远|走太远|步行|就近|"
    r"(?:组合|搭配).{0,16}(?:地点|店|桌球|台球)|"
    r"吃完.{0,16}(?:桌球|台球|下一家|另一家)"
)
_PLACE_RELATION_CAVEAT_RE = re.compile(
    r"(?:距离|步行|走路|远近).{0,24}(?:未返回|无法确认|不能确认|另行查询|建议.{0,8}(?:导航|查询))|"
    r"(?:未返回|无法确认|不能确认).{0,24}(?:距离|步行|走路|远近)"
)
_RELATION_TERM_RE = re.compile(r"步行|相距|距离|车程|驾车|骑行")
_LINE_RE = re.compile(
    r"(?:地铁|轨道交通)?\s*(?:\d+|[一二三四五六七八九十百]+)\s*号线|"
    r"(?:地铁|轨道交通)\s*(?:\d+|[一二三四五六七八九十百]+)\s*线|"
    r"高峰专线\s*\d+\s*号?|(?:[A-Za-z]\d{1,4})(?:路|线)|\d{1,4}路",
    re.IGNORECASE,
)
_EXPLICIT_RECOMMENDATION_RE = re.compile(
    r"(?:首选|推荐|优先选择|建议选择|可以去|选择)\s*[「『“\"']?([^，。；！？!\n]{2,48})"
)
_ROUTE_MODE_TERMS = {
    "driving": ("驾车", "开车", "自驾"),
    "transit": ("公交", "地铁", "公共交通", "轨道交通"),
    "walking": ("步行", "走路"),
    "bicycling": ("骑行", "单车", "自行车"),
}
_ROUTE_MODE_PATTERN = r"(?:驾车|开车|自驾|公交|地铁|公共交通|轨道交通|步行|走路|骑行|单车|自行车)"
_ROUTE_RANKING_CLAIM_RE = re.compile(
    r"用时最短|耗时最短|时间最短|最快|用时最长|耗时最长|时间最长|最慢|"
    r"距离最短|路程最短|最近|距离最长|路程最长|最远"
)
_ROUTE_RELATIVE_COMPARISON_RE = re.compile(
    rf"(?P<left>{_ROUTE_MODE_PATTERN}).{{0,8}}比(?P<right>{_ROUTE_MODE_PATTERN}).{{0,8}}"
    r"(?P<claim>更?快|更?慢|用时更短|耗时更短|时间更短|用时更长|耗时更长|时间更长|"
    r"距离更短|路程更短|更近|距离更长|路程更长|更远)"
)
_ROUTE_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_ROUTE_STATION_MENTION_RE = re.compile(
    r"(?:在|从|到|至|经|由|途经|抵达|前往|经过)"
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）()]{1,24}?站)"
)
_ROUTE_STATION_ACTION_RE = re.compile(
    r"(?:^|[，,。！？!?；;\s])"
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）()]{1,24}?站)"
    r"(?=换乘|转乘|乘坐|上车|下车)"
)
_ROUTE_ACCESS_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<name>(?:[A-Za-z]\d{0,2}|[东南西北])\s*(?:出入口|入口|出口|口))",
    re.IGNORECASE,
)
_GENERIC_ROUTE_STATIONS = {"站", "车站", "公交站", "地铁站", "进站", "出站", "到站"}
_TRANSIT_MODE_RE = re.compile(r"公交|地铁|公共交通|轨道交通")
_WALKING_DISTANCE_RE = re.compile(r"步行|走路")
_HOUR_MINUTE_RE = re.compile(r"(?P<hours>\d+(?:\.\d+)?)\s*小时(?:\s*(?P<minutes>\d+(?:\.\d+)?)\s*分(?:钟)?)?")
_NUMBER_UNIT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>分钟|公里|米|元|次|分)")
_USER_MONEY_CONSTRAINT_RE = re.compile(r"预算|预算上限|总预算")
_DIRECT_PLACE_FACT_RE = re.compile(
    r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9·（）()]{2,32})(?:的)?"
    r"(?:综合)?(?:评分|参考消费|距离)"
)
_GENERIC_PLACE_REFERENCES = {"这家店", "该店", "这个地点", "该地点", "第一家", "第二家", "第三家"}
_PLACE_NAME_SUFFIX_RE = re.compile(r"(?:店|馆|中心|公园|站|广场|城|吧|餐厅|咖啡|火锅)$")
_TRANSIT_TOTAL_DISTANCE_RE = re.compile(
    r"(?:公交|地铁|公共交通|轨道交通).{0,12}(?:全程|总距离).{0,12}\d+(?:\.\d+)?\s*(?:公里|米)"
)
_DIFFERENCE_CUE_RE = re.compile(r"相差|差(?:了)?|快(?:了)?|慢(?:了)?|多(?:了)?|少(?:了)?|节省|缩短|增加")
_SAME_SCOPE_DIFFERENCE_RE = re.compile(r"两个?方案|两种方案|两条路线|主方案|备选|替代方案")
_UNSCOPED_SUPERLATIVE_RE = re.compile(
    r"(?:评分|消费|价格|距离|用时).{0,6}(?:最高|最低|最短|最长|最近|最远|最便宜)|"
    r"(?:最高|最低|最短|最长|最近|最远|最便宜).{0,6}(?:评分|消费|价格|距离|用时)"
)
_RETURNED_SCOPE_RE = re.compile(r"本次|此次|返回|候选|所列|卡片|这些|结果中")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
_AMAP_REALTIME_ROUTE_DATA_RE = re.compile(r"高德(?:地图)?(?:本次)?返回(?:的|了)?实时路线数据")
_REPAIR_SENTENCE_RE = re.compile(r"[^。！？!?]+(?:[。！？!?]+|$)")
_NUMBERED_LIST_PREFIX_RE = re.compile(r"^\s*\d+[.、)]\s*")
_REPAIR_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}|[-*>])\s*")
_REPAIR_SOURCE_ONLY_RE = re.compile(r"^(?:根据|结合)?(?:本次|此次)(?:查询|返回)?结果(?:显示|来看)?$")
_REPAIR_DANGLING_PREDICATE_RE = re.compile(r"^(?:是|为|属于)(?:非常|很|较|比较|更|最|也)?")
_SAFE_CAVEATS = {
    "realtime": "实时排队、空位、预约、停车、拥堵、候车和票价等未返回信息，本次查询结果无法确认，建议出发前核实。",
    "cost": "未返回的费用信息，本次查询结果无法确认，建议以实际信息为准。",
    "numeric": "未返回的时间、距离和费用信息，本次查询结果无法确认，请以卡片数值为准。",
    "relation": "地点之间的距离和步行时间本次查询结果无法确认，如需组合出行建议应另行查询路线。",
    "attribute": "口味、适合人群和转场体验等未返回属性，本次查询结果无法确认。",
    "scope": "最高、最低等排序只代表本次返回候选，不能扩展为区域整体结论。",
    "transit_total_distance": "公共交通全程距离本次查询结果无法确认，请以卡片已展示的步行距离和线路信息为准。",
}


def _empty_numeric_values() -> dict[str, set[float]]:
    return {
        "duration_minutes": set(),
        "distance_m": set(),
        "walking_distance_m": set(),
        "money_yuan": set(),
        "reference_cost_yuan": set(),
        "toll_yuan": set(),
        "transfers": set(),
        "rating": set(),
    }


@dataclass
class _FactIndex:
    searched_place_names: set[str]
    entity_names: set[str]
    route_lines: set[str]
    route_stop_names: set[str]
    route_access_names: set[str]
    numeric_values: dict[str, set[float]]
    route_numeric_values: dict[str, dict[str, set[float]]]
    route_primary_numeric_values: dict[str, dict[str, set[float]]]
    place_numeric_values: dict[str, dict[str, set[float]]]
    route_endpoint_pairs: set[frozenset[str]]
    has_place_results: bool
    has_route_results: bool


def validate_product_answer(
    answer: str,
    content_blocks: list[Any],
    *,
    messages: list[dict] | None = None,
    _enforce_completeness: bool = True,
) -> ProductAnswerValidation:
    """验证高置信硬事实；无法可靠判断的自然语言交给前置事实边界约束。"""

    normalized_answer = answer.strip() if isinstance(answer, str) else ""
    if not normalized_answer:
        return ProductAnswerValidation(False, "empty_answer")
    if _MARKDOWN_TABLE_SEPARATOR_RE.search(normalized_answer):
        return ProductAnswerValidation(False, "unsupported_format")

    product_blocks = [block for block in content_blocks if _value(block, "type") in _PRODUCT_RESULT_TYPES]
    if not product_blocks:
        return ProductAnswerValidation(False, "missing_product_result")

    facts = _build_fact_index(product_blocks)
    user_text = _latest_user_text(messages)
    if _has_unsupported_claim(normalized_answer, facts):
        return ProductAnswerValidation(False, "unsupported_claim")
    if _has_unreturned_place_relation(normalized_answer, facts):
        return ProductAnswerValidation(False, "unsupported_place_relation")
    if _has_unknown_line(normalized_answer, facts.route_lines):
        return ProductAnswerValidation(False, "unknown_line")
    if facts.has_route_results and _has_unknown_route_entity(
        normalized_answer,
        allowed_stops=facts.route_stop_names,
        allowed_accesses=facts.route_access_names,
    ):
        return ProductAnswerValidation(False, "unknown_route_entity")
    if facts.searched_place_names and _has_unknown_recommended_place(
        normalized_answer,
        facts.searched_place_names,
    ):
        return ProductAnswerValidation(False, "unknown_place")
    if facts.searched_place_names and _has_unknown_place_fact(
        normalized_answer,
        facts.searched_place_names,
    ):
        return ProductAnswerValidation(False, "unknown_place")
    if _has_numeric_mismatch(normalized_answer, facts, user_text):
        return ProductAnswerValidation(False, "numeric_mismatch")
    if _has_route_comparison_mismatch(normalized_answer, facts):
        return ProductAnswerValidation(False, "numeric_mismatch")
    if (
        _enforce_completeness
        and _needs_place_relation_caveat(user_text, facts)
        and not _PLACE_RELATION_CAVEAT_RE.search(normalized_answer)
    ):
        return ProductAnswerValidation(False, "missing_place_relation_caveat")
    return ProductAnswerValidation(True, "ok")


def repair_unsupported_product_answer(
    answer: str,
    content_blocks: list[Any],
    *,
    messages: list[dict] | None = None,
) -> tuple[str | None, str]:
    """按完整语义单元移除越界内容，避免删除分句后把事实拼到错误方案。"""

    product_blocks = [block for block in content_blocks if _value(block, "type") in _PRODUCT_RESULT_TYPES]
    if not answer.strip() or not product_blocks:
        return None, "not_repairable"
    if _MARKDOWN_TABLE_SEPARATOR_RE.search(answer):
        return None, "unsupported_format"
    answer, label_rewritten = _rewrite_repairable_labels(answer)
    facts = _build_fact_index(product_blocks)
    kept_units: list[str] = []
    safe_text_length = 0
    caveat_codes: set[str] = set()
    user_text = _latest_user_text(messages)
    if _needs_place_relation_caveat(user_text, facts) and not _PLACE_RELATION_CAVEAT_RE.search(answer):
        caveat_codes.add("relation")
    for unit in _iter_repair_units(answer):
        validation = validate_product_answer(
            unit,
            content_blocks,
            messages=messages,
            _enforce_completeness=False,
        )
        if validation.is_valid:
            kept_units.append(unit)
            safe_text_length += len(re.sub(r"\s+", "", unit))
            continue
        if validation.reason_code == "unsupported_claim":
            reasons = {
                reason
                for clause in _CLAUSE_SPLIT_RE.split(unit)
                if (reason := _unsupported_clause_reason(clause, facts)) is not None
            }
            caveat_codes.update(reasons or {"realtime"})
            salvaged = _salvage_safe_subclauses(
                unit,
                content_blocks,
                facts,
                messages=messages,
            )
            kept_units.extend(salvaged)
            safe_text_length += sum(len(re.sub(r"\s+", "", item)) for item in salvaged)
            continue
        if validation.reason_code == "unsupported_place_relation":
            caveat_codes.add("relation")
            salvaged = _salvage_safe_subclauses(
                unit,
                content_blocks,
                facts,
                messages=messages,
            )
            kept_units.extend(salvaged)
            safe_text_length += sum(len(re.sub(r"\s+", "", item)) for item in salvaged)
            continue
        if validation.reason_code == "numeric_mismatch":
            caveat_codes.add("numeric")
            salvaged = _salvage_safe_subclauses(
                unit,
                content_blocks,
                facts,
                messages=messages,
            )
            kept_units.extend(salvaged)
            safe_text_length += sum(len(re.sub(r"\s+", "", item)) for item in salvaged)
            continue
        return None, validation.reason_code
    if safe_text_length < 8 or (not caveat_codes and not label_rewritten):
        return None, "not_repairable"
    repaired = "\n".join(_normalize_repaired_unit(unit) for unit in kept_units)
    repaired = repaired.strip(" ，,。！？!?；;\n")
    if caveat_codes:
        caveats = [_SAFE_CAVEATS[code] for code in sorted(caveat_codes)]
        repaired = f"{repaired.rstrip('。')}。\n\n{' '.join(caveats)}"
    else:
        repaired = f"{repaired.rstrip('。')}。"
    validation = validate_product_answer(repaired, content_blocks, messages=messages)
    if not validation.is_valid:
        return None, validation.reason_code
    if not _has_sufficient_repair_coverage(repaired, facts):
        return None, "insufficient_coverage"
    return repaired, "ok"


def _iter_repair_units(answer: str):
    """句号与换行是安全删除边界；逗号内的主语和数值必须一起保留或一起删除。"""

    for line in answer.splitlines():
        for match in _REPAIR_SENTENCE_RE.finditer(line):
            unit = match.group(0).strip()
            if unit:
                yield unit


def _rewrite_repairable_labels(answer: str) -> tuple[str, bool]:
    """只修正不改变路线事实的来源标签，避免因一个错误形容词丢弃整段比较。"""

    rewritten = _AMAP_REALTIME_ROUTE_DATA_RE.sub("本次返回的路线数据", answer)
    return rewritten, rewritten != answer


def _salvage_safe_subclauses(
    unit: str,
    content_blocks: list[Any],
    facts: _FactIndex,
    *,
    messages: list[dict] | None,
) -> list[str]:
    """只保留能独立成立的安全子句，并补句号阻断主语与数值重新串接。"""

    salvaged: list[str] = []
    for raw_clause in _CLAUSE_SPLIT_RE.split(unit):
        clause = _NUMBERED_LIST_PREFIX_RE.sub("", raw_clause).strip()
        if len(re.sub(r"\s+", "", clause)) < 4:
            continue
        if not _is_independent_repair_clause(clause):
            continue
        validation = validate_product_answer(
            clause,
            content_blocks,
            messages=messages,
            _enforce_completeness=False,
        )
        if not validation.is_valid:
            continue
        has_number = _HOUR_MINUTE_RE.search(clause) or _NUMBER_UNIT_RE.search(clause)
        if has_number and not _has_explicit_numeric_scope(clause, facts):
            continue
        if facts.has_place_results and not _has_safe_place_repair_scope(clause, facts):
            continue
        salvaged.append(f"{clause.rstrip('。！？!?；;')}。")
    return salvaged


def _is_independent_repair_clause(clause: str) -> bool:
    """过滤逗号拆分后失去主语或只剩来源提示的病句。"""

    plain_clause = _REPAIR_MARKDOWN_PREFIX_RE.sub("", clause).strip()
    return not (_REPAIR_SOURCE_ONLY_RE.fullmatch(plain_clause) or _REPAIR_DANGLING_PREDICATE_RE.match(plain_clause))


def _has_safe_place_repair_scope(clause: str, facts: _FactIndex) -> bool:
    compact_clause = _compact_text(clause)
    if any(_compact_text(name) in compact_clause for name in facts.searched_place_names):
        return True
    if _LIMITATION_CUE_RE.search(clause) or _USER_MONEY_CONSTRAINT_RE.search(clause):
        return True
    return bool(re.search(r"高德|本次|此次|返回|候选|卡片", clause))


def _needs_place_relation_caveat(user_text: str, facts: _FactIndex) -> bool:
    return bool(
        facts.has_place_results and not facts.route_endpoint_pairs and _USER_PLACE_RELATION_REQUEST_RE.search(user_text)
    )


def _has_explicit_numeric_scope(clause: str, facts: _FactIndex) -> bool:
    if _DIFFERENCE_CUE_RE.search(clause):
        return True
    if any(term in clause for terms in _ROUTE_MODE_TERMS.values() for term in terms):
        return True
    compact_clause = _compact_text(clause)
    if any(_compact_text(name) in compact_clause for name in facts.searched_place_names):
        return True
    return any(_compact_text(value) in compact_clause for value in _GENERIC_PLACE_REFERENCES)


def _normalize_repaired_unit(unit: str) -> str:
    """删除列表项后统一改为无序列表，避免出现 1、3 这样的断号。"""

    return _NUMBERED_LIST_PREFIX_RE.sub("- ", unit)


def _has_sufficient_repair_coverage(answer: str, facts: _FactIndex) -> bool:
    """修整后的路线回答不能只剩标题或单一方案，否则完整兜底比残缺正文更可靠。"""

    available_modes = set(facts.route_numeric_values)
    if available_modes:
        grounded_modes: set[str] = set()
        for unit in _iter_repair_units(answer):
            if not (_HOUR_MINUTE_RE.search(unit) or _NUMBER_UNIT_RE.search(unit)):
                continue
            for mode, terms in _ROUTE_MODE_TERMS.items():
                if mode in available_modes and any(term in unit for term in terms):
                    grounded_modes.add(mode)
        required_modes = min(2, len(available_modes))
        return len(grounded_modes) >= required_modes

    if facts.searched_place_names:
        compact_answer = _compact_text(answer)
        return any(_compact_text(name) in compact_answer for name in facts.searched_place_names)
    return True


def _build_fact_index(blocks: list[Any]) -> _FactIndex:
    searched_place_names: set[str] = set()
    entity_names: set[str] = set()
    route_lines: set[str] = set()
    route_stop_names: set[str] = set()
    route_access_names: set[str] = set()
    numeric_values = _empty_numeric_values()
    route_numeric_values: dict[str, dict[str, set[float]]] = {}
    route_primary_numeric_values: dict[str, dict[str, set[float]]] = {}
    place_numeric_values: dict[str, dict[str, set[float]]] = {}
    route_endpoint_pairs: set[frozenset[str]] = set()
    has_place_results = False
    has_route_results = False

    for block in blocks:
        block_type = _value(block, "type")
        if block_type == "place_results":
            has_place_results = True
            for place in (_value(block, "places") or [])[:10]:
                name = _value(place, "name")
                _add_text(searched_place_names, name)
                _add_text(entity_names, name)
                place_values = place_numeric_values.setdefault(
                    _compact_text(name) if isinstance(name, str) else "",
                    _empty_numeric_values(),
                )
                _add_scoped_number(numeric_values, place_values, "distance_m", _value(place, "distance_m"))
                _add_scoped_number(
                    numeric_values,
                    place_values,
                    "money_yuan",
                    _value(place, "reference_cost_yuan"),
                )
                _add_scoped_number(
                    numeric_values,
                    place_values,
                    "reference_cost_yuan",
                    _value(place, "reference_cost_yuan"),
                )
                _add_scoped_number(numeric_values, place_values, "rating", _value(place, "rating"))
            continue

        has_route_results = True
        origin = _value(_value(block, "origin"), "label")
        destination = _value(_value(block, "destination"), "label")
        _add_text(entity_names, origin)
        _add_text(entity_names, destination)
        _add_text(route_stop_names, origin)
        _add_text(route_stop_names, destination)
        if isinstance(origin, str) and origin.strip() and isinstance(destination, str) and destination.strip():
            route_endpoint_pairs.add(frozenset({_compact_text(origin), _compact_text(destination)}))
        for route in (_value(block, "routes") or [])[:6]:
            _collect_route_facts(
                route,
                route_lines,
                route_stop_names,
                route_access_names,
                numeric_values,
                route_numeric_values,
                route_primary_numeric_values,
                parent_mode=None,
            )

    return _FactIndex(
        searched_place_names=searched_place_names,
        entity_names=entity_names,
        route_lines=route_lines,
        route_stop_names=route_stop_names,
        route_access_names=route_access_names,
        numeric_values=numeric_values,
        route_numeric_values=route_numeric_values,
        route_primary_numeric_values=route_primary_numeric_values,
        place_numeric_values=place_numeric_values,
        route_endpoint_pairs=route_endpoint_pairs,
        has_place_results=has_place_results,
        has_route_results=has_route_results,
    )


def _collect_route_facts(
    route: Any,
    route_lines: set[str],
    route_stop_names: set[str],
    route_access_names: set[str],
    numeric_values: dict[str, set[float]],
    route_numeric_values: dict[str, dict[str, set[float]]],
    route_primary_numeric_values: dict[str, dict[str, set[float]]],
    *,
    parent_mode: str | None,
) -> None:
    mode = _value(route, "mode") or parent_mode
    scoped_values = route_numeric_values.setdefault(mode, _empty_numeric_values()) if mode else None
    primary_values = (
        route_primary_numeric_values.setdefault(mode, _empty_numeric_values()) if mode and parent_mode is None else None
    )
    duration_s = _number(_value(route, "duration_s"))
    if duration_s is not None:
        _add_scoped_value(numeric_values, scoped_values, "duration_minutes", duration_s / 60)
        if primary_values is not None:
            primary_values["duration_minutes"].add(duration_s / 60)
    if mode != "transit":
        _add_scoped_number(numeric_values, scoped_values, "distance_m", _value(route, "distance_m"))
        if primary_values is not None:
            _add_bucket_number(primary_values, "distance_m", _value(route, "distance_m"))
    _add_scoped_number(
        numeric_values,
        scoped_values,
        "walking_distance_m",
        _value(route, "walking_distance_m"),
    )
    _add_scoped_number(numeric_values, scoped_values, "money_yuan", _value(route, "toll_yuan"))
    _add_scoped_number(numeric_values, scoped_values, "toll_yuan", _value(route, "toll_yuan"))
    _add_scoped_number(numeric_values, scoped_values, "transfers", _value(route, "transfers"))

    for leg in (_value(route, "legs") or [])[:12]:
        _add_text(route_lines, _value(leg, "line_name"))
        _add_text(route_stop_names, _value(leg, "departure_stop"))
        _add_text(route_stop_names, _value(leg, "arrival_stop"))
        _add_text(route_access_names, _value(leg, "entrance"))
        _add_text(route_access_names, _value(leg, "exit"))

    for alternative in (_value(route, "alternatives") or [])[:4]:
        _collect_route_facts(
            alternative,
            route_lines,
            route_stop_names,
            route_access_names,
            numeric_values,
            route_numeric_values,
            route_primary_numeric_values,
            parent_mode=mode,
        )


def _has_unsupported_claim(answer: str, facts: _FactIndex) -> bool:
    for clause in _CLAUSE_SPLIT_RE.split(answer):
        if _unsupported_clause_reason(clause, facts) is not None:
            return True
    return False


def _unsupported_clause_reason(clause: str, facts: _FactIndex) -> str | None:
    if _TRANSIT_TOTAL_DISTANCE_RE.search(clause):
        return "transit_total_distance"
    if facts.has_place_results and _PLACE_UNGROUNDED_EXPERIENCE_RE.search(clause):
        return "relation"
    if facts.has_place_results and _PLACE_NAME_INFERENCE_RE.search(clause):
        return "attribute"
    if _UNSCOPED_SUPERLATIVE_RE.search(clause) and not _RETURNED_SCOPE_RE.search(clause):
        return "scope"
    if _RISK_TERM_RE.search(clause) and not _LIMITATION_CUE_RE.search(clause):
        return "realtime"
    if _COST_TERM_RE.search(clause) and not _LIMITATION_CUE_RE.search(clause):
        if "过路费" in clause:
            category = "toll_yuan"
        elif "参考消费" in clause:
            category = "reference_cost_yuan"
        else:
            return "cost"
        allowed_money = _allowed_numeric_values(facts, category, clause, len(clause))
        if not _has_supported_money_value(clause, allowed_money):
            return "cost"
    return None


def _has_supported_money_value(clause: str, allowed_money: set[float]) -> bool:
    for match in _NUMBER_UNIT_RE.finditer(clause):
        if match.group("unit") != "元":
            continue
        if _matches_allowed(float(match.group("value")), allowed_money, category="money_yuan"):
            return True
    return False


def _has_unreturned_place_relation(answer: str, facts: _FactIndex) -> bool:
    compact_entities = {_compact_text(name) for name in facts.entity_names}
    for clause in _CLAUSE_SPLIT_RE.split(answer):
        if facts.has_place_results and _PLACE_PROXIMITY_RE.search(clause):
            return True
        if facts.has_place_results and _GENERIC_PLACE_RELATION_RE.search(clause):
            return True
        if not _RELATION_TERM_RE.search(clause):
            continue
        compact_clause = _compact_text(clause)
        mentioned = {name for name in compact_entities if name and name in compact_clause}
        if len(mentioned) >= 2 and not any(pair <= mentioned for pair in facts.route_endpoint_pairs):
            return True
    return False


def _has_unknown_line(answer: str, allowed_lines: set[str]) -> bool:
    normalized_lines = {_canonical_line(match.group(0)) for line in allowed_lines for match in _LINE_RE.finditer(line)}
    for match in _LINE_RE.finditer(answer):
        if _canonical_line(match.group(0)) not in normalized_lines:
            return True
    return False


def _has_unknown_route_entity(
    answer: str,
    *,
    allowed_stops: set[str],
    allowed_accesses: set[str],
) -> bool:
    normalized_stops = {_canonical_station_name(value) for value in allowed_stops if _canonical_station_name(value)}
    normalized_accesses = {_canonical_access_name(value) for value in allowed_accesses if _canonical_access_name(value)}
    for pattern in (_ROUTE_STATION_MENTION_RE, _ROUTE_STATION_ACTION_RE):
        for match in pattern.finditer(answer):
            name = match.group("name")
            if name in _GENERIC_ROUTE_STATIONS or name.endswith(("进站", "出站", "到站", "离站")):
                continue
            if _canonical_station_name(name) not in normalized_stops:
                return True
    for match in _ROUTE_ACCESS_MENTION_RE.finditer(answer):
        if _canonical_access_name(match.group("name")) not in normalized_accesses:
            return True
    return False


def _canonical_station_name(value: str) -> str:
    compact = _compact_text(value)
    return re.sub(r"(?:地铁)?站$", "", compact)


def _canonical_access_name(value: str) -> str:
    compact = _compact_text(value).upper()
    return re.sub(r"(?:出入口|入口|出口|口)$", "", compact)


def _canonical_line(value: str) -> str:
    compact = _compact_text(value)
    bus_code_match = re.fullmatch(r"([a-z]\d+)(?:路|线)", compact)
    if bus_code_match:
        return bus_code_match.group(1).upper()
    number_match = re.search(r"(\d+|[一二三四五六七八九十百]+)号?线$", compact)
    if number_match:
        raw_number = number_match.group(1)
        number = raw_number if raw_number.isdigit() else str(_chinese_number(raw_number))
        return f"{number}号线"
    return compact.upper()


def _chinese_number(value: str) -> int:
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    return digits.get(value, -1)


def _has_unknown_recommended_place(answer: str, allowed_places: set[str]) -> bool:
    normalized_places = {_compact_text(place) for place in allowed_places}
    for match in _EXPLICIT_RECOMMENDATION_RE.finditer(answer):
        candidate = _compact_text(match.group(1))
        if any(term in candidate for terms in _ROUTE_MODE_TERMS.values() for term in terms):
            continue
        if not any(place in candidate or candidate in place for place in normalized_places):
            return True
    return False


def _has_unknown_place_fact(answer: str, allowed_places: set[str]) -> bool:
    normalized_places = {_compact_text(place) for place in allowed_places}
    for match in _DIRECT_PLACE_FACT_RE.finditer(answer):
        subject = _compact_text(match.group("subject"))
        if subject in {_compact_text(value) for value in _GENERIC_PLACE_REFERENCES}:
            continue
        subject_parts = [part for part in re.split(r"和|与|及|、", subject) if part]
        if len(subject_parts) > 1:
            for part in subject_parts:
                if any(place in part or part in place for place in normalized_places):
                    continue
                if _PLACE_NAME_SUFFIX_RE.search(part):
                    return True
            continue
        if any(place in subject or subject in place for place in normalized_places):
            continue
        if _PLACE_NAME_SUFFIX_RE.search(subject):
            return True
    return False


def _has_numeric_mismatch(answer: str, facts: _FactIndex, user_text: str) -> bool:
    for clause in _CLAUSE_SPLIT_RE.split(answer):
        compound_spans: list[tuple[int, int]] = []
        for match in _HOUR_MINUTE_RE.finditer(clause):
            compound_spans.append(match.span())
            value = float(match.group("hours")) * 60
            if match.group("minutes") is not None:
                value += float(match.group("minutes"))
            allowed = _allowed_numeric_values(facts, "duration_minutes", clause, match.start())
            if not _matches_allowed(value, allowed, category="duration_minutes"):
                return True

        for match in _NUMBER_UNIT_RE.finditer(clause):
            if any(start <= match.start() and match.end() <= end for start, end in compound_spans):
                continue
            value = float(match.group("value"))
            unit = match.group("unit")
            if unit == "分钟":
                category = "duration_minutes"
            elif unit == "公里":
                category = _distance_category(clause)
                value *= 1000
            elif unit == "米":
                category = _distance_category(clause)
            elif unit == "元":
                category = "money_yuan"
                if _is_user_money_constraint(clause, match.group(0), user_text):
                    continue
            elif unit == "次":
                category = "transfers"
            else:
                category = "rating"
            allowed = _allowed_numeric_values(facts, category, clause, match.start())
            if not _matches_allowed(value, allowed, category=category):
                return True
    return False


def _distance_category(clause: str) -> str:
    if _TRANSIT_MODE_RE.search(clause) and _WALKING_DISTANCE_RE.search(clause):
        return "walking_distance_m"
    return "distance_m"


def _has_route_comparison_mismatch(answer: str, facts: _FactIndex) -> bool:
    primary = facts.route_primary_numeric_values
    if not primary:
        return False

    for match in _ROUTE_RELATIVE_COMPARISON_RE.finditer(answer):
        left_mode = _route_mode_from_term(match.group("left"))
        right_mode = _route_mode_from_term(match.group("right"))
        metric, direction = _comparison_metric_direction(match.group("claim"))
        if not _ordered_route_values_match(primary, left_mode, right_mode, metric, direction):
            return True

    for sentence in _ROUTE_SENTENCE_SPLIT_RE.split(answer):
        for match in _ROUTE_RANKING_CLAIM_RE.finditer(sentence):
            mode = _nearest_route_mode(sentence, match.start())
            if mode is None:
                continue
            metric, direction = _comparison_metric_direction(match.group(0))
            if not _ranked_route_value_matches(primary, mode, metric, direction):
                return True
    return False


def _route_mode_from_term(term: str) -> str:
    for mode, terms in _ROUTE_MODE_TERMS.items():
        if term in terms:
            return mode
    return ""


def _comparison_metric_direction(claim: str) -> tuple[str, str]:
    if any(term in claim for term in ("距离", "路程", "最近", "最远", "更近", "更远")):
        metric = "distance_m"
    else:
        metric = "duration_minutes"
    direction = "max" if any(term in claim for term in ("最长", "最慢", "更长", "更慢", "最远", "更远")) else "min"
    return metric, direction


def _ordered_route_values_match(
    primary: dict[str, dict[str, set[float]]],
    left_mode: str,
    right_mode: str,
    metric: str,
    direction: str,
) -> bool:
    left = _primary_route_value(primary, left_mode, metric)
    right = _primary_route_value(primary, right_mode, metric)
    if left is None or right is None:
        return False
    return left < right if direction == "min" else left > right


def _ranked_route_value_matches(
    primary: dict[str, dict[str, set[float]]],
    mode: str,
    metric: str,
    direction: str,
) -> bool:
    target = _primary_route_value(primary, mode, metric)
    candidates = [
        value
        for candidate_mode in primary
        if (value := _primary_route_value(primary, candidate_mode, metric)) is not None
    ]
    if target is None or not candidates:
        return False
    expected = min(candidates) if direction == "min" else max(candidates)
    return target == expected


def _primary_route_value(
    primary: dict[str, dict[str, set[float]]],
    mode: str,
    metric: str,
) -> float | None:
    values = primary.get(mode, _empty_numeric_values()).get(metric, set())
    if not values:
        return None
    return min(values)


def _allowed_numeric_values(
    facts: _FactIndex,
    category: str,
    clause: str,
    number_position: int,
) -> set[float]:
    if _DIFFERENCE_CUE_RE.search(clause):
        return _allowed_difference_values(facts, category, clause)
    place = _nearest_place(clause, number_position, facts.place_numeric_values)
    if place is not None:
        return facts.place_numeric_values[place][category]
    if category == "walking_distance_m":
        return facts.route_numeric_values.get("transit", _empty_numeric_values())[category]
    mode = _nearest_route_mode(clause, number_position)
    if mode is not None:
        return facts.route_numeric_values.get(mode, _empty_numeric_values())[category]
    return facts.numeric_values[category]


def _nearest_place(
    clause: str,
    number_position: int,
    place_values: dict[str, dict[str, set[float]]],
) -> str | None:
    compact_clause = _compact_text(clause)
    candidates: list[tuple[int, str]] = []
    for place in place_values:
        position = compact_clause.rfind(place)
        if position >= 0:
            candidates.append((abs(number_position - position), place))
    if not candidates:
        return None
    distance, place = min(candidates)
    return place if distance <= 48 else None


def _nearest_route_mode(clause: str, number_position: int) -> str | None:
    candidates: list[tuple[int, str]] = []
    for mode, terms in _ROUTE_MODE_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term), clause):
                candidates.append((abs(number_position - match.start()), mode))
    if not candidates:
        return None
    distance, mode = min(candidates)
    return mode if distance <= 16 else None


def _is_user_money_constraint(clause: str, raw_value: str, user_text: str) -> bool:
    if not user_text or not _USER_MONEY_CONSTRAINT_RE.search(clause):
        return False
    return re.sub(r"\s+", "", raw_value) in re.sub(r"\s+", "", user_text)


def _latest_user_text(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _matches_allowed(value: float, allowed: set[float], *, category: str) -> bool:
    if category == "duration_minutes":
        return any(abs(value - candidate) <= max(1.0, abs(candidate) * 0.03) for candidate in allowed)
    if category in {"distance_m", "walking_distance_m"}:
        return any(abs(value - candidate) <= max(50.0, abs(candidate) * 0.03) for candidate in allowed)
    if category == "rating":
        return any(abs(value - candidate) <= 0.05 for candidate in allowed)
    if category == "money_yuan":
        return any(abs(value - candidate) <= 0.1 for candidate in allowed)
    return any(abs(value - candidate) <= 0.001 for candidate in allowed)


def _allowed_difference_values(
    facts: _FactIndex,
    category: str,
    clause: str,
) -> set[float]:
    compact_clause = _compact_text(clause)
    mentioned_places = [name for name in facts.place_numeric_values if name and name in compact_clause]
    if mentioned_places:
        if len(mentioned_places) != 2:
            return set()
        return _cross_differences(
            facts.place_numeric_values[mentioned_places[0]][category],
            facts.place_numeric_values[mentioned_places[1]][category],
        )

    mentioned_modes = [mode for mode, terms in _ROUTE_MODE_TERMS.items() if any(term in clause for term in terms)]
    if len(mentioned_modes) == 2:
        if category == "distance_m" and "transit" in mentioned_modes:
            return set()
        return _cross_differences(
            facts.route_numeric_values.get(mentioned_modes[0], _empty_numeric_values())[category],
            facts.route_numeric_values.get(mentioned_modes[1], _empty_numeric_values())[category],
        )
    if len(mentioned_modes) == 1:
        if not _SAME_SCOPE_DIFFERENCE_RE.search(clause):
            return set()
        return _differences(facts.route_numeric_values.get(mentioned_modes[0], _empty_numeric_values())[category])
    if len(mentioned_modes) > 2:
        return set()

    route_scopes = [(mode, values[category]) for mode, values in facts.route_numeric_values.items() if values[category]]
    if len(route_scopes) == 2 and all(len(values) == 1 for _, values in route_scopes):
        if category == "distance_m" and any(mode == "transit" for mode, _ in route_scopes):
            return set()
        return _cross_differences(route_scopes[0][1], route_scopes[1][1])

    place_scopes = [values[category] for values in facts.place_numeric_values.values() if values[category]]
    if len(place_scopes) == 2 and all(len(values) == 1 for values in place_scopes):
        return _cross_differences(place_scopes[0], place_scopes[1])
    return set()


def _cross_differences(left_values: set[float], right_values: set[float]) -> set[float]:
    return {abs(left - right) for left in left_values for right in right_values if left != right}


def _differences(values: set[float]) -> set[float]:
    return {abs(left - right) for left, right in combinations(values, 2) if left != right}


def _add_scoped_number(
    all_values: dict[str, set[float]],
    scoped_values: dict[str, set[float]] | None,
    category: str,
    value: Any,
) -> None:
    parsed = _number(value)
    if parsed is not None:
        _add_scoped_value(all_values, scoped_values, category, parsed)


def _add_bucket_number(bucket: dict[str, set[float]], category: str, value: Any) -> None:
    parsed = _number(value)
    if parsed is not None:
        bucket[category].add(parsed)


def _add_scoped_value(
    all_values: dict[str, set[float]],
    scoped_values: dict[str, set[float]] | None,
    category: str,
    value: float,
) -> None:
    all_values[category].add(value)
    if scoped_values is not None:
        scoped_values[category].add(value)


def _add_text(values: set[str], value: Any) -> None:
    if isinstance(value, str) and value.strip():
        values.add(value.strip())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if parsed >= 0 and parsed == parsed else None


def _compact_text(value: str) -> str:
    return re.sub(r"[\s·•（）()\-—_]+", "", value).lower()


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
