from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from app.services.stream.agent_task_policy import AgentTaskPolicy
from app.services.stream.run_capability_router import (
    RunCapabilityResolution,
    resolve_run_capability_route,
    serialize_capability_resolution,
)

ALL_TOOLS = [
    "unrelated_mcp_tool",
    "search_trains",
    "web_search",
    "route_compare",
    "url_read",
    "weather_forecast",
    "local_place_search",
    "search_flights",
]


def _task_policy(*, task_mode: str = "standard", plan_mode: str = "auto") -> AgentTaskPolicy:
    if task_mode == "deep_research":
        return AgentTaskPolicy(
            task_mode="deep_research",
            plan_mode="on",
            network_profile="deep_research",
            evidence_policy="deep_research_v1",
        )
    return AgentTaskPolicy(
        task_mode="standard",
        plan_mode=plan_mode,
        network_profile="standard",
        evidence_policy="standard",
    )


def _resolve(
    message: str,
    *,
    task_context_messages: list[object] | None = None,
    available_tool_names: list[str] | None = None,
    requested_plan_mode: str = "auto",
    task_mode: str = "standard",
    capabilities: dict | None = None,
    tools_disabled: bool = False,
    knowledge_grounded: bool = False,
) -> RunCapabilityResolution:
    return resolve_run_capability_route(
        original_message=message,
        task_context_messages=task_context_messages,
        available_tool_names=available_tool_names or ALL_TOOLS,
        requested_plan_mode=requested_plan_mode,
        task_policy=_task_policy(task_mode=task_mode, plan_mode=requested_plan_mode),
        capabilities=capabilities
        or {"functionCalling": True, "searchCapable": True},
        tools_disabled=tools_disabled,
        knowledge_grounded=knowledge_grounded,
    )


@pytest.mark.parametrize(
    (
        "message",
        "expected_package",
        "expected_tools",
        "expected_plan_mode",
        "expected_include_date",
        "expected_network_boundary",
        "expected_confidence",
        "expected_reason_codes",
    ),
    [
        (
            "你好，很高兴见到你",
            "direct",
            (),
            "off",
            False,
            False,
            "high",
            ("direct_greeting",),
        ),
        (
            "为什么天空通常看起来是蓝色的？",
            "direct",
            (),
            "off",
            False,
            False,
            "high",
            ("stable_knowledge_question",),
        ),
        (
            "把 See you tomorrow 翻译成中文",
            "transform",
            (),
            "off",
            False,
            False,
            "high",
            ("text_transform_request",),
        ),
        (
            "把北京到上海翻译成英文",
            "transform",
            (),
            "off",
            False,
            False,
            "high",
            ("text_transform_request",),
        ),
        (
            "今天是几月几日、星期几？",
            "date",
            (),
            "off",
            True,
            False,
            "high",
            ("current_date_question",),
        ),
        (
            "今天上海证券交易所开市吗？",
            "fresh_web",
            ("web_search",),
            "off",
            True,
            False,
            "high",
            ("fresh_external_fact",),
        ),
        (
            "OpenAI 今天发布了什么？阅读官方公告后总结",
            "verified_web",
            ("web_search", "url_read"),
            "auto",
            True,
            False,
            "high",
            ("verified_source_request",),
        ),
        (
            "总结 https://example.com/report，只依据该页面",
            "url_read",
            ("url_read",),
            "off",
            False,
            False,
            "high",
            ("explicit_url_read",),
        ),
        (
            "明天上海天气怎样？",
            "weather",
            ("weather_forecast",),
            "off",
            True,
            False,
            "high",
            ("explicit_weather_request",),
        ),
        (
            "找人民广场附近评分较高的咖啡店",
            "place_discovery",
            ("local_place_search",),
            "off",
            False,
            False,
            "high",
            ("explicit_place_discovery",),
        ),
        (
            "从上海虹桥站到外滩怎么坐公共交通？",
            "mobility_route",
            ("route_compare",),
            "auto",
            False,
            False,
            "high",
            ("explicit_route_task",),
        ),
        (
            "查 2026-09-10 上海到北京的机票",
            "flight",
            ("search_flights",),
            "off",
            True,
            False,
            "high",
            ("explicit_flight_request",),
        ),
        (
            "查 2026-09-10 上海到北京的高铁",
            "train",
            ("search_trains",),
            "off",
            True,
            False,
            "high",
            ("explicit_train_request",),
        ),
        (
            "北京去上海，飞机还是高铁好？",
            "travel_air_rail",
            ("search_flights", "search_trains"),
            "auto",
            True,
            False,
            "high",
            ("air_rail_comparison",),
        ),
        (
            "我现在在北京，我想去上海，你可以帮我吗",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
            "auto",
            True,
            False,
            "medium",
            ("origin_destination_relation", "intercity_locations"),
        ),
        (
            "帮我查一下这个",
            "clarification_only",
            (),
            "off",
            False,
            False,
            "low",
            ("insufficient_capability_signal",),
        ),
    ],
)
def test_route_matrix(
    message,
    expected_package,
    expected_tools,
    expected_plan_mode,
    expected_include_date,
    expected_network_boundary,
    expected_confidence,
    expected_reason_codes,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools
    assert route.effective_plan_mode == expected_plan_mode
    assert route.include_current_date is expected_include_date
    assert route.network_boundary_required is expected_network_boundary
    assert route.confidence == expected_confidence
    assert route.reason_codes == expected_reason_codes
    assert len(route.external_tool_names) <= 3 or route.package_id == "deep_research"


def test_deep_research_has_fixed_tools_and_forces_plan_on():
    route = _resolve(
        "用可靠一手来源深入研究 2026 年 AI Agent 浏览器安全现状",
        task_mode="deep_research",
        requested_plan_mode="off",
    )

    assert route.package_id == "deep_research"
    assert route.external_tool_names == ("web_search", "url_read")
    assert route.effective_plan_mode == "on"
    assert route.include_current_date is True
    assert route.network_boundary_required is False
    assert route.confidence == "high"
    assert route.reason_codes == ("deep_research_mode",)


@pytest.mark.parametrize(
    ("message", "kwargs", "expected_reason"),
    [
        (
            "查一下今天最新的 OpenAI 新闻",
            {"tools_disabled": True},
            "tools_disabled",
        ),
        (
            "查今天上海天气",
            {"capabilities": {"functionCalling": False, "searchCapable": True}},
            "function_calling_unavailable",
        ),
        (
            "查一下今天最新的 OpenAI 新闻",
            {"capabilities": {"functionCalling": True, "searchCapable": False}},
            "search_capability_unavailable",
        ),
    ],
)
def test_tool_degradation_uses_network_boundary(message, kwargs, expected_reason):
    route = _resolve(message, **kwargs)

    assert route.package_id == "tools_unavailable"
    assert route.external_tool_names == ()
    assert route.effective_plan_mode == "off"
    assert route.include_current_date is True
    assert route.network_boundary_required is True
    assert route.resolution_mode == "degraded"
    assert route.reason_codes == (expected_reason,)


def test_knowledge_grounded_marks_blocked_fresh_request_with_date_and_boundary():
    route = _resolve(
        "查一下最新的 OpenAI 新闻",
        knowledge_grounded=True,
        tools_disabled=True,
        requested_plan_mode="on",
    )

    assert route.package_id == "knowledge_grounded"
    assert route.external_tool_names == ()
    assert route.effective_plan_mode == "off"
    assert route.include_current_date is True
    assert route.network_boundary_required is True
    assert route.reason_codes == ("knowledge_grounded_mode",)


def test_knowledge_grounded_stable_question_does_not_add_network_boundary():
    route = _resolve(
        "为什么天空通常看起来是蓝色的？",
        knowledge_grounded=True,
    )

    assert route.package_id == "knowledge_grounded"
    assert route.include_current_date is False
    assert route.network_boundary_required is False


@pytest.mark.parametrize("available_tools", [["web_search"], ["url_read"]])
def test_deep_research_requires_complete_search_and_read_tool_set(available_tools):
    route = _resolve(
        "深入研究 AI Agent 浏览器安全现状",
        task_mode="deep_research",
        available_tool_names=available_tools,
    )

    assert route.package_id == "tools_unavailable"
    assert route.external_tool_names == ()
    assert route.effective_plan_mode == "off"
    assert route.include_current_date is True
    assert route.network_boundary_required is True
    assert route.resolution_mode == "degraded"
    assert route.reason_codes == ("required_tools_unavailable",)


def test_explicit_plan_mode_overrides_package_auto_policy():
    forced_on = _resolve("你好", requested_plan_mode="on")
    forced_off = _resolve(
        "从上海虹桥站到外滩怎么坐公共交通？",
        requested_plan_mode="off",
    )

    assert forced_on.package_id == "direct"
    assert forced_on.effective_plan_mode == "on"
    assert forced_off.package_id == "mobility_route"
    assert forced_off.effective_plan_mode == "off"


def test_adjacent_route_result_enables_elliptical_route_followup():
    route = _resolve(
        "哪个更适合通勤？",
        task_context_messages=[
            {"role": "user", "content": "从虹桥站到外滩怎么走？"},
            {"role": "assistant", "content": [{"type": "route_results", "schema_version": 1}]},
            {"role": "user", "content": "哪个更适合通勤？"},
        ],
    )

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)
    assert route.effective_plan_mode == "auto"
    assert route.reason_codes == ("adjacent_route_followup",)


def test_topic_switch_does_not_inherit_old_route_capability():
    route = _resolve(
        "把 See you tomorrow 翻译成中文",
        task_context_messages=[
            {"role": "user", "content": "从北京到上海怎么走？"},
            {"role": "assistant", "content": [{"type": "route_results", "schema_version": 1}]},
            {"role": "user", "content": "讲讲 Python 3.14"},
            {"role": "assistant", "content": [{"type": "text", "text": "Python 3.14 的变化如下。"}]},
            {"role": "user", "content": "把 See you tomorrow 翻译成中文"},
        ],
    )

    assert route.package_id == "transform"
    assert route.external_tool_names == ()


def test_destination_without_origin_requests_clarification():
    route = _resolve("我想去上海，你可以帮我吗")

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()
    assert route.resolution_mode == "clarification"


def test_bare_intercity_relation_with_transport_choice_is_routed():
    route = _resolve("北京到上海哪种方式好？")

    assert route.package_id == "mobility_intercity"
    assert route.external_tool_names == (
        "route_compare",
        "search_flights",
        "search_trains",
    )
    assert route.reason_codes == (
        "origin_destination_relation",
        "intercity_locations",
    )


@pytest.mark.parametrize(
    "message",
    [
        "从北京大学到上海交通大学申请哪个更适合我？",
        "从北京公司到上海公司比较哪家发展更好？",
        "比较从北京到上海两篇文章的写作风格",
    ],
)
def test_city_names_in_non_travel_context_do_not_expose_intercity_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_greeting_prefix_does_not_turn_an_ambiguous_request_into_direct():
    route = _resolve("你好，帮我查一下这个")

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_tools_are_canonical_and_intersect_available_names():
    route = _resolve(
        "北京去上海，飞机还是高铁好？",
        available_tool_names=["search_trains", "search_flights", "web_search"],
    )

    assert route.external_tool_names == ("search_flights", "search_trains")


def test_mixed_itinerary_keeps_only_three_travel_tools():
    route = _resolve(
        "从北京到上海，比较飞机和高铁，并规划落地后的市内接驳路线"
    )

    assert route.package_id == "mixed_itinerary"
    assert route.external_tool_names == (
        "route_compare",
        "search_flights",
        "search_trains",
    )
    assert route.effective_plan_mode == "auto"
    assert route.reason_codes == ("mixed_itinerary_request",)


def test_summary_of_provided_text_stays_transform():
    route = _resolve("摘要以下内容：天空通常看起来是蓝色的。")

    assert route.package_id == "transform"
    assert route.external_tool_names == ()
    assert route.include_current_date is False


def test_summary_of_fresh_official_announcement_uses_verified_web():
    route = _resolve("摘要 OpenAI 今天发布的官方公告")

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")
    assert route.effective_plan_mode == "auto"
    assert route.include_current_date is True


@pytest.mark.parametrize(
    "message",
    [
        "请查证明天上海天气并给官方来源",
        "请核验 2026-09-10 上海到北京航班并阅读官方原文",
    ],
)
def test_verified_request_has_priority_over_product_keywords(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")
    assert route.effective_plan_mode == "auto"
    assert route.include_current_date is True
    assert route.reason_codes == ("verified_source_request",)


def test_exact_authorized_mcp_alias_can_select_only_that_tool():
    route = _resolve(
        "请调用 unrelated_mcp_tool 处理这份数据",
        available_tool_names=["unrelated_mcp_tool", "web_search", "url_read"],
    )

    assert route.package_id == "mcp_explicit"
    assert route.external_tool_names == ("unrelated_mcp_tool",)
    assert route.effective_plan_mode == "off"
    assert route.reason_codes == ("explicit_authorized_tool_alias",)


def test_unmentioned_authorized_mcp_alias_is_not_exposed():
    route = _resolve(
        "帮我处理这份数据",
        available_tool_names=["unrelated_mcp_tool", "web_search", "url_read"],
    )

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_serialization_only_contains_safe_protocol_fields():
    route = _resolve("我现在在北京，我想去上海，你可以帮我吗")

    payload = serialize_capability_resolution(route)

    assert payload == {
        "schema_version": 1,
        "router_version": "2026-08-27.1",
        "package_id": "mobility_intercity",
        "confidence": "medium",
        "resolution_mode": "routed",
        "reason_codes": ["origin_destination_relation", "intercity_locations"],
        "external_tool_names": ["route_compare", "search_flights", "search_trains"],
        "effective_plan_mode": "auto",
        "include_current_date": True,
        "network_boundary_required": False,
    }
    assert "original_message" not in payload


def test_resolution_is_immutable():
    route = _resolve("你好")

    with pytest.raises(FrozenInstanceError):
        route.package_id = "weather"

    assert replace(route, package_id="weather").package_id == "weather"
