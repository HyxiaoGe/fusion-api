from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from app.services.stream.agent_task_policy import AgentTaskPolicy
from app.services.stream.run_capability_router import (
    RunCapabilityResolution,
    resolve_run_capability_route,
    serialize_capability_resolution,
)
from app.utils.run_capability_contract import validate_capability_resolution_semantics

ALL_TOOLS = [
    "mcp_unrelated_tool",
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
        capabilities=capabilities or {"functionCalling": True, "searchCapable": True},
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
        (
            "Translate this into Chinese: See you tomorrow.",
            "transform",
            (),
            "off",
            False,
            False,
            "high",
            ("text_transform_request",),
        ),
        (
            "What is today's date?",
            "date",
            (),
            "off",
            True,
            False,
            "high",
            ("current_date_question",),
        ),
        (
            "What day is it today?",
            "date",
            (),
            "off",
            True,
            False,
            "high",
            ("current_date_question",),
        ),
        (
            "What is the latest OpenAI release?",
            "fresh_web",
            ("web_search",),
            "off",
            True,
            False,
            "high",
            ("fresh_external_fact",),
        ),
        (
            "Check the official OpenAI announcement and summarize it.",
            "verified_web",
            ("web_search", "url_read"),
            "auto",
            True,
            False,
            "high",
            ("verified_source_request",),
        ),
        (
            "Summarize https://example.com/report using only that page.",
            "url_read",
            ("url_read",),
            "off",
            False,
            False,
            "high",
            ("explicit_url_read",),
        ),
        (
            "What is the weather in Shanghai today?",
            "weather",
            ("weather_forecast",),
            "off",
            True,
            False,
            "high",
            ("explicit_weather_request",),
        ),
        (
            "Find highly rated coffee shops near People's Square.",
            "place_discovery",
            ("local_place_search",),
            "off",
            False,
            False,
            "high",
            ("explicit_place_discovery",),
        ),
        (
            "How do I get from Shanghai Hongqiao Station to the Bund by public transit?",
            "mobility_route",
            ("route_compare",),
            "auto",
            False,
            False,
            "high",
            ("explicit_route_task",),
        ),
        (
            "Find flights from Shanghai to Beijing on 2026-09-10.",
            "flight",
            ("search_flights",),
            "off",
            True,
            False,
            "high",
            ("explicit_flight_request",),
        ),
        (
            "Find trains from Shanghai to Beijing on 2026-09-10.",
            "train",
            ("search_trains",),
            "off",
            True,
            False,
            "high",
            ("explicit_train_request",),
        ),
        (
            "Compare flights and trains from Beijing to Shanghai.",
            "travel_air_rail",
            ("search_flights", "search_trains"),
            "auto",
            True,
            False,
            "high",
            ("air_rail_comparison",),
        ),
        (
            "I'm in Beijing and I want to go to Shanghai. Can you help me?",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
            "auto",
            True,
            False,
            "medium",
            ("origin_destination_relation", "intercity_locations"),
        ),
        (
            "How do I get from Beijing to Shanghai?",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
            "auto",
            True,
            False,
            "medium",
            ("origin_destination_relation", "intercity_locations"),
        ),
        (
            "I need to travel from Beijing to Shanghai.",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
            "auto",
            True,
            False,
            "medium",
            ("origin_destination_relation", "intercity_locations"),
        ),
        (
            "联网查一下量子计算的入门资料",
            "fresh_web",
            ("web_search",),
            "off",
            True,
            False,
            "high",
            ("fresh_external_fact",),
        ),
        (
            "打开 https://example.com/report 看看",
            "url_read",
            ("url_read",),
            "off",
            False,
            False,
            "high",
            ("explicit_url_read",),
        ),
        (
            "从虹桥机场到外滩怎么走？",
            "mobility_route",
            ("route_compare",),
            "auto",
            False,
            False,
            "high",
            ("explicit_route_task",),
        ),
        (
            "从上海火车站到人民广场怎么坐地铁？",
            "mobility_route",
            ("route_compare",),
            "auto",
            False,
            False,
            "high",
            ("explicit_route_task",),
        ),
        (
            "从北京到上海怎么去？",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
            "auto",
            True,
            False,
            "medium",
            ("origin_destination_relation", "intercity_locations"),
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


@pytest.mark.parametrize(
    "message",
    [
        "What is photosynthesis?",
        "What is a flight recorder?",
        "How does train scheduling work?",
        "What is weathering in geology?",
        "What is wind?",
        "How does rain form?",
        "What is temperature?",
    ],
)
def test_english_stable_knowledge_does_not_trigger_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "direct"
    assert route.external_tool_names == ()
    assert route.reason_codes == ("stable_knowledge_question",)


def test_english_place_value_request_does_not_trigger_place_discovery():
    route = _resolve("Find the place value of 7 in 7,042.")

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_chinese_search_algorithm_noun_does_not_trigger_web_search():
    route = _resolve("搜索算法的时间复杂度是什么？")

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "不要联网搜索，解释一下二分查找",
        "Do not search the web. Explain binary search.",
        "Don’t search the web. Explain binary search.",
        "别联网搜索，解释一下二分查找",
        "不用打开 https://example.com/report，根据我提供的标题回答",
        "请勿打开 https://example.com/report，根据我提供的标题回答",
    ],
)
def test_negated_network_actions_do_not_announce_external_tools(message):
    route = _resolve(message)

    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "不要联网，告诉我 OpenAI 今天最新发布了什么",
        "不要使用互联网，告诉我 OpenAI 今天最新发布了什么",
        "请在不联网的情况下，告诉我今天的最新新闻",
        "Please don't use the internet; what is the latest OpenAI release?",
        "Don't go online; what is the latest OpenAI release?",
        "No web search: What is the latest OpenAI release?",
        "In offline mode, what is the latest OpenAI release?",
        "离线情况下告诉我今天最新新闻",
    ],
)
def test_common_network_negations_do_not_announce_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()
    assert route.include_current_date is False


@pytest.mark.parametrize(
    "message",
    [
        "The app is in offline mode; search the web for fixes",
        "Don't use the internet for old docs, but search the web for the latest docs",
        "不要使用互联网查旧公告，但请联网搜索最新公告",
        "不要搜索旧新闻而要搜索最新新闻",
        "During this request, do not use the internet for weather; then search the latest AI news.",
        "For this request, do not use the internet for weather; then search the latest AI news.",
    ],
)
def test_later_explicit_web_authorization_overrides_scoped_description_or_denial(message):
    route = _resolve(message)

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not search the web.",
        "Read https://example.com/a; do not open https://example.com/a.",
        "不要调用 web_search，解释最新新闻",
        "Do not use url_read; summarize https://example.com/report",
    ],
)
def test_final_or_explicit_tool_denial_wins_over_earlier_positive_signal(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_negated_mcp_alias_is_not_treated_as_explicit_authorization():
    route = _resolve(
        "Do not call mcp_unrelated_tool",
        available_tool_names=["mcp_unrelated_tool", "web_search", "url_read"],
    )

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Don't search the web; what's the latest OpenAI release?",
            "clarification_only",
            (),
        ),
        (
            "I'm in Beijing and I want to go to Shanghai. It's urgent.",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
        ),
    ],
)
def test_ascii_contractions_are_not_mistaken_for_single_quoted_literals(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "不要联网查旧新闻，但查明天上海天气",
            "weather",
            ("weather_forecast",),
        ),
        (
            "Don't use the internet for news, but find flights from Beijing to Shanghai tomorrow",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_scoped_all_network_denial_does_not_lock_later_product_capability(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Don't call web_search for old news; search the web for the latest AI news",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Do not call url_read for https://example.com/a; read https://example.com/b",
            "url_read",
            ("url_read",),
        ),
        (
            "Do not call search_flights for old routes; find flights from Beijing to Shanghai tomorrow",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_scoped_explicit_tool_denial_can_be_reauthorized_by_later_request(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "查北京到上海高铁，但不要调用 search_trains",
        "Find flights from Beijing to Shanghai tomorrow, but do not call search_flights",
        "How do I get from Beijing to Shanghai by car, but do not call route_compare",
    ],
)
def test_final_product_tool_denial_prevents_requested_tool_announcement(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Do not call mcp_unrelated_tool; then call mcp_unrelated_tool",
            "mcp_explicit",
            ("mcp_unrelated_tool",),
        ),
        (
            "Call mcp_unrelated_tool; then do not call mcp_unrelated_tool",
            "clarification_only",
            (),
        ),
    ],
)
def test_mcp_alias_authorization_uses_the_last_explicit_directive(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Read and translate https://example.com/report, then search the web for updates",
        "Summarize https://example.com/report and cross-check it with official sources",
    ],
)
def test_url_read_and_independent_web_action_keep_both_tools(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by train, not by plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, not by plane?",
            "mobility_route",
            ("route_compare",),
        ),
    ],
)
def test_negated_english_route_mode_is_not_authorized(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Please work offline and tell me the latest OpenAI news.",
        "请离线回答今天最新新闻。",
    ],
)
def test_polite_offline_constraint_still_blocks_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            'Open "https://example.com/report" and summarize it.',
            "url_read",
            ("url_read",),
        ),
        (
            "Summarize `https://example.com/report`.",
            "url_read",
            ("url_read",),
        ),
        (
            "Please call `mcp_unrelated_tool`.",
            "mcp_explicit",
            ("mcp_unrelated_tool",),
        ),
    ],
)
def test_explicitly_operated_quoted_resource_remains_routable(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        'Translate "OpenAI" and do a web search for context.',
        'Translate "OpenAI" and browse the web for context.',
    ],
)
def test_transform_with_common_explicit_web_action_keeps_search_tool(message):
    route = _resolve(message)

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by car, train, or plane?",
            "mixed_itinerary",
            ("route_compare", "search_flights", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train, plane, or air?",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
    ],
)
def test_comma_enumerated_route_modes_preserve_the_full_union(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Search the latest AI news; do not search old quantum news.",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Read https://example.com/b; do not open https://example.com/a.",
            "url_read",
            ("url_read",),
        ),
    ],
)
def test_later_denial_for_a_different_object_does_not_cancel_allowed_work(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_scoped_internet_denial_does_not_block_later_verified_request():
    route = _resolve("Don't use the internet for old docs; verify the latest official OpenAI announcement")

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


def test_trailing_network_negation_keeps_stable_knowledge_direct():
    route = _resolve("解释二分查找，不要联网搜索。")

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "不要联网搜索，只读取 https://example.com/report",
            "url_read",
            ("url_read",),
        ),
        (
            "Do not search the web; only read https://example.com/report",
            "url_read",
            ("url_read",),
        ),
        (
            "不要联网搜索但只读取 https://example.com/report",
            "url_read",
            ("url_read",),
        ),
        (
            "Do not search the web but only read https://example.com/report",
            "url_read",
            ("url_read",),
        ),
        (
            "不用打开网页，请联网搜索 OpenAI 最新发布",
            "fresh_web",
            ("web_search",),
        ),
        (
            "不要打开网页但请联网搜索 OpenAI 最新发布",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Do not open https://example.com/a; read https://example.com/b",
            "url_read",
            ("url_read",),
        ),
        (
            "Do not search old news; search the latest AI news",
            "fresh_web",
            ("web_search",),
        ),
    ],
)
def test_scoped_network_negation_does_not_disable_other_explicit_capability(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "把“最新新闻”翻译成英文",
        'Translate "latest news" into Chinese.',
        "把“官方公告”翻译成英文",
        'Translate "official announcement" into Chinese.',
        "Translate 'latest news' into Chinese.",
        'Translate the phrase "latest official announcement" into Chinese.',
        'Translate: "latest news"',
        "“latest news” 翻译成中文",
        'Rewrite "Don’t search the web; use only local knowledge."',
        'Translate "don\'t use the internet" into Chinese.',
    ],
)
def test_given_text_with_external_source_words_stays_local_transform(message):
    route = _resolve(message)

    assert route.package_id == "transform"
    assert route.external_tool_names == ()
    assert route.include_current_date is False


@pytest.mark.parametrize(
    "message",
    [
        'What does "latest news" mean?',
        'What does the phrase "official announcement" mean?',
        'Explain the sentence "How do I get from Beijing to Shanghai by train?"',
        'What does "How do I get from Beijing to Shanghai?" mean?',
    ],
)
def test_quoted_literals_do_not_activate_external_or_route_capabilities(message):
    route = _resolve(message)

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "把 OpenAI 今天最新发布的官方公告翻译成英文",
        "将 OpenAI 官方公告翻译成中文",
    ],
)
def test_unprovided_external_announcement_translation_keeps_verified_tools(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    "message",
    [
        "Translate this page into Chinese: https://example.com/report",
        "把这个页面翻译成中文：https://example.com/report",
        "Translate the following URL into Chinese: https://example.com/report",
    ],
)
def test_page_translation_with_url_keeps_url_read_tool(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


def test_quoted_translation_with_independent_search_keeps_search_tool():
    route = _resolve('Translate "OpenAI" into Chinese and search the web for its latest announcement')

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            'Translate "official announcement" using the official source.',
            "verified_web",
            ("web_search", "url_read"),
        ),
        (
            'Translate "headline" after reading https://example.com/report.',
            "url_read",
            ("url_read",),
        ),
        (
            "Translate the words latest news into Chinese.",
            "transform",
            (),
        ),
        (
            "将最新新闻这四个字翻译成英文",
            "transform",
            (),
        ),
    ],
)
def test_transform_literal_and_explicit_external_actions_use_correct_precedence(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from draft to publication?",
        "Explain the route from junior engineer to architect.",
    ],
)
def test_english_abstract_from_to_relations_do_not_trigger_routes(message):
    route = _resolve(message)

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Pudong Airport to Disneyland?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Tiananmen to the Forbidden City?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Harbin to Beijing?",
            "mobility_intercity",
            ("route_compare", "search_flights", "search_trains"),
        ),
    ],
)
def test_english_physical_routes_are_not_swallowed_by_stable_knowledge(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by train?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by plane?",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_english_explicit_travel_mode_uses_only_the_requested_tool(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by train or plane?",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by airplane?",
            "flight",
            ("search_flights",),
        ),
        (
            "How do I get from Beijing to Shanghai by train or bus?",
            "mixed_itinerary",
            ("route_compare", "search_trains"),
        ),
    ],
)
def test_english_parallel_travel_modes_preserve_every_requested_tool(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from Pudong Airport to Disneyland by train?",
        "How do I get from Tiananmen to the Forbidden City by train?",
    ],
)
def test_local_train_request_uses_route_compare_instead_of_intercity_ticket_search(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from Beijing to Shanghai by car?",
        "How do I get from Beijing to Shanghai by public transit?",
    ],
)
def test_intercity_explicit_ground_route_does_not_announce_ticket_tools(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)


def test_comma_separated_intercity_modes_preserve_flight_and_train_tools():
    route = _resolve("How do I get from Beijing to Shanghai by train, or by plane?")

    assert route.package_id == "travel_air_rail"
    assert route.external_tool_names == ("search_flights", "search_trains")


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai via rail, then by air?",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train or coach?",
            "mixed_itinerary",
            ("route_compare", "search_trains"),
        ),
    ],
)
def test_multiple_mode_segments_and_coach_preserve_all_tools(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package"),
    [
        ("How do I get from draft to publication by train?", "direct"),
        ("How do I get from Red Building to Blue Hall by plane?", "clarification_only"),
    ],
)
def test_english_travel_mode_cannot_bypass_endpoint_trust(message, expected_package):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == ()


def test_english_abstract_center_suffix_does_not_trigger_route_tool():
    route = _resolve("Explain the process route from cost center to profit center.")

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Explain the escalation route from support center to engineering center.",
        "Explain the reporting route from sales office to finance office.",
        "Explain the education route from primary school to graduate school.",
        "Explain the organizational escalation route from support district to engineering district.",
    ],
)
def test_ambiguous_english_suffixes_are_not_positive_physical_evidence(message):
    route = _resolve(message)

    assert route.external_tool_names == ()


def test_unknown_english_from_to_route_requires_clarification_instead_of_direct_answer():
    route = _resolve("How do I get from Red Building to Blue Hall?")

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "把 https://example.com/report 翻译成中文",
            "url_read",
            ("url_read",),
        ),
        (
            "翻译 OpenAI 今天最新发布的官方公告",
            "verified_web",
            ("web_search", "url_read"),
        ),
        (
            "查明天上海天气和上海到北京的机票",
            "mixed_itinerary",
            ("weather_forecast", "search_flights"),
        ),
        (
            "查明天上海天气，并找人民广场附近的咖啡店",
            "mixed_itinerary",
            ("weather_forecast", "local_place_search"),
        ),
        (
            "规划从上海虹桥站到外滩的公共交通路线，并查明天天气",
            "mixed_itinerary",
            ("weather_forecast", "route_compare"),
        ),
    ],
)
def test_external_source_and_multi_capability_requests_do_not_silently_drop_tools(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_contract_rejects_more_than_three_external_tools_for_mixed_itinerary():
    with pytest.raises(ValueError, match="最多三个"):
        validate_capability_resolution_semantics(
            package_id="mixed_itinerary",
            confidence="high",
            resolution_mode="routed",
            reason_codes=("mixed_itinerary_request",),
            external_tool_names=(
                "weather_forecast",
                "local_place_search",
                "route_compare",
                "search_flights",
            ),
            effective_plan_mode="auto",
            include_current_date=True,
            network_boundary_required=False,
        )


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
        "从北京到上海",
        "从北京去上海",
        "住在北京，公司在上海",
    ],
)
def test_structured_intercity_relation_is_itself_a_mobility_signal(message):
    route = _resolve(message)

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


def test_business_development_route_does_not_expose_mobility_tools():
    route = _resolve("比较北京到上海两家公司的发展路线")

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "从北京团队到上海团队的职责如何划分？",
        "从北京部门到上海部门的协作关系是什么？",
    ],
)
def test_organization_slots_are_not_treated_as_places_without_route_action(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_institution_slots_are_valid_when_route_action_is_explicit():
    route = _resolve("从北京大学到上海交通大学哪个路线更快？")

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)
    assert route.reason_codes == ("explicit_route_task",)


@pytest.mark.parametrize(
    "message",
    [
        "从故宫到颐和园怎么走？",
        "从迪士尼到东方明珠如何去？",
        "从奥体中心到市民中心坐地铁怎么走？",
        "从故宫到颐和园哪条路线？",
        "从北京公交集团到上海地铁公司怎么走？",
    ],
)
def test_explicit_mobility_accepts_confirmed_physical_endpoint_slots(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)
    assert route.reason_codes == ("explicit_route_task",)


@pytest.mark.parametrize(
    "message",
    [
        "从亏损到盈利怎么走？",
        "请规划从冷启动到规模化的路线",
        "从需求评审到正式上线怎么走流程？",
        "从初级工程师到架构师怎么走？",
        "从初级工程师到架构师怎么走职业路径？",
    ],
)
def test_abstract_process_and_career_paths_do_not_expose_mobility_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "请给我从故宫到颐和园的路线",
        "请给出从故宫到颐和园的路线",
        "请规划从故宫到颐和园的路线",
        "请推荐从故宫到颐和园的路线",
        "查询从故宫到颐和园的路线",
        "帮我查询从故宫到颐和园的路线",
        "帮我查询一下从故宫到颐和园的路线",
        "获取从故宫到颐和园的路线",
        "请帮我查下从故宫到颐和园的路线",
        "帮我找一条从故宫到颐和园的路线",
        "提供一条从故宫到颐和园的路线",
    ],
)
def test_plain_route_request_structure_exposes_route_compare(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)
    assert route.reason_codes == ("explicit_route_task",)


@pytest.mark.parametrize(
    "message",
    [
        "请给我从前端团队到后端团队的技术路线",
        "请规划从初创团队到成熟团队的发展路线",
        "请推荐从销售线索到正式签约的业务路线",
    ],
)
def test_plain_route_request_structure_rejects_abstract_route_types(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "从 到颐和园怎么走？",
        f"从故宫到{'颐' * 61}怎么走？",
    ],
)
def test_explicit_mobility_rejects_empty_or_overlong_endpoint_slots(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "从北京产品中心到上海研发中心，比较两边职责",
        "从北京运营中心到上海研发中心的协作关系是什么？",
        "从北京公交集团到上海地铁公司的职责如何划分？",
        "从前端团队到后端团队比较技术路线",
        "从北京产品中心到上海研发中心比较发展路线",
    ],
)
def test_ambiguous_organization_and_abstract_route_slots_do_not_expose_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_safe_natural_location_slots_keep_structured_intercity_package():
    route = _resolve("从广州南站到深圳北站")

    assert route.package_id == "mobility_intercity"
    assert route.external_tool_names == (
        "route_compare",
        "search_flights",
        "search_trains",
    )


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


def test_three_tool_package_keeps_canonical_partial_subsequence():
    route = _resolve(
        "我现在在北京，我想去上海，你可以帮我吗",
        available_tool_names=["search_trains", "route_compare"],
    )

    assert route.package_id == "mobility_intercity"
    assert route.external_tool_names == ("route_compare", "search_trains")


def test_mixed_itinerary_keeps_only_three_travel_tools():
    route = _resolve("从北京到上海，比较飞机和高铁，并规划落地后的市内接驳路线")

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


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not search it.",
        "Search the latest AI news; do not search that topic.",
    ],
)
def test_final_anaphoric_web_denial_cancels_the_prior_search(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Please stay offline and tell me the latest OpenAI news.",
        "Please remain offline and tell me the latest OpenAI news.",
        "请保持离线并回答今天最新新闻。",
        "Tell me the latest OpenAI news without going online.",
        "I'd prefer an offline answer about the latest OpenAI news.",
    ],
)
def test_offline_language_variants_block_all_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Summarize https://example.com/latest-news",
        'Summarize "https://example.com/latest-news"',
        "Read https://example.com/official-announcement",
        "Summarize https://example.com/latest-news using only this page",
    ],
)
def test_url_path_words_do_not_synthesize_an_independent_web_search(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai, not by train, but by plane?",
            "flight",
            ("search_flights",),
        ),
        (
            "How do I get from Beijing to Shanghai by train or plane, but not by plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, train, or plane, but not by plane?",
            "mixed_itinerary",
            ("route_compare", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train rather than plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car instead of train or plane?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, not train or plane?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Beijing to Shanghai by car or train, but not train?",
            "mobility_route",
            ("route_compare",),
        ),
    ],
)
def test_english_route_modes_use_ordered_last_directive_semantics(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Call search_flights for Beijing to Shanghai tomorrow",
        "Do not call search_flights; then call search_flights for Beijing to Shanghai tomorrow",
    ],
)
def test_explicit_positive_product_tool_directive_authorizes_that_product(message):
    route = _resolve(message)

    assert route.package_id == "flight"
    assert route.external_tool_names == ("search_flights",)


@pytest.mark.parametrize(
    "message",
    [
        "Find flights from Beijing to Shanghai tomorrow; do not call search_flights for old routes",
        "Find flights from Beijing to Shanghai tomorrow and do not call search_flights for old routes",
    ],
)
def test_scoped_product_denial_does_not_cancel_prior_positive_request(message):
    route = _resolve(message)

    assert route.package_id == "flight"
    assert route.external_tool_names == ("search_flights",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Search the latest AI news and do not search old quantum news",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Read https://example.com/b and do not open https://example.com/a",
            "url_read",
            ("url_read",),
        ),
    ],
)
def test_same_clause_denial_for_other_object_keeps_authorized_work(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Don't search the web for old docs; verify the latest official OpenAI announcement",
        "不要联网搜索旧文档；核验 OpenAI 最新官方公告",
        "Don't call url_read for old pages; verify the latest official OpenAI announcement",
        "不要调用 url_read 用于旧页面；核验 OpenAI 最新官方公告",
    ],
)
def test_scoped_web_or_url_denial_can_be_reauthorized_by_verified_request(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    "message",
    [
        "调用 mcp_unrelated_tool；但不要再调用 mcp_unrelated_tool",
        "Use mcp_unrelated_tool; then do not use the mcp_unrelated_tool",
    ],
)
def test_mcp_last_denial_syntax_variants_remove_alias(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            'Translate "OpenAI" and do a quick web search for context.',
            "fresh_web",
            ("web_search",),
        ),
        (
            'Translate "OpenAI" and browse online for context.',
            "fresh_web",
            ("web_search",),
        ),
        (
            "Summarize https://example.com/report and do a quick web search for updates",
            "verified_web",
            ("web_search", "url_read"),
        ),
    ],
)
def test_additional_explicit_web_action_variants_route_external_work(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_product_alias_scope_events_do_not_crash_real_route_resolution():
    route = _resolve(
        "Do not call search_flights for this request; call search_flights for Beijing to Shanghai tomorrow."
    )

    assert route.package_id == "flight"
    assert route.external_tool_names == ("search_flights",)


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not use the internet for this request.",
        "Do not use the internet for this task; search the latest AI news.",
    ],
)
def test_all_network_denial_for_current_request_is_not_treated_as_other_object_scope(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "本次请求不要联网；随后联网查询最新 AI 新闻。",
        "For this request, do not use the internet; then search the latest AI news.",
        "本次请求，请不要联网；随后联网查询最新 AI 新闻。",
        "For this request, please do not use the internet; then search the latest AI news.",
        "本次任务，麻烦务必不要联网；随后联网查询最新 AI 新闻。",
        "For this task, kindly do not access the internet; then search the latest AI news.",
        "请在本次请求中不要联网；随后联网查询最新 AI 新闻。",
        "Within this request, do not use the internet; then search the latest AI news.",
        "During this task, do not access the network; then search the latest AI news.",
        "Do not use the internet during this request; then search the latest AI news.",
        "Do not access the network within this task; then search the latest AI news.",
        "Do not use the internet in this request; then search the latest AI news.",
        "Never access the network in this conversation; later search the latest AI news.",
        "Avoid using the internet in this task; afterwards search the latest AI news.",
        "不要在本次请求期间联网；随后联网查询最新 AI 新闻。",
    ],
)
def test_current_request_scope_prefix_cannot_be_reauthorized_later(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not search it anymore.",
        "Search the latest AI news; do not look it up.",
        "搜索最新 AI 新闻；别再查这个话题。",
    ],
)
def test_final_web_denial_accepts_common_anaphoric_variants(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "I prefer an offline answer about the latest OpenAI news.",
        "My preference is an offline answer about the latest OpenAI news.",
        "Tell me the latest OpenAI news without connecting to the internet.",
        "Please keep this offline and tell me the latest OpenAI news.",
        "Please answer entirely offline about the latest OpenAI news.",
    ],
)
def test_additional_offline_constraints_block_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "总结 https://example.com/report，并搜索最新消息。",
        "Read https://example.com/report,and search the web for updates.",
    ],
)
def test_url_delimiters_do_not_swallow_independent_web_action(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by car with air conditioning?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Beijing to Shanghai by railway with air conditioning?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by not only train but also plane?",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train, no stops, or plane?",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train as opposed to plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, avoiding train and plane?",
            "mobility_route",
            ("route_compare",),
        ),
    ],
)
def test_route_mode_lexemes_and_negative_scope_do_not_over_authorize(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Find flights from Beijing to Shanghai tomorrow; do not call search_flights for this request.",
            "clarification_only",
            (),
        ),
        (
            "Call search_flights for tomorrow; do not call search_flights for old routes.",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_product_alias_scope_distinguishes_current_request_from_other_objects(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_mcp_alias_allows_the_tool_filler_between_action_and_alias():
    route = _resolve("Call the tool `mcp_unrelated_tool`.")

    assert route.package_id == "mcp_explicit"
    assert route.external_tool_names == ("mcp_unrelated_tool",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Never search the web for the latest OpenAI news",
            "clarification_only",
            (),
        ),
        (
            "Never open https://example.com/report and summarize it",
            "clarification_only",
            (),
        ),
        (
            "Do not verify the latest official OpenAI announcement",
            "clarification_only",
            (),
        ),
        (
            "Read https://example.com/report and do not cross-check it with official sources",
            "url_read",
            ("url_read",),
        ),
    ],
)
def test_network_and_verified_negative_actions_follow_ordered_permissions(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_browse_the_internet_is_an_explicit_web_action_for_transform():
    route = _resolve('Translate "OpenAI" and browse the internet for context')

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "Do not find flights from Beijing to Shanghai tomorrow",
        "Do not look for restaurants near People's Square",
        "Do not give directions from Beijing to Shanghai",
    ],
)
def test_negated_natural_product_actions_do_not_announce_product_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Call search_flights for old routes; do not call search_flights for old routes.",
            "clarification_only",
            (),
        ),
        (
            "How do I get from Beijing to Shanghai by car while monitoring air quality?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "Avoid giving directions from Beijing to Shanghai",
            "clarification_only",
            (),
        ),
        (
            "Without finding flights from Beijing to Shanghai tomorrow, explain the cities",
            "clarification_only",
            (),
        ),
        (
            "Do not fact check the latest official OpenAI announcement",
            "clarification_only",
            (),
        ),
        (
            "Read https://example.com/report and do not cross check it with official sources",
            "url_read",
            ("url_read",),
        ),
        (
            "Read https://example.com/report?lang=en&v=2",
            "url_read",
            ("url_read",),
        ),
    ],
)
def test_neighboring_scope_and_lexical_boundaries_remain_closed(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not use the internet for this answer.",
        "Search the latest AI news; do not use the internet for this response.",
        "Search the latest AI news; do not use the internet for this query.",
        "Search the latest AI news; do not use the internet for the entire request.",
        "Search the latest AI news; do not use the internet for any request.",
        "搜索最新 AI 新闻；不要联网用于本次回答。",
        "搜索最新 AI 新闻；不要联网针对这次查询。",
    ],
)
def test_current_request_network_scope_synonyms_remain_hard_denials(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "No flights from Beijing to Shanghai tomorrow.",
        "No trains from Beijing to Shanghai tomorrow.",
        "No need for flights from Beijing to Shanghai tomorrow.",
        "Exclude flights from Beijing to Shanghai tomorrow.",
        "避免查询北京到上海的航班。",
        "Skip finding flights from Beijing to Shanghai tomorrow.",
        "无需预订北京到上海的航班。",
        "不需要推荐从故宫到颐和园的路线。",
        "不必找人民广场附近咖啡店。",
        "无需比较北京到上海的航班和高铁。",
        "没必要预订北京到上海的航班。",
        "用不着找人民广场附近咖啡店。",
        "没有必要查询上海天气。",
        "不需要查询上海天气。",
    ],
)
def test_additional_natural_product_denials_do_not_announce_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "总结（https://example.com/report）并搜索最新消息。",
        "Read (https://example.com/report)and search the web for updates.",
        "Read https://example.com/report—then search the web for updates.",
    ],
)
def test_url_parenthesis_and_dash_delimiters_keep_following_web_action(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from Beijing airport to downtown by air-conditioned car?",
        "How do I get from Beijing airport to downtown by air conditioned car?",
    ],
)
def test_air_conditioned_car_does_not_authorize_flight_search(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai by train, excluding plane, and car?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, other than plane, and car?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, other than plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, excluding train and plane?",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "How do I get from Beijing to Shanghai by train or plane, with plane excluded?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by car, skipping train and plane?",
            "mobility_route",
            ("route_compare",),
        ),
    ],
)
def test_additional_route_mode_exclusion_grammar_is_ordered(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_mcp_alias_allows_mcp_tool_filler():
    route = _resolve("Call the MCP tool `mcp_unrelated_tool`.")

    assert route.package_id == "mcp_explicit"
    assert route.external_tool_names == ("mcp_unrelated_tool",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Don't use the internet. Actually, search the web for the latest OpenAI news",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Do not call web_search; then call web_search for OpenAI background",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Do not call url_read; then call url_read for https://example.com/report",
            "url_read",
            ("url_read",),
        ),
    ],
)
def test_later_explicit_network_tool_directive_can_reauthorize_generic_denial(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Avoid searching the web for the latest OpenAI news.",
        "Refrain from searching the web for the latest OpenAI news.",
        "Avoid opening https://example.com/report.",
        "Refrain from opening https://example.com/report.",
        "Avoid verifying the latest official OpenAI announcement.",
        "Don't consult official sources for the latest OpenAI announcement.",
    ],
)
def test_avoid_refrain_and_consult_denials_block_network_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Not only find flights from Beijing to Shanghai tomorrow, but also find trains.",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "Don't forget to find flights from Beijing to Shanghai tomorrow.",
            "flight",
            ("search_flights",),
        ),
        (
            "Do not just find flights from Beijing to Shanghai tomorrow; also compare trains.",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "Don't only look for restaurants near People's Square; also check weather.",
            "mixed_itinerary",
            ("weather_forecast", "local_place_search"),
        ),
    ],
)
def test_positive_expansion_idioms_are_not_mistaken_for_product_denial(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Please keep it offline and tell me the latest OpenAI news.",
        "Please stay completely offline and tell me the latest OpenAI news.",
        "Use local knowledge only; tell me the latest OpenAI news.",
    ],
)
def test_more_offline_constraints_block_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Read https://example.com/report and search within the page for references",
        "Read https://example.com/report and search the document for references",
    ],
)
def test_search_inside_provided_page_does_not_add_public_web_search(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


def test_browse_public_web_is_an_explicit_web_action_for_transform():
    route = _resolve('Translate "OpenAI" and browse the public web for context')

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "Don't use the internet for this; tell me the latest OpenAI news",
        "Don't use the internet for this one; tell me the latest OpenAI news",
        "Don't use the internet for my request; tell me the latest OpenAI news",
    ],
)
def test_pronoun_and_owner_current_scope_keep_network_denial_hard(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Read https://example.com/report and search this page for references",
        "Read https://example.com/report and search inside it for references",
        "读取 https://example.com/report 并搜索页面中的关键词",
    ],
)
def test_page_local_search_pronouns_and_chinese_do_not_add_public_web(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


@pytest.mark.parametrize(
    "message",
    [
        "Don't use official sources for the latest OpenAI announcement",
        "Avoid checking official sources for the latest OpenAI announcement",
    ],
)
def test_direct_official_source_denial_blocks_verified_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Not merely find flights from Beijing to Shanghai tomorrow; also compare trains.",
            "travel_air_rail",
            ("search_flights", "search_trains"),
        ),
        (
            "Don't simply look for restaurants near People's Square; also check weather.",
            "mixed_itinerary",
            ("weather_forecast", "local_place_search"),
        ),
        (
            "Do not fail to find flights from Beijing to Shanghai tomorrow.",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_more_positive_product_idioms_are_not_treated_as_denials(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from Beijing to Shanghai by car, excluding train, bus, and plane?",
        "How do I get from Beijing to Shanghai by car, with train and plane excluded?",
        "How do I get from Beijing to Shanghai by car, with plane and train excluded?",
    ],
)
def test_coordinated_mode_exclusion_lists_keep_only_allowed_route_mode(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)


@pytest.mark.parametrize(
    "message",
    [
        "Answer with local knowledge only: what is the latest OpenAI news?",
        "Rely only on local knowledge and tell me the latest OpenAI news",
    ],
)
def test_local_knowledge_only_word_order_variants_block_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not use the internet for the whole request.",
        "Search the latest AI news; do not use the internet for the full request.",
        "Search the latest AI news; do not use the internet for your answer.",
        "搜索最新 AI 新闻；不要联网用于此次回答。",
        "搜索最新 AI 新闻；不要联网针对本条消息。",
        "搜索最新 AI 新闻；不要联网关于整个请求。",
    ],
)
def test_more_current_scope_synonyms_keep_network_denial_hard(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_chinese_builtin_web_directive_reauthorizes_generic_network_denial():
    route = _resolve("不要联网；但调用 web_search 搜索最新 OpenAI 新闻。")

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "Read <https://example.com/report>and search the web for updates.",
        "Read https://example.com/report：并搜索最新消息。",
        "Read https://example.com/report:search the web for updates.",
    ],
)
def test_more_url_delimiters_keep_following_public_web_action(message):
    route = _resolve(message)

    assert route.package_id == "verified_web"
    assert route.external_tool_names == ("web_search", "url_read")


@pytest.mark.parametrize(
    "message",
    [
        "Read https://example.com/report?q=old,latest-news",
        "读取 https://example.com/report?q=旧，最新消息",
        "Read https://example.com/report?q=old;latest-news",
    ],
)
def test_freshness_words_inside_legal_url_query_do_not_add_public_web_search(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


@pytest.mark.parametrize(
    "message",
    [
        "How do I get from Beijing to Shanghai by car, ranked by air quality?",
        "How do I get from Beijing to Shanghai by car, chosen by air quality exposure?",
    ],
)
def test_air_quality_phrases_do_not_authorize_flight_search(message):
    route = _resolve(message)

    assert route.package_id == "mobility_route"
    assert route.external_tool_names == ("route_compare",)


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "How do I get from Beijing to Shanghai, excluding plane, by train and car?",
            "mixed_itinerary",
            ("route_compare", "search_trains"),
        ),
        (
            "How do I get from Beijing to Shanghai by train, with plane not allowed?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, with plane prohibited?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, except for plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, excluding any plane?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train, avoiding all flights?",
            "train",
            ("search_trains",),
        ),
        (
            "How do I get from Beijing to Shanghai by train or plane, with plane (excluded)?",
            "train",
            ("search_trains",),
        ),
    ],
)
def test_route_mode_exclusion_modifiers_and_postfixes_remove_denied_modes(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Exclude overnight flights from Beijing to Shanghai and show daytime options.",
            "flight",
            ("search_flights",),
        ),
        (
            "No nonstop flights from Beijing to Shanghai; show connecting options.",
            "flight",
            ("search_flights",),
        ),
        (
            "Skip overnight trains from Beijing to Shanghai; show daytime options.",
            "train",
            ("search_trains",),
        ),
    ],
)
def test_product_subset_filters_keep_required_product_capability(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "Use only local knowledge; what's the latest OpenAI announcement?",
        "Answer without web access; what's the latest OpenAI announcement?",
    ],
)
def test_more_offline_language_orderings_block_external_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Read https://example.com/report and search it for references.",
        "Read https://example.com/report and search its contents for references.",
    ],
)
def test_more_page_local_pronouns_do_not_add_public_web_search(message):
    route = _resolve(message)

    assert route.package_id == "url_read"
    assert route.external_tool_names == ("url_read",)


@pytest.mark.parametrize(
    "message",
    [
        "Use no official sources for the latest OpenAI announcement.",
        "Exclude official sources when checking the latest OpenAI announcement.",
    ],
)
def test_more_official_source_exclusions_block_verified_tools(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news without using web_search.",
        "Find flights from Beijing to Shanghai tomorrow; avoid calling search_flights.",
        "Call mcp_unrelated_tool, but avoid calling mcp_unrelated_tool.",
        "搜索最新 AI 新闻；禁止调用 web_search。",
        "查询明天北京到上海的航班；禁止调用 search_flights。",
        "调用 mcp_unrelated_tool；随后禁止调用 mcp_unrelated_tool。",
        "Search the latest AI news; skip using web_search.",
        "Find flights from Beijing to Shanghai tomorrow; do not execute search_flights.",
        "Call mcp_unrelated_tool, then skip using mcp_unrelated_tool.",
    ],
)
def test_avoid_and_without_explicit_tool_denials_are_hard(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "Search the latest AI news; do not call web_search for the latest AI news.",
        "Find flights from Beijing to Shanghai tomorrow; do not call search_flights for Beijing to Shanghai.",
        "搜索最新 AI 新闻；不要调用 web_search 用于最新 AI 新闻。",
        "查询明天北京到上海的航班；不要调用 search_flights 用于北京到上海。",
    ],
)
def test_scoped_tool_denial_for_same_object_cancels_prior_request(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "What is a primary source?",
        "What is an official source?",
        "What is a current price?",
    ],
)
def test_source_definition_questions_remain_direct(message):
    route = _resolve(message)

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "What is an official source for OpenAI's latest announcement, and what did it say?",
            "verified_web",
            ("web_search", "url_read"),
        ),
        (
            "What is a current price for Bitcoin right now, in USD?",
            "fresh_web",
            ("web_search",),
        ),
    ],
)
def test_real_time_queries_with_definition_prefix_still_use_external_capabilities(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools
    assert route.include_current_date is True


@pytest.mark.parametrize(
    "message",
    [
        "What is a primary source in historical research?",
        "What is an official source in journalism?",
        "What is a current price in finance?",
        "What is a primary source, in simple terms?",
    ],
)
def test_definition_questions_with_domain_or_explanation_qualifiers_remain_direct(message):
    route = _resolve(message)

    assert route.package_id == "direct"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "What is a current price in USD for Bitcoin right now?",
            "fresh_web",
            ("web_search",),
        ),
        (
            "What is an official source in journalism for OpenAI's latest announcement?",
            "verified_web",
            ("web_search", "url_read"),
        ),
    ],
)
def test_definition_domain_tail_cannot_absorb_a_following_external_query(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools
    assert route.include_current_date is True


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "Do not execute web_search; then execute web_search to search the latest AI news.",
            "fresh_web",
            ("web_search",),
        ),
        (
            "Do not execute url_read; then execute url_read to read https://example.com/report.",
            "url_read",
            ("url_read",),
        ),
        (
            "查询北京到上海的航班；禁止调用 search_flights；随后调用 search_flights 查询北京到上海的航班。",
            "flight",
            ("search_flights",),
        ),
    ],
)
def test_final_explicit_reauthorization_wins_after_hard_tool_denial(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        ("What is a primary source for beginners?", "direct", ()),
        (
            "What is a weather forecast in Shanghai this evening?",
            "weather",
            ("weather_forecast",),
        ),
    ],
)
def test_definition_qualifiers_distinguish_explanation_from_real_product_query(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


@pytest.mark.parametrize(
    "message",
    [
        "搜索最新 AI 新闻；禁止搜索。",
        "查询北京到上海的航班；禁止查询航班。",
        "查询北京到上海的航班；不要查询航班。",
    ],
)
def test_natural_language_hard_denial_cancels_prior_capability_request(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "不要搜索；随后搜索最新 AI 新闻。",
        "Do not search the web; afterwards search the web for the latest AI news.",
    ],
)
def test_final_natural_reauthorization_accepts_common_sequence_markers(message):
    route = _resolve(message)

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "What is breaking news in China?",
        "What is a current price in USD for Bitcoin?",
    ],
)
def test_definition_noun_type_cannot_absorb_news_or_asset_queries(message):
    route = _resolve(message)

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "搜索最新 AI 新闻；禁止联网。",
        "Search the latest AI news; never use the internet.",
        "搜索最新 AI 新闻；禁止访问网络。",
        "Search the latest AI news; never go online.",
        "不要联网搜索航班事故的最新新闻。",
        "不要联网搜索上海天气的最新新闻。",
        "不要联网搜索北京路线调整的最新新闻。",
        "不要联网搜索附近酒店的最新新闻。",
        "不要查询 OpenAI 最新官方公告。",
        "不要查询最新 AI 新闻。",
        "不要查询 OpenAI 现任 CEO。",
        "不要查询目前的美国总统。",
        "不要查询权威来源。",
        "不要查询一手来源。",
    ],
)
def test_global_network_hard_denials_cover_common_prohibition_language(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "查询北京到上海的航班: 禁止查询航班。",
        "Find flights from Beijing to Shanghai, do not find flights.",
        "Find flights from Beijing to Shanghai — do not find flights.",
        "Find trains from Beijing to Shanghai — do not find trains.",
        "查询北京到上海的航班 — 禁止查询航班。",
        "查询北京到上海的高铁 — 禁止查询高铁。",
        "Find flights then do not find flights.",
        "查询北京到上海的高铁随后不要查询高铁。",
    ],
)
def test_product_requests_do_not_cross_comma_or_colon_into_final_denial(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    ("message", "expected_package", "expected_tools"),
    [
        (
            "不要查询航班——随后查询北京到上海的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "Flights from Paris–Charles de Gaulle to New York tomorrow.",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询航班 — 随后查询北京到上海的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询高铁 — 随后查询北京到上海的高铁。",
            "train",
            ("search_trains",),
        ),
        (
            "不要查询航班随后查询北京到上海的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "Do not find flights——find flights from Beijing to Shanghai tomorrow.",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询航班：随后查询北京到上海的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询高铁:随后查询北京到上海的高铁。",
            "train",
            ("search_trains",),
        ),
        (
            "不要查询北京到上海的航班随后查询广州到深圳的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询北京到上海的航班价格随后查询广州到深圳的航班。",
            "flight",
            ("search_flights",),
        ),
        (
            "不要查询上海天气结果随后查询北京明天天气。",
            "weather",
            ("weather_forecast",),
        ),
        (
            "Do not check weather; then check weather in Shanghai tomorrow.",
            "weather",
            ("weather_forecast",),
        ),
        (
            "Book later flights tomorrow.",
            "flight",
            ("search_flights",),
        ),
        (
            "Show later trains tomorrow.",
            "train",
            ("search_trains",),
        ),
        (
            "Do not show route from City Hall to Downtown; then show route from City Hall to Downtown.",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "不要给出从故宫到颐和园的路线随后给出从故宫到颐和园的路线。",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "不要给出从故宫到颐和园的路线随后查询从故宫到颐和园的路线。",
            "mobility_route",
            ("route_compare",),
        ),
        (
            "Do not find flights then find flights from Beijing to Shanghai tomorrow.",
            "flight",
            ("search_flights",),
        ),
        (
            "Do not check weather then check weather in Shanghai tomorrow.",
            "weather",
            ("weather_forecast",),
        ),
        (
            "不要找咖啡店随后找人民广场附近的咖啡店。",
            "place_discovery",
            ("local_place_search",),
        ),
    ],
)
def test_dash_boundaries_preserve_final_reauthorization_and_entity_names(
    message,
    expected_package,
    expected_tools,
):
    route = _resolve(message)

    assert route.package_id == expected_package
    assert route.external_tool_names == expected_tools


def test_final_query_reauthorization_restores_web_search():
    route = _resolve("不要查询最新 AI 新闻；随后查询最新 AI 新闻。")

    assert route.package_id == "fresh_web"
    assert route.external_tool_names == ("web_search",)


@pytest.mark.parametrize(
    "message",
    [
        "查询明天上海天气；随后不要查询天气。",
        "Find coffee shops near Peoples Square; then do not find coffee shops.",
        "给出从故宫到颐和园的路线；随后不要给出从故宫到颐和园的路线。",
        "找人民广场附近的咖啡店随后不要找咖啡店。",
        (
            "Give me a route from Forbidden City to Summer Palace; "
            "then do not give me a route from Forbidden City to Summer Palace."
        ),
        "预订北京到上海的航班随后无需预订北京到上海的航班。",
        "找人民广场附近咖啡店随后不必找人民广场附近咖啡店。",
        "查询一下从故宫到颐和园的路线随后不需要规划从故宫到颐和园的路线。",
        "查询上海天气随后无需查询上海天气。",
        "预订北京到上海高铁随后不必预订北京到上海高铁。",
        "查询上海天气随后没必要查询上海天气。",
    ],
)
def test_all_product_capabilities_respect_final_natural_denial(message):
    route = _resolve(message)

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


@pytest.mark.parametrize(
    "message",
    [
        "不需要推荐机票但查询上海天气。",
        "不需要查询上海天气但查询北京天气。",
    ],
)
def test_chinese_product_contrast_boundary_preserves_final_positive_request(message):
    route = _resolve(message)

    assert route.package_id == "weather"
    assert route.external_tool_names == ("weather_forecast",)


def test_exact_authorized_mcp_alias_can_select_only_that_tool():
    route = _resolve(
        "请调用 mcp_unrelated_tool 处理这份数据",
        available_tool_names=["mcp_unrelated_tool", "web_search", "url_read"],
    )

    assert route.package_id == "mcp_explicit"
    assert route.external_tool_names == ("mcp_unrelated_tool",)
    assert route.effective_plan_mode == "off"
    assert route.reason_codes == ("explicit_authorized_tool_alias",)


def test_non_mcp_tool_name_cannot_be_treated_as_an_authorized_mcp_alias():
    route = _resolve(
        "请调用 unrelated_tool 处理这份数据",
        available_tool_names=["unrelated_tool", "web_search", "url_read"],
    )

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_unmentioned_authorized_mcp_alias_is_not_exposed():
    route = _resolve(
        "帮我处理这份数据",
        available_tool_names=["mcp_unrelated_tool", "web_search", "url_read"],
    )

    assert route.package_id == "clarification_only"
    assert route.external_tool_names == ()


def test_serialization_only_contains_safe_protocol_fields():
    route = _resolve("我现在在北京，我想去上海，你可以帮我吗")

    payload = serialize_capability_resolution(route)

    assert payload == {
        "schema_version": 1,
        "router_version": "2026-08-27.2",
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
