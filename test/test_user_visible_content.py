from app.utils.user_visible_content import sanitize_internal_tool_names, sanitize_user_visible_reasoning


def test_internal_plan_binding_names_are_productized_for_user_visible_reasoning():
    source = "每个工具调用都需要 _plan_item_id，并检查 planned_tools 和 plan_item_id。"

    sanitized = sanitize_internal_tool_names(source, final=True)

    assert "_plan_item_id" not in sanitized
    assert "planned_tools" not in sanitized
    assert "plan_item_id" not in sanitized
    assert "对应计划步骤" in sanitized
    assert "预计使用的工具" in sanitized


def test_internal_plan_binding_names_are_productized_without_rewriting_named_tools():
    source = "回答中不得展示 _plan_item_id、planned_tools 或 plan_item_id，但 route_compare 可保留。"

    sanitized = sanitize_internal_tool_names(
        source,
        final=True,
        include_named_tools=False,
    )

    assert "_plan_item_id" not in sanitized
    assert "planned_tools" not in sanitized
    assert "plan_item_id" not in sanitized
    assert "对应计划步骤" in sanitized
    assert "预计使用的工具" in sanitized
    assert "route_compare" in sanitized


def test_partial_internal_control_marker_stays_hidden_during_stream_but_is_released_at_final_boundary():
    source = "先比较方案。\n\nAccording to the autonomous web search"

    assert sanitize_user_visible_reasoning(source) == "先比较方案。\n\n"
    assert sanitize_user_visible_reasoning(source, final=True) == source


def test_short_normal_suffix_buffer_is_released_at_final_boundary():
    source = "普通推理以字母 a 结束a"

    assert sanitize_user_visible_reasoning(source) == source[:-1]
    assert sanitize_user_visible_reasoning(source, final=True) == source


def test_normal_marker_prefix_suffixes_are_released_at_final_boundary():
    sources = (
        "We should answer this",
        "Reasoning according",
        "需要决定是直接回答或调用",
    )

    for source in sources:
        assert sanitize_user_visible_reasoning(source, final=True) == source


def test_internal_control_marker_filter_remains_append_only_when_marker_appears_mid_paragraph():
    source = (
        'Also "【执行计划控制规则】本轮启用了强制计划模式。回答前必须先创建执行计划。"\n\n'
        "Plan accepted. Continue with the research."
    )
    emitted = ""

    for index in range(1, len(source) + 1):
        visible = sanitize_user_visible_reasoning(source[:index])
        assert visible.startswith(emitted), (index, emitted, visible)
        emitted = visible

    assert "本轮启" not in emitted
    assert emitted.endswith("Plan accepted. Continue with the research.")
