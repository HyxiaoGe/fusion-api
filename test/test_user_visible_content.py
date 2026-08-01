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


def test_partial_internal_control_marker_stays_hidden_during_stream_and_at_final_boundary():
    source = "先比较方案。\n\nAccording to the autonomous web search"

    assert sanitize_user_visible_reasoning(source) == "先比较方案。\n\n"
    assert sanitize_user_visible_reasoning(source, final=True) == "先比较方案。\n\n"
