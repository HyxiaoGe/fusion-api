from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_plan_builder import build_long_task_plan_items


def test_build_long_task_plan_items_includes_focus_tools_and_budget_for_network_task():
    items = build_long_task_plan_items(
        original_message="请帮我查一下 2026 年暑期旅游哪里最火，顺便给出适合亲子游的推荐",
        tools=["web_search"],
        limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
    )

    assert [item["id"] for item in items] == ["understand", "search", "read", "answer"]
    assert items[0] == {
        "id": "understand",
        "title": "制定执行计划",
        "status": "running",
        "kind": "reasoning",
        "summary": "围绕「2026 年暑期旅游哪里最火，顺便给出适合亲子游的推荐」判断资料需求和回答路径",
        "tool_names": [],
        "evidence_item_ids": [],
    }
    assert items[1] == {
        "id": "search",
        "title": "搜索：2026 年暑期旅游哪里最火，顺便给出适合亲子游的推荐",
        "status": "pending",
        "kind": "search",
        "summary": "工具：联网搜索；预算：最多 4 次搜索，每次 3-10 条结果",
        "tool_names": ["web_search"],
        "evidence_item_ids": [],
    }
    assert items[2] == {
        "id": "read",
        "title": "筛选关键来源",
        "status": "pending",
        "kind": "read",
        "summary": "必要时读取网页核验；预算：最多 5 个网页",
        "tool_names": ["web_search"],
        "evidence_item_ids": [],
    }
    assert items[3] == {
        "id": "answer",
        "title": "整理回答",
        "status": "pending",
        "kind": "answer",
        "summary": "基于可用依据给出结论、推荐和不确定性",
        "tool_names": [],
        "evidence_item_ids": [],
    }


def test_build_long_task_plan_items_uses_direct_answer_plan_without_tools():
    items = build_long_task_plan_items(
        original_message="帮我把这句话润色得更自然",
        tools=[],
        limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
    )

    assert [item["id"] for item in items] == ["understand", "answer"]
    assert items[0]["title"] == "制定执行计划"
    assert items[0]["summary"] == "确认「帮我把这句话润色得更自然」的目标和回答结构"
    assert items[1] == {
        "id": "answer",
        "title": "整理回答",
        "status": "pending",
        "kind": "answer",
        "summary": "基于已有上下文直接回答，不使用联网工具",
        "tool_names": [],
        "evidence_item_ids": [],
    }
