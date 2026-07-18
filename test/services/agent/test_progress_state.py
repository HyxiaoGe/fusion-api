from app.services.agent.progress_state import apply_progress_event, empty_progress_state


def test_run_progress_updated_replaces_progress():
    state = empty_progress_state(run_id="r1", message_id="m1")

    state = apply_progress_event(
        state,
        {
            "type": "run_progress_updated",
            "protocol_version": 2,
            "phase": "researching",
            "label": "正在搜索相关资料",
            "completed_steps": 1,
            "total_steps": 4,
            "completed_tool_calls": 2,
            "max_tool_calls": 20,
        },
    )

    assert state["progress"] == {
        "phase": "researching",
        "label": "正在搜索相关资料",
        "completed_steps": 1,
        "total_steps": 4,
        "completed_tool_calls": 2,
        "max_tool_calls": 20,
    }


def test_context_required_and_result_are_replayable_without_location_payload():
    state = empty_progress_state(run_id="r1", message_id="m1")
    state = apply_progress_event(
        state,
        {
            "type": "context_required",
            "protocol_version": 2,
            "context_type": "geolocation",
            "request_id": "ctx-1",
            "purpose": "nearby_search",
            "reason": "搜索当前位置附近的地点",
            "expires_at": 123.5,
        },
    )

    assert state["context_request"] == {
        "request_id": "ctx-1",
        "context_type": "geolocation",
        "purpose": "nearby_search",
        "reason": "搜索当前位置附近的地点",
        "expires_at": 123.5,
        "status": "pending",
    }
    assert "latitude" not in str(state)

    state = apply_progress_event(
        state,
        {
            "type": "context_result",
            "protocol_version": 2,
            "context_type": "geolocation",
            "request_id": "ctx-1",
            "status": "provided",
        },
    )
    assert state["context_request"]["status"] == "provided"


def test_plan_snapshot_replaces_existing_plan():
    state = empty_progress_state(run_id="r1", message_id="m1")
    state = apply_progress_event(
        state,
        {
            "type": "plan_snapshot",
            "protocol_version": 2,
            "plan_id": "plan-r1",
            "revision": 1,
            "items": [
                {
                    "id": "understand",
                    "title": "理解问题",
                    "status": "running",
                    "kind": "reasoning",
                    "tool_names": [],
                    "evidence_item_ids": [],
                }
            ],
        },
    )
    state = apply_progress_event(
        state,
        {
            "type": "plan_snapshot",
            "protocol_version": 2,
            "plan_id": "plan-r1",
            "revision": 2,
            "items": [
                {
                    "id": "answer",
                    "title": "整理回答",
                    "status": "pending",
                    "kind": "answer",
                    "tool_names": [],
                    "evidence_item_ids": [],
                }
            ],
        },
    )

    assert state["plan"]["revision"] == 2
    assert [item["id"] for item in state["plan"]["items"]] == ["answer"]


def test_plan_step_update_ignores_stale_revision():
    state = empty_progress_state(run_id="r1", message_id="m1")
    state = apply_progress_event(
        state,
        {"type": "plan_snapshot", "protocol_version": 2, "plan_id": "plan-r1", "revision": 2, "items": []},
    )

    state = apply_progress_event(
        state,
        {
            "type": "plan_step_updated",
            "protocol_version": 2,
            "plan_id": "plan-r1",
            "revision": 2,
            "item": {
                "id": "search",
                "title": "搜索资料",
                "status": "running",
                "kind": "search",
                "tool_names": [],
                "evidence_item_ids": [],
            },
        },
    )

    assert state["plan"]["items"] == []


def test_tool_digest_upserts_and_caps_to_twenty_items():
    state = empty_progress_state(run_id="r1", message_id="m1")

    for index in range(22):
        state = apply_progress_event(
            state,
            {
                "type": "tool_result_digest",
                "protocol_version": 2,
                "tool_call_id": f"tc-{index}",
                "tool_name": "web_search",
                "status": "success",
                "title": f"工具结果 {index}",
                "summary": "摘要",
                "key_findings": [f"发现 {index}"],
                "source_refs": [],
                "truncated": False,
            },
        )

    assert len(state["tool_digests"]) == 20
    assert state["tool_digests"][0]["tool_call_id"] == "tc-2"


def test_evidence_upsert_cap_keeps_used_and_truncates_fields():
    state = empty_progress_state(run_id="r1", message_id="m1")

    for index in range(14):
        state = apply_progress_event(
            state,
            {
                "type": "evidence_item_upserted",
                "protocol_version": 2,
                "evidence": {
                    "id": f"ev-{index}",
                    "kind": "web",
                    "status": "used" if index == 0 else "candidate",
                    "title": "t" * 100,
                    "domain": "example.com",
                    "claim": "c" * 200,
                    "snippet": "s" * 300,
                    "used_by_final_answer": index == 0,
                },
            },
        )

    ids = [item["id"] for item in state["evidence"]]
    assert len(ids) == 12
    assert "ev-0" in ids
    kept_used = next(item for item in state["evidence"] if item["id"] == "ev-0")
    assert kept_used["title"] == "t" * 80
    assert kept_used["claim"] == "c" * 120
    assert kept_used["snippet"] == "s" * 180


def test_evidence_upsert_cap_keeps_selected_and_read_success():
    state = empty_progress_state(run_id="r1", message_id="m1")

    for evidence in [
        {
            "id": "selected-id",
            "kind": "web",
            "status": "selected",
            "title": "建议深读来源",
            "url": "https://example.com/selected",
            "domain": "example.com",
            "claim": "建议深读：官方来源",
        },
        {
            "id": "read-success-id",
            "kind": "web",
            "status": "read_success",
            "title": "已深读来源",
            "url": "https://example.com/read",
            "domain": "example.com",
            "claim": "已读取网页内容。",
        },
    ]:
        state = apply_progress_event(
            state,
            {
                "type": "evidence_item_upserted",
                "protocol_version": 2,
                "evidence": evidence,
            },
        )

    for index in range(12):
        state = apply_progress_event(
            state,
            {
                "type": "evidence_item_upserted",
                "protocol_version": 2,
                "evidence": {
                    "id": f"candidate-{index}",
                    "kind": "web",
                    "status": "candidate",
                    "title": f"普通候选 {index}",
                    "url": f"https://example.com/candidate-{index}",
                    "domain": "example.com",
                    "claim": "普通候选",
                },
            },
        )

    ids = [item["id"] for item in state["evidence"]]
    assert len(ids) == 12
    assert "selected-id" in ids
    assert "read-success-id" in ids


def test_terminal_v1_events_update_snapshot_status():
    state = empty_progress_state(run_id="r1", message_id="m1")

    state = apply_progress_event(state, {"type": "run_failed", "message": "boom"})

    assert state["status"] == "failed"
