"""Agent Loop 内部计划控制调用的解析、门禁与安全回执。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.logger import app_logger as logger
from app.services.agent.plan_coordinator import PlanCoordinator

UPDATE_PLAN_TOOL_NAME = "update_plan"
PLAN_ITEM_ARGUMENT_NAME = "_plan_item_id"


@dataclass(frozen=True)
class PlanControlResult:
    external_tool_calls: list[dict]
    tool_responses: dict[str, str] = field(default_factory=dict)
    plan_item_ids: dict[str, str] = field(default_factory=dict)
    repair_exhausted: bool = False


def _parse_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _response(*, status: str, reason: str, revision: int, hint: str | None = None) -> str:
    payload = {
        "status": status,
        "reason": reason,
        "revision": revision,
    }
    if hint:
        payload["hint"] = hint
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _control_rejection_hint(reason: str, coordinator: PlanCoordinator) -> str | None:
    if reason == "missing_required_initial_tool_coverage":
        requirements = "、".join(
            f"{tool_name}×{count}" for tool_name, count in coordinator.required_initial_tool_counts.items()
        )
        return (
            f"重新提交计划，并为所需工具分别保留独立步骤：{requirements}。"
            "这些步骤必须处于 pending 或 in_progress，planned_tools 只声明该步骤实际使用的工具；"
            "同一个 url_read 步骤可以连续读取多个不同来源。"
        )
    if reason == "invalid_plan_structure":
        return (
            "重新提交 2 至 6 个步骤；每项包含稳定 id、非空 step、pending 或 in_progress 状态，以及 planned_tools 数组。"
        )
    if reason == "multiple_running_items":
        return "同一版计划最多只能有一个 in_progress 步骤，其余未完成步骤使用 pending。"
    if reason in {"unknown_dependency", "self_dependency", "dependency_cycle"}:
        return "depends_on 只能引用本次计划中已经声明的其他步骤 ID，并且不能形成循环依赖。"
    return None


def _required_tool_coverage_summary(
    items: list[Any] | None,
    coordinator: PlanCoordinator,
) -> dict[str, int]:
    """只统计服务端声明的工具名，避免把未校验模型内容写入日志。"""

    counts = {tool_name: 0 for tool_name in coordinator.required_initial_tool_counts}
    if not items:
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        planned_tools = item.get("planned_tools")
        if not isinstance(planned_tools, list):
            continue
        declared_tools = {tool_name for tool_name in planned_tools if isinstance(tool_name, str)}
        for tool_name in counts:
            other_required_tools = set(counts) - {tool_name}
            if tool_name in declared_tools and not other_required_tools.intersection(declared_tools):
                counts[tool_name] += 1
    return counts


def _extract_plan_item_binding(call: dict) -> tuple[dict, str | None]:
    """提取并移除只供 Fusion 使用的计划项 ID，避免污染真实工具参数。"""

    raw_arguments = call.get("arguments")
    arguments = _parse_arguments(raw_arguments)
    if arguments is None or PLAN_ITEM_ARGUMENT_NAME not in arguments:
        return call, None
    requested_item_id = arguments.pop(PLAN_ITEM_ARGUMENT_NAME)
    cleaned = dict(call)
    cleaned["arguments"] = (
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if isinstance(raw_arguments, str)
        else arguments
    )
    return cleaned, requested_item_id if isinstance(requested_item_id, str) else ""


async def process_plan_control_calls(
    *,
    tool_calls: list[dict],
    coordinator: PlanCoordinator,
    emitter: Any,
) -> PlanControlResult:
    """先应用控制调用，再决定同轮外部调用；回执只含安全状态码。"""

    control_calls = [call for call in tool_calls if call.get("name") == UPDATE_PLAN_TOOL_NAME]
    external_calls = [call for call in tool_calls if call.get("name") != UPDATE_PLAN_TOOL_NAME]
    responses: dict[str, str] = {}
    accepted_control = False
    repairable_rejection = False

    for call in control_calls:
        call_id = str(call.get("id", ""))
        payload = _parse_arguments(call.get("arguments"))
        result = coordinator.apply_model_update(payload)
        if not result.accepted:
            items = payload.get("plan") if isinstance(payload, dict) else None
            if not isinstance(items, list) and isinstance(payload, dict):
                items = payload.get("items")
            logger.info(
                "计划控制更新被拒绝: run_id=%s reason=%s item_count=%s required_tool_coverage=%s",
                coordinator.run_id,
                result.reason,
                min(len(items), 7) if isinstance(items, list) else None,
                _required_tool_coverage_summary(
                    items if isinstance(items, list) else None,
                    coordinator,
                ),
            )
        responses[call_id] = _response(
            status="accepted" if result.accepted else "rejected",
            reason=result.reason,
            revision=coordinator.revision,
            hint=None if result.accepted else _control_rejection_hint(result.reason, coordinator),
        )
        accepted_control = accepted_control or result.accepted
        repairable_rejection = repairable_rejection or (
            not result.accepted
            and result.reason != "plan_mode_off"
            and not (result.reason == "control_update_limit_reached" and coordinator.has_valid_model_plan)
        )
        if result.accepted and result.snapshot is not None:
            await emitter.plan_snapshot(**result.snapshot)

    round_failed = repairable_rejection and not accepted_control
    if coordinator.mode == "on" and external_calls and not coordinator.has_valid_model_plan:
        round_failed = True
        for call in external_calls:
            responses[str(call.get("id", ""))] = _response(
                status="not_executed",
                reason="plan_required",
                revision=coordinator.revision,
            )
        external_calls = []

    prepared_external_calls: list[dict] = []
    requested_item_ids: list[str | None] = []
    for call in external_calls:
        prepared_call, requested_item_id = _extract_plan_item_binding(call)
        prepared_external_calls.append(prepared_call)
        requested_item_ids.append(requested_item_id)
    external_calls = prepared_external_calls

    plan_item_ids: dict[str, str] = {}
    mapped_item_ids = coordinator.plan_item_ids_for_tools(
        [str(call.get("name", "")) for call in external_calls],
        requested_item_ids=requested_item_ids,
    )
    executable_external_calls: list[dict] = []
    for call, plan_item_id, requested_item_id in zip(
        external_calls,
        mapped_item_ids,
        requested_item_ids,
    ):
        missing_required_binding = (
            coordinator.mode == "on" and coordinator.has_valid_model_plan and requested_item_id is None
        )
        if plan_item_id is not None and not missing_required_binding:
            plan_item_ids[str(call.get("id", ""))] = plan_item_id
            executable_external_calls.append(call)
            continue
        invalid_explicit_binding = requested_item_id is not None and coordinator.has_valid_model_plan
        if (
            missing_required_binding
            or invalid_explicit_binding
            or (coordinator.mode == "on" and coordinator.has_valid_model_plan)
        ):
            round_failed = True
            responses[str(call.get("id", ""))] = _response(
                status="not_executed",
                reason="plan_item_required",
                revision=coordinator.revision,
                hint=(
                    "修订计划后重试：_plan_item_id 必须等于本次调用所属未完成步骤的精确 id，"
                    "且该步骤 planned_tools 必须包含本次真实工具名称。"
                ),
            )
            continue
        executable_external_calls.append(call)
    external_calls = executable_external_calls

    repair_result = coordinator.record_repair_round_with_fallback() if round_failed else None
    if repair_result is not None and repair_result.fallback is not None:
        fallback = repair_result.fallback
        if fallback.snapshot is not None:
            await emitter.plan_snapshot(**fallback.snapshot)

    return PlanControlResult(
        external_tool_calls=[
            {
                **call,
                **(
                    {"plan_item_id": plan_item_ids[str(call.get("id", ""))]}
                    if str(call.get("id", "")) in plan_item_ids
                    else {}
                ),
            }
            for call in external_calls
        ],
        tool_responses=responses,
        plan_item_ids=plan_item_ids,
        repair_exhausted=repair_result.exhausted if repair_result is not None else False,
    )
