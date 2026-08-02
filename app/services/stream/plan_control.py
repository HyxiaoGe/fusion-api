"""Agent Loop 内部计划控制调用的解析、门禁与安全回执。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.logger import app_logger as logger
from app.services.agent.plan_coordinator import PlanCoordinator

UPDATE_PLAN_TOOL_NAME = "update_plan"
PLAN_ITEM_ARGUMENT_NAME = "_plan_item_id"
_STATUS_DRIFT_REPAIR_REASONS = frozenset(
    {
        "multiple_running_items",
        "terminal_status_regression",
    }
)


@dataclass(frozen=True)
class PlanControlResult:
    external_tool_calls: list[dict]
    tool_responses: dict[str, str] = field(default_factory=dict)
    plan_item_ids: dict[str, str] = field(default_factory=dict)
    repair_exhausted: bool = False
    repair_attempt_count: int = 0
    repair_attempt_limit: int = 0


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


def _response(
    *,
    status: str,
    reason: str,
    revision: int,
    hint: str | None = None,
    canonical_plan: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "status": status,
        "reason": reason,
        "revision": revision,
    }
    if hint:
        payload["hint"] = hint
    if canonical_plan:
        payload["canonical_plan"] = canonical_plan
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _control_rejection_hint(reason: str, coordinator: PlanCoordinator) -> str | None:
    if reason in {"attempted_item_removed", "terminal_item_removed"}:
        return (
            "保留 canonical_plan 中全部既有步骤 ID；直接复用其中的 step、kind、depends_on 和 "
            "planned_tools，并将提交状态统一写为 pending；真实状态由系统保留。"
        )
    if reason == "missing_required_initial_tool_coverage":
        requirements = "、".join(
            f"{tool_name}×{count}" for tool_name, count in coordinator.required_initial_tool_counts.items()
        )
        return (
            f"重新提交计划，并为所需工具分别保留独立步骤：{requirements}。"
            "这些步骤必须处于 pending，planned_tools 只声明该步骤实际使用的一个工具；"
            "每个 url_read 计划项负责一个独立来源任务，多个来源必须拆成多个计划项；"
            "仅当服务端将同一任务保持为 retryable/running 时，才可跨轮重试该计划项。"
        )
    if reason == "research_read_missing_search_dependency":
        return (
            "重新提交计划：每个 url_read 步骤都必须通过 depends_on 直接或间接依赖至少一个 "
            "web_search 步骤，不能把 url_read 放在首批可执行步骤中；最终回答需依赖全部读取步骤。"
        )
    if reason == "invalid_plan_structure":
        return "重新提交 2 至 6 个步骤；每项包含稳定 id、非空 step、pending 状态，以及 planned_tools 数组。"
    if reason == "multiple_tools_per_item":
        return "每个执行步骤的 planned_tools 最多声明一个真实工具；需要多个工具时拆成有依赖关系的独立步骤。"
    if reason == "unannounced_planned_tool":
        return "planned_tools 只能使用当前请求实际提供的工具名称；删除不可用工具后重新提交计划。"
    if reason == "missing_answer_phase":
        return (
            "重新提交计划，并增加一个 kind 为 answer 或 synthesis 的最终步骤；"
            "该步骤的 planned_tools 使用空数组，并依赖需要先完成的查询或核验步骤。"
        )
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
    repair_reasons: set[str] = set()

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
            canonical_plan=(
                coordinator.canonical_plan_for_model()
                if not result.accepted and coordinator.has_valid_model_plan
                else None
            ),
        )
        accepted_control = accepted_control or result.accepted
        is_repairable_rejection = (
            not result.accepted
            and result.reason != "plan_mode_off"
            and not (result.reason == "control_update_limit_reached" and coordinator.has_valid_model_plan)
        )
        repairable_rejection = repairable_rejection or is_repairable_rejection
        if is_repairable_rejection:
            repair_reasons.add(result.reason)
        if result.accepted and result.snapshot is not None:
            await emitter.plan_snapshot(**result.snapshot)

    round_failed = repairable_rejection and not accepted_control
    if coordinator.mode == "on" and external_calls and not coordinator.has_valid_model_plan:
        round_failed = True
        repair_reasons.add("plan_required")
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
    # 只阻止同一模型批次把不同调用复用到同一任务；每轮重新建立，
    # 不影响服务端保持 retryable/running 的同一任务跨轮有界重试。
    round_bound_item_ids: set[str] = set()
    for call, plan_item_id, requested_item_id in zip(
        external_calls,
        mapped_item_ids,
        requested_item_ids,
    ):
        missing_required_binding = (
            coordinator.mode == "on" and coordinator.has_valid_model_plan and requested_item_id is None
        )
        if plan_item_id is not None and not missing_required_binding:
            if plan_item_id in round_bound_item_ids:
                round_failed = True
                responses[str(call.get("id", ""))] = _response(
                    status="not_executed",
                    reason="plan_item_already_bound",
                    revision=coordinator.revision,
                    hint=(
                        "同一批次的不同工具调用不能复用同一计划项；请为每个独立工具任务创建不同步骤，"
                        "并分别提交对应的 _plan_item_id。服务端标记为 retryable/running 的同一任务"
                        "可在后续轮次使用原计划项重试。"
                    ),
                )
                continue
            round_bound_item_ids.add(plan_item_id)
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
            repair_reasons.add("plan_item_required")
            responses[str(call.get("id", ""))] = _response(
                status="not_executed",
                reason="plan_item_required",
                revision=coordinator.revision,
                hint=(
                    "修订计划后重试：_plan_item_id 必须等于本次调用所属未完成步骤的精确 id，"
                    "该步骤 planned_tools 必须只包含本次真实工具名称，且所有前置步骤已经完成；"
                    "同一批次的不同调用不能复用同一计划项，只有服务端保持为 retryable/running 的"
                    "同一任务可在后续轮次复用原计划项。"
                ),
            )
            continue
        executable_external_calls.append(call)
    external_calls = executable_external_calls

    tolerate_status_drift = bool(repair_reasons) and repair_reasons.issubset(
        _STATUS_DRIFT_REPAIR_REASONS
    )
    repair_result = (
        coordinator.record_repair_round_with_fallback(
            tolerate_status_drift=tolerate_status_drift,
        )
        if round_failed
        else None
    )
    if repair_result is not None:
        logger.info(
            "计划修复轮次已记录: run_id=%s attempt=%s limit=%s threshold_reached=%s "
            "fallback_adopted=%s has_valid_plan=%s",
            coordinator.run_id,
            repair_result.attempt_count,
            repair_result.attempt_limit,
            repair_result.attempt_count >= repair_result.attempt_limit,
            repair_result.fallback is not None,
            coordinator.has_valid_model_plan,
        )
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
        repair_attempt_count=repair_result.attempt_count if repair_result is not None else 0,
        repair_attempt_limit=repair_result.attempt_limit if repair_result is not None else 0,
    )
