"""Agent Plan Mode 的单一计划状态所有者。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PlanMode = Literal["auto", "on", "off"]
PlanSource = Literal["model", "observed"]
PlanStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
PlanKind = Literal["reasoning", "search", "read", "synthesis", "answer", "other"]

_PLAN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_SYSTEM_FALLBACK_REASON = "system_fallback"
_INITIAL_PLAN_REPAIR_ATTEMPT_LIMIT = 3
_ACTIVE_PLAN_REPAIR_ATTEMPT_LIMIT = 5


class ModelPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1, max_length=120)
    status: PlanStatus
    kind: PlanKind
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    planned_tools: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _PLAN_ID_RE.fullmatch(value):
            raise ValueError("invalid_plan_item_id")
        return value

    @field_validator("depends_on", "planned_tools")
    @classmethod
    def validate_string_list(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 128 for value in values):
            raise ValueError("invalid_string_list")
        return list(dict.fromkeys(values))


class ModelPlanUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=240)
    items: list[ModelPlanItem] = Field(min_length=2, max_length=6)


@dataclass(frozen=True)
class PlanUpdateResult:
    accepted: bool
    reason: str
    snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class PlanRepairResult:
    exhausted: bool
    fallback: PlanUpdateResult | None = None
    attempt_count: int = 0
    attempt_limit: int = 0


@dataclass
class PlanCoordinator:
    """集中管理模型计划 revision；观察型旧计划只能在模型计划出现前兼容写入。"""

    run_id: str
    mode: PlanMode = "auto"
    max_valid_updates: int = 6
    revision: int = 0
    source: PlanSource = "observed"
    reason: str = "legacy_observed"
    items: list[dict[str, Any]] = field(default_factory=list)
    valid_update_count: int = 0
    repair_attempt_count: int = 0
    required_initial_tool_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_valid_model_plan(self) -> bool:
        return (
            (self.source == "model" or (self.source == "observed" and self.reason == _SYSTEM_FALLBACK_REASON))
            and self.revision > 0
            and bool(self.items)
        )

    def apply_model_update(self, payload: Any) -> PlanUpdateResult:
        if self.mode == "off":
            return PlanUpdateResult(False, "plan_mode_off")
        if self.valid_update_count >= min(6, self.max_valid_updates):
            return PlanUpdateResult(False, "control_update_limit_reached")
        payload = _normalize_model_plan_payload(payload, previous_items=self.items)
        try:
            update = ModelPlanUpdate.model_validate(payload)
        except ValidationError:
            return self._reject_repair("invalid_plan_structure")

        item_ids = [item.id for item in update.items]
        if len(item_ids) != len(set(item_ids)):
            return self._reject_repair("duplicate_item_id")
        known_ids = set(item_ids)
        if sum(item.status == "running" for item in update.items) > 1:
            return self._reject_repair("multiple_running_items")
        for item in update.items:
            if item.id in item.depends_on:
                return self._reject_repair("self_dependency")
            if any(dependency not in known_ids for dependency in item.depends_on):
                return self._reject_repair("unknown_dependency")
        dependencies = {item.id: set(item.depends_on) for item in update.items}
        if _has_dependency_cycle(dependencies):
            return self._reject_repair("dependency_cycle")
        if not self.has_valid_model_plan and self.required_initial_tool_counts:
            required_tool_names = set(self.required_initial_tool_counts)
            planned_counts = {
                tool_name: sum(
                    tool_name in item.planned_tools
                    and not (required_tool_names - {tool_name}).intersection(item.planned_tools)
                    for item in update.items
                )
                for tool_name in self.required_initial_tool_counts
            }
            if any(
                planned_counts.get(tool_name, 0) < required_count
                for tool_name, required_count in self.required_initial_tool_counts.items()
            ):
                return self._reject_repair("missing_required_initial_tool_coverage")
        if self.has_valid_model_plan:
            previous_status = {str(item.get("id")): item.get("status") for item in self.items}
            previous_items_by_id = {str(item.get("id")): item for item in self.items}
            previous_terminal_ids = {
                item_id
                for item_id, status in previous_status.items()
                if status in {"completed", "failed", "skipped", "blocked"}
            }
            if previous_terminal_ids - known_ids:
                return self._reject_repair("terminal_item_removed")
            for item in update.items:
                previous_item = previous_items_by_id.get(item.id)
                if previous_status.get(item.id) in {"completed", "failed", "skipped", "blocked"} and previous_item:
                    locked_fields = ("title", "kind", "depends_on", "planned_tools")
                    if any(getattr(item, field) != previous_item.get(field) for field in locked_fields):
                        return self._reject_repair("terminal_item_mutated")
                if (
                    item.status in {"completed", "failed", "skipped", "blocked"}
                    and previous_status.get(item.id) != item.status
                ):
                    return self._reject_repair("unproven_terminal_status")
                if previous_status.get(item.id) in {"completed", "failed", "skipped", "blocked"} and item.status in {
                    "pending",
                    "running",
                }:
                    return self._reject_repair("terminal_status_regression")
        elif any(item.status in {"completed", "failed", "skipped", "blocked"} for item in update.items):
            return self._reject_repair("unproven_terminal_status")

        self.revision += 1
        self.valid_update_count += 1
        self.source = "model"
        self.reason = "model_update"
        self.items = [item.model_dump() for item in update.items]
        self.reset_repair_attempts()
        return PlanUpdateResult(True, "model_update", self.snapshot())

    def _reject_repair(self, reason: str) -> PlanUpdateResult:
        return PlanUpdateResult(False, reason)

    def repair_attempt_limit(self, *, tolerate_status_drift: bool = False) -> int:
        """仅对已有计划的并发/终态漂移放宽阈值，其他错误保持原门禁。"""

        if self.has_valid_model_plan and tolerate_status_drift:
            return _ACTIVE_PLAN_REPAIR_ATTEMPT_LIMIT
        return _INITIAL_PLAN_REPAIR_ATTEMPT_LIMIT

    def record_repair_round(self, *, tolerate_status_drift: bool = False) -> bool:
        self.repair_attempt_count += 1
        return self.repair_attempt_count >= self.repair_attempt_limit(
            tolerate_status_drift=tolerate_status_drift,
        )

    def reset_repair_attempts(self) -> None:
        """有效计划更新或真实工具执行后，重新计算连续修复失败。"""

        self.repair_attempt_count = 0

    def record_repair_round_with_fallback(
        self,
        *,
        tolerate_status_drift: bool = False,
    ) -> PlanRepairResult:
        """记录一次计划修复，并仅在尚无有效计划时采用研究兜底计划。"""

        exhausted = self.record_repair_round(
            tolerate_status_drift=tolerate_status_drift,
        )
        attempt_count = self.repair_attempt_count
        attempt_limit = self.repair_attempt_limit(
            tolerate_status_drift=tolerate_status_drift,
        )
        if not exhausted:
            return PlanRepairResult(
                exhausted=False,
                attempt_count=attempt_count,
                attempt_limit=attempt_limit,
            )
        fallback = self.adopt_research_fallback()
        if fallback.accepted:
            return PlanRepairResult(
                exhausted=False,
                fallback=fallback,
                attempt_count=attempt_count,
                attempt_limit=attempt_limit,
            )
        return PlanRepairResult(
            exhausted=True,
            attempt_count=attempt_count,
            attempt_limit=attempt_limit,
        )

    def adopt_research_fallback(self) -> PlanUpdateResult:
        """连续计划修复失败后采用最小研究计划，避免无证据直接收尾。"""

        if (
            self.has_valid_model_plan
            or self.mode != "on"
            or self.required_initial_tool_counts
            != {
                "web_search": 1,
                "url_read": 1,
            }
        ):
            return PlanUpdateResult(False, "fallback_not_applicable")
        update = ModelPlanUpdate.model_validate(
            {
                "reason": _SYSTEM_FALLBACK_REASON,
                "items": [
                    {
                        "id": "research-search",
                        "title": "搜索候选来源",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "research-read",
                        "title": "核验关键来源",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["research-search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "research-answer",
                        "title": "整理研究结论",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["research-read"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.revision += 1
        self.source = "observed"
        self.reason = _SYSTEM_FALLBACK_REASON
        self.items = [item.model_dump() for item in update.items]
        self.reset_repair_attempts()
        return PlanUpdateResult(True, _SYSTEM_FALLBACK_REASON, self.snapshot())

    def configure_initial_tool_requirements(self, requirements: dict[str, int]) -> None:
        """配置首个模型计划必须预留的工具步骤数量。"""

        self.required_initial_tool_counts = {
            str(tool_name): int(count)
            for tool_name, count in requirements.items()
            if isinstance(tool_name, str)
            and tool_name
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
        }

    def adopt_observed_items(self, items: list[dict[str, Any]], *, reason: str = "legacy_observed") -> None:
        if self.has_valid_model_plan or self.mode == "off":
            return
        normalized: list[dict[str, Any]] = []
        for raw_item in items:
            item_id = str(raw_item.get("id", "")).strip()
            title = str(raw_item.get("title", "")).strip()
            if not item_id or not title:
                continue
            item = dict(raw_item)
            item.setdefault("depends_on", [])
            item.setdefault("planned_tools", list(item.get("tool_names") or []))
            normalized.append(item)
        if not normalized:
            return
        self.source = "observed"
        self.reason = reason
        self.items = normalized

    def snapshot(self, *, reason: str | None = None) -> dict[str, Any]:
        return {
            "plan_id": f"plan-{self.run_id}-model" if self.source == "model" else f"plan-{self.run_id}",
            "mode": self.mode,
            "source": self.source,
            "revision": self.revision,
            "reason": reason or self.reason,
            "items": [dict(item) for item in self.items],
        }

    def terminalize(self, outcome: str, *, has_final_answer: bool = False) -> dict[str, Any] | None:
        if not self.has_valid_model_plan:
            return None
        for item in self.items:
            status = item.get("status")
            if status in {"completed", "failed", "skipped", "blocked"}:
                continue
            if outcome == "stop":
                if has_final_answer and item.get("kind") in {"reasoning", "synthesis", "answer"}:
                    item["status"] = "completed"
                elif item.get("kind") in {"reasoning", "synthesis", "answer"}:
                    item["status"] = "blocked"
                elif status == "running":
                    item["status"] = "blocked"
                else:
                    item["status"] = "skipped"
            elif outcome in {"limit_reached", "incomplete"}:
                item["status"] = "blocked"
            elif outcome in {"interrupted", "superseded"}:
                item["status"] = "skipped"
            elif outcome == "failed":
                item["status"] = "failed" if status == "running" else "skipped"
            else:
                item["status"] = "blocked"
        self.revision += 1
        self.reason = f"terminal_{outcome}"
        return self.snapshot()

    def contains_item(self, item_id: str) -> bool:
        return any(item.get("id") == item_id for item in self.items)

    def has_active_tool_owner(self, tool_name: str) -> bool:
        """判断当前计划是否存在可合法绑定该工具的未完成步骤。"""

        return bool(self.active_plan_item_ids_for_tool(tool_name))

    def active_plan_item_ids_for_tool(self, tool_name: str) -> list[str]:
        """返回当前工具可绑定的未完成计划项 ID，供工具 schema 收窄取值。"""

        return [
            str(item.get("id"))
            for item in self.items
            if tool_name in (item.get("planned_tools") or [])
            and item.get("status") not in {"completed", "failed", "skipped", "blocked"}
        ]

    def plan_item_id_for_tool(
        self,
        tool_name: str,
        *,
        requested_item_id: str | None = None,
    ) -> str | None:
        matches = [
            *self.active_plan_item_ids_for_tool(tool_name),
        ]
        if requested_item_id is not None:
            return requested_item_id if requested_item_id in matches else None
        return matches[0] if len(matches) == 1 else None

    def plan_item_ids_for_tools(
        self,
        tool_names: list[str],
        *,
        requested_item_ids: list[str | None] | None = None,
    ) -> list[str | None]:
        if requested_item_ids is not None:
            return [
                self.plan_item_id_for_tool(tool_name, requested_item_id=requested_item_id)
                for tool_name, requested_item_id in zip(tool_names, requested_item_ids)
            ]
        result: list[str | None] = [None] * len(tool_names)
        for tool_name in dict.fromkeys(tool_names):
            call_indexes = [index for index, name in enumerate(tool_names) if name == tool_name]
            candidates = self.active_plan_item_ids_for_tool(tool_name)
            if len(candidates) == 1:
                for index in call_indexes:
                    result[index] = candidates[0]
        return result

    def mark_tools_started(self, plan_item_ids: list[str]) -> dict[str, Any] | None:
        return self._apply_tool_statuses({item_id: "running" for item_id in plan_item_ids})

    def mark_tool_results(self, statuses: dict[str, PlanStatus]) -> dict[str, Any] | None:
        return self._apply_tool_statuses(statuses)

    def _apply_tool_statuses(self, statuses: dict[str, PlanStatus]) -> dict[str, Any] | None:
        if not self.has_valid_model_plan:
            return None
        changed = False
        for item in self.items:
            item_id = str(item.get("id"))
            status = statuses.get(item_id)
            if status is not None and item.get("status") != status:
                item["status"] = status
                changed = True
        if not changed:
            return None
        self.revision += 1
        self.reason = "tool_progress"
        return self.snapshot()


def normalize_plan_mode(value: Any) -> PlanMode:
    return value if value in {"auto", "on", "off"} else "auto"


def _normalize_model_plan_payload(
    payload: Any,
    *,
    previous_items: list[dict[str, Any]],
) -> Any:
    """兼容主流 update_plan 的 explanation/plan/step/in_progress 形态。

    归一化后仍进入严格的 ModelPlanUpdate 校验；这里只处理字段别名、缺省的
    展示元数据和线性计划的显式依赖，不接受任意嵌套 payload。
    """

    if not isinstance(payload, dict):
        return payload
    raw_items = payload.get("items")
    compatibility_shape = False
    if not isinstance(raw_items, list):
        raw_items = payload.get("plan")
        compatibility_shape = isinstance(raw_items, list)
    if not isinstance(raw_items, list):
        return payload

    previous_ids_by_title = {
        str(item.get("title")): str(item.get("id")) for item in previous_items if item.get("title") and item.get("id")
    }
    previous_items_by_id = {str(item.get("id")): item for item in previous_items if item.get("id")}
    normalized_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            return payload
        title = raw_item.get("title")
        if not isinstance(title, str) or not title.strip():
            title = raw_item.get("step")
        if not isinstance(title, str) or not title.strip():
            return payload
        title = title.strip()

        item_id = raw_item.get("id")
        if not isinstance(item_id, str) or not _PLAN_ID_RE.fullmatch(item_id):
            item_id = previous_ids_by_title.get(title, f"step-{index + 1}")
        previous_item = previous_items_by_id.get(item_id)

        status = raw_item.get("status", "pending")
        if status == "in_progress":
            status = "running"
        elif status == "not_started":
            status = "pending"
        if compatibility_shape and not previous_items and status in {"completed", "failed", "skipped", "blocked"}:
            status = "pending"

        kind = raw_item.get("kind")
        if kind not in {"reasoning", "search", "read", "synthesis", "answer", "other"}:
            kind = (
                previous_item.get("kind")
                if previous_item is not None
                else ("answer" if index == len(raw_items) - 1 else "other")
            )

        depends_on = raw_item.get("depends_on")
        if not isinstance(depends_on, list):
            depends_on = (
                list(previous_item.get("depends_on") or [])
                if previous_item is not None
                else ([normalized_items[-1]["id"]] if normalized_items else [])
            )
        planned_tools = raw_item.get("planned_tools")
        if not isinstance(planned_tools, list):
            planned_tools = raw_item.get("tools")
        if not isinstance(planned_tools, list):
            planned_tools = list(previous_item.get("planned_tools") or []) if previous_item is not None else []

        normalized_items.append(
            {
                "id": item_id,
                "title": title,
                "status": status,
                "kind": kind,
                "depends_on": depends_on,
                "planned_tools": planned_tools,
            }
        )

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = payload.get("explanation")
    if not isinstance(reason, str) or not reason.strip():
        reason = "model_update"
    return {"reason": reason, "items": normalized_items}


def _has_dependency_cycle(dependencies: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item_id: str) -> bool:
        if item_id in visiting:
            return True
        if item_id in visited:
            return False
        visiting.add(item_id)
        if any(visit(dependency) for dependency in dependencies[item_id]):
            return True
        visiting.remove(item_id)
        visited.add(item_id)
        return False

    return any(visit(item_id) for item_id in dependencies)
