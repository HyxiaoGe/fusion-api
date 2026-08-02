"""Agent Plan Mode 的单一计划状态所有者。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PlanMode = Literal["auto", "on", "off"]
PlanSource = Literal["model"]
PlanStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
PlanKind = Literal["reasoning", "search", "read", "synthesis", "answer", "other"]

_PLAN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_SYSTEM_FALLBACK_REASON = "system_fallback"
_TERMINAL_STATUSES = frozenset({"completed", "failed", "skipped", "blocked"})
_FAILED_DEPENDENCY_STATUSES = frozenset({"failed", "skipped", "blocked"})
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
    """集中管理新运行中唯一权威的模型计划 revision。"""

    run_id: str
    mode: PlanMode = "auto"
    max_valid_updates: int = 6
    revision: int = 0
    source: PlanSource = "model"
    reason: str = "not_started"
    items: list[dict[str, Any]] = field(default_factory=list)
    valid_update_count: int = 0
    repair_attempt_count: int = 0
    consecutive_no_progress_updates: int = 0
    required_initial_tool_counts: dict[str, int] = field(default_factory=dict)
    allowed_tool_names: frozenset[str] | None = None
    attempted_tool_item_ids: set[str] = field(default_factory=set)
    successful_tool_item_ids: set[str] = field(default_factory=set)
    failed_tool_item_ids: set[str] = field(default_factory=set)
    synthesis_started: bool = False
    terminal_outcome: str | None = None

    @property
    def has_valid_model_plan(self) -> bool:
        return self.source == "model" and self.revision > 0 and bool(self.items)

    def apply_model_update(self, payload: Any) -> PlanUpdateResult:
        if self.mode == "off":
            return PlanUpdateResult(False, "plan_mode_off")
        if self.terminal_outcome is not None:
            return PlanUpdateResult(False, "run_already_terminal")
        if self.synthesis_started:
            return PlanUpdateResult(False, "synthesis_already_started")
        locked_item_ids = set(self.attempted_tool_item_ids)
        locked_item_ids.update(
            str(item.get("id"))
            for item in self.items
            if item.get("status") in {"completed", "failed", "skipped", "blocked"}
        )
        payload = _normalize_model_plan_payload(
            payload,
            previous_items=self.items,
            locked_item_ids=locked_item_ids,
        )
        try:
            update = ModelPlanUpdate.model_validate(payload)
        except ValidationError:
            return self._reject_repair("invalid_plan_structure")

        item_ids = [item.id for item in update.items]
        if len(item_ids) != len(set(item_ids)):
            return self._reject_repair("duplicate_item_id")
        if any(len(item.planned_tools) > 1 for item in update.items):
            return self._reject_repair("multiple_tools_per_item")
        if self.allowed_tool_names is not None and any(
            tool_name not in self.allowed_tool_names for item in update.items for tool_name in item.planned_tools
        ):
            return self._reject_repair("unannounced_planned_tool")
        known_ids = set(item_ids)
        if self.has_valid_model_plan:
            previous_items_by_id = {str(item.get("id")): item for item in self.items}
            previous_terminal_ids = {
                item_id
                for item_id, item in previous_items_by_id.items()
                if item.get("status") in {"completed", "failed", "skipped", "blocked"}
            }
            if previous_terminal_ids - known_ids:
                return self._reject_repair("terminal_item_removed")
            previous_attempted_ids = self.attempted_tool_item_ids.intersection(previous_items_by_id)
            if previous_attempted_ids - known_ids:
                return self._reject_repair("attempted_item_removed")
        for item in update.items:
            if item.id in item.depends_on:
                return self._reject_repair("self_dependency")
            if any(dependency not in known_ids for dependency in item.depends_on):
                return self._reject_repair("unknown_dependency")
        dependencies = {item.id: set(item.depends_on) for item in update.items}
        if _has_dependency_cycle(dependencies):
            return self._reject_repair("dependency_cycle")
        structure_error, _ = _terminal_answer_phase(update.items)
        if structure_error is not None:
            return self._reject_repair(structure_error)
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
        if _research_dependency_error(update.items, self.required_initial_tool_counts) is not None:
            return self._reject_repair("research_read_missing_search_dependency")
        if self.has_valid_model_plan:
            previous_status = {str(item.get("id")): item.get("status") for item in self.items}
            previous_items_by_id = {str(item.get("id")): item for item in self.items}
            previous_attempted_ids = self.attempted_tool_item_ids.intersection(previous_items_by_id)
            for item in update.items:
                previous_item = previous_items_by_id.get(item.id)
                if previous_status.get(item.id) in {"completed", "failed", "skipped", "blocked"} and previous_item:
                    locked_fields = ("title", "kind", "depends_on", "planned_tools")
                    if any(getattr(item, field) != previous_item.get(field) for field in locked_fields):
                        return self._reject_repair("terminal_item_mutated")
                elif item.id in previous_attempted_ids and previous_item:
                    locked_fields = ("title", "kind", "depends_on", "planned_tools")
                    if any(getattr(item, field) != previous_item.get(field) for field in locked_fields):
                        return self._reject_repair("attempted_item_mutated")

        normalized_items = _assign_user_visible_phases(
            [item.model_dump() for item in update.items],
            previous_items=self.items,
        )
        if self.has_valid_model_plan and normalized_items == self.items:
            self.consecutive_no_progress_updates += 1
            return PlanUpdateResult(True, "no_change", self.snapshot(reason="no_change"))
        if self.valid_update_count >= min(6, self.max_valid_updates):
            return PlanUpdateResult(False, "control_update_limit_reached")

        self.revision += 1
        self.valid_update_count += 1
        self.source = "model"
        self.reason = "model_update"
        self.items = normalized_items
        dependency_blocks = self._dependency_block_statuses({})
        if dependency_blocks:
            for item in self.items:
                item_id = str(item.get("id"))
                if item_id in dependency_blocks:
                    item["status"] = "blocked"
            self.failed_tool_item_ids.update(dependency_blocks)
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
        """有效计划更新或真实工具执行会打断连续修复失败。"""

        self.repair_attempt_count = 0
        self.consecutive_no_progress_updates = 0

    @property
    def plan_control_suppressed(self) -> bool:
        """连续重复同一计划时暂停控制工具，强制模型推进当前真实任务。"""

        return self.has_valid_model_plan and self.consecutive_no_progress_updates >= 2

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
                "url_read": 2,
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
                        "id": "research-read-primary",
                        "title": "核验首个关键来源",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["research-search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "research-read-secondary",
                        "title": "交叉核验独立来源",
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
                        "depends_on": ["research-read-primary", "research-read-secondary"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.revision += 1
        self.source = "model"
        self.reason = _SYSTEM_FALLBACK_REASON
        self.items = _assign_user_visible_phases(
            [item.model_dump() for item in update.items]
        )
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

    def snapshot(self, *, reason: str | None = None) -> dict[str, Any]:
        return {
            "plan_id": f"plan-{self.run_id}-model" if self.source == "model" else f"plan-{self.run_id}",
            "mode": self.mode,
            "source": self.source,
            "revision": self.revision,
            "reason": reason or self.reason,
            "items": [dict(item) for item in self.items],
        }

    def canonical_plan_for_model(self) -> list[dict[str, Any]]:
        """返回可直接用于下一次 update_plan 的安全 canonical 计划。"""

        return [
            {
                "id": str(item.get("id")),
                "step": str(item.get("title", "")),
                "status": "pending",
                "kind": item.get("kind", "other"),
                "depends_on": list(item.get("depends_on") or []),
                "planned_tools": list(item.get("planned_tools") or []),
            }
            for item in self.items
            if item.get("id") and item.get("title")
        ]

    def terminalize(self, outcome: str, *, has_final_answer: bool = False) -> dict[str, Any] | None:
        if not self.has_valid_model_plan or self.terminal_outcome is not None:
            return None
        self.terminal_outcome = outcome
        for item in self.items:
            item_id = str(item.get("id"))
            status = item.get("status")
            if status in {"completed", "failed", "skipped", "blocked"}:
                continue
            if outcome in {"stop", "limit_reached", "incomplete"}:
                if item_id in self.failed_tool_item_ids:
                    item["status"] = "blocked"
                elif item_id in self.successful_tool_item_ids:
                    item["status"] = "completed"
                elif item_id in self.attempted_tool_item_ids:
                    item["status"] = "blocked"
                elif has_final_answer and (
                    item.get("kind") in {"reasoning", "synthesis", "answer"}
                    or (item.get("kind") == "other" and not (item.get("planned_tools") or []))
                ):
                    item["status"] = "completed"
                elif outcome in {"limit_reached", "incomplete"}:
                    item["status"] = "blocked"
                elif item.get("kind") in {"reasoning", "synthesis", "answer"}:
                    item["status"] = "blocked"
                elif status == "running":
                    item["status"] = "blocked"
                else:
                    item["status"] = "skipped"
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

    def _items_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("id")): item for item in self.items if item.get("id")}

    def _running_item_id(self) -> str | None:
        return next(
            (str(item.get("id")) for item in self.items if item.get("status") == "running"),
            None,
        )

    def _auto_completable_prerequisites(self, item_id: str) -> set[str] | None:
        """返回工具启动时可同 revision 完成的无工具前置；None 表示依赖尚不可执行。"""

        items_by_id = self._items_by_id()
        auto_completed: set[str] = set()
        visiting: set[str] = set()

        def visit(dependency_id: str) -> bool:
            dependency = items_by_id.get(dependency_id)
            if dependency is None or dependency_id in visiting:
                return False
            status = dependency.get("status")
            if status == "completed":
                return True
            if status in _FAILED_DEPENDENCY_STATUSES or status == "running":
                return False
            if dependency.get("planned_tools") or dependency.get("kind") not in {
                "reasoning",
                "other",
            }:
                return False
            visiting.add(dependency_id)
            dependencies_ready = all(visit(str(parent_id)) for parent_id in dependency.get("depends_on") or [])
            visiting.remove(dependency_id)
            if not dependencies_ready:
                return False
            auto_completed.add(dependency_id)
            return True

        item = items_by_id.get(item_id)
        if item is None:
            return None
        if all(visit(str(dependency_id)) for dependency_id in item.get("depends_on") or []):
            return auto_completed
        return None

    def _tool_item_is_bindable(
        self,
        item: dict[str, Any],
        tool_name: str,
    ) -> bool:
        item_id = str(item.get("id"))
        if item.get("planned_tools") != [tool_name]:
            return False
        if item.get("status") not in {"pending", "running"}:
            return False
        running_item_id = self._running_item_id()
        if running_item_id is not None and running_item_id != item_id:
            return False
        return self._auto_completable_prerequisites(item_id) is not None

    def _dependency_block_statuses(
        self,
        proposed_statuses: dict[str, PlanStatus],
    ) -> dict[str, PlanStatus]:
        """失败依赖只递归阻塞后继执行项，最终回答仍由综合阶段诚实收口。"""

        statuses = {str(item.get("id")): str(item.get("status")) for item in self.items if item.get("id")}
        statuses.update(proposed_statuses)
        items_by_id = self._items_by_id()
        blocked: dict[str, PlanStatus] = {}

        def has_failed_ancestor(item_id: str, visiting: set[str]) -> bool:
            if item_id in visiting:
                return False
            if statuses.get(item_id) in _FAILED_DEPENDENCY_STATUSES:
                return True
            item = items_by_id.get(item_id)
            if item is None or statuses.get(item_id) == "completed":
                return False
            visiting.add(item_id)
            failed = any(
                has_failed_ancestor(str(dependency_id), visiting) for dependency_id in item.get("depends_on") or []
            )
            visiting.remove(item_id)
            return failed

        changed = True
        while changed:
            changed = False
            for item in self.items:
                item_id = str(item.get("id"))
                if item.get("kind") in {"answer", "synthesis"} or statuses.get(item_id) in _TERMINAL_STATUSES:
                    continue
                if any(
                    has_failed_ancestor(str(dependency_id), set()) for dependency_id in item.get("depends_on") or []
                ):
                    statuses[item_id] = "blocked"
                    blocked[item_id] = "blocked"
                    changed = True
        return blocked

    def has_active_tool_owner(self, tool_name: str) -> bool:
        """判断当前计划是否存在可合法绑定该工具的未完成步骤。"""

        return bool(self.active_plan_item_ids_for_tool(tool_name))

    def active_plan_item_ids_for_tool(self, tool_name: str) -> list[str]:
        """返回当前工具可绑定的未完成计划项 ID，供工具 schema 收窄取值。"""

        return [str(item.get("id")) for item in self.items if self._tool_item_is_bindable(item, tool_name)]

    def unexecuted_plan_item_ids_for_tool(self, tool_name: str) -> list[str]:
        """返回尚未取得成功执行证据的未完成工具步骤。"""

        return [
            item_id
            for item_id in self.active_plan_item_ids_for_tool(tool_name)
            if item_id not in self.successful_tool_item_ids or item_id in self.failed_tool_item_ids
        ]

    def unexecuted_plan_tool_names(self) -> set[str]:
        """返回仍有计划项未取得成功执行证据的工具名。"""

        return {
            str(tool_name)
            for item in self.items
            if (
                str(item.get("id")) not in self.successful_tool_item_ids
                or str(item.get("id")) in self.failed_tool_item_ids
            )
            and item.get("status") not in {"completed", "failed", "skipped", "blocked"}
            for tool_name in (item.get("planned_tools") or [])
            if isinstance(tool_name, str) and tool_name
        }

    def active_plan_tool_names(self) -> set[str]:
        """返回当前依赖已经满足、可以进入下一轮执行的计划工具。"""

        return {
            tool_name
            for item in self.items
            for tool_name in (item.get("planned_tools") or [])
            if isinstance(tool_name, str) and self._tool_item_is_bindable(item, tool_name)
        }

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
        if self.synthesis_started or self.terminal_outcome is not None:
            return None
        items_by_id = self._items_by_id()
        attempted_item_ids: list[str] = []
        auto_completed_item_ids: set[str] = set()
        for item_id in dict.fromkeys(plan_item_ids):
            item = items_by_id.get(item_id)
            planned_tools = item.get("planned_tools") if item is not None else None
            tool_name = planned_tools[0] if isinstance(planned_tools, list) and len(planned_tools) == 1 else None
            if not isinstance(tool_name, str) or item is None or not self._tool_item_is_bindable(item, tool_name):
                continue
            prerequisites = self._auto_completable_prerequisites(item_id)
            if prerequisites is None:
                continue
            attempted_item_ids.append(item_id)
            auto_completed_item_ids.update(prerequisites)
        attempted_item_id_set = set(attempted_item_ids)
        self.attempted_tool_item_ids.update(attempted_item_id_set)
        self.successful_tool_item_ids.difference_update(attempted_item_id_set)
        self.failed_tool_item_ids.difference_update(attempted_item_id_set)
        statuses: dict[str, PlanStatus] = {item_id: "completed" for item_id in auto_completed_item_ids}
        if self._running_item_id() is None and attempted_item_ids:
            statuses[attempted_item_ids[0]] = "running"
        return self._apply_tool_statuses(statuses)

    def mark_tool_results(self, statuses: dict[str, PlanStatus]) -> dict[str, Any] | None:
        if self.synthesis_started or self.terminal_outcome is not None:
            return None
        accepted_result_statuses = {
            item_id: status
            for item_id, status in statuses.items()
            if any(
                item.get("id") == item_id and item.get("status") not in {"completed", "failed", "skipped", "blocked"}
                for item in self.items
            )
        }
        accepted_statuses = {
            **accepted_result_statuses,
            **self._dependency_block_statuses(accepted_result_statuses),
        }
        self.attempted_tool_item_ids.update(accepted_result_statuses)
        for item_id, status in accepted_statuses.items():
            if status == "completed":
                self.successful_tool_item_ids.add(item_id)
                self.failed_tool_item_ids.discard(item_id)
            elif status in {"failed", "blocked"}:
                self.failed_tool_item_ids.add(item_id)
            elif status == "running":
                self.successful_tool_item_ids.discard(item_id)
                self.failed_tool_item_ids.discard(item_id)
        snapshot = self._apply_statuses(
            accepted_statuses,
            reason="tool_result",
        )
        if snapshot is not None or not accepted_statuses:
            return snapshot
        self.revision += 1
        self.reason = "tool_result"
        return self.snapshot()

    def execution_items_terminal(self) -> bool:
        """所有声明真实工具的执行项都已经取得服务端终态。"""

        if not self.has_valid_model_plan:
            return False
        return all(
            item.get("status") in {"completed", "failed", "skipped", "blocked"}
            for item in self.items
            if item.get("planned_tools")
        )

    def pending_execution_items(self) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.items
            if item.get("planned_tools") and item.get("status") not in {"completed", "failed", "skipped", "blocked"}
        ]

    def block_pending_execution(self, *, reason: str) -> dict[str, Any] | None:
        """在额度或协议收口前，将未取得结果的执行项单向收为 blocked。"""

        if self.synthesis_started or self.terminal_outcome is not None:
            return None
        statuses = {
            str(item.get("id")): "blocked"
            for item in self.items
            if item.get("planned_tools") and item.get("status") not in {"completed", "failed", "skipped", "blocked"}
        }
        self.failed_tool_item_ids.update(statuses)
        return self._apply_statuses(statuses, reason=reason)

    def block_tool_owners(
        self,
        tool_names: set[str] | frozenset[str],
        *,
        reason: str,
    ) -> dict[str, Any] | None:
        """把已确认无法再执行的工具 owner 及其下游执行项原子收为 blocked。"""

        if self.synthesis_started or self.terminal_outcome is not None or not tool_names:
            return None
        statuses: dict[str, PlanStatus] = {
            str(item.get("id")): "blocked"
            for item in self.items
            if item.get("status") in {"pending", "running"}
            and any(tool_name in tool_names for tool_name in (item.get("planned_tools") or []))
        }
        if not statuses:
            return None
        statuses.update(self._dependency_block_statuses(statuses))
        self.failed_tool_item_ids.update(statuses)
        return self._apply_statuses(statuses, reason=reason)

    def begin_synthesis(self) -> dict[str, Any] | None:
        """在所有执行项终态后不可逆地进入无工具综合阶段。"""

        if (
            not self.has_valid_model_plan
            or self.synthesis_started
            or self.terminal_outcome is not None
            or not self.execution_items_terminal()
        ):
            return None
        target_item_id = self._answer_phase_item_id()
        if target_item_id is None:
            self.synthesis_started = True
            self.revision += 1
            self.reason = "synthesis_started"
            return self.snapshot()
        statuses: dict[str, PlanStatus] = {}
        for item in self.items:
            item_id = str(item.get("id"))
            if item.get("status") in {"completed", "failed", "skipped", "blocked"}:
                continue
            if item_id == target_item_id:
                statuses[item_id] = "running"
            elif item.get("planned_tools"):
                statuses[item_id] = "blocked"
            elif item.get("kind") in {"reasoning", "other"}:
                statuses[item_id] = "completed"
            else:
                statuses[item_id] = "skipped"
        self.synthesis_started = True
        return self._apply_statuses(statuses, reason="synthesis_started")

    def _answer_phase_item_id(self) -> str | None:
        non_terminal_items = [
            item for item in self.items if item.get("status") not in {"completed", "failed", "skipped", "blocked"}
        ]
        _, terminal_item_id = _terminal_answer_phase(non_terminal_items)
        return terminal_item_id

    def _apply_tool_statuses(self, statuses: dict[str, PlanStatus]) -> dict[str, Any] | None:
        return self._apply_statuses(statuses, reason="tool_progress")

    def _apply_statuses(
        self,
        statuses: dict[str, PlanStatus],
        *,
        reason: str,
    ) -> dict[str, Any] | None:
        if not self.has_valid_model_plan or self.terminal_outcome is not None:
            return None
        changed = False
        for item in self.items:
            item_id = str(item.get("id"))
            status = statuses.get(item_id)
            if item.get("status") in {"completed", "failed", "skipped", "blocked"}:
                continue
            if status is not None and item.get("status") != status:
                item["status"] = status
                changed = True
        if not changed:
            return None
        self.revision += 1
        self.reason = reason
        return self.snapshot()


def normalize_plan_mode(value: Any) -> PlanMode:
    return value if value in {"auto", "on", "off"} else "auto"


def _assign_user_visible_phases(
    items: list[dict[str, Any]],
    *,
    previous_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """把连续同类执行任务归入稳定阶段，并保证阶段 ID 在快照内唯一连续。"""

    previous_by_id = {
        str(item.get("id")): item
        for item in previous_items or []
        if item.get("id")
    }
    current_item_ids = {str(item.get("id")) for item in items if item.get("id")}
    previous_phase_anchors: dict[str, str] = {}
    for item in previous_items or []:
        phase_id = item.get("phase_id")
        item_id = item.get("id")
        if phase_id and item_id and str(item_id) in current_item_ids:
            previous_phase_anchors.setdefault(str(phase_id), str(item_id))

    grouped_items: list[list[dict[str, Any]]] = []
    for item in items:
        phase_kind = _user_visible_phase_kind(item.get("kind"))
        previous_phase_id = previous_by_id.get(str(item.get("id")), {}).get("phase_id")
        current_previous_phase_ids = (
            {
                str(previous_by_id.get(str(grouped_item.get("id")), {}).get("phase_id"))
                for grouped_item in grouped_items[-1]
                if previous_by_id.get(str(grouped_item.get("id")), {}).get("phase_id")
            }
            if grouped_items
            else set()
        )
        keeps_existing_phase_boundary = (
            previous_phase_id is not None
            and bool(current_previous_phase_ids)
            and str(previous_phase_id) not in current_previous_phase_ids
        )
        if (
            grouped_items
            and _user_visible_phase_kind(grouped_items[-1][0].get("kind")) == phase_kind
            and not keeps_existing_phase_boundary
        ):
            grouped_items[-1].append(item)
        else:
            grouped_items.append([item])

    normalized: list[dict[str, Any]] = []
    reserved_previous_phase_ids = set(previous_phase_anchors)
    used_phase_ids: set[str] = set()
    for group in grouped_items:
        first = group[0]
        group_item_ids = {str(item.get("id")) for item in group}
        reusable_phase_id = next(
            (
                phase_id
                for phase_id, anchor_item_id in previous_phase_anchors.items()
                if anchor_item_id in group_item_ids and phase_id not in used_phase_ids
            ),
            None,
        )
        phase_id = reusable_phase_id or _new_user_visible_phase_id(
            str(first.get("id") or "task"),
            unavailable_phase_ids=reserved_previous_phase_ids | used_phase_ids,
        )
        used_phase_ids.add(phase_id)
        phase_title = _user_visible_phase_title(group)
        for item in group:
            normalized.append(
                {
                    **item,
                    "phase_id": phase_id,
                    "phase_title": phase_title,
                }
            )
    return normalized


def _new_user_visible_phase_id(
    first_item_id: str,
    *,
    unavailable_phase_ids: set[str],
) -> str:
    base_phase_id = f"phase-{first_item_id}"
    phase_id = base_phase_id
    suffix = 2
    while phase_id in unavailable_phase_ids:
        phase_id = f"{base_phase_id}-{suffix}"
        suffix += 1
    return phase_id


def _user_visible_phase_kind(kind: Any) -> str:
    return "synthesis" if kind in {"answer", "synthesis"} else str(kind or "other")


def _user_visible_phase_title(group: list[dict[str, Any]]) -> str:
    if len(group) == 1:
        return str(group[0].get("title") or "执行任务")[:80]
    phase_kind = _user_visible_phase_kind(group[0].get("kind"))
    return {
        "reasoning": "分析任务并制定方案",
        "search": "搜索并收集资料",
        "read": "读取并核验关键来源",
        "synthesis": "综合证据并输出结论",
        "other": "执行相关任务",
    }.get(phase_kind, "执行相关任务")


def _terminal_answer_phase(
    items: list[ModelPlanItem] | list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """校验计划终局可调度性，并返回覆盖全部执行分支的最终回答项。"""

    graph: dict[str, dict[str, Any]] = {}
    for raw_item in items:
        if isinstance(raw_item, ModelPlanItem):
            item_id = raw_item.id
            kind = raw_item.kind
            depends_on = list(raw_item.depends_on)
            planned_tools = list(raw_item.planned_tools)
        else:
            item_id = str(raw_item.get("id", ""))
            kind = raw_item.get("kind")
            depends_on = list(raw_item.get("depends_on") or [])
            planned_tools = list(raw_item.get("planned_tools") or [])
        if item_id:
            graph[item_id] = {
                "kind": kind,
                "depends_on": depends_on,
                "planned_tools": planned_tools,
            }

    answer_phase_ids = {item_id for item_id, item in graph.items() if item.get("kind") in {"answer", "synthesis"}}
    if not answer_phase_ids:
        return "missing_answer_phase", None
    if any(graph[item_id]["planned_tools"] for item_id in answer_phase_ids):
        return "answer_phase_has_tools", None

    def ancestors(item_id: str) -> set[str]:
        result: set[str] = set()
        stack = list(graph.get(item_id, {}).get("depends_on") or [])
        while stack:
            dependency_id = str(stack.pop())
            if dependency_id in result:
                continue
            result.add(dependency_id)
            stack.extend(graph.get(dependency_id, {}).get("depends_on") or [])
        return result

    execution_item_ids = {item_id for item_id, item in graph.items() if item.get("planned_tools")}
    if any(answer_phase_ids.intersection(ancestors(item_id)) for item_id in execution_item_ids):
        return "execution_depends_on_answer_phase", None

    depended_on_ids = {str(dependency_id) for item in graph.values() for dependency_id in item.get("depends_on") or []}
    terminal_phase_ids = [
        item_id for item_id in graph if item_id in answer_phase_ids and item_id not in depended_on_ids
    ]
    if not terminal_phase_ids:
        return "missing_terminal_answer_phase", None

    covering_phase_ids = [item_id for item_id in terminal_phase_ids if execution_item_ids.issubset(ancestors(item_id))]
    if not covering_phase_ids:
        return "uncovered_execution_branch", None

    for kind in ("answer", "synthesis"):
        candidates = [item_id for item_id in covering_phase_ids if graph[item_id].get("kind") == kind]
        if candidates:
            return None, candidates[-1]
    return "missing_terminal_answer_phase", None


def _research_dependency_error(
    items: list[ModelPlanItem],
    required_initial_tool_counts: dict[str, int],
) -> str | None:
    """研究证据计划中的读取步骤必须位于搜索步骤之后。"""

    if not {"web_search", "url_read"}.issubset(required_initial_tool_counts):
        return None
    items_by_id = {item.id: item for item in items}
    search_item_ids = {
        item.id for item in items if "web_search" in item.planned_tools
    }

    def has_search_ancestor(item: ModelPlanItem) -> bool:
        visited: set[str] = set()
        stack = list(item.depends_on)
        while stack:
            dependency_id = stack.pop()
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            if dependency_id in search_item_ids:
                return True
            dependency = items_by_id.get(dependency_id)
            if dependency is not None:
                stack.extend(dependency.depends_on)
        return False

    if any(
        "url_read" in item.planned_tools and not has_search_ancestor(item)
        for item in items
    ):
        return "research_read_missing_search_dependency"
    return None


def _normalize_model_plan_payload(
    payload: Any,
    *,
    previous_items: list[dict[str, Any]],
    locked_item_ids: set[str] | None = None,
) -> Any:
    """兼容主流 update_plan 的 explanation/plan/step/in_progress 形态。

    归一化后仍进入严格的 ModelPlanUpdate 校验；这里只处理字段别名、缺省的
    展示元数据和线性计划的显式依赖，不接受任意嵌套 payload。
    """

    if not isinstance(payload, dict):
        return payload
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raw_items = payload.get("plan")
    if not isinstance(raw_items, list):
        return payload

    previous_ids_by_title = {
        str(item.get("title")): str(item.get("id")) for item in previous_items if item.get("title") and item.get("id")
    }
    previous_items_by_id = {str(item.get("id")): item for item in previous_items if item.get("id")}
    canonical_item_ids = locked_item_ids or set()
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
        locked_item = previous_item if item_id in canonical_item_ids else None
        if locked_item is not None:
            title = str(locked_item.get("title", title))

        previous_status = previous_item.get("status") if previous_item is not None else None
        status = previous_status if previous_status is not None else "pending"

        kind = raw_item.get("kind")
        if kind not in {"reasoning", "search", "read", "synthesis", "answer", "other"}:
            kind = (
                previous_item.get("kind")
                if previous_item is not None
                else ("answer" if index == len(raw_items) - 1 else "other")
            )
        if locked_item is not None:
            kind = locked_item.get("kind", kind)

        depends_on = raw_item.get("depends_on")
        if not isinstance(depends_on, list):
            depends_on = (
                list(previous_item.get("depends_on") or [])
                if previous_item is not None
                else ([normalized_items[-1]["id"]] if normalized_items else [])
            )
        if locked_item is not None:
            depends_on = list(locked_item.get("depends_on") or [])
        planned_tools = raw_item.get("planned_tools")
        if not isinstance(planned_tools, list):
            planned_tools = raw_item.get("tools")
        if not isinstance(planned_tools, list):
            planned_tools = list(previous_item.get("planned_tools") or []) if previous_item is not None else []
        if locked_item is not None:
            planned_tools = list(locked_item.get("planned_tools") or [])

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
