"""Agent 任务模式的唯一受控策略解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.schemas.response import ApiException, ErrorCode
from app.services.agent.plan_coordinator import PlanMode, normalize_plan_mode

TaskMode = Literal["standard", "deep_research"]
NetworkProfile = Literal["standard", "deep_research"]
EvidencePolicy = Literal["standard", "deep_research_v1", "knowledge_grounded_v1"]

DEEP_RESEARCH_NETWORK_PROFILE: NetworkProfile = "deep_research"
DEEP_RESEARCH_EVIDENCE_POLICY: EvidencePolicy = "deep_research_v1"


@dataclass(frozen=True)
class AgentTaskPolicy:
    task_mode: TaskMode
    plan_mode: PlanMode
    network_profile: NetworkProfile
    evidence_policy: EvidencePolicy

    def apply_to_options(self, options: dict | None = None) -> dict:
        return {
            **(options or {}),
            "task_mode": self.task_mode,
            "plan_mode": self.plan_mode,
            "network_profile": self.network_profile,
            "evidence_policy": self.evidence_policy,
        }


def normalize_task_mode(value: Any) -> TaskMode:
    return value if value in {"standard", "deep_research"} else "standard"


def resolve_agent_task_policy(
    *,
    options: dict | None,
    capabilities: dict | None,
    enforce_capabilities: bool = True,
) -> AgentTaskPolicy:
    """解析任务模式；深度研究的安全约束不接受客户端降级覆盖。"""

    normalized_options = options or {}
    normalized_capabilities = capabilities or {}
    task_mode = normalize_task_mode(normalized_options.get("task_mode"))
    requested_plan_mode = normalize_plan_mode(normalized_options.get("plan_mode"))

    if task_mode == "deep_research":
        if enforce_capabilities:
            _require_deep_research_capabilities(normalized_options, normalized_capabilities)
        return AgentTaskPolicy(
            task_mode="deep_research",
            plan_mode="on",
            network_profile=DEEP_RESEARCH_NETWORK_PROFILE,
            evidence_policy=DEEP_RESEARCH_EVIDENCE_POLICY,
        )

    return AgentTaskPolicy(
        task_mode="standard",
        plan_mode=requested_plan_mode,
        network_profile="standard",
        evidence_policy="standard",
    )


def restore_agent_task_policy(run_config: dict | None) -> AgentTaskPolicy:
    """从可信 run config 恢复策略；旧 run 缺字段时保持原 plan_mode 语义。"""

    config = run_config if isinstance(run_config, dict) else {}
    policy = resolve_agent_task_policy(
        options={
            "task_mode": config.get("task_mode"),
            "plan_mode": config.get("plan_mode"),
        },
        capabilities={},
        enforce_capabilities=False,
    )
    if policy.task_mode != "deep_research":
        return policy
    return AgentTaskPolicy(
        task_mode=policy.task_mode,
        plan_mode=policy.plan_mode,
        network_profile=(
            DEEP_RESEARCH_NETWORK_PROFILE
            if config.get("network_profile") != DEEP_RESEARCH_NETWORK_PROFILE
            else config["network_profile"]
        ),
        evidence_policy=(
            DEEP_RESEARCH_EVIDENCE_POLICY
            if config.get("evidence_policy") != DEEP_RESEARCH_EVIDENCE_POLICY
            else config["evidence_policy"]
        ),
    )


def _require_deep_research_capabilities(options: dict, capabilities: dict) -> None:
    tools_enabled = options.get("disable_tools") is not True
    function_calling = capabilities.get("functionCalling") is True
    search_capable = capabilities.get("searchCapable") is True
    if tools_enabled and function_calling and search_capable:
        return
    raise ApiException(
        ErrorCode.MODEL_UNAVAILABLE,
        "当前模型不支持深度研究所需的联网工具，请选择支持联网工具的模型",
        400,
    )
