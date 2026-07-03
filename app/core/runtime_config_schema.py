"""运行时配置 schema 校验。

这里刻意使用轻量规则而不是完整 JSON Schema。运行时配置是主链路依赖，
校验失败时应阻断坏配置生效，而不是阻断聊天服务。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeConfigValidationResult:
    valid: bool
    issues: list[str]


def validate_runtime_config_payload(
    namespace: str,
    key: str,
    payload: Any,
) -> RuntimeConfigValidationResult:
    """校验单个 runtime config payload。

    返回全部可读问题，方便 admin 诊断和自动跳过坏版本。
    """

    issues: list[str] = []
    if not isinstance(payload, dict):
        return RuntimeConfigValidationResult(valid=False, issues=["payload 必须是对象"])

    if namespace == "prompt_template":
        _require_non_empty_string(payload, "template", issues)
    elif namespace == "agent_strategy" and key == "default":
        _validate_agent_strategy(payload, issues)
    elif namespace == "model_presentation" and key == "default":
        _validate_model_presentation(payload, issues)

    return RuntimeConfigValidationResult(valid=not issues, issues=issues)


def _validate_agent_strategy(payload: dict[str, Any], issues: list[str]) -> None:
    for field in ("model_runtime", "search", "network", "read_planner", "source_ranker", "tool_context"):
        _require_dict(payload, field, issues)

    model_runtime = payload.get("model_runtime")
    if isinstance(model_runtime, dict):
        aliases = model_runtime.get("agent_tools_disabled_aliases")
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            issues.append("model_runtime.agent_tools_disabled_aliases 必须是字符串数组")

    search = payload.get("search")
    if isinstance(search, dict):
        _require_dict(search, "standard_budget", issues, prefix="search")
        _require_dict(search, "budgets_by_intent", issues, prefix="search")
        _require_dict(search, "followup_budgets_by_name", issues, prefix="search")
        _require_dict(search, "intent_keywords", issues, prefix="search")
        _require_dict(search, "thresholds", issues, prefix="search")

    network = payload.get("network")
    if isinstance(network, dict):
        _require_positive_int(network, "max_search_calls", issues, prefix="network")
        _require_positive_int(network, "max_url_read_calls", issues, prefix="network")

    read_planner = payload.get("read_planner")
    if isinstance(read_planner, dict):
        _require_dict(read_planner, "read_limits", issues, prefix="read_planner")

    source_ranker = payload.get("source_ranker")
    if isinstance(source_ranker, dict):
        _require_dict(source_ranker, "weights", issues, prefix="source_ranker")
        _require_dict(source_ranker, "priority_thresholds", issues, prefix="source_ranker")

    tool_context = payload.get("tool_context")
    if isinstance(tool_context, dict):
        _require_positive_int(tool_context, "max_context_sources", issues, prefix="tool_context")
        _require_positive_int(tool_context, "url_read_max_content_chars", issues, prefix="tool_context")


def _validate_model_presentation(payload: dict[str, Any], issues: list[str]) -> None:
    for field in ("weights", "levels", "copy"):
        _require_dict(payload, field, issues)

    weights = payload.get("weights")
    if isinstance(weights, dict):
        for field in ("base", "network", "vision", "long_context", "deep_thinking"):
            _require_number(weights, field, issues, prefix="weights")

    levels = payload.get("levels")
    if isinstance(levels, dict):
        for field in ("recommended", "capable"):
            _require_number(levels, field, issues, prefix="levels")

    copy_section = payload.get("copy")
    if isinstance(copy_section, dict):
        for field in ("base_reason", "network_tooltip", "no_network_tooltip"):
            _require_non_empty_string(copy_section, field, issues, prefix="copy")


def _require_dict(payload: dict[str, Any], field: str, issues: list[str], *, prefix: str = "") -> None:
    value = payload.get(field)
    if not isinstance(value, dict):
        issues.append(f"{_path(prefix, field)} 必须是对象")


def _require_non_empty_string(payload: dict[str, Any], field: str, issues: list[str], *, prefix: str = "") -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{_path(prefix, field)} 必须是非空字符串")


def _require_number(payload: dict[str, Any], field: str, issues: list[str], *, prefix: str = "") -> None:
    value = payload.get(field)
    if not isinstance(value, int | float):
        issues.append(f"{_path(prefix, field)} 必须是数字")


def _require_positive_int(payload: dict[str, Any], field: str, issues: list[str], *, prefix: str = "") -> None:
    value = payload.get(field)
    if not isinstance(value, int) or value <= 0:
        issues.append(f"{_path(prefix, field)} 必须是正整数")


def _path(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field
