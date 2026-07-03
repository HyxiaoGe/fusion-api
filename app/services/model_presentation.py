"""模型能力展示派生逻辑。"""

from __future__ import annotations

from typing import Any

from app.services.runtime_config_defaults import DEFAULT_MODEL_PRESENTATION_CONFIG
from app.services.runtime_config_service import deep_merge_config, get_runtime_config_payload

CapabilityPresentation = dict[str, Any]


def get_model_presentation_config() -> tuple[dict[str, Any], dict[str, Any]]:
    return get_runtime_config_payload(
        "model_presentation",
        "default",
        DEFAULT_MODEL_PRESENTATION_CONFIG,
    )


def build_model_capability_presentation(
    model: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> CapabilityPresentation:
    resolved_config = _resolve_config(config)
    copy = resolved_config["copy"]
    health = model.get("health") or {}
    if health.get("status") == "unhealthy":
        warning = health.get("error") or copy["unhealthy_fallback"]
        labels = _build_labels(model, resolved_config)
        if not any(label["key"] == "unhealthy" for label in labels):
            labels.append({"key": "unhealthy", "text": "不可用", "tone": "danger"})
        tooltip = "\n".join(
            [
                str(model.get("name") or ""),
                copy["unavailable_headline"],
                warning,
                f"健康状态异常：{warning}",
            ]
        )
        return {
            "score": 0,
            "level": "unavailable",
            "headline": copy["unavailable_headline"],
            "reasons": [],
            "warnings": [warning],
            "labels": labels,
            "tooltip": tooltip,
        }

    capabilities = model.get("capabilities") or {}
    has_network = _supports_agent_tools(capabilities)
    has_vision = bool(capabilities.get("vision"))
    has_deep_thinking = bool(capabilities.get("deepThinking"))
    has_long_context = _supports_long_context(model, resolved_config)
    weights = resolved_config["weights"]

    score = int(weights["base"])
    reasons = [copy["base_reason"]]
    warnings: list[str] = []

    if has_network:
        score += int(weights["network"])
        reasons.append(copy["network_reason"])
    else:
        warnings.append(copy["no_network_warning"])

    if has_vision:
        score += int(weights["vision"])
        reasons.append(copy["vision_reason"])

    if has_long_context:
        score += int(weights["long_context"])
        reasons.append(copy["long_context_reason"])

    if has_deep_thinking:
        score += int(weights["deep_thinking"])
        reasons.append(copy["deep_thinking_reason"])

    normalized_score = min(score, 100)
    return {
        "score": normalized_score,
        "level": _level(normalized_score, resolved_config),
        "headline": _headline(
            copy,
            has_network=has_network,
            has_vision=has_vision,
            has_long_context=has_long_context,
        ),
        "reasons": reasons,
        "warnings": warnings,
        "labels": _build_labels(model, resolved_config),
        "tooltip": _tooltip(
            model,
            resolved_config,
            has_network=has_network,
            has_vision=has_vision,
            has_deep_thinking=has_deep_thinking,
            has_long_context=has_long_context,
            warnings=warnings,
        ),
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        payload, _meta = get_model_presentation_config()
        return payload
    return deep_merge_config(DEFAULT_MODEL_PRESENTATION_CONFIG, config)


def _supports_agent_tools(capabilities: dict[str, Any]) -> bool:
    if isinstance(capabilities.get("searchCapable"), bool):
        return bool(capabilities.get("searchCapable"))
    return bool(capabilities.get("agentTools") or capabilities.get("webSearch"))


def _supports_long_context(model: dict[str, Any], config: dict[str, Any]) -> bool:
    try:
        tokens = int(model.get("contextWindowTokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    return tokens >= int(config["long_context_threshold_tokens"])


def _level(score: int, config: dict[str, Any]) -> str:
    levels = config["levels"]
    if score >= int(levels["recommended"]):
        return "recommended"
    if score >= int(levels["capable"]):
        return "capable"
    return "limited"


def _headline(
    copy: dict[str, str],
    *,
    has_network: bool,
    has_vision: bool,
    has_long_context: bool,
) -> str:
    if has_network and has_vision and has_long_context:
        return copy["network_vision_long_context_headline"]
    if has_network and has_long_context:
        return copy["network_long_context_headline"]
    if has_network and has_vision:
        return copy["network_vision_headline"]
    if has_network:
        return copy["network_headline"]
    if has_vision:
        return copy["vision_headline"]
    return copy["default_headline"]


def _build_labels(model: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    capabilities = model.get("capabilities") or {}
    labels = [
        {"key": "network", "text": "可联网", "tone": "success"}
        if _supports_agent_tools(capabilities)
        else {"key": "no-network", "text": "不可联网", "tone": "muted"}
    ]
    if capabilities.get("vision"):
        labels.append({"key": "vision", "text": "读图", "tone": "info"})
    if _supports_agent_tools(capabilities):
        labels.append({"key": "tools", "text": "工具", "tone": "info"})
    if _supports_long_context(model, config):
        labels.append({"key": "long-context", "text": "长上下文", "tone": "info"})
    if capabilities.get("deepThinking"):
        labels.append({"key": "deep-task", "text": "深度任务", "tone": "warning"})
    if capabilities.get("fileSupport") and not capabilities.get("vision"):
        labels.append({"key": "file", "text": "文件", "tone": "info"})
    if capabilities.get("imageGen"):
        labels.append({"key": "image-gen", "text": "画图", "tone": "info"})
    return labels


def _tooltip(
    model: dict[str, Any],
    config: dict[str, Any],
    *,
    has_network: bool,
    has_vision: bool,
    has_deep_thinking: bool,
    has_long_context: bool,
    warnings: list[str],
) -> str:
    copy = config["copy"]
    lines = [
        str(model.get("name") or ""),
        _headline(copy, has_network=has_network, has_vision=has_vision, has_long_context=has_long_context),
        copy["network_tooltip"] if has_network else copy["no_network_tooltip"],
        copy["vision_tooltip"] if has_vision else copy["no_vision_tooltip"],
    ]
    if model.get("contextWindowTokens"):
        lines.append(f"上下文窗口约 {_format_token_limit(int(model['contextWindowTokens']))}")
    lines.append(copy["deep_thinking_tooltip"] if has_deep_thinking else copy["no_deep_thinking_tooltip"])
    lines.extend(warnings)
    return "\n".join(lines)


def _format_token_limit(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{round(tokens / 10_000)}万 tokens"
    if tokens >= 10_000:
        return f"{round(tokens / 1000)}k tokens"
    return f"{tokens} tokens"
