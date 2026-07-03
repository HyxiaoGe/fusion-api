"""Agent 策略配置访问。"""

from __future__ import annotations

from typing import Any

from app.services.runtime_config_defaults import DEFAULT_AGENT_STRATEGY_CONFIG
from app.services.runtime_config_service import deep_merge_config, get_runtime_config_payload


def get_agent_strategy_config(
    *,
    override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if override is not None:
        return deep_merge_config(DEFAULT_AGENT_STRATEGY_CONFIG, override), {
            "namespace": "agent_strategy",
            "key": "default",
            "source": "override",
            "version": "test-override",
        }
    return get_runtime_config_payload(
        "agent_strategy",
        "default",
        DEFAULT_AGENT_STRATEGY_CONFIG,
    )


def get_agent_tools_disabled_aliases() -> set[str]:
    config, _meta = get_agent_strategy_config()
    aliases = config.get("model_runtime", {}).get("agent_tools_disabled_aliases") or []
    return {str(alias) for alias in aliases}
