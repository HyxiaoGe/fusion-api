from typing import Any, Mapping

APP_TAG = "app:fusion"

ALLOWED_LLM_PHASES = frozenset(
    {
        "chat_non_stream",
        "chat_stream",
        "generate_title",
        "suggest_questions",
        "file_processing",
    }
)


def build_litellm_metadata(phase: str) -> dict[str, list[str]]:
    """构造 LiteLLM SpendLogs 可消费的低基数业务标签。"""
    if phase not in ALLOWED_LLM_PHASES:
        raise ValueError(f"未知 LLM phase: {phase}")
    return {"tags": [APP_TAG, f"phase:{phase}"]}


def merge_litellm_kwargs(phase: str, kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(kwargs or {})
    merged.pop("metadata", None)
    merged["extra_body"] = merge_openai_extra_body(phase, merged.get("extra_body"))
    return merged


def merge_openai_extra_body(phase: str, extra_body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(extra_body or {})
    metadata = dict(merged.get("metadata") or {})
    metadata["tags"] = build_litellm_metadata(phase)["tags"]
    merged["metadata"] = metadata
    return merged
