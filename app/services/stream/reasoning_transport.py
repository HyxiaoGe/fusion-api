"""模型推理流运输策略。"""

from __future__ import annotations

from typing import Literal

ReasoningTransportMode = Literal["delta", "snapshot", "probing"]


def normalize_reasoning_model_id(model_id: str | None) -> str:
    if not model_id:
        return ""
    normalized = model_id.strip().lower().rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def is_k3_reasoning_model(model_id: str | None) -> bool:
    return normalize_reasoning_model_id(model_id) == "kimi-k3"


def initial_reasoning_transport_mode(
    *,
    model_id: str | None,
    explicit_mode: ReasoningTransportMode | None,
) -> ReasoningTransportMode:
    if explicit_mode is not None:
        return explicit_mode
    return "probing" if is_k3_reasoning_model(model_id) else "delta"


def allows_deferred_reasoning_output(model_id: str | None) -> bool:
    """计划前只允许 K3 展示推理；其它模型保持既有 defer 语义。"""

    return is_k3_reasoning_model(model_id)
