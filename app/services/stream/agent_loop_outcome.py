"""Agent loop outcome 类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentLoopExit(Enum):
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    SUMMARY_REQUIRED = "summary_required"
    PRODUCT_RESULT_READY = "product_result_ready"


@dataclass(frozen=True)
class AgentLoopOutcome:
    exit: AgentLoopExit
    error_msg: str | None = None
    summary_finish_reason: str | None = None
