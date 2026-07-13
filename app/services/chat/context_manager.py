"""按模型输入窗口为单次 LLM 调用生成受控消息快照。"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Callable

from app.ai.llm_round_observability import estimate_prompt_tokens, resolve_context_window
from app.schemas.chat import ContextUsage

DEFAULT_TRIGGER_RATIO = 0.85
DEFAULT_TARGET_RATIO = 0.75
DEFAULT_ESTIMATOR_TIMEOUT_SECONDS = 5.0
_TOKEN_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="context-manager-tokenizer")
_TOKEN_ADMISSION = BoundedSemaphore(value=2)

WindowResolver = Callable[[str], tuple[int | None, str, str]]
TokenEstimator = Callable[[str, list[dict], dict], int]


@dataclass(frozen=True)
class ContextPlan:
    messages: list[dict]
    status: str
    context_window_tokens: int | None
    context_window_source: str
    context_window_status: str
    trigger_tokens: int | None = None
    target_tokens: int | None = None
    estimated_tokens_before: int | None = None
    estimated_tokens_after: int | None = None
    removed_turns: int = 0
    removed_tool_transactions: int = 0
    removed_messages: int = 0

    def to_usage_context(
        self,
        *,
        actual_prompt_tokens: int | None = None,
        round_index: int | None = None,
    ) -> ContextUsage:
        """返回可发给客户端并持久化的安全字段白名单。"""
        return ContextUsage(
            status=self.status,
            round_index=round_index,
            window_tokens=self.context_window_tokens,
            estimated_tokens_before=self.estimated_tokens_before,
            estimated_tokens_after=self.estimated_tokens_after,
            actual_prompt_tokens=actual_prompt_tokens,
            removed_turns=self.removed_turns,
            removed_messages=self.removed_messages,
            removed_tool_transactions=self.removed_tool_transactions,
        )

    def telemetry(self) -> dict[str, Any]:
        """返回不包含消息正文的单轮观测字段。"""
        payload = {
            "status": self.status,
            "context_window_tokens": self.context_window_tokens,
            "context_window_source": self.context_window_source,
            "context_window_status": self.context_window_status,
            "trigger_tokens": self.trigger_tokens,
            "target_tokens": self.target_tokens,
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "removed_turns": self.removed_turns,
            "removed_tool_transactions": self.removed_tool_transactions,
            "removed_messages": self.removed_messages,
        }
        return {f"context_management_{key}": value for key, value in payload.items()}


class ContextManagementError(RuntimeError):
    """可安全展示并可结构化写入流终态的 Context 管理错误。"""

    error_code = "context_management_error"

    def __init__(self, message: str, plan: ContextPlan):
        super().__init__(message)
        self.plan = plan


class ContextBudgetExceededError(ContextManagementError):
    """强制保留内容本身已经超过安全输入预算。"""

    error_code = "context_budget_exceeded"

    def __init__(self, plan: ContextPlan):
        super().__init__("当前消息与必要上下文过长，请缩短本次输入或移除较大的文件后重试", plan)


class ContextEstimationUnavailableError(ContextManagementError):
    """已知窗口的长输入无法完成 Token 预算校验。"""

    error_code = "context_estimation_unavailable"

    def __init__(self, plan: ContextPlan):
        super().__init__("上下文预算暂时无法校验，请稍后重试", plan)


@dataclass(frozen=True)
class _RemovalGroup:
    indices: tuple[int, ...]
    kind: str


async def prepare_context(
    *,
    messages: list[dict],
    model_id: str,
    litellm_model: str,
    call_kwargs: dict,
    window_resolver: WindowResolver = resolve_context_window,
    token_estimator: TokenEstimator = estimate_prompt_tokens,
    trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
    target_ratio: float = DEFAULT_TARGET_RATIO,
    estimator_timeout_seconds: float = DEFAULT_ESTIMATOR_TIMEOUT_SECONDS,
    run_in_thread: bool = True,
    use_fast_path: bool = True,
) -> ContextPlan:
    """保留 canonical messages，只返回本次调用使用的 effective snapshot。"""
    snapshot = list(messages)
    window_tokens, window_source, window_status = _safe_resolve_window(model_id, window_resolver)
    if window_tokens is None:
        return _plan(
            snapshot,
            status="bypass_unknown_window",
            window_tokens=None,
            window_source=window_source,
            window_status=window_status,
        )

    trigger_tokens, target_tokens = _validate_budget(window_tokens, trigger_ratio, target_ratio)
    if use_fast_path and _rough_token_upper_bound(snapshot, call_kwargs) <= trigger_tokens:
        return _plan(
            snapshot,
            status="no_op_fast_path",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
        )

    try:
        before = await _estimate(
            litellm_model,
            snapshot,
            call_kwargs,
            token_estimator=token_estimator,
            run_in_thread=run_in_thread,
            timeout_seconds=estimator_timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        plan = _plan(
            snapshot,
            status="estimator_unavailable",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
        )
        raise ContextEstimationUnavailableError(plan) from error

    if before <= trigger_tokens:
        return _plan(
            snapshot,
            status="no_op",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            estimated_before=before,
            estimated_after=before,
        )

    try:
        return await _trim_to_budget(
            messages=snapshot,
            litellm_model=litellm_model,
            call_kwargs=call_kwargs,
            token_estimator=token_estimator,
            run_in_thread=run_in_thread,
            timeout_seconds=estimator_timeout_seconds,
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            estimated_before=before,
        )
    except asyncio.CancelledError:
        raise
    except ContextManagementError:
        raise
    except Exception as error:
        plan = _plan(
            snapshot,
            status="estimator_unavailable",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            estimated_before=before,
        )
        raise ContextEstimationUnavailableError(plan) from error


def _safe_resolve_window(model_id: str, resolver: WindowResolver) -> tuple[int | None, str, str]:
    try:
        value, source, status = resolver(model_id)
    except Exception:
        return None, "unknown", "error"
    if isinstance(value, bool):
        return None, source, "invalid"
    try:
        normalized = int(value) if value is not None else None
    except (TypeError, ValueError):
        normalized = None
    if normalized is None or normalized <= 0:
        return None, source, status if value is None else "invalid"
    return normalized, source, status


def _validate_budget(window_tokens: int, trigger_ratio: float, target_ratio: float) -> tuple[int, int]:
    if not 0 < target_ratio < trigger_ratio < 1:
        raise ValueError("Context 预算比例必须满足 0 < target < trigger < 1")
    return max(1, int(window_tokens * trigger_ratio)), max(1, int(window_tokens * target_ratio))


def _rough_token_upper_bound(messages: list[dict], call_kwargs: dict) -> int:
    """不复制正文的保守上界；仅用于证明短输入无需精确 tokenizer。"""
    return (
        _rough_value_units(messages)
        + _rough_value_units(call_kwargs.get("tools"))
        + _rough_value_units(call_kwargs.get("tool_choice"))
    )


def _rough_value_units(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        # 单个 Unicode 字符最多占四个 UTF-8 字节；按每字节一个 token 给出保守上界。
        return len(value) * 4 + 4
    if isinstance(value, dict):
        return 8 + sum(_rough_value_units(key) + _rough_value_units(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 8 + sum(_rough_value_units(item) for item in value)
    return 8


async def _estimate(
    litellm_model: str,
    messages: list[dict],
    call_kwargs: dict,
    *,
    token_estimator: TokenEstimator,
    run_in_thread: bool,
    timeout_seconds: float,
) -> int:
    if run_in_thread:
        loop = asyncio.get_running_loop()
        if timeout_seconds <= 0:
            raise ValueError("Token 估算超时必须是正数")
        if not _TOKEN_ADMISSION.acquire(blocking=False):
            raise RuntimeError("Token 估算器繁忙")
        try:
            future = loop.run_in_executor(
                _TOKEN_EXECUTOR,
                _estimate_with_admission_release,
                token_estimator,
                litellm_model,
                messages,
                call_kwargs,
            )
        except Exception:
            _TOKEN_ADMISSION.release()
            raise
        value = await asyncio.wait_for(future, timeout=timeout_seconds)
    else:
        value = token_estimator(litellm_model, messages, call_kwargs)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Token 估算结果无效")
    return value


def _estimate_with_admission_release(
    token_estimator: TokenEstimator,
    litellm_model: str,
    messages: list[dict],
    call_kwargs: dict,
) -> int:
    try:
        return token_estimator(litellm_model, messages, call_kwargs)
    finally:
        _TOKEN_ADMISSION.release()


async def _trim_to_budget(
    *,
    messages: list[dict],
    litellm_model: str,
    call_kwargs: dict,
    token_estimator: TokenEstimator,
    run_in_thread: bool,
    timeout_seconds: float,
    window_tokens: int,
    window_source: str,
    window_status: str,
    trigger_tokens: int,
    target_tokens: int,
    estimated_before: int,
) -> ContextPlan:
    turns = _turn_indices(messages)
    groups = [_RemovalGroup(tuple(turn), "turn") for turn in turns[:-1]]
    if turns:
        transactions = _tool_transaction_groups(messages, turns[-1])
        groups.extend(_RemovalGroup(tuple(indices), "tool_transaction") for indices in transactions[:-1])

    best_count: int | None = None
    best_messages: list[dict] | None = None
    best_estimate: int | None = None
    estimates: dict[int, tuple[list[dict], int]] = {}

    async def estimate_after_removing(group_count: int) -> tuple[list[dict], int]:
        cached = estimates.get(group_count)
        if cached is not None:
            return cached
        removed = _indices_for_groups(groups[:group_count])
        candidate = _without_indices(messages, removed)
        estimate = await _estimate(
            litellm_model,
            candidate,
            call_kwargs,
            token_estimator=token_estimator,
            run_in_thread=run_in_thread,
            timeout_seconds=timeout_seconds,
        )
        estimates[group_count] = (candidate, estimate)
        return candidate, estimate

    low, high = 1, len(groups)
    while low <= high:
        middle = (low + high) // 2
        candidate, estimate = await estimate_after_removing(middle)
        if estimate <= target_tokens:
            best_count = middle
            best_messages = candidate
            best_estimate = estimate
            high = middle - 1
        else:
            low = middle + 1

    if best_count is not None and best_messages is not None and best_estimate is not None:
        removed_indices = _indices_for_groups(groups[:best_count])
        removed_turns, removed_transactions = _removed_group_counts(groups[:best_count])
        return _trimmed_plan(
            best_messages,
            status="trimmed",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            estimated_before=estimated_before,
            estimated_after=best_estimate,
            removed_turns=removed_turns,
            removed_tool_transactions=removed_transactions,
            removed_messages=len(removed_indices),
        )

    removed_indices = _indices_for_groups(groups)
    if groups:
        required, estimated_after = await estimate_after_removing(len(groups))
    else:
        required, estimated_after = messages, estimated_before
    removed_turn_count, removed_transaction_count = _removed_group_counts(groups)
    if estimated_after <= trigger_tokens:
        return _trimmed_plan(
            required,
            status="trimmed_required_above_target",
            window_tokens=window_tokens,
            window_source=window_source,
            window_status=window_status,
            trigger_tokens=trigger_tokens,
            target_tokens=target_tokens,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            removed_turns=removed_turn_count,
            removed_tool_transactions=removed_transaction_count,
            removed_messages=len(removed_indices),
        )

    plan = _trimmed_plan(
        required,
        status="required_context_over_budget",
        window_tokens=window_tokens,
        window_source=window_source,
        window_status=window_status,
        trigger_tokens=trigger_tokens,
        target_tokens=target_tokens,
        estimated_before=estimated_before,
        estimated_after=estimated_after,
        removed_turns=removed_turn_count,
        removed_tool_transactions=removed_transaction_count,
        removed_messages=len(removed_indices),
    )
    raise ContextBudgetExceededError(plan)


def _turn_indices(messages: list[dict]) -> list[list[int]]:
    turns: list[list[int]] = []
    current: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            continue
        previous_role = messages[current[-1]].get("role") if current else None
        if message.get("role") == "user" and current and previous_role != "user":
            turns.append(current)
            current = []
        current.append(index)
    if current:
        turns.append(current)
    return turns


def _tool_transaction_groups(messages: list[dict], turn: list[int]) -> list[list[int]]:
    """返回同一 user turn 内协议完整的 assistant.tool_calls + tool results。"""
    groups: list[list[int]] = []
    turn_positions = {index: position for position, index in enumerate(turn)}
    for index in turn:
        message = messages[index]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        call_ids = {
            call.get("id")
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("id")
        }
        if not call_ids:
            continue
        group = [index]
        observed_ids: set[str] = set()
        start = turn_positions[index] + 1
        for following_index in turn[start:]:
            following = messages[following_index]
            if following.get("role") != "tool":
                break
            tool_call_id = following.get("tool_call_id")
            if tool_call_id not in call_ids:
                break
            group.append(following_index)
            observed_ids.add(tool_call_id)
        if observed_ids == call_ids:
            groups.append(group)
    return groups


def _indices_for_groups(groups: list[_RemovalGroup]) -> set[int]:
    return {index for group in groups for index in group.indices}


def _removed_group_counts(groups: list[_RemovalGroup]) -> tuple[int, int]:
    return (
        sum(group.kind == "turn" for group in groups),
        sum(group.kind == "tool_transaction" for group in groups),
    )


def _without_indices(messages: list[dict], removed_indices: set[int]) -> list[dict]:
    return [message for index, message in enumerate(messages) if index not in removed_indices]


def _plan(
    messages: list[dict],
    *,
    status: str,
    window_tokens: int | None,
    window_source: str,
    window_status: str,
    trigger_tokens: int | None = None,
    target_tokens: int | None = None,
    estimated_before: int | None = None,
    estimated_after: int | None = None,
) -> ContextPlan:
    return ContextPlan(
        messages=messages,
        status=status,
        context_window_tokens=window_tokens,
        context_window_source=window_source,
        context_window_status=window_status,
        trigger_tokens=trigger_tokens,
        target_tokens=target_tokens,
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
    )


def _trimmed_plan(
    messages: list[dict],
    *,
    status: str,
    window_tokens: int,
    window_source: str,
    window_status: str,
    trigger_tokens: int,
    target_tokens: int,
    estimated_before: int,
    estimated_after: int,
    removed_turns: int,
    removed_tool_transactions: int,
    removed_messages: int,
) -> ContextPlan:
    return ContextPlan(
        messages=messages,
        status=status,
        context_window_tokens=window_tokens,
        context_window_source=window_source,
        context_window_status=window_status,
        trigger_tokens=trigger_tokens,
        target_tokens=target_tokens,
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
        removed_turns=removed_turns,
        removed_tool_transactions=removed_tool_transactions,
        removed_messages=removed_messages,
    )
