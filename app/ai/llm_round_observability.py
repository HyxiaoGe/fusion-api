"""LLM 单轮 Context、Token 与时延的脱敏可观测性。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import BoundedSemaphore
from typing import Any

import litellm

from app.ai import litellm_catalog
from app.core.logger import app_logger as logger
from app.schemas.chat import Usage

LOG_PREFIX = "LLM_ROUND_CONTEXT"
_ESTIMATE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-context-estimator")
_ESTIMATE_ADMISSION = BoundedSemaphore(value=2)
_ESTIMATE_TIMEOUT_SECONDS = 5.0
_BACKGROUND_LOG_TASKS: set[asyncio.Task[None]] = set()
_CONTEXT_MANAGEMENT_FIELDS = {
    "context_management_status",
    "context_management_context_window_tokens",
    "context_management_context_window_source",
    "context_management_context_window_status",
    "context_management_trigger_tokens",
    "context_management_target_tokens",
    "context_management_estimated_tokens_before",
    "context_management_estimated_tokens_after",
    "context_management_removed_turns",
    "context_management_removed_tool_transactions",
    "context_management_removed_messages",
}


@dataclass(frozen=True)
class RoundMetadata:
    conversation_id: str
    run_id: str
    round_index: int
    step_id: str
    round_kind: str
    model_id: str
    provider: str
    assistant_message_id: str | None = None


@dataclass(frozen=True)
class ContextComputation:
    estimated_prompt_tokens: int | None
    estimator_status: str
    context_window_tokens: int | None
    context_window_source: str
    context_window_status: str


def summarize_messages(messages: list[dict], call_kwargs: dict) -> dict[str, Any]:
    """只统计消息构成，不返回任何正文、链接或工具参数。"""
    role_counts = {role: 0 for role in ("system", "user", "assistant", "tool", "other")}
    content_part_counts = {part: 0 for part in ("text", "image_url", "other")}
    assistant_tool_call_count = 0

    for message in messages:
        role = message.get("role")
        role_counts[role if role in role_counts else "other"] += 1
        assistant_tool_call_count += len(message.get("tool_calls") or [])
        _count_content_parts(message.get("content"), content_part_counts)

    return {
        "message_count": len(messages),
        "role_counts": role_counts,
        "content_part_counts": content_part_counts,
        "assistant_tool_call_count": assistant_tool_call_count,
        "request_tool_definition_count": len(call_kwargs.get("tools") or []),
    }


def _count_content_parts(content: Any, counts: dict[str, int]) -> None:
    if isinstance(content, str):
        if content:
            counts["text"] += 1
        return
    if not isinstance(content, list):
        if content is not None:
            counts["other"] += 1
        return
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == "text":
            counts["text"] += 1
        elif part_type in {"image_url", "input_image"}:
            counts["image_url"] += 1
        else:
            counts["other"] += 1


def estimate_prompt_tokens(litellm_model: str, messages: list[dict], call_kwargs: dict) -> int:
    """使用 LiteLLM tokenizer 估算单轮输入；调用方必须放在线程池执行。"""
    return int(
        litellm.token_counter(
            model=litellm_model,
            messages=messages,
            tools=call_kwargs.get("tools"),
            tool_choice=call_kwargs.get("tool_choice"),
            use_default_image_token_count=True,
            default_token_count=0,
        )
    )


def resolve_context_window(model_id: str) -> tuple[int | None, str, str]:
    """只读 LiteLLM 现有缓存中的正整数输入窗口，不触发目录 HTTP。"""
    entry, cache_status = litellm_catalog.get_cached_model_entry(model_id)
    value = entry.get("max_input_tokens") if entry else None
    if entry is None:
        return None, "unknown" if cache_status == "unavailable" else "litellm_catalog_cache", cache_status
    if isinstance(value, bool):
        value = None
    try:
        tokens = int(value) if value is not None else None
    except (TypeError, ValueError):
        tokens = None
    if tokens is None or tokens <= 0:
        return None, "litellm_catalog_cache", "invalid"
    return tokens, "litellm_catalog_cache", cache_status


def calculate_budget_metrics(
    prompt_tokens: int | None,
    context_window_tokens: int | None,
) -> tuple[float | None, bool | None]:
    if prompt_tokens is None or context_window_tokens is None:
        return None, None
    return round(prompt_tokens / context_window_tokens, 6), prompt_tokens > context_window_tokens


def _has_text_delta(chunk: Any) -> bool:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return False
    reasoning = getattr(delta, "reasoning_content", None)
    model_extra = getattr(delta, "model_extra", None)
    if not reasoning and isinstance(model_extra, dict):
        reasoning = model_extra.get("reasoning_content")
    return bool(reasoning or getattr(delta, "content", None))


class _ObservedAsyncIterator:
    def __init__(self, response: Any, observation: LLMRoundObservation):
        self._iterator = response.__aiter__()
        self._observation = observation

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        chunk = await self._iterator.__anext__()
        self._observation.observe_chunk(chunk)
        return chunk


class LLMRoundObservation:
    """在不改变 streaming response 契约的前提下观测一个 logical round。"""

    def __init__(
        self,
        *,
        metadata: RoundMetadata,
        litellm_model: str,
        messages: list[dict],
        call_kwargs: dict,
        clock: Callable[[], float] = time.perf_counter,
        token_estimator: Callable[[str, list[dict], dict], int] = estimate_prompt_tokens,
        context_window_resolver: Callable[[str], tuple[int | None, str, str]] = resolve_context_window,
        run_context_in_thread: bool = True,
        estimated_prompt_tokens: int | None = None,
        estimator_status: str | None = None,
        context_management: dict[str, Any] | None = None,
    ):
        self.metadata = metadata
        self.litellm_model = litellm_model
        self.messages = list(messages)
        self.call_kwargs = dict(call_kwargs)
        self.clock = clock
        self.token_estimator = token_estimator
        self.context_window_resolver = context_window_resolver
        self.run_context_in_thread = run_context_in_thread
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.estimator_status = estimator_status
        self.context_management = {
            key: value for key, value in (context_management or {}).items() if key in _CONTEXT_MANAGEMENT_FIELDS
        }
        self.started_at: float | None = None
        self.first_text_delta_at: float | None = None
        self._context_result = self._resolve_window()
        self._estimate_task: asyncio.Future[tuple[int | None, str]] | None = None
        self._log_task: asyncio.Task[None] | None = None
        self.last_payload: dict[str, Any] | None = None

    def start(self) -> None:
        self.started_at = self.clock()
        if self.estimator_status is not None:
            self._context_result = self._with_estimate(
                self.estimated_prompt_tokens,
                self.estimator_status,
            )
            return
        if self.estimated_prompt_tokens is not None:
            self._context_result = self._with_estimate(self.estimated_prompt_tokens, "reused_context_manager")
            return
        if self.run_context_in_thread:
            if not _ESTIMATE_ADMISSION.acquire(blocking=False):
                self._context_result = self._with_estimate(None, "skipped_overload")
                return
            try:
                loop = asyncio.get_running_loop()
                self._estimate_task = asyncio.ensure_future(
                    loop.run_in_executor(_ESTIMATE_EXECUTOR, self._estimate_with_admission_release)
                )
            except Exception:
                _ESTIMATE_ADMISSION.release()
                self._context_result = self._with_estimate(None, "error")
        else:
            estimated_tokens, status = self._estimate_tokens()
            self._context_result = self._with_estimate(estimated_tokens, status)

    def wrap_response(self, response: Any) -> Any:
        if not hasattr(response, "__aiter__"):
            return response
        return _ObservedAsyncIterator(response, self)

    def observe_chunk(self, chunk: Any) -> None:
        if self.first_text_delta_at is None and _has_text_delta(chunk):
            self.first_text_delta_at = self.clock()

    async def finish_success(self, *, usage: Usage | None, finish_reason: str) -> None:
        outcome = "cancelled" if finish_reason == "cancelled" else "success"
        await self._finish(
            usage=usage,
            finish_reason=finish_reason,
            outcome=outcome,
            error_type=None,
        )

    async def finish_error(self, error: BaseException) -> None:
        await self._finish(
            usage=None,
            finish_reason=None,
            outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "error",
            error_type=type(error).__name__,
        )

    def _resolve_window(self) -> ContextComputation:
        try:
            window_tokens, window_source, window_status = self.context_window_resolver(self.metadata.model_id)
        except Exception:
            window_tokens, window_source, window_status = None, "unknown", "error"
        return ContextComputation(
            estimated_prompt_tokens=None,
            estimator_status="pending",
            context_window_tokens=window_tokens,
            context_window_source=window_source,
            context_window_status=window_status,
        )

    def _estimate_tokens(self) -> tuple[int | None, str]:
        try:
            estimated_tokens = self.token_estimator(self.litellm_model, self.messages, self.call_kwargs)
            if isinstance(estimated_tokens, bool) or estimated_tokens < 0:
                return None, "error"
            return int(estimated_tokens), "success"
        except Exception:
            return None, "error"

    def _estimate_with_admission_release(self) -> tuple[int | None, str]:
        try:
            return self._estimate_tokens()
        finally:
            _ESTIMATE_ADMISSION.release()

    def _with_estimate(self, estimated_tokens: int | None, status: str) -> ContextComputation:
        return ContextComputation(
            estimated_prompt_tokens=estimated_tokens,
            estimator_status=status,
            context_window_tokens=self._context_result.context_window_tokens,
            context_window_source=self._context_result.context_window_source,
            context_window_status=self._context_result.context_window_status,
        )

    async def _await_context_result(self) -> ContextComputation:
        if self._estimate_task is None:
            return self._context_result
        try:
            estimated_tokens, status = await asyncio.wait_for(
                asyncio.shield(self._estimate_task),
                timeout=_ESTIMATE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            estimated_tokens, status = None, "timeout"
        except (asyncio.CancelledError, Exception):
            estimated_tokens, status = None, "error"
        return self._with_estimate(estimated_tokens, status)

    async def _finish(
        self,
        *,
        usage: Usage | None,
        finish_reason: str | None,
        outcome: str,
        error_type: str | None,
    ) -> None:
        try:
            finished_at = self.clock()
            self._log_task = asyncio.create_task(
                self._finish_log(
                    finished_at=finished_at,
                    usage=usage,
                    finish_reason=finish_reason,
                    outcome=outcome,
                    error_type=error_type,
                )
            )
            _BACKGROUND_LOG_TASKS.add(self._log_task)
            self._log_task.add_done_callback(_release_background_log_task)
        except Exception:
            logger.warning("LLM 单轮可观测性收尾失败，已忽略")

    async def _finish_log(
        self,
        *,
        finished_at: float,
        usage: Usage | None,
        finish_reason: str | None,
        outcome: str,
        error_type: str | None,
    ) -> None:
        try:
            context = await self._await_context_result()
            self.last_payload = self._build_payload(
                finished_at=finished_at,
                context=context,
                usage=usage,
                finish_reason=finish_reason,
                outcome=outcome,
                error_type=error_type,
            )
            emit_llm_round_log(self.last_payload)
        except Exception:
            logger.warning("LLM 单轮可观测性后台日志失败，已忽略")

    async def wait_for_log(self) -> None:
        """仅供测试与受控验收等待旁路日志，不在聊天主链路调用。"""
        if self._log_task is not None:
            await self._log_task

    def _build_payload(
        self,
        *,
        finished_at: float,
        context: ContextComputation,
        usage: Usage | None,
        finish_reason: str | None,
        outcome: str,
        error_type: str | None,
    ) -> dict[str, Any]:
        estimated_ratio, estimated_over = calculate_budget_metrics(
            context.estimated_prompt_tokens,
            context.context_window_tokens,
        )
        round_prompt_tokens = usage.input_tokens if usage is not None else None
        round_completion_tokens = usage.output_tokens if usage is not None else None
        actual_ratio, actual_over = calculate_budget_metrics(round_prompt_tokens, context.context_window_tokens)
        first_model_text_delta_ms = None
        total_duration_ms = None
        if self.started_at is not None:
            total_duration_ms = round((finished_at - self.started_at) * 1000, 3)
            if self.first_text_delta_at is not None:
                first_model_text_delta_ms = round((self.first_text_delta_at - self.started_at) * 1000, 3)
        return {
            "event": "llm_round_context",
            "schema_version": 1,
            "usage_scope": "llm_round",
            **asdict(self.metadata),
            **summarize_messages(self.messages, self.call_kwargs),
            "estimated_prompt_tokens": context.estimated_prompt_tokens,
            "estimator_method": "litellm_token_counter",
            "estimator_status": context.estimator_status,
            "context_window_tokens": context.context_window_tokens,
            "context_window_source": context.context_window_source,
            "context_window_status": context.context_window_status,
            "estimated_utilization_ratio": estimated_ratio,
            "estimated_over_budget": estimated_over,
            "round_prompt_tokens": round_prompt_tokens,
            "round_completion_tokens": round_completion_tokens,
            "actual_utilization_ratio": actual_ratio,
            "actual_over_budget": actual_over,
            "first_model_text_delta_ms": first_model_text_delta_ms,
            "total_duration_ms": total_duration_ms,
            "finish_reason": finish_reason,
            "outcome": outcome,
            "error_type": error_type,
            **self.context_management,
        }


def _release_background_log_task(task: asyncio.Task[None]) -> None:
    _BACKGROUND_LOG_TASKS.discard(task)
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        return


def emit_llm_round_log(payload: dict[str, Any]) -> None:
    try:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        logger.info(f"{LOG_PREFIX} {serialized}")
    except Exception:
        logger.warning("LLM 单轮可观测性日志写入失败，已忽略")


def create_llm_round_observation(
    *,
    conversation_id: str,
    run_id: str,
    round_index: int,
    step_id: str,
    round_kind: str,
    model_id: str,
    provider: str,
    litellm_model: str,
    messages: list[dict],
    call_kwargs: dict,
    assistant_message_id: str | None = None,
    estimated_prompt_tokens: int | None = None,
    estimator_status: str | None = None,
    context_management: dict[str, Any] | None = None,
) -> LLMRoundObservation:
    return LLMRoundObservation(
        metadata=RoundMetadata(
            conversation_id=conversation_id,
            run_id=run_id,
            round_index=round_index,
            step_id=step_id,
            round_kind=round_kind,
            model_id=model_id,
            provider=provider,
            assistant_message_id=assistant_message_id,
        ),
        litellm_model=litellm_model,
        messages=messages,
        call_kwargs=call_kwargs,
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimator_status=estimator_status,
        context_management=context_management,
    )
