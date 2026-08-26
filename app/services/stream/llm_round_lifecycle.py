"""把 LLM round 测量结果映射为 Agent 生命周期事件。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.logger import app_logger as logger
from app.schemas.chat import Usage
from app.services.agent.llm_round_detail_recorder import LlmRoundDetailDraft


def _measured_int(observation: Any, name: str) -> int | None:
    value = getattr(observation, name, None)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return None


def _measured_delta_ms(observation: Any, delta_kind: str) -> int | None:
    by_kind = getattr(observation, "output_delta_ms", None)
    if callable(by_kind):
        value = by_kind(delta_kind)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    if getattr(observation, "first_output_delta_kind", None) == delta_kind:
        return _measured_int(observation, "first_output_delta_ms")
    return None


def _llm_error_code(error: BaseException) -> str:
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if type(error).__name__ == "RateLimitError" or getattr(error, "status_code", None) == 429:
        return "rate_limit"
    return "provider_error"


def _optional_sum(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def accumulate_token_usage(current: Usage, addition: Usage | None) -> Usage:
    """累加公开 token 统计；供应商未报告的 cache 字段保持 ``None``。"""

    if addition is None:
        return current
    return Usage(
        input_tokens=current.input_tokens + addition.input_tokens,
        output_tokens=current.output_tokens + addition.output_tokens,
        cache_read_tokens=_optional_sum(current.cache_read_tokens, addition.cache_read_tokens),
        cache_write_tokens=_optional_sum(current.cache_write_tokens, addition.cache_write_tokens),
        reasoning_tokens=_optional_sum(current.reasoning_tokens, addition.reasoning_tokens),
    )


@dataclass
class LLMRoundLifecycle:
    """单个真实 LLM logical round 的无敏感生命周期句柄。"""

    emitter: Any
    observation: Any
    llm_round_id: str
    parent_step_id: str
    usage: Usage | None = None
    finish_reason: str | None = None
    first_output_emitted: bool = False
    published_delta_kind: str | None = None
    terminal_emitted: bool = False
    conversation_id: str | None = None
    run_id: str | None = None
    message_id: str | None = None
    detail_scheduler: Callable[[LlmRoundDetailDraft], Any] | None = None
    reasoning_text: str = ""
    content_text: str = ""
    detail_scheduled: bool = False

    @classmethod
    async def start(
        cls,
        *,
        emitter: Any | None,
        observation: Any,
        round_index: int,
        model: str,
        provider: str,
        parent_step_id: str,
        conversation_id: str | None = None,
        run_id: str | None = None,
        message_id: str | None = None,
        detail_scheduler: Callable[[LlmRoundDetailDraft], Any] | None = None,
        system_prompt_fingerprint: str | None = None,
    ) -> LLMRoundLifecycle | None:
        emit = getattr(emitter, "llm_round_started", None)
        if not callable(emit):
            return None
        lifecycle = cls(
            emitter=emitter,
            observation=observation,
            llm_round_id=str(uuid.uuid4()),
            parent_step_id=parent_step_id,
            conversation_id=conversation_id,
            run_id=run_id,
            message_id=message_id,
            detail_scheduler=detail_scheduler,
        )
        await emit(
            llm_round_id=lifecycle.llm_round_id,
            round_index=round_index,
            model=model,
            provider=provider,
            parent_step_id=parent_step_id,
            **(
                {"system_prompt_fingerprint": system_prompt_fingerprint}
                if system_prompt_fingerprint is not None
                else {}
            ),
        )
        return lifecycle

    def record_result(self, *, usage: Usage | None, finish_reason: str | None) -> None:
        self.usage = usage
        self.finish_reason = finish_reason

    def record_detail(self, *, reasoning_text: str, content_text: str) -> None:
        """保存该 logical round 已对用户可见的正文，等待终态后异步落库。"""

        self.reasoning_text = reasoning_text
        self.content_text = content_text

    async def publish_visible_output(self, visible_kind: str | None = None) -> None:
        """在对应可见 chunk 已写 Redis 后发送首次输出事件。"""

        if self.first_output_emitted or self.terminal_emitted:
            return
        delta_kind = visible_kind
        if delta_kind not in {"reasoning", "content"}:
            return
        ttft_ms = _measured_delta_ms(self.observation, delta_kind)
        if ttft_ms is None:
            return
        await self._publish_first(delta_kind=delta_kind, ttft_ms=ttft_ms)

    async def publish_tool_output(self) -> None:
        if self.first_output_emitted or self.terminal_emitted:
            return
        delta_kind = "tool_call"
        ttft_ms = _measured_delta_ms(self.observation, delta_kind)
        if ttft_ms is None:
            return
        await self._publish_first(delta_kind=delta_kind, ttft_ms=ttft_ms)

    async def _publish_first(self, *, delta_kind: str, ttft_ms: int) -> None:
        await self.emitter.llm_round_first_output_delta(
            llm_round_id=self.llm_round_id,
            delta_kind=delta_kind,
            ttft_ms=ttft_ms,
            parent_step_id=self.parent_step_id,
        )
        self.first_output_emitted = True
        self.published_delta_kind = delta_kind

    async def finish_success(self, *, output_visible: bool = False) -> None:
        if self.terminal_emitted:
            return
        if output_visible:
            await self.publish_visible_output(getattr(self.observation, "first_output_delta_kind", None))
        usage = self.usage or Usage()
        await self.emitter.llm_round_completed(
            llm_round_id=self.llm_round_id,
            finish_reason=self.finish_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            ttft_ms=(
                _measured_delta_ms(self.observation, self.published_delta_kind or "")
                if self.first_output_emitted
                else None
            ),
            duration_ms=_measured_int(self.observation, "duration_ms") or 0,
            parent_step_id=self.parent_step_id,
        )
        self.terminal_emitted = True
        self._schedule_detail()

    async def finish_failed(self, error: BaseException) -> None:
        if self.terminal_emitted:
            return
        await self.emitter.llm_round_failed(
            llm_round_id=self.llm_round_id,
            error_code=_llm_error_code(error),
            message=None,
            parent_step_id=self.parent_step_id,
        )
        self.terminal_emitted = True
        self._schedule_detail()

    async def finish_cancelled(self, *, reason: str) -> None:
        if self.terminal_emitted:
            return
        await self.emitter.llm_round_cancelled(
            llm_round_id=self.llm_round_id,
            reason=reason,
            parent_step_id=self.parent_step_id,
        )
        self.terminal_emitted = True
        self._schedule_detail()

    def _schedule_detail(self) -> None:
        if self.detail_scheduled or self.detail_scheduler is None:
            return
        if self.conversation_id is None or self.run_id is None:
            return
        self.detail_scheduled = True
        try:
            self.detail_scheduler(
                LlmRoundDetailDraft(
                    conversation_id=self.conversation_id,
                    run_id=self.run_id,
                    message_id=self.message_id,
                    llm_round_id=self.llm_round_id,
                    reasoning_text=self.reasoning_text,
                    content_text=self.content_text,
                )
            )
        except BaseException as error:  # noqa: BLE001 — 详情是辅助投影，不得替换生命周期终态
            logger.warning(
                "调度 LLM round 详情失败: run_id=%s llm_round_id=%s error_type=%s",
                self.run_id,
                self.llm_round_id,
                type(error).__name__,
            )
