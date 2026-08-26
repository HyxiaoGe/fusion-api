"""AgentEventEmitter — 控制面事件唯一发送方."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from app.services.agent import events as ev
from app.services.agent.sanitizer import cap_and_truncate, sanitize_arguments

# Sentinel 用于 _envelope 区分"未传 step_id（用 current）"vs"显式传 None"
_USE_CURRENT_STEP = object()
_LLM_ERROR_SUMMARIES = {
    "rate_limit": "模型服务请求过于频繁，请稍后重试。",
    "timeout": "模型服务响应超时，请稍后重试。",
    "provider_error": "模型服务暂时不可用，请稍后重试。",
}
_RETRIEVAL_ERROR_SUMMARIES = {
    "timeout": "知识库检索超时，请稍后重试。",
    "knowledge_retrieval_unavailable": "知识库检索暂时不可用，请稍后重试。",
    "provider_error": "知识库检索暂时不可用，请稍后重试。",
}


class _RedisWriter(Protocol):
    """emitter 的 Redis 写入抽象。

    本仓库目前没有具体实现：Task 9 (stream_handler) 会提供一个 adapter，
    把 (conv_id, chunk_type, payload: dict) 桥接到现有
    stream_state_service.append_chunk(conv_id, chunk_type, content, block_id, task_id=task_id)
    （payload JSON 序列化进 content，block_id 留空）。
    单元测试用 unittest.mock.AsyncMock 满足此 Protocol。
    """

    async def append_chunk(
        self,
        conversation_id: str,
        task_id: str,
        chunk_type: str,
        payload: dict[str, Any],
    ) -> None: ...


class AgentEventEmitter:
    """单 run 内发 agent_event；并发安全；维护 step 上下文。"""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        conversation_id: str,
        task_id: str,
        redis_writer: _RedisWriter,
    ) -> None:
        self._run_id = run_id
        self._trace_id = trace_id
        self._conv_id = conversation_id
        self._task_id = task_id
        self._writer = redis_writer
        self._sequence = 0
        self._current_step_id: str | None = None
        self._message_id: str | None = None
        self._lock = asyncio.Lock()
        self._sealed = False

    async def _emit(self, event: ev.AgentEventBase, *, max_payload_bytes: int | None = None) -> None:
        """在 lock 内校验后预留 sequence，再写入 Redis。

        依赖 Pydantic v2 默认行为：模型字段可赋值且不重新校验
        （未启用 frozen / validate_assignment）。extra="forbid" 只拒绝额外字段，
        不阻塞已声明字段的 mutation。若未来在 AgentEventBase 启用
        validate_assignment，本方法的 mutation 会触发额外校验开销。
        """
        async with self._lock:
            if self._sealed:
                raise RuntimeError("agent_event emitter 已封口")
            event.sequence = self._sequence
            event.ts = time.time()
            payload = event.model_dump(mode="json")
            if max_payload_bytes is not None and len(event.model_dump_json().encode("utf-8")) > max_payload_bytes:
                raise ValueError("agent_event 超过允许的体积上限")
            self._sequence += 1
            await self._writer.append_chunk(self._conv_id, self._task_id, "agent_event", payload)

    async def seal_and_get_last_sequence(self) -> int:
        """封口当前 emitter，并返回最后已预留的序号。"""
        async with self._lock:
            self._sealed = True
            return self._sequence - 1

    def _envelope(
        self,
        *,
        tool_call_id: str | None = None,
        step_id: Any = _USE_CURRENT_STEP,
        parent_step_id: str | None = None,
    ) -> dict[str, Any]:
        """构造 envelope 字段；sequence 与 ts 用占位值，由 _emit 在 lock 内回填。

        返回的 dict 不可直接发出 — sequence/ts 必须由 _emit 在 lock 内回填，
        否则会和真实顺序错位。

        step_id 默认从 _current_step_id 派生；run-level 事件需显式传 None。
        """
        return dict(
            run_id=self._run_id,
            trace_id=self._trace_id,
            sequence=0,  # 占位，_emit 回填
            ts=0.0,  # 占位，_emit 回填
            step_id=self._current_step_id if step_id is _USE_CURRENT_STEP else step_id,
            tool_call_id=tool_call_id,
            parent_run_id=None,
            parent_step_id=parent_step_id,
        )

    @staticmethod
    def _controlled_error(error_code: str | None, summaries: dict[str, str]) -> tuple[str | None, str | None]:
        """只将白名单错误码及其公开摘要发往用户事件。"""
        if error_code not in summaries:
            return None, None
        return error_code, summaries[error_code]

    @staticmethod
    def _opaque_retrieval_summary(query_summary: str | None) -> str | None:
        """隔离无法在 emitter 边界证明安全的检索原文。"""
        if query_summary is None or not query_summary.strip():
            return None
        return "已发起知识库检索"

    async def run_started(self, *, message_id: str, model: str, tools: list[str], config: dict[str, Any]) -> None:
        await self._emit(
            ev.RunStarted(
                type="run_started",
                conversation_id=self._conv_id,
                message_id=message_id,
                task_id=self._task_id,
                model=model,
                tools=tools,
                config=config,
                **self._envelope(step_id=None),
            )
        )
        self._message_id = message_id

    async def step_started(self, *, step_number: int) -> str:
        step_id = str(uuid.uuid4())
        # 在发事件前先设 current_step_id，让 envelope 带上自己
        self._current_step_id = step_id
        await self._emit(
            ev.StepStarted(
                type="step_started",
                step_number=step_number,
                **self._envelope(),
            )
        )
        return step_id

    async def tool_call_started(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        plan_item_id: str | None = None,
    ) -> None:
        sanitized = sanitize_arguments(tool_name, arguments)
        await self._emit(
            ev.ToolCallStarted(
                type="tool_call_started",
                tool_name=tool_name,
                arguments=sanitized,
                plan_item_id=plan_item_id,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def tool_call_delta(self, *, tool_call_id: str, tool_name: str, delta: dict[str, Any]) -> None:
        await self._emit(
            ev.ToolCallDelta(
                type="tool_call_delta",
                tool_name=tool_name,
                delta=delta,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def tool_call_completed(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        status: str,
        duration_ms: int,
        result_summary: dict[str, Any],
        error: str | None = None,
        plan_item_id: str | None = None,
    ) -> None:
        capped = cap_and_truncate(result_summary, max_bytes=1024)
        await self._emit(
            ev.ToolCallCompleted(
                type="tool_call_completed",
                tool_name=tool_name,
                status=status,
                duration_ms=duration_ms,
                result_summary=capped,
                error=error,
                plan_item_id=plan_item_id,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def step_completed(self, *, step_number: int, tool_call_count: int, duration_ms: int) -> None:
        await self._emit(
            ev.StepCompleted(
                type="step_completed",
                step_number=step_number,
                tool_call_count=tool_call_count,
                duration_ms=duration_ms,
                **self._envelope(),
            )
        )
        self._current_step_id = None

    async def run_limit_reached(self, *, reason: str) -> None:
        await self._emit(
            ev.RunLimitReached(
                type="run_limit_reached",
                reason=reason,
                **self._envelope(step_id=None),
            )
        )

    async def run_interrupted(self, *, reason: str) -> None:
        await self._emit(
            ev.RunInterrupted(
                type="run_interrupted",
                reason=reason,
                **self._envelope(step_id=None),
            )
        )

    async def run_failed(self, *, error_code: str, message: str) -> None:
        await self._emit(
            ev.RunFailed(
                type="run_failed",
                error_code=error_code,
                message=message,
                **self._envelope(step_id=None),
            )
        )

    async def run_completed(self, *, total_steps: int, total_tool_calls: int, finish_reason: str) -> None:
        await self._emit(
            ev.RunCompleted(
                type="run_completed",
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                finish_reason=finish_reason,
                **self._envelope(step_id=None),
            )
        )

    async def llm_round_started(
        self,
        *,
        llm_round_id: str,
        round_index: int,
        model: str,
        provider: str,
        parent_step_id: str | None = None,
        system_prompt_fingerprint: str | None = None,
    ) -> None:
        await self._emit(
            ev.LLMRoundStarted(
                type="llm_round_started",
                llm_round_id=llm_round_id,
                round_index=round_index,
                model=model,
                provider=provider,
                system_prompt_fingerprint=system_prompt_fingerprint,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def llm_round_first_output_delta(
        self,
        *,
        llm_round_id: str,
        delta_kind: str,
        ttft_ms: int,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.LLMRoundFirstOutputDelta(
                type="llm_round_first_output_delta",
                llm_round_id=llm_round_id,
                delta_kind=delta_kind,
                ttft_ms=ttft_ms,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def llm_round_completed(
        self,
        *,
        llm_round_id: str,
        finish_reason: str | None,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cache_read_tokens: int | None,
        cache_write_tokens: int | None,
        reasoning_tokens: int | None = None,
        ttft_ms: int | None,
        duration_ms: int,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.LLMRoundCompleted(
                type="llm_round_completed",
                llm_round_id=llm_round_id,
                status="success",
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                reasoning_tokens=reasoning_tokens,
                ttft_ms=ttft_ms,
                duration_ms=duration_ms,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def llm_round_failed(
        self,
        *,
        llm_round_id: str,
        error_code: str | None,
        message: str | None,
        parent_step_id: str | None = None,
    ) -> None:
        safe_error_code, safe_message = self._controlled_error(error_code, _LLM_ERROR_SUMMARIES)
        await self._emit(
            ev.LLMRoundFailed(
                type="llm_round_failed",
                llm_round_id=llm_round_id,
                status="failed",
                error_code=safe_error_code,
                message=safe_message,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def llm_round_cancelled(
        self,
        *,
        llm_round_id: str,
        reason: str,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.LLMRoundCancelled(
                type="llm_round_cancelled",
                llm_round_id=llm_round_id,
                status="cancelled",
                reason=reason,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def retrieval_started(
        self,
        *,
        retrieval_id: str,
        query_summary: str | None,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.RetrievalStarted(
                type="retrieval_started",
                retrieval_id=retrieval_id,
                query_summary=self._opaque_retrieval_summary(query_summary),
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def retrieval_completed(
        self,
        *,
        retrieval_id: str,
        document_count: int,
        duration_ms: int,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.RetrievalCompleted(
                type="retrieval_completed",
                retrieval_id=retrieval_id,
                status="success",
                document_count=document_count,
                duration_ms=duration_ms,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def retrieval_failed(
        self,
        *,
        retrieval_id: str,
        error_code: str | None,
        message: str | None,
        parent_step_id: str | None = None,
    ) -> None:
        safe_error_code, safe_message = self._controlled_error(error_code, _RETRIEVAL_ERROR_SUMMARIES)
        await self._emit(
            ev.RetrievalFailed(
                type="retrieval_failed",
                retrieval_id=retrieval_id,
                status="failed",
                error_code=safe_error_code,
                message=safe_message,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def retrieval_cancelled(
        self,
        *,
        retrieval_id: str,
        reason: str,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.RetrievalCancelled(
                type="retrieval_cancelled",
                retrieval_id=retrieval_id,
                status="cancelled",
                reason=reason,
                **self._envelope(parent_step_id=parent_step_id),
            )
        )

    async def tool_attempt_started(
        self,
        *,
        tool_attempt_id: str,
        tool_call_id: str,
        tool_name: str,
        attempt_index: int,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.ToolAttemptStarted(
                type="tool_attempt_started",
                tool_attempt_id=tool_attempt_id,
                tool_name=tool_name,
                attempt_index=attempt_index,
                **self._envelope(tool_call_id=tool_call_id, parent_step_id=parent_step_id),
            )
        )

    async def tool_attempt_completed(
        self,
        *,
        tool_attempt_id: str,
        status: str,
        error_code: str | None,
        duration_ms: int,
        tool_call_id: str | None = None,
        parent_step_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.ToolAttemptCompleted(
                type="tool_attempt_completed",
                tool_attempt_id=tool_attempt_id,
                status=status,
                error_code=error_code,
                duration_ms=duration_ms,
                **self._envelope(tool_call_id=tool_call_id, parent_step_id=parent_step_id),
            )
        )

    async def suggested_questions_pending(self, *, message_id: str, revision: int) -> None:
        await self._emit(
            ev.SuggestedQuestionsPending(
                type="suggested_questions_pending",
                protocol_version=2,
                message_id=message_id,
                revision=revision,
                status="pending",
                **self._envelope(step_id=None),
            )
        )

    async def run_progress_updated(
        self,
        *,
        phase: str,
        label: str,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        completed_tool_calls: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        await self._emit(
            ev.RunProgressUpdated(
                type="run_progress_updated",
                protocol_version=2,
                phase=phase,
                label=label,
                completed_steps=completed_steps,
                total_steps=total_steps,
                completed_tool_calls=completed_tool_calls,
                max_tool_calls=max_tool_calls,
                **self._envelope(step_id=None),
            )
        )

    async def plan_snapshot(
        self,
        *,
        plan_id: str,
        revision: int,
        items: list[dict[str, Any]],
        mode: str = "auto",
        source: str = "observed",
        reason: str = "legacy_observed",
    ) -> None:
        await self._emit(
            ev.PlanSnapshot(
                type="plan_snapshot",
                protocol_version=2,
                plan_id=plan_id,
                mode=mode,
                source=source,
                revision=revision,
                reason=reason,
                items=items,
                **self._envelope(step_id=None),
            )
        )

    async def plan_step_updated(
        self,
        *,
        plan_id: str,
        revision: int,
        item: dict[str, Any],
        mode: str = "auto",
        source: str = "observed",
        reason: str = "legacy_observed",
    ) -> None:
        await self._emit(
            ev.PlanStepUpdated(
                type="plan_step_updated",
                protocol_version=2,
                plan_id=plan_id,
                mode=mode,
                source=source,
                revision=revision,
                reason=reason,
                item=item,
                **self._envelope(),
            )
        )

    async def tool_result_digest(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        status: str,
        title: str,
        summary: str,
        key_findings: list[str] | None = None,
        source_refs: list[str] | None = None,
        truncated: bool = False,
        repair_state: str | None = None,
        repair_id: str | None = None,
        plan_item_id: str | None = None,
    ) -> None:
        await self._emit(
            ev.ToolResultDigest(
                type="tool_result_digest",
                protocol_version=2,
                tool_name=tool_name,
                status=status,
                title=title,
                summary=summary,
                key_findings=key_findings or [],
                source_refs=source_refs or [],
                truncated=truncated,
                repair_state=repair_state,
                repair_id=repair_id,
                plan_item_id=plan_item_id,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def evidence_item_upserted(
        self,
        *,
        tool_call_id: str | None = None,
        evidence: dict[str, Any],
    ) -> None:
        await self._emit(
            ev.EvidenceItemUpserted(
                type="evidence_item_upserted",
                protocol_version=2,
                evidence=evidence,
                **self._envelope(tool_call_id=tool_call_id),
            )
        )

    async def content_block_upserted(self, *, tool_call_id: str, content_block: Any) -> None:
        await self._emit(
            ev.ContentBlockUpserted(
                type="content_block_upserted",
                protocol_version=2,
                content_block=content_block,
                **self._envelope(tool_call_id=tool_call_id),
            ),
            max_payload_bytes=65_536,
        )

    async def content_block_discarded(self, *, block_id: str) -> None:
        await self._emit(
            ev.ContentBlockDiscarded(
                type="content_block_discarded",
                protocol_version=2,
                block_id=block_id,
                **self._envelope(),
            )
        )

    async def system_prompt_prepared(
        self,
        *,
        status: str,
        source: str,
        template_version: str,
        section_ids: list[str],
        duration_ms: int,
        fingerprint: str | None = None,
        char_count: int | None = None,
        detail_status: str | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> None:
        if self._message_id is None:
            raise RuntimeError("system_prompt_prepared 必须在 run_started 之后发送")
        # 错误原文可能含偏好或模板内容；事件只发送固定文案。
        safe_error = "assembly_failed" if status == "failed" else None
        await self._emit(
            ev.SystemPromptPrepared(
                type="system_prompt_prepared",
                protocol_version=2,
                status=status,
                source=source,
                template_version=template_version,
                section_ids=section_ids,
                fingerprint=fingerprint,
                char_count=char_count,
                detail_status=detail_status,
                duration_ms=duration_ms,
                error_code=safe_error,
                message="系统提示词组装失败，请稍后重试。" if safe_error else None,
                **self._envelope(step_id=None),
            )
        )

    async def context_status_updated(
        self,
        *,
        phase: str,
        status: str,
        round_index: int,
        window_tokens: int | None,
        estimated_tokens_before: int | None,
        estimated_tokens_after: int | None,
        actual_prompt_tokens: int | None,
        removed_turns: int,
        removed_messages: int,
        removed_tool_transactions: int,
    ) -> None:
        if self._message_id is None:
            raise RuntimeError("context_status_updated 必须在 run_started 之后发送")
        await self._emit(
            ev.ContextStatusUpdated(
                type="context_status_updated",
                protocol_version=2,
                message_id=self._message_id,
                phase=phase,
                status=status,
                round_index=round_index,
                window_tokens=window_tokens,
                estimated_tokens_before=estimated_tokens_before,
                estimated_tokens_after=estimated_tokens_after,
                actual_prompt_tokens=actual_prompt_tokens,
                removed_turns=removed_turns,
                removed_messages=removed_messages,
                removed_tool_transactions=removed_tool_transactions,
                **self._envelope(),
            )
        )

    async def context_required(
        self,
        *,
        request_id: str,
        context_type: str,
        purpose: str,
        reason: str,
        expires_at: float,
    ) -> None:
        await self._emit(
            ev.ContextRequired(
                type="context_required",
                protocol_version=2,
                context_type=context_type,
                request_id=request_id,
                purpose=purpose,
                reason=reason,
                expires_at=expires_at,
                **self._envelope(),
            )
        )

    async def context_result(
        self,
        *,
        request_id: str,
        context_type: str,
        status: str,
    ) -> None:
        await self._emit(
            ev.ContextResult(
                type="context_result",
                protocol_version=2,
                context_type=context_type,
                request_id=request_id,
                status=status,
                **self._envelope(),
            )
        )
