"""Agent loop 单次运行生命周期 facade。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.ai.prompts.system_prompt import TEMPLATE_VERSION, SystemPromptAssemblyError
from app.core.logger import app_logger as logger
from app.core.prompt_bundle import get_active_prompt_bundle_revision
from app.schemas.chat import TextBlock
from app.schemas.response import ApiException
from app.schemas.trajectory import TrajectoryCapabilityResolution
from app.services.agent.session_cache import write_system_prompt_snapshot
from app.services.agent_strategy_config import get_agent_strategy_config
from app.services.knowledge.chat_grounding import (
    KnowledgeGroundingStreamError,
    inject_knowledge_grounding_messages,
    max_explicit_citation_index,
    prepare_knowledge_grounding,
    to_stream_grounding_error,
)
from app.services.stream.agent_loop_execution import AgentLoopExecutionContext
from app.services.stream.agent_loop_outcome import AgentLoopExit
from app.services.stream.agent_loop_policy import AgentLoopLimits, map_run_terminal_state
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig
from app.services.stream.research_evidence import assign_missing_source_reference_metadata
from app.services.stream.run_capability_router import serialize_capability_resolution
from app.services.stream_state_service import StreamOwnershipLostError

AsyncFn = Callable[..., Awaitable[Any]]
PersistMessageFn = Callable[..., Any]
LogFn = Callable[[str], None]
TRAJECTORY_BARRIER_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class AgentLoopLifecycleRequest:
    raw_messages: list
    has_vision: bool
    file_ids: list | None
    original_message: str
    call_config: AgentLoopCallConfig
    limits: AgentLoopLimits
    initial_content_blocks: list[Any] = field(default_factory=list)
    extra_system_prompts: list[str] = field(default_factory=list)
    preprocess_user_input: bool = True
    knowledge_base_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentLoopLifecycleDependencies:
    append_chunk_fn: AsyncFn
    start_agent_run_fn: AsyncFn
    prepare_messages_fn: AsyncFn
    run_agent_loop_fn: AsyncFn
    finalize_completed_run_fn: AsyncFn
    finalize_superseded_run_fn: AsyncFn
    finalize_cancelled_run_fn: AsyncFn
    finalize_failed_run_fn: AsyncFn
    write_fallback_run_error_fn: AsyncFn
    persist_message_fn: PersistMessageFn
    complete_agent_run_fn: AsyncFn
    interrupt_agent_run_fn: AsyncFn
    fail_agent_run_fn: AsyncFn
    finalize_stream_fn: AsyncFn
    write_fallback_error_status_fn: AsyncFn
    info_fn: LogFn
    error_fn: LogFn
    warning_fn: LogFn
    claim_suggested_questions_fn: Callable[..., Any] | None = None
    generate_suggested_questions_fn: Callable[..., Any] | None = None
    fail_suggested_questions_fn: Callable[..., Any] | None = None
    write_system_prompt_snapshot_fn: AsyncFn = write_system_prompt_snapshot


async def run_agent_loop_lifecycle(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    primary_error: BaseException | None = None
    try:
        await _run_success_path(request=request, execution=execution, dependencies=dependencies)
    except asyncio.CancelledError as error:
        primary_error = error
        if not execution.state.superseded_terminal_decided:
            await _finalize_cancelled(execution=execution, dependencies=dependencies)
        raise
    except StreamOwnershipLostError:
        # stop 接口或后续请求已经原子接管 Redis 终态时，后台任务可能先观察到
        # 写入权失效，再收到 asyncio cancellation。这属于正常中断，不应记为生成失败。
        if not execution.state.superseded_terminal_decided:
            await _finalize_cancelled(execution=execution, dependencies=dependencies)
    except Exception as error:
        primary_error = error
        if not execution.state.superseded_terminal_decided:
            await _finalize_failed(error=error, execution=execution, dependencies=dependencies)
        raise
    finally:
        fallback_error: BaseException | None = None
        try:
            await _write_fallback(execution=execution, dependencies=dependencies)
        except BaseException as error:
            fallback_error = error
            raise
        finally:
            cancelled_during_barrier = await commit_trajectory_barrier(
                execution=execution,
                warning_fn=dependencies.warning_fn,
            )
            if cancelled_during_barrier and primary_error is None and fallback_error is None:
                raise asyncio.CancelledError


async def commit_trajectory_barrier(
    *,
    execution: AgentLoopExecutionContext,
    warning_fn: LogFn,
) -> bool:
    """幂等执行 emitter seal 与 Recorder finalize，并保留调用方取消语义。"""
    state = execution.trajectory_barrier_state
    if state.started:
        return False
    state.started = True

    async def _commit() -> None:
        last_sequence = await execution.emitter.seal_and_get_last_sequence()
        await execution.trajectory_recorder.finalize(last_sequence)

    barrier_task = asyncio.create_task(_commit())
    cancelled, error, timed_out = await _wait_for_trajectory_barrier(barrier_task)
    if timed_out:
        barrier_task.add_done_callback(_consume_trajectory_barrier_result)
        warning_fn(f"轨迹完整性提交屏障超时: run_id={execution.run_id}")
    elif error is not None:
        warning_fn(f"轨迹完整性提交屏障失败: run_id={execution.run_id}, error_type={type(error).__name__}")
    return cancelled


async def _wait_for_trajectory_barrier(
    barrier_task: asyncio.Task[None],
) -> tuple[bool, BaseException | None, bool]:
    """在绝对 deadline 内等待；外层二次取消不取消底层 barrier。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TRAJECTORY_BARRIER_TIMEOUT_SECONDS
    cancelled = False
    while not barrier_task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return cancelled, None, True
        try:
            done, _pending = await asyncio.wait({barrier_task}, timeout=remaining)
        except asyncio.CancelledError:
            cancelled = True
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            continue
        if not done:
            return cancelled, None, True

    try:
        barrier_task.result()
    except BaseException as error:  # barrier 是辅助路径，任何异常都只降级并记录类型
        return cancelled, error, False
    return cancelled, None, False


def _consume_trajectory_barrier_result(barrier_task: asyncio.Task[None]) -> None:
    """消费外层超时后的迟到异常，避免 never-retrieved 告警。"""
    try:
        barrier_task.exception()
    except asyncio.CancelledError:
        pass


async def _run_success_path(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await _start_run(request=request, execution=execution, dependencies=dependencies)
    grounding = await _prepare_knowledge_grounding(request=request, execution=execution)
    if grounding is not None:
        execution.state.content_blocks.append(grounding.evidence_block)
        await execution.emitter.content_block_upserted(
            tool_call_id="knowledge_retrieval",
            content_block=grounding.evidence_block,
        )
        if grounding.no_evidence:
            answer = grounding.deterministic_answer or "未在所选知识库中找到足够依据"
            block_id = f"blk_knowledge_empty_{execution.run_id[:12]}"
            await execution.emitter.run_progress_updated(
                phase="answering",
                label="未找到足够依据",
            )
            await dependencies.append_chunk_fn(
                execution.completion_context.conversation_id,
                "answering",
                answer,
                block_id,
                task_id=execution.completion_context.task_id,
                run_id=execution.run_id,
            )
            execution.state.content_blocks.append(TextBlock(type="text", id=block_id, text=answer))
            await _finalize_completed(
                execution=execution,
                dependencies=dependencies,
                generate_suggestions=False,
            )
            return
    prepared_messages = await _prepare_messages(request=request, execution=execution, dependencies=dependencies)
    execution.state.content_blocks.extend(request.initial_content_blocks)
    execution.state.content_blocks.extend(prepared_messages.initial_content_blocks)
    if grounding is not None:
        prepared_messages.messages[:] = inject_knowledge_grounding_messages(prepared_messages.messages, grounding)
        await execution.emitter.run_progress_updated(
            phase="synthesizing",
            label="正在基于知识库整理回答",
        )
    configure_research_state(
        state=execution.state,
        call_config=request.call_config,
        file_ids=request.file_ids,
        content_blocks=request.initial_content_blocks,
        allow_read_success=False,
    )
    configure_research_state(
        state=execution.state,
        call_config=request.call_config,
        file_ids=request.file_ids,
        content_blocks=prepared_messages.initial_content_blocks,
        allow_read_success=True,
    )

    loop_outcome = await dependencies.run_agent_loop_fn(
        db=execution.completion_context.db,
        messages=prepared_messages.messages,
        state=execution.state,
        runtime=execution.runtime,
    )
    if loop_outcome.exit == AgentLoopExit.SUPERSEDED:
        await _finalize_superseded(
            error_msg=loop_outcome.error_msg,
            execution=execution,
            dependencies=dependencies,
        )
        return

    await _finalize_completed(
        execution=execution,
        dependencies=dependencies,
        generate_suggestions=grounding is None,
    )


async def _start_run(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    context = execution.completion_context
    await dependencies.append_chunk_fn(
        context.conversation_id,
        "preparing",
        "",
        "",
        task_id=context.task_id,
    )
    await dependencies.start_agent_run_fn(
        emitter=execution.emitter,
        session_cache=context.session_cache,
        run_id=execution.run_id,
        conversation_id=context.conversation_id,
        user_id=execution.runtime.user_id,
        model_id=context.model_id,
        provider=execution.runtime.provider,
        message_id=context.assistant_message_id,
        turn_message_id=execution.turn_message_id,
        previous_run_id=execution.previous_run_id,
        run_attempt_kind=execution.run_attempt_kind,
        tools=request.call_config.announced_tools,
        config=_run_config(request.limits, request.call_config),
    )


def configure_research_state(
    *,
    state: Any,
    call_config: AgentLoopCallConfig,
    file_ids: list | None,
    content_blocks: list[Any],
    allow_read_success: bool = True,
) -> None:
    if getattr(call_config, "task_mode", "standard") != "deep_research":
        return
    state.configure_research_mode(network_required=True)
    state.plan_coordinator.configure_initial_tool_requirements(
        {
            "web_search": 1,
            "url_read": 2,
        }
    )
    research_blocks = [
        block
        for block in content_blocks
        if getattr(block, "type", None) in {"search", "url_read"}
        or (isinstance(block, dict) and block.get("type") in {"search", "url_read"})
    ]
    assign_missing_source_reference_metadata(research_blocks)
    state.record_research_content_blocks(
        research_blocks,
        allow_read_success=allow_read_success,
    )


async def _prepare_messages(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> Any:
    try:
        prepared = await dependencies.prepare_messages_fn(
            db=execution.completion_context.db,
            user_id=execution.runtime.user_id,
            conversation_id=execution.completion_context.conversation_id,
            raw_messages=request.raw_messages,
            has_vision=request.has_vision,
            file_ids=request.file_ids,
            original_message=request.original_message,
            call_config=request.call_config,
            extra_system_prompts=request.extra_system_prompts,
            preprocess_user_input=request.preprocess_user_input,
        )
    except SystemPromptAssemblyError as error:
        try:
            await execution.emitter.system_prompt_prepared(**error.metadata)
        except Exception:
            # 保留组装原始失败，不让诊断写入异常覆盖主错误。
            dependencies.warning_fn("系统提示词失败事件写入失败")
        raise
    metadata = getattr(prepared, "prompt_assembly", None)
    if metadata is not None:
        metadata = dict(metadata)
        snapshot = getattr(prepared, "prompt_snapshot", None)
        if snapshot is not None:
            try:
                await dependencies.write_system_prompt_snapshot_fn(
                    run_id=execution.run_id,
                    conversation_id=execution.completion_context.conversation_id,
                    user_id=execution.runtime.user_id,
                    snapshot=snapshot,
                )
                metadata["detail_status"] = "available"
            except Exception as error:
                # 辅助正文失败不终止生成，也不将异常中的提示词写入日志。
                metadata["detail_status"] = "degraded"
                dependencies.warning_fn(f"系统提示词正文保存失败: error_type={type(error).__name__}")
        await execution.emitter.system_prompt_prepared(**metadata)
    return prepared


async def _prepare_knowledge_grounding(
    *,
    request: AgentLoopLifecycleRequest,
    execution: AgentLoopExecutionContext,
) -> Any | None:
    if not request.knowledge_base_ids:
        return None
    await execution.emitter.run_progress_updated(
        phase="researching",
        label="正在检索所选知识库",
    )
    retrieval_id = str(uuid.uuid4())
    started_at = time.monotonic()
    await execution.emitter.retrieval_started(
        retrieval_id=retrieval_id,
        query_summary=request.original_message,
        parent_step_id=None,
    )
    try:
        grounding = await prepare_knowledge_grounding(
            db=execution.completion_context.db,
            user_id=execution.runtime.user_id,
            query=request.original_message,
            knowledge_base_ids=request.knowledge_base_ids,
            citation_start=max_explicit_citation_index(request.initial_content_blocks) + 1,
        )
    except asyncio.CancelledError:
        await _emit_retrieval_terminal_preserving_primary(
            execution.emitter.retrieval_cancelled,
            retrieval_id=retrieval_id,
            reason="shutdown",
            parent_step_id=None,
        )
        raise
    except ApiException as error:
        mapped_error = to_stream_grounding_error(error)
        await _emit_retrieval_terminal_preserving_primary(
            execution.emitter.retrieval_failed,
            retrieval_id=retrieval_id,
            error_code=mapped_error.error_code,
            message=None,
            parent_step_id=None,
        )
        raise mapped_error from error
    except KnowledgeGroundingStreamError as error:
        await _emit_retrieval_terminal_preserving_primary(
            execution.emitter.retrieval_failed,
            retrieval_id=retrieval_id,
            error_code=error.error_code,
            message=None,
            parent_step_id=None,
        )
        raise
    except Exception as error:
        mapped_error = KnowledgeGroundingStreamError("knowledge_retrieval_unavailable")
        await _emit_retrieval_terminal_preserving_primary(
            execution.emitter.retrieval_failed,
            retrieval_id=retrieval_id,
            error_code=mapped_error.error_code,
            message=None,
            parent_step_id=None,
        )
        raise mapped_error from error
    await execution.emitter.retrieval_completed(
        retrieval_id=retrieval_id,
        document_count=grounding.evidence_block.source_count,
        duration_ms=max(0, int(round((time.monotonic() - started_at) * 1000))),
        parent_step_id=None,
    )
    return grounding


async def _emit_retrieval_terminal_preserving_primary(emit: AsyncFn, **kwargs: Any) -> None:
    try:
        await emit(**kwargs)
    except BaseException as secondary:
        logger.warning(
            "知识检索生命周期收尾失败，保留主异常: error_type=%s",
            type(secondary).__name__,
        )


async def _finalize_completed(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
    generate_suggestions: bool = True,
) -> None:
    terminal_state = map_run_terminal_state(
        unknown_terminated=execution.state.unknown_terminated,
        limit_reason=execution.state.limit_reason,
    )
    await dependencies.finalize_completed_run_fn(
        context=execution.completion_context,
        terminal_state=terminal_state,
        persist_message_fn=dependencies.persist_message_fn,
        complete_agent_run_fn=dependencies.complete_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        claim_suggested_questions_fn=(dependencies.claim_suggested_questions_fn if generate_suggestions else None),
        generate_suggested_questions_fn=(
            dependencies.generate_suggested_questions_fn if generate_suggestions else None
        ),
        fail_suggested_questions_fn=(dependencies.fail_suggested_questions_fn if generate_suggestions else None),
        warning_fn=dependencies.warning_fn,
    )


async def _finalize_superseded(
    *,
    error_msg: str | None,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await dependencies.finalize_superseded_run_fn(
        context=execution.completion_context,
        error_msg=error_msg,
        persist_message_fn=dependencies.persist_message_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
    )


async def _finalize_cancelled(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    dependencies.info_fn(f"Agent 任务被取消: conv_id={execution.completion_context.conversation_id}")
    await dependencies.finalize_cancelled_run_fn(
        context=execution.completion_context,
        persist_message_fn=dependencies.persist_message_fn,
        interrupt_agent_run_fn=dependencies.interrupt_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        warning_fn=dependencies.warning_fn,
    )


async def _finalize_failed(
    *,
    error: Exception,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    dependencies.error_fn(
        f"Agent 生成异常: conv_id={execution.completion_context.conversation_id}, error_type={type(error).__name__}"
    )
    await dependencies.finalize_failed_run_fn(
        context=execution.completion_context,
        error=error,
        persist_message_fn=dependencies.persist_message_fn,
        fail_agent_run_fn=dependencies.fail_agent_run_fn,
        finalize_stream_fn=dependencies.finalize_stream_fn,
        warning_fn=dependencies.warning_fn,
    )


async def _write_fallback(
    *,
    execution: AgentLoopExecutionContext,
    dependencies: AgentLoopLifecycleDependencies,
) -> None:
    await dependencies.write_fallback_run_error_fn(
        context=execution.completion_context,
        write_fallback_error_status_fn=dependencies.write_fallback_error_status_fn,
        warning_fn=dependencies.warning_fn,
    )


def _run_config(limits: AgentLoopLimits, call_config: AgentLoopCallConfig | None = None) -> dict:
    _strategy_config, strategy_meta = get_agent_strategy_config()
    runtime_config_versions = {
        "agent_strategy/default": strategy_meta.get("version", "code-default"),
    }
    prompt_revision = get_active_prompt_bundle_revision()
    if prompt_revision is not None:
        runtime_config_versions["prompt_bundle/fusion"] = prompt_revision
    config = {
        "max_steps": limits.max_steps,
        "max_tool_calls": limits.max_tool_calls,
        "timeout_s": limits.total_timeout_s,
        "plan_mode": getattr(call_config, "plan_mode", "auto"),
        "task_mode": getattr(call_config, "task_mode", "standard"),
        "network_profile": getattr(call_config, "network_profile", "standard"),
        "evidence_policy": getattr(call_config, "evidence_policy", "standard"),
        "runtime_config_versions": runtime_config_versions,
    }
    binding_fields = (
        "alias",
        "server_id",
        "remote_tool_name",
        "provider",
        "config_version",
        "tool_label",
        "definition_sha256",
    )
    bindings = []
    for binding in getattr(call_config, "tool_bindings", []) or []:
        if not isinstance(binding, dict):
            continue
        safe_binding = {field: binding[field] for field in binding_fields if field in binding}
        if safe_binding.get("alias"):
            bindings.append(safe_binding)
    if bindings:
        config["mcp_tool_bindings"] = bindings
    resolution = getattr(call_config, "capability_resolution", None)
    if resolution is not None:
        resolution_payload = serialize_capability_resolution(resolution)
        TrajectoryCapabilityResolution.model_validate(
            {
                **resolution_payload,
                "bundle_fingerprint": "sha256:" + "0" * 64,
            }
        )
        announced_tools = list(getattr(call_config, "announced_tools", []) or [])
        if resolution_payload["external_tool_names"] != announced_tools:
            raise ValueError("能力路由工具与 Run 公告工具不一致")
        fingerprint_input = {
            "router_version": resolution_payload["router_version"],
            "prompt_template_version": TEMPLATE_VERSION,
            "package_id": resolution_payload["package_id"],
            "external_tool_names": announced_tools,
            "effective_plan_mode": resolution_payload["effective_plan_mode"],
            "task_mode": config["task_mode"],
            "evidence_policy": config["evidence_policy"],
        }
        serialized_fingerprint_input = json.dumps(
            fingerprint_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        resolution_payload["bundle_fingerprint"] = "sha256:" + hashlib.sha256(serialized_fingerprint_input).hexdigest()
        config["capability_resolution"] = TrajectoryCapabilityResolution.model_validate(resolution_payload).model_dump()
    return config
