"""StreamHandler 类与 generate_to_redis 编排。

spec §4.1。本模块只负责 agent loop 的控制流编排，所有"做事"的逻辑
（LLM 流消费 / 工具执行 / 落库 / SSE 编码）都委派给同子包内的兄弟模块。
"""

import time
from typing import Optional

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.services.agent import session_cache
from app.services.mcp.agent_tools import load_mcp_agent_tools
from app.services.stream.agent_loop_driver import run_agent_loop
from app.services.stream.agent_loop_execution import build_agent_loop_execution
from app.services.stream.agent_loop_lifecycle import (
    run_agent_loop_lifecycle,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits
from app.services.stream.agent_loop_request_prep import (
    build_agent_loop_call_config,
    prepare_agent_loop_messages,
)
from app.services.stream.agent_loop_run_completion import (
    finalize_cancelled_run,
    finalize_completed_run,
    finalize_failed_run,
    finalize_superseded_run,
    write_fallback_run_error,
)
from app.services.stream.agent_loop_wiring import (
    AgentLoopLifecycleCall,
    AgentLoopRunInput,
    AgentLoopWiringDependencies,
    build_agent_loop_lifecycle_call,
)
from app.services.stream.agent_round import run_agent_round
from app.services.stream.limit_summary import run_limit_summary_step
from app.services.stream.llm_stream import llm_call_with_retry, stream_round
from app.services.stream.persistence import persist_message
from app.services.stream.run_finalizer import (
    complete_agent_run,
    fail_agent_run,
    interrupt_agent_run,
    start_agent_run,
    write_fallback_error_status,
)
from app.services.stream.step_lifecycle import complete_agent_step, start_agent_step
from app.services.stream.tool_executor import AgentEventRedisWriter, execute_tools_parallel
from app.services.stream.tool_round import handle_tool_calls_round
from app.services.stream_state_service import (
    append_chunk,
    finalize_stream,
)

# Agent Loop 限制
AGENT_MAX_STEPS = 8  # LLM 调用轮次上限
AGENT_MAX_TOOL_CALLS = 20  # 工具执行总次数上限
AGENT_TOTAL_TIMEOUT = 300  # 5 分钟硬超时


def _log_agent_round_summary(
    *,
    conversation_id: str,
    run_id: str,
    step_number: int,
    model_id: str,
    provider: str,
    finish_reason: str,
    tool_calls_count: int,
    reasoning_buf: str,
    content_buf: str,
) -> None:
    logger.info(
        "AGENT_ROUND_SUMMARY "
        f"conv_id={conversation_id} run_id={run_id} step={step_number} "
        f"model_id={model_id} provider={provider} finish_reason={finish_reason} "
        f"tool_calls={tool_calls_count} reasoning_chars={len(reasoning_buf)} "
        f"content_chars={len(content_buf)}"
    )


def _agent_loop_limits() -> AgentLoopLimits:
    return AgentLoopLimits(
        max_steps=AGENT_MAX_STEPS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        total_timeout_s=AGENT_TOTAL_TIMEOUT,
    )


def _agent_loop_wiring_dependencies() -> AgentLoopWiringDependencies:
    return AgentLoopWiringDependencies(
        build_call_config_fn=build_agent_loop_call_config,
        build_execution_fn=build_agent_loop_execution,
        session_cache=session_cache,
        redis_writer_factory=AgentEventRedisWriter,
        start_step_fn=start_agent_step,
        complete_step_fn=complete_agent_step,
        run_round_fn=run_agent_round,
        handle_tool_calls_round_fn=handle_tool_calls_round,
        run_limit_summary_step_fn=run_limit_summary_step,
        llm_call_fn=llm_call_with_retry,
        stream_round_fn=stream_round,
        execute_tools_fn=execute_tools_parallel,
        persist_message_fn=persist_message,
        log_round_summary_fn=_log_agent_round_summary,
        clock=time.time,
        append_chunk_fn=append_chunk,
        start_agent_run_fn=start_agent_run,
        prepare_messages_fn=prepare_agent_loop_messages,
        run_agent_loop_fn=run_agent_loop,
        finalize_completed_run_fn=finalize_completed_run,
        finalize_superseded_run_fn=finalize_superseded_run,
        finalize_cancelled_run_fn=finalize_cancelled_run,
        finalize_failed_run_fn=finalize_failed_run,
        write_fallback_run_error_fn=write_fallback_run_error,
        complete_agent_run_fn=complete_agent_run,
        interrupt_agent_run_fn=interrupt_agent_run,
        fail_agent_run_fn=fail_agent_run,
        finalize_stream_fn=finalize_stream,
        write_fallback_error_status_fn=write_fallback_error_status,
        info_fn=logger.info,
        error_fn=logger.error,
        warning_fn=logger.warning,
        load_dynamic_tools_fn=load_mcp_agent_tools,
    )


async def _run_agent_loop_lifecycle_call(lifecycle_call: AgentLoopLifecycleCall) -> None:
    await run_agent_loop_lifecycle(
        request=lifecycle_call.request,
        execution=lifecycle_call.execution,
        dependencies=lifecycle_call.dependencies,
    )


class StreamHandler:
    """流式处理器"""

    async def generate_to_redis(
        self,
        conversation_id: str,
        user_id: str,
        model_id: str,
        litellm_model: str,
        litellm_kwargs: dict,
        provider: str,
        raw_messages: list,
        has_vision: bool,
        file_ids: Optional[list],
        original_message: str,
        assistant_message_id: str,
        task_id: str,
        assistant_message_sequence: int | None = None,
        options: Optional[dict] = None,
        capabilities: Optional[dict] = None,
        trace_id: Optional[str] = None,
        initial_content_blocks: Optional[list] = None,
        extra_system_prompts: Optional[list[str]] = None,
        preprocess_user_input: bool = True,
        limits: Optional[AgentLoopLimits] = None,
    ) -> None:
        """后台任务：调用 LLM，chunk 写入 Redis Stream，并由 agent loop 完成落库。"""
        run_input = AgentLoopRunInput(
            conversation_id=conversation_id,
            user_id=user_id,
            model_id=model_id,
            litellm_model=litellm_model,
            litellm_kwargs=litellm_kwargs,
            provider=provider,
            raw_messages=raw_messages,
            has_vision=has_vision,
            file_ids=file_ids,
            original_message=original_message,
            assistant_message_id=assistant_message_id,
            assistant_message_sequence=assistant_message_sequence,
            task_id=task_id,
            options=options,
            capabilities=capabilities,
            trace_id=trace_id,
            initial_content_blocks=initial_content_blocks,
            extra_system_prompts=extra_system_prompts,
            preprocess_user_input=preprocess_user_input,
        )

        db = SessionLocal()
        try:
            lifecycle_call = build_agent_loop_lifecycle_call(
                run_input=run_input,
                db=db,
                limits=limits or _agent_loop_limits(),
                dependencies=_agent_loop_wiring_dependencies(),
            )
            await _run_agent_loop_lifecycle_call(lifecycle_call)
        finally:
            db.close()
