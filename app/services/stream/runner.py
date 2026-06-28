"""StreamHandler 类与 generate_to_redis 编排。

spec §4.1。本模块只负责 agent loop 的控制流编排，所有"做事"的逻辑
（LLM 流消费 / 工具执行 / 落库 / SSE 编码）都委派给同子包内的兄弟模块。
"""

import asyncio
import time
from typing import Optional

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.services.agent import session_cache
from app.services.stream.agent_loop_driver import AgentLoopExit, run_agent_loop
from app.services.stream.agent_loop_execution import (
    AgentLoopDependencies,
    AgentLoopExecutionRequest,
    build_agent_loop_execution,
)
from app.services.stream.agent_loop_policy import (
    AgentLoopLimits,
    map_run_terminal_state,
)
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
        options: Optional[dict] = None,
        capabilities: Optional[dict] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        后台任务：调用 LLM，chunk 写入 Redis Stream，完成后落库 PostgreSQL。
        支持 Agent Loop：LLM 可多轮调用工具，直到自行决定 stop。
        """
        if options is None:
            options = {}
        if capabilities is None:
            capabilities = {}

        call_config = build_agent_loop_call_config(
            provider=provider,
            options=options,
            capabilities=capabilities,
        )

        db = SessionLocal()

        limits = AgentLoopLimits(
            max_steps=AGENT_MAX_STEPS,
            max_tool_calls=AGENT_MAX_TOOL_CALLS,
            total_timeout_s=AGENT_TOTAL_TIMEOUT,
        )
        execution = build_agent_loop_execution(
            request=AgentLoopExecutionRequest(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                litellm_model=litellm_model,
                litellm_kwargs=litellm_kwargs,
                provider=provider,
                assistant_message_id=assistant_message_id,
                task_id=task_id,
                call_config=call_config,
                trace_id=trace_id,
            ),
            limits=limits,
            dependencies=AgentLoopDependencies(
                session_cache=session_cache,
                redis_writer=AgentEventRedisWriter(),
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
                warning_fn=logger.warning,
                clock=time.time,
            ),
        )

        try:
            await append_chunk(conversation_id, "preparing", "", "")

            await start_agent_run(
                emitter=execution.emitter,
                session_cache=session_cache,
                run_id=execution.run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                provider=provider,
                message_id=assistant_message_id,
                tools=call_config.announced_tools,
                config={
                    "max_steps": limits.max_steps,
                    "max_tool_calls": limits.max_tool_calls,
                    "timeout_s": limits.total_timeout_s,
                },
            )

            prepared_messages = await prepare_agent_loop_messages(
                db=db,
                user_id=user_id,
                raw_messages=raw_messages,
                has_vision=has_vision,
                file_ids=file_ids,
                original_message=original_message,
                call_config=call_config,
            )
            messages = prepared_messages.messages
            execution.state.content_blocks.extend(prepared_messages.initial_content_blocks)

            # ═══════════════════════════════════════
            # Agent Loop
            # ═══════════════════════════════════════

            loop_outcome = await run_agent_loop(
                db=db,
                messages=messages,
                state=execution.state,
                runtime=execution.runtime,
            )
            if loop_outcome.exit == AgentLoopExit.SUPERSEDED:
                await finalize_superseded_run(
                    context=execution.completion_context,
                    error_msg=loop_outcome.error_msg,
                    persist_message_fn=persist_message,
                    interrupt_agent_run_fn=interrupt_agent_run,
                    finalize_stream_fn=finalize_stream,
                )
                return

            # ═══════════════════════════════════════
            # 最终落库 + run_completed
            # ═══════════════════════════════════════
            terminal_state = map_run_terminal_state(
                unknown_terminated=execution.state.unknown_terminated,
                limit_reason=execution.state.limit_reason,
            )
            await finalize_completed_run(
                context=execution.completion_context,
                terminal_state=terminal_state,
                persist_message_fn=persist_message,
                complete_agent_run_fn=complete_agent_run,
                finalize_stream_fn=finalize_stream,
            )

        except asyncio.CancelledError:
            logger.info(f"Agent 任务被取消: conv_id={conversation_id}")
            await finalize_cancelled_run(
                context=execution.completion_context,
                persist_message_fn=persist_message,
                interrupt_agent_run_fn=interrupt_agent_run,
                finalize_stream_fn=finalize_stream,
                warning_fn=logger.warning,
            )
            raise

        except Exception as e:
            logger.error(f"Agent 生成异常: conv_id={conversation_id}, error={e}")
            await finalize_failed_run(
                context=execution.completion_context,
                error=e,
                persist_message_fn=persist_message,
                fail_agent_run_fn=fail_agent_run,
                finalize_stream_fn=finalize_stream,
                warning_fn=logger.warning,
            )
            # 完成协议层 + DB cache + SSE 收尾后 re-raise，让 background task scheduler 拿到失败信号；
            # 与 CancelledError 路径行为对齐（spec §5.3）。
            raise

        finally:
            # 兜底：极端路径（例如未匹配任何 except 又没走 try 终段）补一次终态
            await write_fallback_run_error(
                context=execution.completion_context,
                write_fallback_error_status_fn=write_fallback_error_status,
                warning_fn=logger.warning,
            )
            db.close()
