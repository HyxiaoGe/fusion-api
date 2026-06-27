"""StreamHandler 类与 generate_to_redis 编排。

spec §4.1。本模块只负责 agent loop 的控制流编排，所有"做事"的逻辑
（LLM 流消费 / 工具执行 / 落库 / SSE 编码）都委派给同子包内的兄弟模块。
"""

import asyncio
import time
import uuid
from typing import Optional

from app.ai.litellm_utils import merge_extra_body
from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.repositories import FileRepository
from app.schemas.chat import (
    TextBlock,
    ThinkingBlock,
    Usage,
)
from app.services.agent import session_cache
from app.services.agent.emitter import AgentEventEmitter
from app.services.chat.message_builder import (
    build_llm_messages,
    inject_file_content,
    is_image_file,
)
from app.services.stream.agent_loop_policy import (
    AgentLoopLimitReason,
    AgentLoopLimits,
    check_agent_loop_limit,
    map_run_terminal_state,
)
from app.services.stream.limit_summary import run_limit_summary_step
from app.services.stream.llm_stream import llm_call_with_retry, stream_round
from app.services.stream.network_budget import NetworkToolBudget
from app.services.stream.persistence import persist_message, preprocess_url_in_message
from app.services.stream.run_finalizer import (
    AgentRunStats,
    complete_agent_run,
    fail_agent_run,
    interrupt_agent_run,
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

_WEB_SEARCH_TOOL_CONTRACT_PROMPT = (
    "【工具调用一致性规则】\n"
    "如果你判断需要联网搜索，必须调用 web_search 工具，不能只在文字里说要搜索。\n"
    "不要在思考过程或最终回答中声称「我将搜索」「正在搜索」「让我搜索一下」「根据搜索结果」等，"
    "除非你已经实际调用工具并收到了工具结果。\n"
    "没有调用工具时，请直接基于已有知识回答，并明确避免暗示已经联网或即将联网。"
)


def _tool_names_from_call_kwargs(call_kwargs: dict) -> set[str]:
    names: set[str] = set()
    for tool in call_kwargs.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _inject_tool_usage_contract(messages: list[dict], call_kwargs: dict) -> list[dict]:
    """工具模式下补一条 system 约束，避免 reasoning 口头承诺搜索但不发 tool_call。"""
    if "web_search" not in _tool_names_from_call_kwargs(call_kwargs):
        return messages
    if any(msg.get("role") == "system" and "【工具调用一致性规则】" in str(msg.get("content", "")) for msg in messages):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    contract_msg = {"role": "system", "content": _WEB_SEARCH_TOOL_CONTRACT_PROMPT}
    return [*messages[:insert_at], contract_msg, *messages[insert_at:]]


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

    # 哪些 provider（底层 LiteLLM 路由识别出的）需要走 volcengine 的 disabled-thinking 兼容；
    # 其余靠 capabilities.deepThinking 推断是否开 reasoning。
    _VOLCENGINE_PROVIDERS = {"volcengine"}

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

        # use_reasoning 默认跟随 capabilities.deepThinking — 模型本身声明支持 thinking
        # 就开，不再按 provider 硬编码（refactor 后 qwen/doubao/xiaomi 的底层 provider
        # 都报告为 "openai"，硬编码集合会漏掉它们）。
        use_reasoning = options.get("use_reasoning")
        supports_thinking = bool(capabilities.get("deepThinking", False))
        should_use_reasoning = use_reasoning is True or (use_reasoning is None and supports_thinking)

        # 判断是否启用工具
        supports_fc = capabilities.get("functionCalling", False)
        call_kwargs = {}
        if supports_fc:
            from app.ai.tools import build_web_search_tool

            call_kwargs["tools"] = [build_web_search_tool()]
            call_kwargs["tool_choice"] = "auto"
            if should_use_reasoning and provider in self._VOLCENGINE_PROVIDERS:
                merge_extra_body(call_kwargs, {"thinking": {"type": "disabled"}})

        db = SessionLocal()

        # agent loop 状态
        content_blocks: list = []
        accumulated_usage = Usage(input_tokens=0, output_tokens=0)
        step = 0
        total_tool_calls = 0
        network_budget = NetworkToolBudget()
        finish_reason = "stop"

        # ─────── Agent 控制面：emitter + session_cache ───────
        # trace_id 必须非空（events.py 强约束），缺失时本地 UUID fallback
        run_id = trace_id or str(uuid.uuid4())
        run_start = time.time()
        emitter = AgentEventEmitter(
            run_id=run_id,
            trace_id=run_id,
            conversation_id=conversation_id,
            redis_writer=AgentEventRedisWriter(),
        )
        # 当前正在执行的 step_id（用于异常路径回填 step terminal 状态）
        current_step_id: Optional[str] = None
        # 标记 finally 路径已经写过终态 / run_completed 事件，避免重复
        terminal_emitted = False

        def _mark_current_step(step_id: str) -> None:
            nonlocal current_step_id
            current_step_id = step_id

        def _run_stats() -> AgentRunStats:
            return AgentRunStats(
                run_id=run_id,
                total_steps=step,
                total_tool_calls=total_tool_calls,
            )

        def _record_executed_tool_calls(tool_call_count: int) -> None:
            nonlocal total_tool_calls
            total_tool_calls += tool_call_count

        def _run_duration_ms() -> int:
            return int((time.time() - run_start) * 1000)

        try:
            await append_chunk(conversation_id, "preparing", "", "")

            # session_cache 行：必须传齐 user_id / model_id / provider（NOT NULL）
            await session_cache.write_session_started(
                run_id=run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                provider=provider,
                message_id=assistant_message_id,
            )

            # 构建 LLM 消息
            file_repo = FileRepository(db)
            from app.db.models import User as UserModel

            user_record = db.query(UserModel).filter(UserModel.id == user_id).first()
            user_system_prompt = user_record.system_prompt if user_record else None
            messages = await build_llm_messages(raw_messages, has_vision, file_repo, user_system_prompt)

            if file_ids:
                non_image_ids = [fid for fid in file_ids if not is_image_file(fid, file_repo)]
                if non_image_ids:
                    file_contents = file_repo.get_parsed_file_content(non_image_ids)
                    if file_contents:
                        messages = inject_file_content(messages, original_message, file_contents)

            # ── URL 自动检测预处理（路径 A，已抽到 services/stream/persistence）──
            url_read_block, url_context_msg, _auto_detected_url = await preprocess_url_in_message(
                original_message, supports_fc, call_kwargs
            )
            if url_context_msg:
                messages.insert(-1, url_context_msg)
            if url_read_block:
                content_blocks.append(url_read_block)

            messages = _inject_tool_usage_contract(messages, call_kwargs)

            # ═══════════════════════════════════════
            # Agent Loop
            # ═══════════════════════════════════════

            start_time = run_start

            # run_started 必须在 while 之前 emit（即使 0 step / 直接 stop 也算一个 run）
            agent_tools_announced: list[str] = []
            for _t in call_kwargs.get("tools", []) or []:
                fn = _t.get("function") if isinstance(_t, dict) else None
                if isinstance(fn, dict) and fn.get("name"):
                    agent_tools_announced.append(fn["name"])
            await emitter.run_started(
                message_id=assistant_message_id,
                model=model_id,
                tools=agent_tools_announced,
                config={
                    "max_steps": AGENT_MAX_STEPS,
                    "max_tool_calls": AGENT_MAX_TOOL_CALLS,
                    "timeout_s": AGENT_TOTAL_TIMEOUT,
                },
            )

            limit_reason: AgentLoopLimitReason | None = None  # 记录触顶原因，决定后续是否走强制总结
            unknown_terminated = False  # 雷点 3 修复：退化分支标记，决定 run_finish_reason

            while True:
                # ─── 三段触顶检查（顺序：timeout > max_steps > max_tool_calls）───
                limit_reason = check_agent_loop_limit(
                    elapsed_seconds=time.time() - start_time,
                    step=step,
                    total_tool_calls=total_tool_calls,
                    limits=AgentLoopLimits(
                        max_steps=AGENT_MAX_STEPS,
                        max_tool_calls=AGENT_MAX_TOOL_CALLS,
                        total_timeout_s=AGENT_TOTAL_TIMEOUT,
                    ),
                )
                if limit_reason is not None:
                    finish_reason = "timeout" if limit_reason == "timeout" else "tool_calls"
                    await emitter.run_limit_reached(reason=limit_reason)
                    break

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 每轮 LLM call 之前先开 step（spec §4.2 "1 step = 1 LLM round"）
                # 这样本轮 reasoning / answering / tool_call_* chunk 都能挂到正确的 step_id；
                # stop / cancelled / tool_calls 三路径都在分支末尾闭合本 step。
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                step += 1
                step_context = await start_agent_step(
                    emitter=emitter,
                    session_cache=session_cache,
                    run_id=run_id,
                    step_number=step,
                    clock=time.time,
                    on_step_started=_mark_current_step,
                )
                current_step_id = step_context.step_id
                thinking_block_id = step_context.thinking_block_id
                text_block_id = step_context.text_block_id

                # LiteLLM Proxy 自己管 health / 重试，这里不再追踪 provider/credential 健康
                response = await llm_call_with_retry(
                    litellm_model,
                    litellm_kwargs,
                    messages,
                    **call_kwargs,
                )

                reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data = await stream_round(
                    response,
                    conversation_id,
                    task_id,
                    should_use_reasoning,
                    thinking_block_id,
                    text_block_id,
                    run_id=run_id,
                    step_id=current_step_id,
                )
                _log_agent_round_summary(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    step_number=step,
                    model_id=model_id,
                    provider=provider,
                    finish_reason=finish_reason,
                    tool_calls_count=len(tool_calls_list),
                    reasoning_buf=reasoning_buf,
                    content_buf=content_buf,
                )

                # 累积 usage
                if usage_data:
                    accumulated_usage = Usage(
                        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
                        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
                    )

                # ── 情况 1: LLM 直接回答 ──
                if finish_reason == "stop":
                    if reasoning_buf:
                        content_blocks.append(
                            ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf)
                        )
                    if content_buf:
                        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                    # 闭合本 step：直接回答路径不调工具
                    await complete_agent_step(
                        context=step_context,
                        emitter=emitter,
                        session_cache=session_cache,
                        tool_names=[],
                        tool_call_count=0,
                        clock=time.time,
                    )
                    current_step_id = None
                    break

                # ── 情况 2: 被踢掉（superseded）──
                if finish_reason == "cancelled":
                    if reasoning_buf:
                        content_blocks.append(
                            ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf)
                        )
                    if content_buf:
                        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                    persist_message(
                        db,
                        assistant_message_id,
                        conversation_id,
                        model_id,
                        content_blocks,
                        accumulated_usage if accumulated_usage.input_tokens > 0 else None,
                    )
                    await interrupt_agent_run(
                        emitter=emitter,
                        session_cache=session_cache,
                        stats=_run_stats(),
                        duration_ms_factory=_run_duration_ms,
                        current_step_id=current_step_id,
                        reason="superseded",
                    )
                    terminal_emitted = True
                    await finalize_stream(conversation_id, success=False, error_msg="被新请求取代", task_id=task_id)
                    return

                # ── 情况 3: LLM 请求工具调用 ──
                if finish_reason == "tool_calls" and tool_calls_list:
                    # step / step_started / write_step_started 已在 while 顶部完成
                    await handle_tool_calls_round(
                        db=db,
                        assistant_message_id=assistant_message_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        model_id=model_id,
                        provider=provider,
                        content_blocks=content_blocks,
                        messages=messages,
                        tool_calls=tool_calls_list,
                        reasoning_buf=reasoning_buf,
                        should_use_reasoning=should_use_reasoning,
                        step_context=step_context,
                        step_number=step,
                        run_id=run_id,
                        emitter=emitter,
                        session_cache=session_cache,
                        network_budget=network_budget,
                        call_kwargs=call_kwargs,
                        persist_message_fn=persist_message,
                        execute_tools_fn=execute_tools_parallel,
                        complete_step_fn=complete_agent_step,
                        on_tools_executed=_record_executed_tool_calls,
                        clock=time.time,
                    )
                    current_step_id = None  # 离开 step 上下文

                    continue

                # 未知 finish_reason（含 tool_calls 但 list 为空的退化情况）→ 保留已收集内容，闭合本 step 后跳出
                if reasoning_buf:
                    content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                if content_buf:
                    content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                unknown_terminated = True
                await complete_agent_step(
                    context=step_context,
                    emitter=emitter,
                    session_cache=session_cache,
                    tool_names=[],
                    tool_call_count=0,
                    clock=time.time,
                )
                current_step_id = None
                break

            # ═══════════════════════════════════════
            # 触顶强制总结（timeout / max_steps / max_tool_calls）
            # ═══════════════════════════════════════
            if limit_reason is not None:
                # 触顶总结作为一个独立 step（不带 tools）
                step += 1
                summary_outcome = await run_limit_summary_step(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    run_id=run_id,
                    step_number=step,
                    model_id=model_id,
                    provider=provider,
                    litellm_model=litellm_model,
                    litellm_kwargs=litellm_kwargs,
                    messages=messages,
                    should_use_reasoning=should_use_reasoning,
                    content_blocks=content_blocks,
                    call_kwargs=call_kwargs,
                    accumulated_usage=accumulated_usage,
                    emitter=emitter,
                    session_cache=session_cache,
                    total_timeout_s=AGENT_TOTAL_TIMEOUT,
                    run_start=run_start,
                    start_step_fn=start_agent_step,
                    complete_step_fn=complete_agent_step,
                    llm_call_fn=llm_call_with_retry,
                    stream_round_fn=stream_round,
                    log_round_summary_fn=_log_agent_round_summary,
                    warning_fn=logger.warning,
                    clock=time.time,
                    on_step_started=_mark_current_step,
                )
                accumulated_usage = summary_outcome.accumulated_usage
                current_step_id = None

            # ═══════════════════════════════════════
            # 最终落库 + run_completed
            # ═══════════════════════════════════════
            final_usage = accumulated_usage if accumulated_usage.input_tokens > 0 else None
            persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, final_usage)

            terminal_state = map_run_terminal_state(
                unknown_terminated=unknown_terminated,
                limit_reason=limit_reason,
            )
            await complete_agent_run(
                emitter=emitter,
                session_cache=session_cache,
                stats=_run_stats(),
                duration_ms_factory=_run_duration_ms,
                session_status=terminal_state.session_status,
                finish_reason=terminal_state.run_finish_reason,
            )
            terminal_emitted = True

            await finalize_stream(conversation_id, success=True, task_id=task_id)

        except asyncio.CancelledError:
            logger.info(f"Agent 任务被取消: conv_id={conversation_id}")
            if content_blocks:
                persist_message(
                    db,
                    assistant_message_id,
                    conversation_id,
                    model_id,
                    content_blocks,
                    accumulated_usage if accumulated_usage.input_tokens > 0 else None,
                )
            # emitter / session_cache 终态：interrupted（user_cancelled）
            try:
                await interrupt_agent_run(
                    emitter=emitter,
                    session_cache=session_cache,
                    stats=_run_stats(),
                    duration_ms_factory=_run_duration_ms,
                    current_step_id=current_step_id,
                    reason="user_cancelled",
                )
                terminal_emitted = True
            except Exception as _emit_exc:  # noqa: BLE001 — 终态事件失败不能阻塞 cancel 传播
                logger.warning(f"emit run_interrupted 失败: {_emit_exc}")
            await finalize_stream(conversation_id, success=False, error_msg="用户中止", task_id=task_id)
            raise

        except Exception as e:
            logger.error(f"Agent 生成异常: conv_id={conversation_id}, error={e}")
            if content_blocks:
                persist_message(
                    db,
                    assistant_message_id,
                    conversation_id,
                    model_id,
                    content_blocks,
                    accumulated_usage if accumulated_usage.input_tokens > 0 else None,
                )
            try:
                await fail_agent_run(
                    emitter=emitter,
                    session_cache=session_cache,
                    stats=_run_stats(),
                    duration_ms_factory=_run_duration_ms,
                    current_step_id=current_step_id,
                    error_code=type(e).__name__,
                    message=str(e),
                )
                terminal_emitted = True
            except Exception as _emit_exc:  # noqa: BLE001
                logger.warning(f"emit run_failed 失败: {_emit_exc}")
            await finalize_stream(conversation_id, success=False, error_msg=str(e), task_id=task_id)
            # 完成协议层 + DB cache + SSE 收尾后 re-raise，让 background task scheduler 拿到失败信号；
            # 与 CancelledError 路径行为对齐（spec §5.3）。
            raise

        finally:
            # 兜底：极端路径（例如未匹配任何 except 又没走 try 终段）补一次终态
            if not terminal_emitted:
                try:
                    await write_fallback_error_status(
                        session_cache=session_cache,
                        stats=_run_stats(),
                        duration_ms_factory=_run_duration_ms,
                    )
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(f"finally 兜底 write_session_status 失败: {_exc}")
            db.close()
