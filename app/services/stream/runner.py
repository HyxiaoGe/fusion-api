"""StreamHandler 类与 generate_to_redis 编排。

spec §4.1。本模块只负责 agent loop 的控制流编排，所有"做事"的逻辑
（LLM 流消费 / 工具执行 / 落库 / SSE 编码）都委派给同子包内的兄弟模块。
"""

import asyncio
import time
import uuid
from typing import Optional

from app.ai.litellm_utils import ProviderOfflineError, merge_extra_body
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
from app.services.error_categorizer import ErrorKind, categorize
from app.services.provider_health import ProviderHealthService
from app.services.stream.llm_stream import llm_call_with_retry, stream_round
from app.services.stream.persistence import persist_message, preprocess_url_in_message
from app.services.stream.tool_executor import AgentEventRedisWriter, execute_tools_parallel
from app.services.stream_state_service import (
    append_chunk,
    finalize_stream,
)
from app.services.user_credential_health import UserCredentialHealthService

# Agent Loop 限制
AGENT_MAX_STEPS = 8  # LLM 调用轮次上限
AGENT_MAX_TOOL_CALLS = 20  # 工具执行总次数上限
AGENT_TOTAL_TIMEOUT = 300  # 5 分钟硬超时


class StreamHandler:
    """流式处理器"""

    REASONING_PROVIDERS = {"deepseek", "qwen", "xai", "volcengine"}

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

        use_reasoning = options.get("use_reasoning")
        should_use_reasoning = use_reasoning is True or (use_reasoning is None and provider in self.REASONING_PROVIDERS)

        # 判断是否启用工具
        supports_fc = capabilities.get("functionCalling", False)
        call_kwargs = {}
        if supports_fc:
            from app.ai.tools import build_web_search_tool

            call_kwargs["tools"] = [build_web_search_tool()]
            call_kwargs["tool_choice"] = "auto"
            if should_use_reasoning and provider == "volcengine":
                merge_extra_body(call_kwargs, {"thinking": {"type": "disabled"}})

        db = SessionLocal()

        # 从 litellm_kwargs metadata 提取凭证来源信息，供 health 标记使用
        _metadata = litellm_kwargs.get("metadata", {})
        credential_source = _metadata.get("credential_source", "system")
        _provider_id = _metadata.get("provider_id", provider)

        # agent loop 状态
        content_blocks: list = []
        accumulated_usage = Usage(input_tokens=0, output_tokens=0)
        step = 0
        total_tool_calls = 0
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

        try:
            await append_chunk(conversation_id, "preparing", "", "")

            # session_cache 行：必须传齐 user_id / model_id / provider（NOT NULL）
            await session_cache.write_session_started(
                run_id=run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                provider=provider,
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

            limit_reason: Optional[str] = None  # 记录触顶原因，决定后续是否走强制总结
            unknown_terminated = False  # 雷点 3 修复：退化分支标记，决定 run_finish_reason

            while True:
                # ─── 三段触顶检查（顺序：timeout > max_steps > max_tool_calls）───
                if time.time() - start_time > AGENT_TOTAL_TIMEOUT:
                    finish_reason = "timeout"
                    limit_reason = "timeout"
                    await emitter.run_limit_reached(reason="timeout")
                    break
                if step >= AGENT_MAX_STEPS:
                    finish_reason = "tool_calls"
                    limit_reason = "max_steps"
                    await emitter.run_limit_reached(reason="max_steps")
                    break
                if total_tool_calls >= AGENT_MAX_TOOL_CALLS:
                    finish_reason = "tool_calls"
                    limit_reason = "max_tool_calls"
                    await emitter.run_limit_reached(reason="max_tool_calls")
                    break

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 每轮 LLM call 之前先开 step（spec §4.2 "1 step = 1 LLM round"）
                # 这样本轮 reasoning / answering / tool_call_* chunk 都能挂到正确的 step_id；
                # stop / cancelled / tool_calls 三路径都在分支末尾闭合本 step。
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                step += 1
                step_start_time = time.time()
                current_step_id = await emitter.step_started(step_number=step)
                await session_cache.write_step_started(
                    run_id=run_id,
                    step_id=current_step_id,
                    step_number=step,
                )

                thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
                text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                try:
                    response = await llm_call_with_retry(
                        litellm_model,
                        litellm_kwargs,
                        messages,
                        **call_kwargs,
                    )
                    # LLM 调用成功，更新 health 状态
                    if credential_source == "system":
                        ProviderHealthService(db).mark_success(_provider_id)
                    else:
                        UserCredentialHealthService(db).mark_success(user_id, _provider_id)
                except Exception as _llm_exc:
                    kind, _msg = categorize(_llm_exc)
                    if kind in {ErrorKind.KEY_INVALID, ErrorKind.QUOTA_EXCEEDED, ErrorKind.TOS_BLOCKED}:
                        if credential_source == "system":
                            ProviderHealthService(db).mark_failure(_provider_id, kind, _msg)
                        else:
                            UserCredentialHealthService(db).mark_failure(user_id, _provider_id, kind, _msg)
                    raise

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
                    stop_step_duration = int((time.time() - step_start_time) * 1000)
                    await emitter.step_completed(
                        step_number=step,
                        tool_call_count=0,
                        duration_ms=stop_step_duration,
                    )
                    await session_cache.write_step_completed(
                        step_id=current_step_id,
                        tool_names=[],
                        tool_calls_count=0,
                        duration_ms=stop_step_duration,
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
                    # 闭合本 step：标记为 interrupted（在 run_interrupted 之前）
                    if current_step_id is not None:
                        await session_cache.write_step_terminal(step_id=current_step_id, status="interrupted")
                    await emitter.run_interrupted(reason="superseded")
                    await session_cache.write_session_status(
                        run_id=run_id,
                        status="interrupted",
                        total_steps=step,
                        total_tool_calls=total_tool_calls,
                        total_duration_ms=int((time.time() - run_start) * 1000),
                    )
                    terminal_emitted = True
                    await finalize_stream(conversation_id, success=False, error_msg="被新请求取代", task_id=task_id)
                    return

                # ── 情况 3: LLM 请求工具调用 ──
                if finish_reason == "tool_calls" and tool_calls_list:
                    # step / step_started / write_step_started 已在 while 顶部完成

                    # 收集本轮 reasoning（tool_call 决策推理）
                    if reasoning_buf:
                        content_blocks.append(
                            ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf)
                        )

                    # 并行执行工具（走 execute_with_emitter，tool_call_started/completed 由 base 统一发）
                    results = await execute_tools_parallel(
                        tool_calls_list,
                        conversation_id,
                        user_id,
                        model_id,
                        provider,
                        trace_id=run_id,
                        step_number=step,
                        emitter=emitter,
                    )
                    total_tool_calls += len(tool_calls_list)

                    # 构建 assistant tool_call message
                    assistant_tool_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                            for tc in tool_calls_list
                        ],
                    }
                    if should_use_reasoning and reasoning_buf:
                        assistant_tool_msg["reasoning_content"] = reasoning_buf
                    messages.append(assistant_tool_msg)

                    # 每个工具结果注入 messages + 收集 content blocks
                    for tc, result, handler, block_id, log_id in results:
                        if handler and result.status == "success":
                            tool_context = handler.format_llm_context(result)
                            content_blocks.append(handler.build_content_block(result, block_id, log_id))
                        elif handler:
                            tool_context = f"工具调用失败：{result.error_message}"
                            content_blocks.append(handler.build_content_block(result, block_id, log_id))
                        else:
                            tool_context = f"工具调用失败：{result.error_message}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_context,
                            }
                        )

                    # Checkpoint：每步写入 DB，进程崩溃不丢已完成步骤
                    persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, partial=True)

                    # step_completed：emitter + session_cache（step_completed 会清空 _current_step_id）
                    step_duration = int((time.time() - step_start_time) * 1000)
                    step_tool_names = [tc["name"] for tc, *_ in results]
                    await emitter.step_completed(
                        step_number=step,
                        tool_call_count=len(results),
                        duration_ms=step_duration,
                    )
                    await session_cache.write_step_completed(
                        step_id=current_step_id,
                        tool_names=step_tool_names,
                        tool_calls_count=len(results),
                        duration_ms=step_duration,
                    )
                    current_step_id = None  # 离开 step 上下文

                    # 恢复 volcengine thinking（首轮 tool_call 决策已完成）
                    if (
                        "extra_body" in call_kwargs
                        and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled"
                    ):
                        del call_kwargs["extra_body"]

                    continue

                # 未知 finish_reason（含 tool_calls 但 list 为空的退化情况）→ 保留已收集内容，闭合本 step 后跳出
                if reasoning_buf:
                    content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                if content_buf:
                    content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                unknown_terminated = True
                unknown_step_duration = int((time.time() - step_start_time) * 1000)
                await emitter.step_completed(
                    step_number=step,
                    tool_call_count=0,
                    duration_ms=unknown_step_duration,
                )
                await session_cache.write_step_completed(
                    step_id=current_step_id,
                    tool_names=[],
                    tool_calls_count=0,
                    duration_ms=unknown_step_duration,
                )
                current_step_id = None
                break

            # ═══════════════════════════════════════
            # 触顶强制总结（timeout / max_steps / max_tool_calls）
            # ═══════════════════════════════════════
            if limit_reason is not None:
                # 触顶总结作为一个独立 step（不带 tools）
                step += 1
                summary_step_id = await emitter.step_started(step_number=step)
                await session_cache.write_step_started(
                    run_id=run_id,
                    step_id=summary_step_id,
                    step_number=step,
                )
                current_step_id = summary_step_id
                summary_step_start = time.time()

                messages.append(
                    {
                        "role": "system",
                        "content": "你已达到工具调用上限，请基于已收集的信息给出最终回答。不要再调用任何工具。",
                    }
                )
                final_call_kwargs = {k: v for k, v in call_kwargs.items() if k not in ("tools", "tool_choice")}
                thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
                text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                try:

                    async def _do_summary():
                        # 健康标记成功路径放在内部，外层 except 仍能捕获
                        response = await llm_call_with_retry(
                            litellm_model,
                            litellm_kwargs,
                            messages,
                            **final_call_kwargs,
                        )
                        # 触顶总结 LLM 调用成功，更新 health 状态
                        if credential_source == "system":
                            ProviderHealthService(db).mark_success(_provider_id)
                        else:
                            UserCredentialHealthService(db).mark_success(user_id, _provider_id)
                        return await stream_round(
                            response,
                            conversation_id,
                            task_id,
                            should_use_reasoning,
                            thinking_block_id,
                            text_block_id,
                            run_id=run_id,
                            step_id=summary_step_id,
                        )

                    # 给触顶总结独立 timeout：剩余 run 预算（兜底 10s 避免负数）
                    remaining = max(10, AGENT_TOTAL_TIMEOUT - (time.time() - run_start))
                    reasoning_buf, content_buf, _, _, usage_data = await asyncio.wait_for(
                        _do_summary(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"触顶总结超出剩余预算: conv_id={conversation_id}, budget={remaining}s")
                    reasoning_buf, content_buf, usage_data = "", "", None
                except Exception as _llm_exc:
                    kind, _msg = categorize(_llm_exc)
                    if kind in {ErrorKind.KEY_INVALID, ErrorKind.QUOTA_EXCEEDED, ErrorKind.TOS_BLOCKED}:
                        if credential_source == "system":
                            ProviderHealthService(db).mark_failure(_provider_id, kind, _msg)
                        else:
                            UserCredentialHealthService(db).mark_failure(user_id, _provider_id, kind, _msg)
                    raise

                if usage_data:
                    accumulated_usage = Usage(
                        input_tokens=accumulated_usage.input_tokens + usage_data.input_tokens,
                        output_tokens=accumulated_usage.output_tokens + usage_data.output_tokens,
                    )
                if reasoning_buf:
                    content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                if content_buf:
                    content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))

                summary_step_duration = int((time.time() - summary_step_start) * 1000)
                await emitter.step_completed(
                    step_number=step,
                    tool_call_count=0,
                    duration_ms=summary_step_duration,
                )
                await session_cache.write_step_completed(
                    step_id=summary_step_id,
                    tool_names=[],
                    tool_calls_count=0,
                    duration_ms=summary_step_duration,
                )
                current_step_id = None

            # ═══════════════════════════════════════
            # 最终落库 + run_completed
            # ═══════════════════════════════════════
            final_usage = accumulated_usage if accumulated_usage.input_tokens > 0 else None
            persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, final_usage)

            if unknown_terminated:
                run_finish_reason = "incomplete"
            elif limit_reason is not None:
                run_finish_reason = "limit_reached"
            else:
                run_finish_reason = "stop"
            await emitter.run_completed(
                total_steps=step,
                total_tool_calls=total_tool_calls,
                finish_reason=run_finish_reason,
            )
            session_status = (
                "incomplete" if unknown_terminated else "limit_reached" if limit_reason is not None else "completed"
            )
            await session_cache.write_session_status(
                run_id=run_id,
                status=session_status,
                total_steps=step,
                total_tool_calls=total_tool_calls,
                total_duration_ms=int((time.time() - run_start) * 1000),
            )
            terminal_emitted = True

            await finalize_stream(conversation_id, success=True, task_id=task_id)

        except ProviderOfflineError as e:
            logger.warning(
                f"Provider 离线，终止生成: conv_id={conversation_id}, provider={e.provider_id}, reason={e.reason}"
            )
            error_message = e.message or f"Provider {e.provider_id} 当前不可用"
            # 协议层终结事件 + cache 终态：补完 agent_event timeline，避免断尾
            try:
                if current_step_id is not None:
                    await session_cache.write_step_terminal(step_id=current_step_id, status="failed")
                await emitter.run_failed(
                    error_code="PROVIDER_OFFLINE",
                    message=error_message,
                )
                await session_cache.write_session_status(
                    run_id=run_id,
                    status="error",
                    total_steps=step,
                    total_tool_calls=total_tool_calls,
                    total_duration_ms=int((time.time() - run_start) * 1000),
                )
                terminal_emitted = True
            except Exception as _emit_exc:  # noqa: BLE001
                logger.warning(f"emit run_failed (PROVIDER_OFFLINE) 失败: {_emit_exc}")
            await finalize_stream(
                conversation_id,
                success=False,
                error_msg=error_message,
                error_code="PROVIDER_OFFLINE",
                error_data={"provider_id": e.provider_id, "reason": e.reason},
                task_id=task_id,
            )
            return

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
                if current_step_id is not None:
                    await session_cache.write_step_terminal(step_id=current_step_id, status="interrupted")
                await emitter.run_interrupted(reason="user_cancelled")
                await session_cache.write_session_status(
                    run_id=run_id,
                    status="interrupted",
                    total_steps=step,
                    total_tool_calls=total_tool_calls,
                    total_duration_ms=int((time.time() - run_start) * 1000),
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
                if current_step_id is not None:
                    await session_cache.write_step_terminal(step_id=current_step_id, status="failed")
                await emitter.run_failed(
                    error_code=type(e).__name__,
                    message=str(e),
                )
                await session_cache.write_session_status(
                    run_id=run_id,
                    status="error",
                    total_steps=step,
                    total_tool_calls=total_tool_calls,
                    total_duration_ms=int((time.time() - run_start) * 1000),
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
                    await session_cache.write_session_status(
                        run_id=run_id,
                        status="error",
                        total_steps=step,
                        total_tool_calls=total_tool_calls,
                        total_duration_ms=int((time.time() - run_start) * 1000),
                    )
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(f"finally 兜底 write_session_status 失败: {_exc}")
            db.close()
