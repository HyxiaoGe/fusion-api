# app/services/stream_handler.py
"""
流式处理器 — 基于 Redis Stream 的两段式架构 + Agent Loop

Part A: generate_to_redis() — 后台任务，Agent Loop 多轮调用 LLM/工具，写 Redis Stream + 落库 PostgreSQL
Part B: stream_redis_as_sse() — SSE 读取器，从 Redis Stream 消费推送给客户端
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

import litellm

import app.services.stream_state_service as sss
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
from app.services.stream_state_service import (
    append_chunk,
    check_lock_owner,
    finalize_stream,
    read_stream_chunks,
)
from app.services.user_credential_health import UserCredentialHealthService


class _AgentEventRedisWriter:
    """把 emitter 的 (conv_id, chunk_type, payload:dict) 调用转成
    stream_state_service.append_chunk(conv_id, chunk_type, content:str, block_id:str)。

    Task 9 引入的 adapter — emitter 不直接知道 stream_state_service 的接口形态，
    通过本 adapter 桥接：payload JSON 序列化进 content 字段，block_id 留空。
    """

    async def append_chunk(
        self, conversation_id: str, chunk_type: str, payload: dict
    ) -> None:
        await sss.append_chunk(
            conversation_id,
            chunk_type,
            json.dumps(payload, ensure_ascii=False),
            "",  # block_id 不适用于 agent_event chunk
        )


# 每 N 个 chunk 检查一次锁状态
LOCK_CHECK_INTERVAL = 20

# Agent Loop 限制
AGENT_MAX_STEPS = 8  # LLM 调用轮次上限
AGENT_MAX_TOOL_CALLS = 20  # 工具执行总次数上限
AGENT_TOTAL_TIMEOUT = 300  # 5 分钟硬超时
AGENT_TOOL_TIMEOUT = 30  # 单次工具调用超时
AGENT_TOOL_MAX_RETRIES = 1  # 瞬时故障重试次数
AGENT_LLM_MAX_RETRIES = 1  # LLM 调用重试次数


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
            redis_writer=_AgentEventRedisWriter(),
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

            # ── URL 自动检测预处理（路径 A）──
            # 注：旧的 url_read_start / url_read_complete 实时 chunk 已删除，
            # FE 由 agent_event 统一渲染（路径 A 直接走 LLM 上下文注入，不发事件，
            # 因为它在 agent run 之外完成）。block 仍写入 content_blocks 以便落库。
            url_read_content = None
            auto_detected_url = None
            url_read_block_id = None
            if supports_fc:
                import re

                url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')
                urls_in_message = url_pattern.findall(original_message)
                if urls_in_message:
                    auto_detected_url = urls_in_message[0]
                    url_read_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                    try:
                        from app.services.reader_client import read_url

                        read_result = await asyncio.wait_for(read_url(auto_detected_url, timeout=8.0), timeout=8.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"URL 自动抓取超时: {auto_detected_url}")
                        read_result = None

                    if read_result:
                        url_read_content = read_result

            if url_read_content:
                from app.services.tool_handlers.url_read import MAX_CONTENT_CHARS

                content_text = url_read_content.content
                truncation_note = ""
                if len(content_text) > MAX_CONTENT_CHARS:
                    content_text = content_text[:MAX_CONTENT_CHARS]
                    truncation_note = "\n（内容已截断，仅展示前部分）"
                url_context_msg = {
                    "role": "system",
                    "content": (
                        f"以下是用户消息中提到的网页 {auto_detected_url} 的内容：\n"
                        f"标题：{url_read_content.title or '未知'}\n\n"
                        f"{content_text}{truncation_note}\n\n"
                        "请基于以上网页内容回答用户的问题。"
                    ),
                }
                messages.insert(-1, url_context_msg)
                if (
                    "extra_body" in call_kwargs
                    and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled"
                ):
                    del call_kwargs["extra_body"]
            else:
                if supports_fc:
                    from app.ai.tools import URL_READ_TOOL

                    if URL_READ_TOOL not in call_kwargs.get("tools", []):
                        call_kwargs.setdefault("tools", []).append(URL_READ_TOOL)

            # 路径 A 成功时，URL block 加入 content_blocks
            if url_read_content and auto_detected_url:
                from app.schemas.chat import UrlBlock

                content_blocks.append(
                    UrlBlock(
                        type="url_read",
                        id=url_read_block_id,
                        url=auto_detected_url,
                        title=url_read_content.title,
                        favicon=url_read_content.favicon,
                    )
                )

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
                model=model_id,
                tools=agent_tools_announced,
                config={
                    "max_steps": AGENT_MAX_STEPS,
                    "max_tool_calls": AGENT_MAX_TOOL_CALLS,
                    "timeout_s": AGENT_TOTAL_TIMEOUT,
                },
            )

            limit_reason: Optional[str] = None  # 记录触顶原因，决定后续是否走强制总结

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

                thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
                text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                try:
                    response = await self._llm_call_with_retry(
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

                reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data = await self._stream_round(
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
                    break

                # ── 情况 2: 被踢掉（superseded）──
                if finish_reason == "cancelled":
                    if reasoning_buf:
                        content_blocks.append(
                            ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf)
                        )
                    if content_buf:
                        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                    self._persist_message(
                        db,
                        assistant_message_id,
                        conversation_id,
                        model_id,
                        content_blocks,
                        accumulated_usage if accumulated_usage.input_tokens > 0 else None,
                    )
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
                    step += 1
                    step_start_time = time.time()

                    # emitter.step_started 内部会派发 step_id 并 set 到 _current_step_id
                    current_step_id = await emitter.step_started(step_number=step)
                    await session_cache.write_step_started(
                        run_id=run_id,
                        step_id=current_step_id,
                        step_number=step,
                    )

                    # 收集本轮 reasoning（tool_call 决策推理）
                    if reasoning_buf:
                        content_blocks.append(
                            ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf)
                        )

                    # 并行执行工具（走 execute_with_emitter，tool_call_started/completed 由 base 统一发）
                    results = await self._execute_tools_parallel(
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
                    self._persist_message(
                        db, assistant_message_id, conversation_id, model_id, content_blocks, partial=True
                    )

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

                # 未知 finish_reason → 跳出循环
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
                    response = await self._llm_call_with_retry(
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
                except Exception as _llm_exc:
                    kind, _msg = categorize(_llm_exc)
                    if kind in {ErrorKind.KEY_INVALID, ErrorKind.QUOTA_EXCEEDED, ErrorKind.TOS_BLOCKED}:
                        if credential_source == "system":
                            ProviderHealthService(db).mark_failure(_provider_id, kind, _msg)
                        else:
                            UserCredentialHealthService(db).mark_failure(user_id, _provider_id, kind, _msg)
                    raise

                reasoning_buf, content_buf, _, _, usage_data = await self._stream_round(
                    response,
                    conversation_id,
                    task_id,
                    should_use_reasoning,
                    thinking_block_id,
                    text_block_id,
                    run_id=run_id,
                    step_id=summary_step_id,
                )
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
            self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, final_usage)

            run_finish_reason = "limit_reached" if limit_reason is not None else "stop"
            await emitter.run_completed(
                total_steps=step,
                total_tool_calls=total_tool_calls,
                finish_reason=run_finish_reason,
            )
            await session_cache.write_session_status(
                run_id=run_id,
                status="limit_reached" if limit_reason is not None else "completed",
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
            await finalize_stream(
                conversation_id,
                success=False,
                error_msg=e.message or f"Provider {e.provider_id} 当前不可用",
                error_code="PROVIDER_OFFLINE",
                error_data={"provider_id": e.provider_id, "reason": e.reason},
                task_id=task_id,
            )
            return

        except asyncio.CancelledError:
            logger.info(f"Agent 任务被取消: conv_id={conversation_id}")
            if content_blocks:
                self._persist_message(
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
                    await session_cache.write_step_terminal(
                        step_id=current_step_id, status="interrupted"
                    )
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
                self._persist_message(
                    db,
                    assistant_message_id,
                    conversation_id,
                    model_id,
                    content_blocks,
                    accumulated_usage if accumulated_usage.input_tokens > 0 else None,
                )
            try:
                if current_step_id is not None:
                    await session_cache.write_step_terminal(
                        step_id=current_step_id, status="failed"
                    )
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

    async def _stream_round(
        self,
        response,
        conversation_id: str,
        task_id: str,
        should_use_reasoning: bool,
        thinking_block_id: str,
        text_block_id: str,
        run_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> tuple[str, str, list[dict], str, Optional[Usage]]:
        """
        通用 LLM 流式响应处理。
        返回 (reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)。
        tool_calls_list 格式: [{"id": str, "name": str, "arguments": str}, ...]

        run_id / step_id 透传给 reasoning / answering chunk，让 FE 把 token 流挂回
        agent_event 控制面对应的 step（spec §4.6）。两者均可为空（非 agent 路径）。
        """
        reasoning_buf = ""
        content_buf = ""
        usage_data: Optional[Usage] = None
        chunk_count = 0
        finish_reason = "stop"

        # tool_call 累积缓冲区（支持多个并行 tool_calls）
        tool_calls_acc: dict[int, dict] = {}  # index → {"id", "name", "arguments"}

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None

            if not choice:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )
                continue

            delta = choice.delta
            fr = choice.finish_reason

            # ===== tool_call 累积（支持多个）=====
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if hasattr(tc, "index") and tc.index is not None else 0
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": None, "name": None, "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

            if fr == "tool_calls":
                finish_reason = "tool_calls"
                continue

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                continue

            # ===== reasoning + content =====
            reasoning_delta = ""
            if should_use_reasoning:
                reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
                    reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""

            content_delta = delta.content or ""

            if reasoning_delta and content_delta == reasoning_delta:
                content_delta = ""

            if reasoning_delta:
                reasoning_buf += reasoning_delta
                await append_chunk(
                    conversation_id,
                    "reasoning",
                    reasoning_delta,
                    thinking_block_id,
                    run_id=run_id,
                    step_id=step_id,
                )

            if content_delta:
                content_buf += content_delta
                await append_chunk(
                    conversation_id,
                    "answering",
                    content_delta,
                    text_block_id,
                    run_id=run_id,
                    step_id=step_id,
                )

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )

            if fr == "stop":
                finish_reason = "stop"

            chunk_count += 1
            if chunk_count % LOCK_CHECK_INTERVAL == 0:
                if not await check_lock_owner(conversation_id, task_id):
                    logger.info(f"流式调用被踢掉: conv_id={conversation_id}")
                    finish_reason = "cancelled"
                    break

        tool_calls_list = [tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys()) if tool_calls_acc[idx]["name"]]

        return reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data

    async def _llm_call_with_retry(
        self,
        litellm_model: str,
        litellm_kwargs: dict,
        messages: list[dict],
        max_retries: int = AGENT_LLM_MAX_RETRIES,
        **call_kwargs,
    ):
        """带重试的 LLM 调用，返回 streaming response"""
        for attempt in range(max_retries + 1):
            try:
                return await litellm.acompletion(
                    model=litellm_model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **litellm_kwargs,
                    **call_kwargs,
                )
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(kw in error_str for kw in ["429", "rate", "503", "502", "timeout"])
                if is_retryable and attempt < max_retries:
                    logger.warning(f"LLM 调用失败（{attempt + 1}/{max_retries + 1}），2s 后重试: {e}")
                    await asyncio.sleep(2)
                    continue
                raise

    @staticmethod
    async def _execute_tool_with_retry(
        handler,
        args: dict,
        max_retries: int = AGENT_TOOL_MAX_RETRIES,
    ):
        """带重试的工具执行（仅瞬时故障重试），返回 ToolResult"""
        from app.services.tool_handlers import ToolResult

        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    handler.execute(args),
                    timeout=AGENT_TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                result = ToolResult(status="failed", error_message="工具调用超时")

            if result.status == "success":
                return result

            # 永久性错误不重试
            err = (result.error_message or "").lower()
            is_permanent = any(kw in err for kw in ["not_found", "invalid", "rate_limit", "400", "401", "403", "404"])
            if is_permanent or attempt >= max_retries:
                return result

            logger.warning(f"工具 {handler.tool_name} 执行失败（{attempt + 1}/{max_retries + 1}），1s 后重试")
            await asyncio.sleep(1)

        return result

    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict],
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
        trace_id: str = None,
        step_number: int = None,
        emitter: Optional[AgentEventEmitter] = None,
    ) -> list:
        """
        并行执行所有 tool_calls。

        统一走 handler.execute_with_emitter，由 base 发 tool_call_started / completed
        agent_event；旧 push_sse_start / push_sse_complete 不再调用（同名 chunk
        type 已废弃）。tool_call_logs 仍通过 handler.log 写入。

        返回 [(tool_call: dict, result: ToolResult, handler: BaseToolHandler|None, block_id: str, log_id: str), ...]
        """
        from app.services.tool_handlers import ToolResult
        from app.services.tool_handlers import get_handler as _get_handler

        async def _run_one(tc: dict):
            handler = _get_handler(tc["name"])
            block_id = f"blk_{uuid.uuid4().hex[:12]}"
            log_id = str(uuid.uuid4())

            if not handler:
                logger.warning(f"未知的 tool_call: {tc['name']}")
                result = ToolResult(status="failed", error_message=f"未知工具: {tc['name']}")
                return tc, result, None, block_id, log_id

            # 解析参数
            try:
                args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except json.JSONDecodeError:
                args = {}

            # ── emitter 路径 ──
            # 直接复用 handler.execute_with_emitter 的发 start/completed 协议，
            # 但在中间塞入 _execute_tool_with_retry（瞬时重试 + 30s timeout）。
            # 不能 monkey-patch handler.execute（handler 是模块级 singleton，
            # 同 gather 内并发 tool_call 会相互覆盖）。这里手动复刻
            # execute_with_emitter 的契约：tool_call_started → 重试执行 → tool_call_completed。
            if emitter is not None:
                import time as _t

                await emitter.tool_call_started(
                    tool_call_id=tc["id"],
                    tool_name=handler.tool_name,
                    arguments=args,
                )
                _start_mono = _t.monotonic()
                try:
                    result = await self._execute_tool_with_retry(handler, args)
                except BaseException as _exc:  # noqa: BLE001 — 必须先 emit completed 再 raise
                    _dur_ms = int((_t.monotonic() - _start_mono) * 1000)
                    synthetic_failed = ToolResult(
                        status="failed",
                        error_message=f"{type(_exc).__name__}: {_exc}",
                    )
                    await emitter.tool_call_completed(
                        tool_call_id=tc["id"],
                        tool_name=handler.tool_name,
                        status="failed",
                        duration_ms=_dur_ms,
                        result_summary=handler._build_result_summary(synthetic_failed),
                        error=f"{type(_exc).__name__}: {_exc}",
                    )
                    # 上层在 results 中需要 result 对象继续构建消息，不向上 re-raise；
                    # 这里返回 failed ToolResult，让消息流仍能完成。
                    result = synthetic_failed
                else:
                    _dur_ms = int((_t.monotonic() - _start_mono) * 1000)
                    # 同步 ToolResult.duration_ms（部分 handler 不写）
                    if result.duration_ms is None:
                        result.duration_ms = _dur_ms
                    await emitter.tool_call_completed(
                        tool_call_id=tc["id"],
                        tool_name=handler.tool_name,
                        status=result.status,
                        duration_ms=_dur_ms,
                        result_summary=handler._build_result_summary(result),
                        error=result.error_message if result.status != "success" else None,
                    )
            else:
                # 兼容 emitter 缺省路径（不应在 generate_to_redis 内触发）
                result = await self._execute_tool_with_retry(handler, args)

            # 异步记录日志（tool_call_logs 路径不变）
            await handler.log(
                log_id=log_id,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                provider=provider,
                result=result,
                input_params=args,
                trace_id=trace_id,
                step_number=step_number,
            )

            return tc, result, handler, block_id, log_id

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
        return list(results)

    # ──────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────

    def _persist_message(
        self,
        db,
        assistant_message_id: str,
        conversation_id: str,
        model_id: str,
        content_blocks: list,
        usage_data: Optional[Usage] = None,
        partial: bool = False,
    ) -> None:
        """
        将 assistant 消息写入 PostgreSQL。
        partial=True 时增量更新 content blocks（checkpoint），不写 usage。
        partial=False 时写入完整数据（最终落库）。
        """
        try:
            from app.db.models import Message as MessageModel

            existing = db.query(MessageModel).filter_by(id=assistant_message_id).first()
            if existing:
                existing.content = [block.model_dump() for block in content_blocks]
                if usage_data and not partial:
                    existing.usage = usage_data.model_dump()
            else:
                db_message = MessageModel(
                    id=assistant_message_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=[block.model_dump() for block in content_blocks],
                    model_id=model_id,
                    usage=usage_data.model_dump() if usage_data and not partial else None,
                )
                db.add(db_message)
            db.commit()
        except Exception as e:
            logger.error(f"写入 assistant 消息失败: {e}")
            db.rollback()


def _entry_to_sse_envelope(entry_fields: dict) -> dict:
    """把 Redis Stream entry 的 hash 字段转成 {chunk_type, data} envelope。

    spec §4.6 SSE 顶层契约：每条 SSE message 形如 {"chunk_type": <type>, "data": {...}}。
    本函数不负责 SSE 包装（id: 行、data: 前缀、[DONE]）— 这由 stream_redis_as_sse 处理。
    """
    chunk_type = entry_fields.get("type", "")
    content = entry_fields.get("content", "")
    block_id = entry_fields.get("block_id", "")

    if chunk_type == "agent_event":
        # agent_event 的 content 由 emitter 序列化为 JSON dict
        data = json.loads(content) if content else {}
    elif chunk_type in ("reasoning", "answering"):
        data = {"block_id": block_id, "delta": content}
        # 透传可选关联字段（emitter 通过 append_chunk 的 **extras 写入）
        for k in ("run_id", "step_id"):
            if k in entry_fields:
                data[k] = entry_fields[k]
    elif chunk_type == "thinking_pending":
        # 思考中占位事件：FE 用来显示脉冲动画
        data = {"block_id": block_id}
    elif chunk_type == "error":
        # BYOK 结构化 error_code: content 是 {"code":..., ...} JSON 时提升进 data
        data = {}
        if content and content.startswith("{") and content.endswith("}"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    data = parsed
            except (ValueError, TypeError):
                pass
    else:
        # done / preparing / 其它已知 type 用空 data
        data = {}

    return {"chunk_type": chunk_type, "data": data}


async def stream_redis_as_sse(
    conversation_id: str,
    message_id: str,
    last_entry_id: str = "0",
) -> AsyncGenerator[str, None]:
    """SSE 读取器：从 Redis Stream 读 chunk，按 spec §4.6 顶层 envelope 输出。

    每条 SSE 事件包含 id: 行（Redis entry ID），供断线重连使用。
    Redis 不可用时立即返回 error 帧 + [DONE]。
    """
    from app.core.redis import get_redis_pool

    if not get_redis_pool():
        # 维持新外层 envelope 形态
        error_envelope = {
            "chunk_type": "error",
            "data": {
                "code": "redis_unavailable",
                "message": "Redis 不可用，无法读取流",
            },
        }
        yield f"data: {json.dumps(error_envelope, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    async for chunk in read_stream_chunks(conversation_id, last_entry_id):
        entry_id = chunk.pop("entry_id")
        chunk_type = chunk.get("type", "")

        # 跳过内部 start 标记
        if chunk_type == "start":
            continue

        envelope = _entry_to_sse_envelope(chunk)
        yield f"id: {entry_id}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
