# app/services/stream_handler.py
"""
流式处理器 — 基于 Redis Stream 的两段式架构 + Agent Loop

Part A: generate_to_redis() — 后台任务，Agent Loop 多轮调用 LLM/工具，写 Redis Stream + 落库 PostgreSQL
Part B: stream_redis_as_sse() — SSE 读取器，从 Redis Stream 消费推送给客户端
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional

import litellm

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.repositories import FileRepository
from app.schemas.chat import (
    TextBlock,
    ThinkingBlock,
    Usage,
)
from app.services.chat.message_builder import (
    build_llm_messages,
    inject_file_content,
    is_image_file,
)
from app.services.stream_state_service import (
    append_chunk,
    check_lock_owner,
    finalize_stream,
    read_stream_chunks,
)

# 每 N 个 chunk 检查一次锁状态
LOCK_CHECK_INTERVAL = 20

# Agent Loop 限制
AGENT_MAX_STEPS = 8               # LLM 调用轮次上限
AGENT_MAX_TOOL_CALLS = 20         # 工具执行总次数上限
AGENT_TOTAL_TIMEOUT = 300         # 5 分钟硬超时
AGENT_TOOL_TIMEOUT = 30           # 单次工具调用超时
AGENT_TOOL_MAX_RETRIES = 1        # 瞬时故障重试次数
AGENT_LLM_MAX_RETRIES = 1         # LLM 调用重试次数


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
            from app.ai.tools import WEB_SEARCH_TOOL
            call_kwargs["tools"] = [WEB_SEARCH_TOOL]
            call_kwargs["tool_choice"] = "auto"
            if should_use_reasoning and provider == "volcengine":
                call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        db = SessionLocal()

        # agent loop 状态
        content_blocks: list = []
        accumulated_usage = Usage(input_tokens=0, output_tokens=0)
        step = 0
        total_tool_calls = 0
        finish_reason = "stop"

        try:
            await append_chunk(conversation_id, "preparing", "", "")

            # 构建 LLM 消息
            file_repo = FileRepository(db)
            from app.db.repositories import MemoryRepository
            memory_repo = MemoryRepository(db)
            user_memories = memory_repo.get_active(user_id)
            messages = await build_llm_messages(raw_messages, has_vision, file_repo, user_memories)

            if file_ids:
                non_image_ids = [fid for fid in file_ids if not is_image_file(fid, file_repo)]
                if non_image_ids:
                    file_contents = file_repo.get_parsed_file_content(non_image_ids)
                    if file_contents:
                        messages = inject_file_content(messages, original_message, file_contents)

            # ── URL 自动检测预处理（路径 A，保持不变）──
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

                    await append_chunk(
                        conversation_id, "url_read_start",
                        json.dumps({"url": auto_detected_url, "source": "auto"}, ensure_ascii=False),
                        url_read_block_id,
                    )

                    try:
                        from app.services.reader_client import read_url
                        read_result = await asyncio.wait_for(read_url(auto_detected_url, timeout=8.0), timeout=8.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"URL 自动抓取超时: {auto_detected_url}")
                        read_result = None

                    if read_result:
                        url_read_content = read_result
                        await append_chunk(
                            conversation_id, "url_read_complete",
                            json.dumps({
                                "url": auto_detected_url, "title": read_result.title,
                                "favicon": read_result.favicon, "status": "success",
                            }, ensure_ascii=False),
                            url_read_block_id,
                        )
                    else:
                        await append_chunk(
                            conversation_id, "url_read_complete",
                            json.dumps({"url": auto_detected_url, "status": "failed"}, ensure_ascii=False),
                            url_read_block_id,
                        )

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
                if "extra_body" in call_kwargs and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled":
                    del call_kwargs["extra_body"]
            else:
                if supports_fc:
                    from app.ai.tools import URL_READ_TOOL
                    if URL_READ_TOOL not in call_kwargs.get("tools", []):
                        call_kwargs.setdefault("tools", []).append(URL_READ_TOOL)

            # 路径 A 成功时，URL block 加入 content_blocks
            if url_read_content and auto_detected_url:
                from app.schemas.chat import UrlBlock
                content_blocks.append(UrlBlock(
                    type="url_read", id=url_read_block_id,
                    url=auto_detected_url,
                    title=url_read_content.title,
                    favicon=url_read_content.favicon,
                ))

            # ═══════════════════════════════════════
            # Agent Loop
            # ═══════════════════════════════════════

            # 注入 agent 行为引导（仅当启用工具时）
            if supports_fc:
                agent_guidance = {
                    "role": "system",
                    "content": (
                        f"你可以多次使用工具来收集信息，总预算为 {AGENT_MAX_TOOL_CALLS} 次工具调用。策略建议：\n"
                        "- 需要对比多个主题时，为每个主题分别搜索以获取更全面的信息\n"
                        "- 搜索结果不够详细时，可以使用 url_read 深入阅读关键网页\n"
                        "- 可以在一次回复中调用多个工具并行搜索\n"
                        "- 请合理规划搜索策略，避免重复搜索相似关键词\n"
                        "- 当你认为已收集到足够信息时，直接给出回答即可"
                    ),
                }
                # 插入到最后一条 user 消息之前
                messages.insert(-1, agent_guidance)

            import time
            start_time = time.time()

            while step < AGENT_MAX_STEPS and total_tool_calls < AGENT_MAX_TOOL_CALLS:
                if time.time() - start_time > AGENT_TOTAL_TIMEOUT:
                    finish_reason = "timeout"
                    break

                thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
                text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                # 调试日志：LLM 调用前记录消息结构
                msg_summary = [(m.get("role"), m.get("content", "")[:50] if m.get("content") else None, bool(m.get("tool_calls"))) for m in messages]
                logger.info(f"Agent round {step+1}: conv={conversation_id}, model={litellm_model}, msgs={len(messages)}, structure={msg_summary[-5:]}, call_kwargs_keys={list(call_kwargs.keys())}")

                response = await self._llm_call_with_retry(
                    litellm_model, litellm_kwargs, messages, **call_kwargs,
                )

                reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data = \
                    await self._stream_round(
                        response, conversation_id, task_id,
                        should_use_reasoning, thinking_block_id, text_block_id,
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
                        content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                    if content_buf:
                        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                    break

                # ── 情况 2: 被踢掉 ──
                if finish_reason == "cancelled":
                    if reasoning_buf:
                        content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                    if content_buf:
                        content_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                    self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks,
                                          accumulated_usage if accumulated_usage.input_tokens > 0 else None)
                    await finalize_stream(conversation_id, success=False, error_msg="被新请求取代", task_id=task_id)
                    return

                # ── 情况 3: LLM 请求工具调用 ──
                if finish_reason == "tool_calls" and tool_calls_list:
                    step += 1

                    await append_chunk(
                        conversation_id, "agent_step_start",
                        json.dumps({"step": step, "max_steps": AGENT_MAX_STEPS, "tool_count": len(tool_calls_list)}, ensure_ascii=False),
                        "",
                    )

                    # 收集本轮 reasoning（tool_call 决策推理）
                    if reasoning_buf:
                        content_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))

                    # 并行执行工具
                    results = await self._execute_tools_parallel(
                        tool_calls_list, conversation_id, user_id, model_id, provider,
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

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_context,
                        })

                    # TODO: 预算提示暂时禁用，待定位 MiniMax 兼容性问题后再启用
                    # remaining = AGENT_MAX_TOOL_CALLS - total_tool_calls
                    # if remaining > 0 and messages and messages[-1].get("role") == "tool":
                    #     messages[-1]["content"] += f"\n\n[工具调用预算] 已用 {total_tool_calls}/{AGENT_MAX_TOOL_CALLS} 次，剩余 {remaining} 次。请据此规划后续搜索策略。"

                    # Checkpoint：每步写入 DB，进程崩溃不丢已完成步骤
                    self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, partial=True)

                    await append_chunk(
                        conversation_id, "agent_step_end",
                        json.dumps({"step": step, "total_tool_calls": total_tool_calls}, ensure_ascii=False),
                        "",
                    )

                    # 恢复 volcengine thinking（首轮 tool_call 决策已完成）
                    if "extra_body" in call_kwargs and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled":
                        del call_kwargs["extra_body"]

                    continue

                # 未知 finish_reason → 跳出循环
                break

            # ═══════════════════════════════════════
            # 触顶强制总结
            # ═══════════════════════════════════════
            if finish_reason == "tool_calls" or finish_reason == "timeout":
                reason = "timeout" if finish_reason == "timeout" else (
                    "max_steps" if step >= AGENT_MAX_STEPS else "max_tool_calls"
                )
                await append_chunk(
                    conversation_id, "agent_limit_reached",
                    json.dumps({"reason": reason}, ensure_ascii=False),
                    "",
                )

                messages.append({
                    "role": "system",
                    "content": "你已达到工具调用上限，请基于已收集的信息给出最终回答。不要再调用任何工具。",
                })
                final_call_kwargs = {k: v for k, v in call_kwargs.items() if k not in ("tools", "tool_choice")}
                thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
                text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

                response = await self._llm_call_with_retry(
                    litellm_model, litellm_kwargs, messages, **final_call_kwargs,
                )
                reasoning_buf, content_buf, _, _, usage_data = await self._stream_round(
                    response, conversation_id, task_id,
                    should_use_reasoning, thinking_block_id, text_block_id,
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

            # ═══════════════════════════════════════
            # 最终落库
            # ═══════════════════════════════════════
            final_usage = accumulated_usage if accumulated_usage.input_tokens > 0 else None
            self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, final_usage)
            await finalize_stream(conversation_id, success=True, task_id=task_id)

            asyncio.create_task(self._extract_user_memories(conversation_id, user_id))

        except asyncio.CancelledError:
            logger.info(f"Agent 任务被取消: conv_id={conversation_id}")
            if content_blocks:
                self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks,
                                      accumulated_usage if accumulated_usage.input_tokens > 0 else None)
            await finalize_stream(conversation_id, success=False, error_msg="用户中止", task_id=task_id)
            raise

        except Exception as e:
            logger.error(f"Agent 生成异常: conv_id={conversation_id}, error={e}")
            if content_blocks:
                self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks,
                                      accumulated_usage if accumulated_usage.input_tokens > 0 else None)
            await finalize_stream(conversation_id, success=False, error_msg=str(e), task_id=task_id)

        finally:
            db.close()

    async def _stream_round(
        self,
        response,
        conversation_id: str,
        task_id: str,
        should_use_reasoning: bool,
        thinking_block_id: str,
        text_block_id: str,
    ) -> tuple[str, str, list[dict], str, Optional[Usage]]:
        """
        通用 LLM 流式响应处理。
        返回 (reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)。
        tool_calls_list 格式: [{"id": str, "name": str, "arguments": str}, ...]
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
                await append_chunk(conversation_id, "reasoning", reasoning_delta, thinking_block_id)

            if content_delta:
                content_buf += content_delta
                await append_chunk(conversation_id, "answering", content_delta, text_block_id)

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

        tool_calls_list = [
            tool_calls_acc[idx]
            for idx in sorted(tool_calls_acc.keys())
            if tool_calls_acc[idx]["name"]
        ]

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
                    logger.warning(f"LLM 调用失败（{attempt+1}/{max_retries+1}），2s 后重试: {e}")
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

            logger.warning(f"工具 {handler.tool_name} 执行失败（{attempt+1}/{max_retries+1}），1s 后重试")
            await asyncio.sleep(1)

        return result

    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict],
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
    ) -> list:
        """
        并行执行所有 tool_calls。
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

            # 推送 start SSE
            sse_start_data = {**args}
            if handler.tool_name == "url_read":
                sse_start_data["source"] = "tool_call"
            sse_start_data["tool_call_id"] = tc["id"]
            await handler.push_sse_start(conversation_id, block_id, sse_start_data)

            # 执行（带重试）
            result = await self._execute_tool_with_retry(handler, args)

            # 推送 complete SSE
            sse_complete_data = {**args, "status": result.status, "tool_call_id": tc["id"]}
            if handler.tool_name == "web_search" and result.status == "success":
                sources = result.data.get("sources", [])
                sse_complete_data["sources"] = [s.model_dump() for s in sources]
            elif handler.tool_name == "url_read" and result.status == "success":
                sse_complete_data.update({
                    "title": result.data.get("title"),
                    "favicon": result.data.get("favicon"),
                })
            await handler.push_sse_complete(conversation_id, block_id, sse_complete_data)

            # 异步记录日志
            await handler.log(
                log_id=log_id,
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                provider=provider,
                result=result,
                input_params=args,
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

    @staticmethod
    async def _extract_user_memories(conversation_id: str, user_id: str) -> None:
        """异步提取用户记忆，使用独立的 DB Session"""
        logger.info(f"开始记忆提取: conv_id={conversation_id}, user_id={user_id}")
        db = SessionLocal()
        try:
            from app.services.user_memory_service import UserMemoryService

            service = UserMemoryService(db)
            await service.extract_memories(conversation_id, user_id)
            logger.info(f"记忆提取完成: conv_id={conversation_id}")
        except Exception as e:
            logger.warning(f"异步记忆提取失败: conv_id={conversation_id}, error={e}", exc_info=True)
        finally:
            db.close()


async def stream_redis_as_sse(
    conversation_id: str,
    message_id: str,
    last_entry_id: str = "0",
) -> AsyncGenerator[str, None]:
    """
    SSE 读取器：从 Redis Stream 读 chunk，格式化为 SSE 事件推送给客户端。
    不调用 LLM，只读 Redis。

    每条 SSE 事件包含 id 行（Redis entry ID），供断线重连使用。
    Redis 不可用时立即返回 error 帧。
    """
    from app.core.redis import get_redis_pool

    if not get_redis_pool():
        error_payload = {
            "id": message_id,
            "conversation_id": conversation_id,
            "choices": [{"delta": {}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    async for chunk in read_stream_chunks(conversation_id, last_entry_id):
        entry_id = chunk.pop("entry_id")
        chunk_type = chunk.get("type")

        # 跳过 start 标记
        if chunk_type == "start":
            continue

        # preparing 事件：后台任务已启动，前端收到此帧即触发 onReady
        # 不携带 content，仅作为 SSE 首帧让前端感知到流已开始
        if chunk_type == "preparing":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"id: {entry_id}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            continue

        if chunk_type == "reasoning":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "id": chunk.get("block_id", ""),
                                    "thinking": chunk["content"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "thinking_pending":
            # 思考中占位事件：前端显示脉冲动画但不展示具体内容
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "id": chunk.get("block_id", ""),
                                    "thinking": "",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "answering":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "text",
                                    "id": chunk.get("block_id", ""),
                                    "text": chunk["content"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "search_start":
            # 搜索开始事件
            search_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "search",
                                    "id": chunk.get("block_id", ""),
                                    "search_event": "start",
                                    "query": search_data.get("query", ""),
                                    "tool_call_id": search_data.get("tool_call_id"),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "search_complete":
            # 搜索完成事件
            search_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "search",
                                    "id": chunk.get("block_id", ""),
                                    "search_event": "complete",
                                    "query": search_data.get("query", ""),
                                    "sources": search_data.get("sources", []),
                                    "tool_call_id": search_data.get("tool_call_id"),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "url_read_start":
            # URL 读取开始事件
            url_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "url_read",
                                    "id": chunk.get("block_id", ""),
                                    "url_read_event": "start",
                                    "url": url_data.get("url", ""),
                                    "source": url_data.get("source", "auto"),
                                    "tool_call_id": url_data.get("tool_call_id"),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "url_read_complete":
            # URL 读取完成事件
            url_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "url_read",
                                    "id": chunk.get("block_id", ""),
                                    "url_read_event": "complete",
                                    "url": url_data.get("url", ""),
                                    "title": url_data.get("title"),
                                    "favicon": url_data.get("favicon"),
                                    "status": url_data.get("status", "success"),
                                    "tool_call_id": url_data.get("tool_call_id"),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "agent_step_start":
            step_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "agent_step",
                                    "agent_event": "step_start",
                                    "step": step_data.get("step", 0),
                                    "max_steps": step_data.get("max_steps", AGENT_MAX_STEPS),
                                    "tool_count": step_data.get("tool_count", 0),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "agent_step_end":
            step_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "agent_step",
                                    "agent_event": "step_end",
                                    "step": step_data.get("step", 0),
                                    "total_tool_calls": step_data.get("total_tool_calls", 0),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "agent_limit_reached":
            limit_data = json.loads(chunk.get("content", "{}"))
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "agent_step",
                                    "agent_event": "limit_reached",
                                    "reason": limit_data.get("reason", ""),
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        elif chunk_type == "done":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        elif chunk_type == "error":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "error",
                    }
                ],
            }
        else:
            continue

        yield f"id: {entry_id}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
