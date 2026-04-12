# app/services/stream_handler.py
"""
流式处理器 — 基于 Redis Stream 的两段式架构

Part A: generate_to_redis() — 后台任务，调用 LLM 写 Redis Stream + 落库 PostgreSQL
Part B: stream_redis_as_sse() — SSE 读取器，从 Redis Stream 消费推送给客户端
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, List, Optional

import litellm

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.repositories import FileRepository
from app.schemas.chat import (
    SearchBlock,
    SearchSource,
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
        生命周期完全独立于 HTTP 连接。

        消息构建（含图片 base64 编码）在此后台任务中完成，
        不阻塞 SSE 首字节返回。

        支持 web_search tool_call 检测：
        - capabilities.functionCalling=true 时，传 tools=[web_search]
        - 模型返回 tool_call 时，调搜索 → 注入上下文 → 第二轮流式调用
        """
        if options is None:
            options = {}
        if capabilities is None:
            capabilities = {}

        use_reasoning = options.get("use_reasoning")
        should_use_reasoning = use_reasoning is True or (use_reasoning is None and provider in self.REASONING_PROVIDERS)

        thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
        text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

        reasoning_buf = ""
        content_buf = ""
        usage_data: Optional[Usage] = None
        chunk_count = 0

        # 判断是否启用搜索工具
        supports_fc = capabilities.get("functionCalling", False)
        call_kwargs = {}
        if supports_fc:
            from app.ai.tools import WEB_SEARCH_TOOL

            call_kwargs["tools"] = [WEB_SEARCH_TOOL]
            call_kwargs["tool_choice"] = "auto"
            # 仅 volcengine（豆包）：第一轮禁用 thinking，避免 tool_call 决策噪音
            # 其他 provider 保持默认，不影响 tool_call 决策质量
            if should_use_reasoning and provider == "volcengine":
                call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        # 后台任务独立管理 DB Session
        db = SessionLocal()

        try:
            # 立即推送 preparing 事件，让前端收到 SSE 首帧
            # → 新对话：触发 onReady → 页面跳转 + 显示用户消息
            # → 已有对话：前端显示 AI 加载状态
            await append_chunk(conversation_id, "preparing", "", "")

            # 在后台任务中构建 LLM 消息（含图片 base64 编码），不阻塞主请求
            file_repo = FileRepository(db)
            # 查询用户记忆，注入 system prompt
            from app.db.repositories import MemoryRepository
            memory_repo = MemoryRepository(db)
            user_memories = memory_repo.get_active(user_id)
            messages = await build_llm_messages(raw_messages, has_vision, file_repo, user_memories)

            # 非图片文件内容注入
            if file_ids:
                non_image_ids = [fid for fid in file_ids if not is_image_file(fid, file_repo)]
                if non_image_ids:
                    file_contents = file_repo.get_parsed_file_content(non_image_ids)
                    if file_contents:
                        messages = inject_file_content(messages, original_message, file_contents)

            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **litellm_kwargs,
                **call_kwargs,
            )

            # tool_call 累积缓冲区
            tool_call_id = None
            tool_call_name = None
            tool_call_args = ""

            # 第一轮 tool_call 判断：thinking 实时推送给前端，同时缓冲用于持久化决策
            # 搜索路径：前端 startSearch 清理噪音 + 后端不持久化第一轮 thinking
            # 非搜索路径：thinking 正常展示和持久化
            first_round_buffering = supports_fc
            first_round_thinking_buf = ""

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
                finish_reason = choice.finish_reason

                # ===== 分支 1: tool_call 累积 =====
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tc = delta.tool_calls[0]
                    if tc.id:
                        tool_call_id = tc.id
                    if tc.function and tc.function.name:
                        tool_call_name = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_call_args += tc.function.arguments

                # tool_call 完整了
                if finish_reason == "tool_calls" and tool_call_name:
                    # 传入第一轮 thinking：前端展示已丢弃，但 DeepSeek 等推理模型
                    # 要求 assistant tool_call 消息必须包含实际的 reasoning_content
                    await self._handle_tool_call(
                        db=db,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        model_id=model_id,
                        litellm_model=litellm_model,
                        litellm_kwargs=litellm_kwargs,
                        provider=provider,
                        messages=messages,
                        assistant_message_id=assistant_message_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        tool_call_name=tool_call_name,
                        tool_call_args=tool_call_args,
                        should_use_reasoning=should_use_reasoning,
                        thinking_block_id=thinking_block_id,
                        text_block_id=text_block_id,
                        first_round_reasoning=first_round_thinking_buf,
                    )
                    return

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    continue

                # ===== 分支 2: 正常文本/thinking =====

                reasoning_delta = ""
                if should_use_reasoning:
                    reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                    if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
                        reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""

                content_delta = delta.content or ""

                if reasoning_delta and content_delta == reasoning_delta:
                    content_delta = ""

                # thinking 处理：始终实时推送，缓冲模式下额外记录用于持久化决策
                if reasoning_delta:
                    await append_chunk(conversation_id, "reasoning", reasoning_delta, thinking_block_id)
                    if first_round_buffering:
                        first_round_thinking_buf += reasoning_delta
                    else:
                        reasoning_buf += reasoning_delta

                # content 出现意味着第一轮结束且没有 tool_call → 缓冲的 thinking 转入正式记录
                if content_delta:
                    if first_round_buffering:
                        reasoning_buf += first_round_thinking_buf
                        first_round_thinking_buf = ""
                        first_round_buffering = False

                    content_buf += content_delta
                    await append_chunk(conversation_id, "answering", content_delta, text_block_id)

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

                chunk_count += 1
                if chunk_count % LOCK_CHECK_INTERVAL == 0:
                    if not await check_lock_owner(conversation_id, task_id):
                        logger.info(f"任务被踢掉，主动退出: conv_id={conversation_id}")
                        # 合并已确认的 reasoning 和可能残留的第一轮 thinking
                        all_reasoning = reasoning_buf + first_round_thinking_buf
                        if all_reasoning or content_buf:
                            self._persist_message(
                                db,
                                assistant_message_id,
                                conversation_id,
                                model_id,
                                self._build_content_blocks(
                                    all_reasoning, content_buf, thinking_block_id, text_block_id
                                ),
                                usage_data,
                            )
                        await finalize_stream(conversation_id, success=False, error_msg="被新请求取代", task_id=task_id)
                        return

            # 回放未消费的第一轮 thinking（极端情况：没有 tool_call 也没有 content）
            if first_round_thinking_buf:
                await append_chunk(conversation_id, "reasoning", first_round_thinking_buf, thinking_block_id)
                reasoning_buf += first_round_thinking_buf

            # 生成完成，落库
            self._persist_message(
                db,
                assistant_message_id,
                conversation_id,
                model_id,
                self._build_content_blocks(reasoning_buf, content_buf, thinking_block_id, text_block_id),
                usage_data,
            )
            await finalize_stream(conversation_id, success=True, task_id=task_id)

            # 异步提取用户记忆（fire-and-forget，失败不影响主流程）
            asyncio.create_task(
                self._extract_user_memories(conversation_id, user_id)
            )

        except asyncio.CancelledError:
            logger.info(f"任务被取消: conv_id={conversation_id}")
            if reasoning_buf or content_buf:
                self._persist_message(
                    db,
                    assistant_message_id,
                    conversation_id,
                    model_id,
                    self._build_content_blocks(reasoning_buf, content_buf, thinking_block_id, text_block_id),
                    usage_data,
                )
            await finalize_stream(conversation_id, success=False, error_msg="用户中止", task_id=task_id)
            raise

        except Exception as e:
            logger.error(f"生成异常: conv_id={conversation_id}, error={e}")
            if reasoning_buf or content_buf:
                self._persist_message(
                    db,
                    assistant_message_id,
                    conversation_id,
                    model_id,
                    self._build_content_blocks(reasoning_buf, content_buf, thinking_block_id, text_block_id),
                    usage_data,
                )
            await finalize_stream(conversation_id, success=False, error_msg=str(e), task_id=task_id)

        finally:
            db.close()

    # ──────────────────────────────────────────────
    # Tool Call 处理
    # ──────────────────────────────────────────────

    async def _handle_tool_call(
        self,
        db,
        conversation_id: str,
        user_id: str,
        model_id: str,
        litellm_model: str,
        litellm_kwargs: dict,
        provider: str,
        messages: list[dict],
        assistant_message_id: str,
        task_id: str,
        tool_call_id: str,
        tool_call_name: str,
        tool_call_args: str,
        should_use_reasoning: bool,
        thinking_block_id: str,
        text_block_id: str,
        first_round_reasoning: str = "",
    ) -> None:
        """
        处理 LLM 返回的 tool_call。当前仅支持 web_search。
        完整流程：search_start SSE → 调搜索 → search_complete SSE → 第二轮 LLM → 落库 + finalize。
        """
        from app.services.search_client import search_web

        if tool_call_name != "web_search":
            logger.warning(f"未知的 tool_call: {tool_call_name}，降级为无搜索回答")
            await self._fallback_no_search(
                db,
                conversation_id,
                model_id,
                litellm_model,
                litellm_kwargs,
                provider,
                messages,
                assistant_message_id,
                task_id,
                should_use_reasoning,
                thinking_block_id,
                text_block_id,
            )
            return

        # 1. 解析搜索 query
        try:
            args = json.loads(tool_call_args)
            query = args.get("query", "")
        except json.JSONDecodeError:
            query = ""

        if not query:
            logger.warning("tool_call web_search 的 query 为空，降级为无搜索回答")
            await self._fallback_no_search(
                db,
                conversation_id,
                model_id,
                litellm_model,
                litellm_kwargs,
                provider,
                messages,
                assistant_message_id,
                task_id,
                should_use_reasoning,
                thinking_block_id,
                text_block_id,
            )
            return

        # 2. 推送 search_start SSE 事件
        search_block_id = f"blk_{uuid.uuid4().hex[:12]}"
        await append_chunk(
            conversation_id,
            "search_start",
            json.dumps({"query": query}, ensure_ascii=False),
            search_block_id,
        )

        # 3. 调用 search-service
        sources = await search_web(query, count=5)

        # 4. 推送 search_complete SSE 事件
        await append_chunk(
            conversation_id,
            "search_complete",
            json.dumps(
                {
                    "query": query,
                    "sources": [s.model_dump() for s in sources],
                },
                ensure_ascii=False,
            ),
            search_block_id,
        )

        # 5. 构造第二轮 LLM 调用的消息（注入搜索上下文）
        search_context = self._format_search_context(sources)
        # 推理模型（DeepSeek 等）要求 assistant 消息包含 reasoning_content，
        # 否则 API 报 400：thinking is enabled but reasoning_content is missing
        assistant_tool_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "web_search", "arguments": tool_call_args},
                }
            ],
        }
        if should_use_reasoning:
            assistant_tool_msg["reasoning_content"] = first_round_reasoning or ""
        augmented_messages = messages + [
            assistant_tool_msg,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": search_context,
            },
        ]

        # 6. 第二轮 LLM 流式调用（不再传 tools，避免无限循环）
        # 第一轮 reasoning 是"决定要不要搜索"的内部推理（含 tool_call 细节），
        # 对用户没有价值，丢弃。用新的 block ID 接收第二轮的有效 reasoning。
        second_thinking_id = f"blk_{uuid.uuid4().hex[:12]}"
        reasoning_buf, content_buf, usage_data = await self._stream_llm_to_redis(
            litellm_model=litellm_model,
            litellm_kwargs=litellm_kwargs,
            messages=augmented_messages,
            conversation_id=conversation_id,
            task_id=task_id,
            should_use_reasoning=should_use_reasoning,
            thinking_block_id=second_thinking_id,
            text_block_id=text_block_id,
        )

        # 7. 落库：content 数组包含 SearchBlock
        # 只保留第二轮 reasoning（基于搜索结果的分析思考），不保留第一轮
        content_blocks = self._build_content_blocks(
            reasoning_buf,
            content_buf,
            second_thinking_id,
            text_block_id,
            search_query=query,
            search_sources=sources,
            search_block_id=search_block_id,
        )
        self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, usage_data)
        await finalize_stream(conversation_id, success=True, task_id=task_id)

        # 异步提取用户记忆
        asyncio.create_task(
            self._extract_user_memories(conversation_id, user_id)
        )

    async def _fallback_no_search(
        self,
        db,
        conversation_id: str,
        model_id: str,
        litellm_model: str,
        litellm_kwargs: dict,
        provider: str,
        messages: list[dict],
        assistant_message_id: str,
        task_id: str,
        should_use_reasoning: bool,
        thinking_block_id: str,
        text_block_id: str,
    ) -> None:
        """tool_call 失败时的降级路径：不带 tools 重新调用 LLM"""
        reasoning_buf, content_buf, usage_data = await self._stream_llm_to_redis(
            litellm_model=litellm_model,
            litellm_kwargs=litellm_kwargs,
            messages=messages,
            conversation_id=conversation_id,
            task_id=task_id,
            should_use_reasoning=should_use_reasoning,
            thinking_block_id=thinking_block_id,
            text_block_id=text_block_id,
        )
        content_blocks = self._build_content_blocks(reasoning_buf, content_buf, thinking_block_id, text_block_id)
        self._persist_message(db, assistant_message_id, conversation_id, model_id, content_blocks, usage_data)
        await finalize_stream(conversation_id, success=True, task_id=task_id)

    async def _stream_llm_to_redis(
        self,
        litellm_model: str,
        litellm_kwargs: dict,
        messages: list[dict],
        conversation_id: str,
        task_id: str,
        should_use_reasoning: bool,
        thinking_block_id: str,
        text_block_id: str,
    ) -> tuple:
        """
        通用的 LLM 流式调用 + 写 Redis Stream 方法。
        不传 tools，纯流式输出。用于第二轮调用和 fallback 场景。
        返回 (reasoning_buf, content_buf, usage_data)。
        """
        reasoning_buf = ""
        content_buf = ""
        usage_data: Optional[Usage] = None
        chunk_count = 0

        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **litellm_kwargs,
        )

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

            chunk_count += 1
            if chunk_count % LOCK_CHECK_INTERVAL == 0:
                if not await check_lock_owner(conversation_id, task_id):
                    logger.info(f"第二轮调用被踢掉: conv_id={conversation_id}")
                    break

        return reasoning_buf, content_buf, usage_data

    # ──────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _format_search_context(sources: List[SearchSource]) -> str:
        """将搜索结果格式化为 LLM 可消费的上下文文本"""
        if not sources:
            return "搜索未返回结果。请基于你的知识回答用户的问题。"

        parts = ["以下是从网络搜索获取的参考信息，请结合这些信息回答用户的问题。"]
        parts.append("如果引用了某条信息，请在相关内容后标注来源编号，格式为 [1]、[2] 等。\n")

        for i, source in enumerate(sources, 1):
            parts.append(f"[{i}] {source.title}")
            parts.append(f"    来源: {source.url}")
            # 优先使用正文内容，没有则用 description 摘要
            if source.content:
                # 截取前 1000 字，避免上下文过长
                content_text = source.content[:1000]
                parts.append(f"    正文: {content_text}")
            else:
                parts.append(f"    摘要: {source.description}")
            parts.append("")

        parts.append("注意：")
        parts.append("- 优先使用搜索结果中的信息回答")
        parts.append("- 如果搜索结果不足以回答，可以结合自身知识补充")
        parts.append("- 引用时使用 [n] 格式标注来源编号")
        parts.append("- 直接回答问题，不要再发起搜索或输出任何工具调用指令")

        return "\n".join(parts)

    @staticmethod
    def _build_content_blocks(
        reasoning_buf: str,
        content_buf: str,
        thinking_block_id: str,
        text_block_id: str,
        search_query: str = "",
        search_sources: Optional[List[SearchSource]] = None,
        search_block_id: str = "",
    ) -> list:
        """构建 assistant 消息的 content blocks 数组"""
        blocks = []
        if reasoning_buf:
            blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
        if search_query and search_sources is not None:
            blocks.append(SearchBlock(type="search", id=search_block_id, query=search_query, sources=search_sources))
        if content_buf:
            blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
        return blocks

    def _persist_message(
        self,
        db,
        assistant_message_id: str,
        conversation_id: str,
        model_id: str,
        content_blocks: list,
        usage_data: Optional[Usage],
    ) -> None:
        """将 assistant 消息写入 PostgreSQL"""
        try:
            from app.db.models import Message as MessageModel

            db_message = MessageModel(
                id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                content=[block.model_dump() for block in content_blocks],
                model_id=model_id,
                usage=usage_data.model_dump() if usage_data else None,
            )
            db.add(db_message)
            db.commit()
        except Exception as e:
            logger.error(f"写入 assistant 消息失败: {e}")
            db.rollback()

    @staticmethod
    async def _extract_user_memories(conversation_id: str, user_id: str) -> None:
        """异步提取用户记忆，使用独立的 DB Session"""
        db = SessionLocal()
        try:
            from app.services.user_memory_service import UserMemoryService
            service = UserMemoryService(db)
            await service.extract_memories(conversation_id, user_id)
        except Exception as e:
            logger.warning(f"异步记忆提取失败: conv_id={conversation_id}, error={e}")
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
