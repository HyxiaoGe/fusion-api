# app/services/stream_handler.py
"""
流式处理器 — 基于 Redis Stream 的两段式架构

Part A: generate_to_redis() — 后台任务，调用 LLM 写 Redis Stream + 落库 PostgreSQL
Part B: stream_redis_as_sse() — SSE 读取器，从 Redis Stream 消费推送给客户端
"""
import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional

import litellm

from app.constants.chat import FinishReasons
from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.schemas.chat import (
    Message, TextBlock, ThinkingBlock, Usage,
    StreamChunk, StreamChoice, StreamDelta,
)
from app.services.stream_state_service import (
    init_stream,
    append_chunk,
    finalize_stream,
    check_lock_owner,
    get_stream_meta,
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
        messages: list[dict],
        assistant_message_id: str,
        task_id: str,
        options: Optional[dict] = None,
    ) -> None:
        """
        后台任务：调用 LLM，chunk 写入 Redis Stream，完成后落库 PostgreSQL。
        生命周期完全独立于 HTTP 连接。
        """
        if options is None:
            options = {}

        use_reasoning = options.get("use_reasoning")
        should_use_reasoning = (
            use_reasoning is True
            or (use_reasoning is None and provider in self.REASONING_PROVIDERS)
        )

        thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
        text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

        reasoning_buf = ""
        content_buf = ""
        usage_data: Optional[Usage] = None
        chunk_count = 0

        # 初始化 Redis Stream
        await init_stream(conversation_id, str(user_id), model_id, assistant_message_id, task_id)

        # 后台任务独立管理 DB Session
        db = SessionLocal()

        try:
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

                # 提取 reasoning_content
                reasoning_delta = ""
                if should_use_reasoning:
                    reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                    if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
                        reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""

                content_delta = delta.content or ""

                # 去重
                if reasoning_delta and content_delta == reasoning_delta:
                    content_delta = ""

                # 写 reasoning chunk 到 Redis Stream
                if reasoning_delta:
                    reasoning_buf += reasoning_delta
                    await append_chunk(conversation_id, "reasoning", reasoning_delta, thinking_block_id)

                # 写 content chunk 到 Redis Stream
                if content_delta:
                    content_buf += content_delta
                    await append_chunk(conversation_id, "answering", content_delta, text_block_id)

                # usage
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

                # 每 N 个 chunk 检查锁
                chunk_count += 1
                if chunk_count % LOCK_CHECK_INTERVAL == 0:
                    if not await check_lock_owner(conversation_id, task_id):
                        logger.info(f"任务被踢掉，主动退出: conv_id={conversation_id}")
                        return

            # 生成完成，落库 PostgreSQL
            final_blocks = []
            if reasoning_buf:
                final_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
            if content_buf:
                final_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))

            assistant_message = Message(
                id=assistant_message_id,
                role="assistant",
                content=final_blocks,
                model_id=model_id,
                usage=usage_data,
            )
            self._persist_message(db, assistant_message, conversation_id)

            # 标记 Stream 正常结束
            await finalize_stream(conversation_id, success=True)

        except asyncio.CancelledError:
            # 任务被取消（用户手动 stop 或新消息踢掉）
            # 把已有内容落库，保证不丢数据
            logger.info(f"任务被取消: conv_id={conversation_id}")
            if reasoning_buf or content_buf:
                final_blocks = []
                if reasoning_buf:
                    final_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                if content_buf:
                    final_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                assistant_message = Message(
                    id=assistant_message_id,
                    role="assistant",
                    content=final_blocks,
                    model_id=model_id,
                    usage=usage_data,
                )
                self._persist_message(db, assistant_message, conversation_id)
            await finalize_stream(conversation_id, success=False, error_msg="用户中止")
            raise  # 必须 re-raise

        except Exception as e:
            logger.error(f"生成异常: conv_id={conversation_id}, error={e}")
            # 异常时也尝试保存已有内容
            if reasoning_buf or content_buf:
                final_blocks = []
                if reasoning_buf:
                    final_blocks.append(ThinkingBlock(type="thinking", id=thinking_block_id, thinking=reasoning_buf))
                if content_buf:
                    final_blocks.append(TextBlock(type="text", id=text_block_id, text=content_buf))
                assistant_message = Message(
                    id=assistant_message_id,
                    role="assistant",
                    content=final_blocks,
                    model_id=model_id,
                    usage=usage_data,
                )
                self._persist_message(db, assistant_message, conversation_id)
            await finalize_stream(conversation_id, success=False, error_msg=str(e))

        finally:
            db.close()

    def _persist_message(self, db, message: Message, conversation_id: str) -> None:
        """将 assistant 消息写入 PostgreSQL"""
        try:
            from app.db.models import Message as MessageModel
            db_message = MessageModel(
                id=message.id,
                conversation_id=conversation_id,
                role=message.role,
                content=[block.model_dump() for block in message.content],
                model_id=message.model_id,
                usage=message.usage.model_dump() if message.usage else None,
            )
            db.add(db_message)
            db.commit()
        except Exception as e:
            logger.error(f"写入 assistant 消息失败: {e}")
            db.rollback()


async def stream_redis_as_sse(
    conversation_id: str,
    message_id: str,
    last_entry_id: str = "0",
) -> AsyncGenerator[str, None]:
    """
    SSE 读取器：从 Redis Stream 读 chunk，格式化为 SSE 事件推送给客户端。
    不调用 LLM，只读 Redis。

    每条 SSE 事件包含 id 行（Redis entry ID），供断线重连使用。
    """
    async for chunk in read_stream_chunks(conversation_id, last_entry_id):
        entry_id = chunk.pop("entry_id")
        chunk_type = chunk.get("type")

        # 跳过 start 标记
        if chunk_type == "start":
            continue

        # 构造对齐前端 StreamChunkPayload 格式的 payload
        if chunk_type == "reasoning":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [{
                    "delta": {
                        "content": [{
                            "type": "thinking",
                            "id": chunk.get("block_id", ""),
                            "thinking": chunk["content"],
                        }]
                    },
                    "finish_reason": None,
                }]
            }
        elif chunk_type == "answering":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [{
                    "delta": {
                        "content": [{
                            "type": "text",
                            "id": chunk.get("block_id", ""),
                            "text": chunk["content"],
                        }]
                    },
                    "finish_reason": None,
                }]
            }
        elif chunk_type == "done":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop",
                }]
            }
        elif chunk_type == "error":
            payload = {
                "id": message_id,
                "conversation_id": conversation_id,
                "choices": [{
                    "delta": {},
                    "finish_reason": "error",
                }]
            }
        else:
            continue

        yield f"id: {entry_id}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
