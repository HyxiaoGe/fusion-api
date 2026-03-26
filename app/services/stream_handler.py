# app/services/stream_handler.py
import uuid
from typing import AsyncGenerator, Optional

import litellm
from sqlalchemy.orm import Session

from app.constants.chat import FinishReasons
from app.core.logger import app_logger as logger
from app.schemas.chat import (
    Message, TextBlock, ThinkingBlock, Usage,
    StreamChunk, StreamChoice, StreamDelta,
)


class StreamHandler:
    """
    负责 SSE 流式响应的生成和落库。
    输出格式对齐 StreamChunk schema，流式和历史消息结构统一。
    """

    # 原生支持 reasoning_content 的 provider
    REASONING_PROVIDERS = {"deepseek", "qwen", "xai", "volcengine"}

    def __init__(self, db: Session, memory_service):
        self.db = db
        self.memory_service = memory_service

    async def generate_stream(
        self,
        litellm_model: str,
        provider: str,
        model_id: str,
        litellm_kwargs: dict,
        messages: list[dict],
        conversation_id: str,
        options: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        生成 SSE 流式响应。
        messages 为标准 OpenAI 格式的 dict 列表。
        """
        if options is None:
            options = {}

        use_reasoning = options.get("use_reasoning")
        should_use_reasoning = (
            use_reasoning is True
            or (use_reasoning is None and provider in self.REASONING_PROVIDERS)
        )

        # 预分配 message id 和 block id，整个流保持不变
        assistant_message_id = str(uuid.uuid4())
        thinking_block_id = f"blk_{uuid.uuid4().hex[:12]}"
        text_block_id = f"blk_{uuid.uuid4().hex[:12]}"

        # 发送初始心跳帧，客户端据此确认连接建立
        yield _serialize(StreamChunk(
            id=assistant_message_id,
            conversation_id=conversation_id,
            choices=[StreamChoice(delta=StreamDelta())],
        ))

        reasoning_buf = ""
        content_buf = ""
        usage_data: Optional[Usage] = None

        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},  # 要求最后一帧带 usage
                **litellm_kwargs,
            )

            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None

                # 最后一帧可能只有 usage，没有 choice
                if not choice:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = Usage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                        )
                    continue

                delta = choice.delta

                # 提取 reasoning_content（部分 provider 通过 model_extra 返回）
                reasoning_delta = ""
                if should_use_reasoning:
                    reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                    if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
                        reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""

                content_delta = delta.content or ""

                # 去重：部分 provider 会在 content 里重复 reasoning 内容
                if reasoning_delta and content_delta == reasoning_delta:
                    content_delta = ""

                # 发送 thinking 增量
                if reasoning_delta:
                    reasoning_buf += reasoning_delta
                    yield _serialize(StreamChunk(
                        id=assistant_message_id,
                        conversation_id=conversation_id,
                        choices=[StreamChoice(delta=StreamDelta(content=[
                            ThinkingBlock(id=thinking_block_id, thinking=reasoning_delta)
                        ]))],
                    ))

                # 发送 text 增量
                if content_delta:
                    content_buf += content_delta
                    yield _serialize(StreamChunk(
                        id=assistant_message_id,
                        conversation_id=conversation_id,
                        choices=[StreamChoice(delta=StreamDelta(content=[
                            TextBlock(id=text_block_id, text=content_delta)
                        ]))],
                    ))

                # 部分 provider 在中间帧就带 usage
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

            # 构造完整 assistant 消息落库
            final_blocks = []
            if reasoning_buf:
                final_blocks.append(ThinkingBlock(id=thinking_block_id, thinking=reasoning_buf))
            if content_buf:
                final_blocks.append(TextBlock(id=text_block_id, text=content_buf))

            assistant_message = Message(
                id=assistant_message_id,
                role="assistant",
                content=final_blocks,
                model_id=model_id,
                usage=usage_data,
            )
            await self._persist_message(assistant_message, conversation_id)

            # 发送结束帧（携带完整 usage）
            yield _serialize(StreamChunk(
                id=assistant_message_id,
                conversation_id=conversation_id,
                choices=[StreamChoice(
                    delta=StreamDelta(),
                    finish_reason=FinishReasons.STOP,
                )],
                usage=usage_data,
            ))

        except Exception as e:
            logger.error(f"流式处理异常 [{litellm_model}]: {e}")
            yield _serialize(StreamChunk(
                id=assistant_message_id,
                conversation_id=conversation_id,
                choices=[StreamChoice(
                    delta=StreamDelta(),
                    finish_reason=FinishReasons.ERROR,
                )],
            ))

        finally:
            yield "data: [DONE]\n\n"

    async def _persist_message(self, message: Message, conversation_id: str) -> None:
        """将 assistant 消息写入数据库"""
        try:
            self.memory_service.create_message(message, conversation_id)
            self.db.commit()
        except Exception as e:
            logger.error(f"写入 assistant 消息失败: {e}")
            self.db.rollback()


def _serialize(chunk: StreamChunk) -> str:
    """将 StreamChunk 序列化为 SSE 字符串"""
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
