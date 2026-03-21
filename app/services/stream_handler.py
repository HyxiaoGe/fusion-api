import asyncio
from typing import AsyncGenerator, Optional

from app.ai.llm_manager import llm_manager
from app.constants import FinishReasons, MessageRoles, MessageTypes
from app.core.logger import app_logger as logger
from app.schemas.chat import Message
from app.services.stream_serializer import StreamSerializer


class StreamHandler:
    REASONING_PROVIDERS = {"deepseek", "qwen", "xai", "volcengine"}

    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service

    async def generate_stream(
        self,
        provider,
        model,
        messages,
        conversation_id,
        options=None,
        turn_id=None,
    ) -> AsyncGenerator[str, None]:
        if options is None:
            options = {}

        use_reasoning = options.get("use_reasoning")
        should_use_reasoning = (
            use_reasoning is True
            or (use_reasoning is None and provider in self.REASONING_PROVIDERS)
        )

        stream = (
            self._stream_with_reasoning(provider, model, messages, conversation_id, options, turn_id)
            if should_use_reasoning
            else self._stream_normal(provider, model, messages, conversation_id, options, turn_id)
        )

        async for event in stream:
            yield event

    @staticmethod
    def _normalize_direct_stream_messages(messages) -> list[dict]:
        """将多种消息对象归一化为 OpenAI 兼容消息格式。"""
        openai_messages = []

        for msg in messages:
            msg_dict = None

            try:
                if hasattr(msg, "dict") and callable(msg.dict):
                    msg_dict = msg.dict()
                elif hasattr(msg, "model_dump") and callable(msg.model_dump):
                    msg_dict = msg.model_dump()

                if msg_dict and "type" in msg_dict:
                    msg_type = msg_dict.get("type", "")
                    msg_content = msg_dict.get("content", "")

                    if msg_type == "human":
                        openai_messages.append({"role": "user", "content": msg_content})
                    elif msg_type == "ai":
                        openai_messages.append({"role": "assistant", "content": msg_content})
                    elif msg_type == "system":
                        openai_messages.append({"role": "system", "content": msg_content})
                    continue
            except Exception as exc:
                logger.warning(f"跳过无法转换的消息对象: {exc}")

            class_name = msg.__class__.__name__
            if class_name == "HumanMessage":
                openai_messages.append({"role": "user", "content": msg.content})
            elif class_name == "AIMessage":
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif class_name == "SystemMessage":
                openai_messages.append({"role": "system", "content": msg.content})
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                if msg.type == "human":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                elif msg.type == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                else:
                    logger.warning(f"跳过无法识别的消息 type: {msg.type}")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                openai_messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                openai_messages.append(msg)
            else:
                logger.warning(f"跳过无法识别的消息格式: {type(msg)}")

        return openai_messages

    @staticmethod
    def _extract_chunk_content(chunk) -> str:
        if hasattr(chunk, "content"):
            return chunk.content or ""
        return chunk if isinstance(chunk, str) else ""

    @staticmethod
    def _extract_openai_delta(chunk):
        choices = getattr(chunk, "choices", None)
        if not choices:
            return None
        choice = choices[0]
        return getattr(choice, "delta", None)

    def _extract_reasoning_content(self, chunk) -> str:
        additional_kwargs = getattr(chunk, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            reasoning_content = additional_kwargs.get("reasoning_content")
            if reasoning_content:
                return reasoning_content

        delta = self._extract_openai_delta(chunk)
        if delta is None:
            return ""

        if isinstance(delta, dict):
            return delta.get("reasoning_content") or ""
        return getattr(delta, "reasoning_content", "") or ""

    def _extract_openai_content(self, chunk) -> str:
        delta = self._extract_openai_delta(chunk)
        if delta is None:
            return ""

        if isinstance(delta, dict):
            return delta.get("content") or ""
        return getattr(delta, "content", "") or ""

    def _extract_finish_reason(self, chunk) -> Optional[str]:
        choices = getattr(chunk, "choices", None)
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
            if finish_reason:
                return finish_reason

        response_metadata = getattr(chunk, "response_metadata", None)
        if isinstance(response_metadata, dict) and response_metadata.get("finish_reason"):
            return response_metadata["finish_reason"]

        generation_info = getattr(chunk, "generation_info", None)
        if isinstance(generation_info, dict) and generation_info.get("finish_reason"):
            return generation_info["finish_reason"]

        additional_kwargs = getattr(chunk, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and additional_kwargs.get("finish_reason"):
            return additional_kwargs["finish_reason"]

        return None

    async def _create_placeholder_message(self, conversation_id: str, message_type: str, turn_id: Optional[str]) -> Message:
        placeholder_message = Message(
            role=MessageRoles.ASSISTANT,
            type=message_type,
            content="",
            turn_id=turn_id,
        )
        return self.memory_service.create_message(placeholder_message, conversation_id)

    async def _persist_accumulated_content(
        self,
        assistant_message_id: str,
        answer_text: str,
        reasoning_message_id: Optional[str] = None,
        reasoning_text: str = "",
    ):
        if reasoning_message_id and reasoning_text:
            await self.update_stream_response(reasoning_message_id, reasoning_text)
        if answer_text:
            await self.update_stream_response(assistant_message_id, answer_text)

    async def update_stream_response(self, message_id: str, response_text: str):
        try:
            self.memory_service.update_message(message_id, {"content": response_text})
            self.db.commit()
        except Exception as exc:
            logger.error(f"更新流式响应失败: {exc}")

    async def _stream_normal(
        self,
        provider,
        model,
        messages,
        conversation_id,
        options=None,
        turn_id=None,
    ) -> AsyncGenerator[str, None]:
        if options is None:
            options = {}

        assistant_message = await self._create_placeholder_message(
            conversation_id,
            MessageTypes.ASSISTANT_CONTENT,
            turn_id,
        )

        answer_result = ""
        finish_reason = FinishReasons.STOP

        yield StreamSerializer.init_chunk(assistant_message.id, conversation_id)

        try:
            llm = llm_manager.get_model(provider=provider, model=model, options=options)
            for chunk in llm.stream(messages):
                observed_finish_reason = self._extract_finish_reason(chunk)
                if observed_finish_reason:
                    finish_reason = observed_finish_reason

                content = self._extract_chunk_content(chunk) or self._extract_openai_content(chunk)
                if not content:
                    continue

                answer_result += content
                yield StreamSerializer.content_chunk(assistant_message.id, conversation_id, content)
                await asyncio.sleep(0)

            await self._persist_accumulated_content(assistant_message.id, answer_result)
            yield StreamSerializer.finish_chunk(
                assistant_message.id,
                conversation_id,
                finish_reason=finish_reason,
            )
            yield StreamSerializer.done_marker()
        except Exception as exc:
            await self._persist_accumulated_content(assistant_message.id, answer_result)
            logger.exception(f"普通流式响应失败: {exc}")
            yield StreamSerializer.error_chunk(assistant_message.id, conversation_id, str(exc))
            yield StreamSerializer.done_marker()

    async def _stream_with_reasoning(
        self,
        provider,
        model,
        messages,
        conversation_id,
        options=None,
        turn_id=None,
    ) -> AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        if options is None:
            options = {}

        reasoning_message = await self._create_placeholder_message(
            conversation_id,
            MessageTypes.REASONING_CONTENT,
            turn_id,
        )
        assistant_message = await self._create_placeholder_message(
            conversation_id,
            MessageTypes.ASSISTANT_CONTENT,
            turn_id,
        )

        reasoning_result = ""
        answer_result = ""
        finish_reason = FinishReasons.STOP

        yield StreamSerializer.init_chunk(assistant_message.id, conversation_id)

        if provider == "volcengine":
            try:
                credentials = llm_manager._get_model_credentials(provider, model)
                if not credentials:
                    raise ValueError(f"未找到{provider}的API凭证")

                client = AsyncOpenAI(
                    api_key=credentials.get("api_key"),
                    base_url=credentials.get("base_url"),
                    timeout=60 * 30,
                )
                openai_messages = self._normalize_direct_stream_messages(messages)
                if not openai_messages:
                    raise ValueError("无有效消息可发送，请检查消息格式")

                stream = await client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    stream=True,
                )

                async for chunk in stream:
                    observed_finish_reason = self._extract_finish_reason(chunk)
                    if observed_finish_reason:
                        finish_reason = observed_finish_reason

                    reasoning_content = self._extract_reasoning_content(chunk)
                    if reasoning_content:
                        reasoning_result += reasoning_content
                        yield StreamSerializer.reasoning_chunk(
                            assistant_message.id,
                            conversation_id,
                            reasoning_content,
                        )

                    content = self._extract_openai_content(chunk) or self._extract_chunk_content(chunk)
                    if content and content != reasoning_result:
                        answer_result += content
                        yield StreamSerializer.content_chunk(
                            assistant_message.id,
                            conversation_id,
                            content,
                        )

                    await asyncio.sleep(0)

                await self._persist_accumulated_content(
                    assistant_message.id,
                    answer_result,
                    reasoning_message.id,
                    reasoning_result,
                )
                yield StreamSerializer.finish_chunk(
                    assistant_message.id,
                    conversation_id,
                    finish_reason=finish_reason,
                )
                yield StreamSerializer.done_marker()
            except Exception as exc:
                await self._persist_accumulated_content(
                    assistant_message.id,
                    answer_result,
                    reasoning_message.id,
                    reasoning_result,
                )
                logger.exception(f"推理流式响应失败: {exc}")
                yield StreamSerializer.error_chunk(assistant_message.id, conversation_id, str(exc))
                yield StreamSerializer.done_marker()
            return

        stream_kwargs = {"reasoning_effort": "medium"} if provider == "deepseek" else {}

        try:
            llm = llm_manager.get_model(provider=provider, model=model, options=options)
            for chunk in llm.stream(messages, **stream_kwargs):
                observed_finish_reason = self._extract_finish_reason(chunk)
                if observed_finish_reason:
                    finish_reason = observed_finish_reason

                reasoning_content = self._extract_reasoning_content(chunk)
                if reasoning_content:
                    reasoning_result += reasoning_content
                    yield StreamSerializer.reasoning_chunk(
                        assistant_message.id,
                        conversation_id,
                        reasoning_content,
                    )

                content = self._extract_chunk_content(chunk) or self._extract_openai_content(chunk)
                if content and content != reasoning_result:
                    answer_result += content
                    yield StreamSerializer.content_chunk(
                        assistant_message.id,
                        conversation_id,
                        content,
                    )

                await asyncio.sleep(0)

            await self._persist_accumulated_content(
                assistant_message.id,
                answer_result,
                reasoning_message.id,
                reasoning_result,
            )
            yield StreamSerializer.finish_chunk(
                assistant_message.id,
                conversation_id,
                finish_reason=finish_reason,
            )
            yield StreamSerializer.done_marker()
        except Exception as exc:
            await self._persist_accumulated_content(
                assistant_message.id,
                answer_result,
                reasoning_message.id,
                reasoning_result,
            )
            logger.exception(f"推理流式响应失败: {exc}")
            yield StreamSerializer.error_chunk(assistant_message.id, conversation_id, str(exc))
            yield StreamSerializer.done_marker()
