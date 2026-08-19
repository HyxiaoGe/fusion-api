# app/services/chat_service.py
import asyncio
import uuid as uuid_mod
from typing import Any, Dict, List, Mapping, Optional, Union

import litellm
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.ai import litellm_catalog
from app.ai.llm_manager import llm_manager
from app.ai.llm_observability import merge_litellm_kwargs
from app.ai.prompts import prompt_manager
from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.model_catalog_control_repository import ModelCatalogControlRepository
from app.db.repositories import ConversationRepository, FileRepository
from app.schemas.chat import (
    ChatResponse,
    ClientPartialContentBlock,
    Conversation,
    FileBlock,
    KnowledgeEvidenceBlock,
    Message,
    TextBlock,
    Usage,
)
from app.schemas.response import ApiException, ErrorCode
from app.services.agent.context_broker import submit_context_result
from app.services.agent.continuation import (
    build_continuation_context,
    get_continuation_system_prompt,
)
from app.services.agent_strategy_config import get_agent_tools_disabled_aliases
from app.services.chat.context_manager import (
    ContextBudgetExceededError,
    ContextEstimationUnavailableError,
    prepare_context,
)
from app.services.chat.message_builder import (
    build_llm_messages,
    inject_file_content,
    is_image_file,
)
from app.services.chat.model_call_language_policy import finalize_model_call_language_policy
from app.services.conversation_service import ConversationService
from app.services.file_service import FileService, is_image_mime
from app.services.knowledge.chat_grounding import (
    KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT,
    inject_knowledge_grounding_messages,
    prepare_knowledge_grounding,
    validate_grounded_answer,
    validate_knowledge_query,
)
from app.services.storage import get_storage_for_backend
from app.services.stream import StreamHandler, stream_redis_as_sse
from app.services.stream.agent_loop_request_prep import (
    inject_no_tool_network_boundary,
    inject_no_vision_file_boundary,
    normalize_controlled_max_tokens,
)
from app.services.stream.agent_task_policy import resolve_agent_task_policy
from app.services.stream.persistence import acquire_message_persistence_lock, merge_partial_content_blocks
from app.services.stream.runner import _agent_loop_limits
from app.services.stream_state_service import StreamInitResult, finalize_stream, get_stream_meta, init_stream
from app.services.suggested_question_service import (
    SuggestedQuestionGenerationResult,
    SuggestedQuestionService,
)
from app.services.task_manager import register_task


def _require_stream_initialized(result: StreamInitResult) -> None:
    if result.ok:
        return
    logger.error(
        "Redis Stream 初始化失败，拒绝启动生成: code=%s, error=%s",
        result.error_code,
        result.message,
    )
    raise ApiException.service_unavailable(
        result.message
        if result.error_code == "stream_stop_in_progress" and result.message
        else "生成服务暂时不可用，请稍后重试",
        code=ErrorCode.STREAM_UNAVAILABLE,
    )


def _get_model_capabilities(model_id: str) -> dict[str, Any]:
    return litellm_catalog.get_capabilities(
        model_id,
        agent_tools_disabled_aliases=get_agent_tools_disabled_aliases(),
    )


def _continuation_original_user_text(
    messages: list[Any],
    *,
    assistant_message_id: str,
) -> str:
    """找出被续写回答对应的上一条用户原文；只读取 text block。"""

    assistant_index = next(
        (index for index, message in enumerate(messages) if str(getattr(message, "id", "")) == assistant_message_id),
        len(messages),
    )
    for message in reversed(messages[:assistant_index]):
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return ""
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                text = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                text = getattr(block, "text", None)
            if block_type == "text" and isinstance(text, str) and text:
                text_parts.append(text)
        return "\n".join(text_parts)[:4_000]
    return ""


def _retry_user_payload(message: Message) -> tuple[str, list[str]]:
    """从已持久化 user 消息恢复重试输入，禁止客户端改写历史问题。"""

    text_parts: list[str] = []
    file_ids: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, FileBlock):
            file_ids.append(block.file_id)
    return "".join(text_parts), file_ids


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.conversation_service = ConversationService(db)
        self.file_repo = FileRepository(db)
        self.model_control_repository = ModelCatalogControlRepository(db)
        self.stream_handler = StreamHandler()
        self.suggested_question_service = SuggestedQuestionService(db)

    def _validate_message_files(self, file_ids: List[str], user_id: str, conversation_id: str) -> List[Any]:
        """校验本次消息引用的文件，并按传入顺序返回文件记录。"""
        validated_files: List[Any] = []
        for file_id in file_ids:
            file_info = self.file_repo.get_file_by_id(file_id, user_id=user_id)
            if not file_info:
                raise ApiException.bad_request("文件不存在或无权访问")

            if not self.file_repo.is_file_linked_to_conversation(conversation_id, file_id):
                raise ApiException.bad_request("文件不属于当前会话")

            if is_image_mime(file_info.mimetype or ""):
                if file_info.status != "processed" or not getattr(file_info, "storage_key", None):
                    raise ApiException.bad_request("图片文件不可用，请重新上传")
            elif file_info.status != "processed":
                raise ApiException.bad_request("文件仍在处理，请稍后再发送")

            validated_files.append(file_info)
        return validated_files

    def persist_stream_partial_before_stop(
        self,
        *,
        conversation_id: str,
        user_id: str,
        message_id: str,
        partial_content: List[ClientPartialContentBlock],
        stream_meta: Dict[str, str],
    ) -> bool:
        """在 stop 冻结流程中持久化客户端已确认展示的 partial blocks。"""
        if not partial_content:
            return False
        if stream_meta.get("status") != "streaming":
            return False
        if stream_meta.get("user_id") != str(user_id):
            raise ApiException.not_found("无进行中的流")
        if not message_id or stream_meta.get("message_id") != message_id:
            raise ApiException.conflict("当前生成已被新请求取代")

        from app.db.models import Message as MessageModel

        if stream_meta.get("stream_mode") == "retry":
            # 只有「替换已有 assistant」的 retry 才需要保留上一版完整回答；
            # 未回答 retry（尚无 assistant）必须落库 partial，否则刷新后整段回答消失。
            existing = (
                self.db.query(MessageModel)
                .filter(
                    MessageModel.id == message_id,
                    MessageModel.conversation_id == conversation_id,
                )
                .first()
            )
            if existing is not None:
                return False

        serialized_content = merge_partial_content_blocks([], partial_content)
        if not serialized_content:
            return False
        try:
            acquire_message_persistence_lock(self.db, message_id)
            conversation = self.conversation_service.get_conversation(conversation_id, str(user_id))
            if not conversation:
                raise ApiException.not_found("会话不存在或无权访问")

            existing = (
                self.db.query(MessageModel)
                .populate_existing()
                .filter(
                    MessageModel.id == message_id,
                    MessageModel.conversation_id == conversation_id,
                )
                .first()
            )
            if existing and existing.role != "assistant":
                raise ApiException.conflict("目标消息不是助手消息")

            if existing:
                existing.content = merge_partial_content_blocks(existing.content or [], serialized_content)
            else:
                reserved_sequence = stream_meta.get("message_sequence")
                self.db.add(
                    MessageModel(
                        id=message_id,
                        conversation_id=conversation_id,
                        sequence=int(reserved_sequence) if reserved_sequence else None,
                        role="assistant",
                        content=serialized_content,
                        model_id=stream_meta.get("model") or conversation.model_id,
                        generation_task_id=stream_meta.get("task_id"),
                    )
                )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    async def _build_file_block_from_record(self, file_info: Any) -> FileBlock:
        """根据已校验文件记录构造消息内容块。"""
        block_kwargs = {
            "type": "file",
            "file_id": file_info.id,
            "filename": file_info.original_filename,
            "mime_type": file_info.mimetype,
        }
        if is_image_mime(file_info.mimetype or ""):
            if getattr(file_info, "thumbnail_key", None):
                try:
                    storage = get_storage_for_backend(getattr(file_info, "storage_backend", None))
                    if await storage.exists(file_info.thumbnail_key):
                        thumb_url = await storage.get_url(
                            file_info.thumbnail_key,
                            expires=settings.MINIO_PRESIGN_EXPIRES,
                        )
                        block_kwargs["thumbnail_url"] = FileService._sign_local_url(
                            thumb_url,
                            file_info.id,
                            settings.MINIO_PRESIGN_EXPIRES,
                        )
                    else:
                        logger.warning("图片缩略图实体缺失: file_id=%s, key=%s", file_info.id, file_info.thumbnail_key)
                except Exception:
                    logger.warning("图片缩略图 URL 构造失败: file_id=%s", file_info.id)
            block_kwargs["width"] = getattr(file_info, "width", None)
            block_kwargs["height"] = getattr(file_info, "height", None)
        return FileBlock(**block_kwargs)

    async def process_message(
        self,
        model_id: str,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_message_id: Optional[str] = None,
        assistant_message_id: Optional[str] = None,
        retry_user_message_id: Optional[str] = None,
        retry_assistant_message_id: Optional[str] = None,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        file_ids: Optional[List[str]] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息，路由到流式或非流式响应"""
        options = dict(options or {})
        # strict grounding 只能由已验证的知识库选择开启，不能信任客户端内部开关。
        options.pop("knowledge_grounded", None)

        existing_conversation = None
        if conversation_id:
            candidate = self.conversation_service.get_conversation(conversation_id, user_id)
            if candidate is not None and isinstance(getattr(candidate, "model_id", None), str):
                existing_conversation = candidate

        retry_user_message: Message | None = None
        retry_assistant_message: Message | None = None
        if retry_assistant_message_id is not None and retry_user_message_id is None:
            raise ApiException.bad_request("retry_assistant_message_id 必须与 retry_user_message_id 同时提供")
        if retry_user_message_id is not None:
            if existing_conversation is None or conversation_id is None:
                raise ApiException.not_found("会话不存在或无权访问")
            retry_user_message, retry_assistant_message = self.conversation_service.prepare_message_retry(
                conversation_id=conversation_id,
                user_id=user_id,
                user_message_id=retry_user_message_id,
                assistant_message_id=retry_assistant_message_id,
            )
            stored_message, stored_file_ids = _retry_user_payload(retry_user_message)
            if message != stored_message or list(file_ids or []) != stored_file_ids:
                raise ApiException.conflict("重试内容与原用户消息不一致")
            message = stored_message
            file_ids = stored_file_ids
            user_message_id = retry_user_message.id
            # 历史消息只提供待校验的 file_id，重试仍必须重新验证所有权、会话关联与处理状态。
            self._validate_message_files(file_ids, user_id, conversation_id)
            if retry_assistant_message is not None:
                assistant_message_id = retry_assistant_message.id
                existing_conversation.messages = [
                    item for item in existing_conversation.messages if item.id != retry_assistant_message.id
                ]

        # 已有会话的模型绑定由服务端持久记录决定，忽略客户端覆盖；新会话才接受请求模型。
        effective_model_id = existing_conversation.model_id if existing_conversation is not None else model_id
        catalog_entry = litellm_catalog.get_model_entry(effective_model_id)
        if not isinstance(catalog_entry, Mapping) or not catalog_entry.get("db_model"):
            catalog_status = litellm_catalog.get_cache_status()
            if catalog_status.get("availability") == "available" or catalog_status.get("has_cache"):
                raise ApiException.service_unavailable("当前模型尚未注册", code=ErrorCode.MODEL_UNAVAILABLE)
        control = self.model_control_repository.get(effective_model_id)
        if control is not None and getattr(control, "routable", True) is False:
            raise ApiException.service_unavailable("当前模型暂不可调用", code=ErrorCode.MODEL_UNAVAILABLE)
        existing_messages = getattr(existing_conversation, "messages", None)
        is_upload_placeholder = (
            existing_conversation is not None and isinstance(existing_messages, list) and not existing_messages
        )
        if (
            (existing_conversation is None or is_upload_placeholder)
            and control is not None
            and getattr(control, "selectable", True) is False
        ):
            raise ApiException.service_unavailable("当前模型不可用于新会话", code=ErrorCode.MODEL_UNAVAILABLE)
        model_id = effective_model_id

        effective_knowledge_base_ids = list(
            knowledge_base_ids
            if knowledge_base_ids is not None
            else (getattr(existing_conversation, "knowledge_base_ids", None) or [])
        )
        if effective_knowledge_base_ids:
            validate_knowledge_query(message)
        # 解析模型调用参数（薄代理 LiteLLM，不再走本地 DB）
        litellm_model, provider, litellm_kwargs = llm_manager.resolve_model(model_id)

        # 模型能力来自 LiteLLM metadata（vision / functionCalling 影响消息构造和工具开关）
        capabilities = _get_model_capabilities(model_id)
        has_vision = capabilities.get("vision", False)
        requested_task_policy = resolve_agent_task_policy(
            options=options,
            capabilities=capabilities,
            enforce_capabilities=False,
        )
        if effective_knowledge_base_ids and requested_task_policy.task_mode == "deep_research":
            raise ApiException.bad_request("知识库问答不能与深度研究模式同时使用")
        task_policy = (
            resolve_agent_task_policy(options=options, capabilities=capabilities)
            if requested_task_policy.task_mode == "deep_research"
            else requested_task_policy
        )
        options = task_policy.apply_to_options(options)
        if effective_knowledge_base_ids and file_ids:
            raise ApiException.bad_request("知识库问答暂不支持同时附加文件")
        if effective_knowledge_base_ids:
            options = {
                **options,
                "knowledge_grounded": True,
                "disable_tools": True,
                "plan_mode": "off",
                "evidence_policy": "knowledge_grounded_v1",
            }
        if task_policy.task_mode == "deep_research" and not stream:
            raise ApiException.bad_request("深度研究模式仅支持流式对话")

        # 获取或创建会话
        if existing_conversation is not None:
            conversation, is_new_conversation = existing_conversation, False
        else:
            conversation, is_new_conversation = self._get_or_create_conversation(
                conversation_id, user_id, model_id, message
            )

        # 新建会话、选择替换与就绪校验必须先于消息序号预留和消息写入，并共享同一事务。
        if is_new_conversation:
            self.conversation_service.save_conversation(conversation)
        if knowledge_base_ids is not None:
            try:
                self.conversation_service.replace_knowledge_base_selection(
                    conversation_id=conversation.id,
                    user_id=user_id,
                    knowledge_base_ids=effective_knowledge_base_ids,
                )
            except Exception:
                self.db.rollback()
                raise
            conversation.knowledge_base_ids = effective_knowledge_base_ids
        elif effective_knowledge_base_ids:
            try:
                self.conversation_service.validate_knowledge_base_selection(
                    conversation_id=conversation.id,
                    user_id=user_id,
                    knowledge_base_ids=effective_knowledge_base_ids,
                )
            except Exception:
                self.db.rollback()
                raise

        if existing_conversation is not None and retry_user_message is None:
            refreshed_conversation = self.conversation_service.lock_conversation_for_message_write(
                conversation.id,
                user_id,
            )
            if isinstance(refreshed_conversation, Conversation):
                conversation = refreshed_conversation

        if retry_user_message is not None:
            user_message = retry_user_message
            if retry_assistant_message is not None and retry_assistant_message.sequence is not None:
                assistant_sequence = retry_assistant_message.sequence
            else:
                _, assistant_sequence = self.conversation_service.reserve_message_sequence_pair()
            assistant_message_id = assistant_message_id or str(uuid_mod.uuid4())
        else:
            # 普通发送仍构造并持久化一条全新的 user 消息。
            validated_files = self._validate_message_files(file_ids or [], user_id, conversation.id)
            user_content = [TextBlock(type="text", text=message)]
            for file_info in validated_files:
                user_content.append(await self._build_file_block_from_record(file_info))

            user_sequence, assistant_sequence = self.conversation_service.reserve_message_sequence_pair()
            user_message = Message(
                id=user_message_id or str(uuid_mod.uuid4()),
                role="user",
                content=user_content,
                sequence=user_sequence,
            )
            assistant_message_id = assistant_message_id or str(uuid_mod.uuid4())
            self.conversation_service.create_message(user_message, conversation.id)

        retry_generation_task_id: str | None = None
        if retry_user_message is not None:
            retry_generation_task_id = str(uuid_mod.uuid4())
            if retry_assistant_message is not None:
                # 已有回答时在 assistant 行换代，后返回的旧调用不能覆盖新回答。
                self.conversation_service.claim_assistant_message_generation(
                    conversation_id=conversation.id,
                    message_id=retry_assistant_message.id,
                    task_id=retry_generation_task_id,
                )
            else:
                # 尚无回答时把代际挂在原 user 行，完成阶段再 CAS 创建唯一 assistant。
                self.conversation_service.claim_unanswered_user_generation(
                    conversation_id=conversation.id,
                    message_id=retry_user_message.id,
                    task_id=retry_generation_task_id,
                )
        else:
            # 首次生成（无论流式或非流式）都必须拥有 user 代际；
            # 若其间被另一标签重试接管，迟到的原请求不得无条件创建 assistant。
            retry_generation_task_id = str(uuid_mod.uuid4())
            self.conversation_service.claim_unanswered_user_generation(
                conversation_id=conversation.id,
                message_id=user_message.id,
                task_id=retry_generation_task_id,
            )

        if stream:
            # 预分配 assistant 消息 ID 和 task ID
            task_id = retry_generation_task_id or str(uuid_mod.uuid4())

            # 先初始化 Redis Stream（清除旧数据 + 写 start 标记），
            # 必须在 SSE 读取器启动之前完成，否则读取器会读到上一轮残留数据
            try:
                init_result = await init_stream(
                    conversation.id,
                    str(user_id),
                    model_id,
                    assistant_message_id,
                    task_id,
                    stream_mode="retry" if retry_user_message is not None else "initial",
                    message_sequence=assistant_sequence,
                )
            except Exception:
                self.db.rollback()
                raise
            if not init_result.ok:
                self.db.rollback()
            _require_stream_initialized(init_result)

            try:
                self.db.commit()
            except Exception:
                self.db.rollback()
                finalized = await finalize_stream(
                    conversation.id,
                    success=False,
                    error_msg="消息持久化失败",
                    task_id=task_id,
                    error_code="generation_init_failed",
                )
                if not finalized:
                    logger.error(
                        "数据库提交失败后的 Redis Stream CAS 收尾失败: conv_id=%s, task_id=%s",
                        conversation.id,
                        task_id,
                    )
                raise
            if retry_user_message is None:
                conversation.messages.append(user_message)

            # 启动后台生成任务（独立于 HTTP 连接生命周期）
            # 图片 base64 编码等耗时操作在后台任务中完成，不阻塞 SSE 首字节
            task = asyncio.create_task(
                self.stream_handler.generate_to_redis(
                    conversation_id=conversation.id,
                    user_id=user_id,
                    model_id=model_id,
                    litellm_model=litellm_model,
                    litellm_kwargs=litellm_kwargs,
                    provider=provider,
                    raw_messages=conversation.messages,
                    has_vision=has_vision and not effective_knowledge_base_ids,
                    file_ids=None if effective_knowledge_base_ids else file_ids,
                    original_message=message,
                    assistant_message_id=assistant_message_id,
                    assistant_message_sequence=assistant_sequence,
                    task_id=task_id,
                    options=options,
                    capabilities=capabilities,
                    knowledge_base_ids=effective_knowledge_base_ids,
                    trace_id=trace_id,
                    defer_partial_persistence=retry_user_message is not None,
                    replace_on_success=retry_assistant_message is not None,
                    create_after_retry_user_id=(
                        user_message.id
                        if retry_assistant_message is None and retry_generation_task_id is not None
                        else None
                    ),
                )
            )
            register_task(conversation.id, task, task_id)

            # SSE 从 Redis Stream 读取，不直接调 LLM
            return StreamingResponse(
                stream_redis_as_sse(
                    conversation_id=conversation.id,
                    message_id=assistant_message_id,
                    task_id=task_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            self.db.commit()
            if retry_user_message is None:
                conversation.messages.append(user_message)
            # 非流式模式：同步构建消息（含图片 base64）
            from app.db.models import User as UserModel

            user_record = self.db.query(UserModel).filter(UserModel.id == user_id).first()
            user_system_prompt = user_record.system_prompt if user_record else None
            lm_messages = await build_llm_messages(
                conversation.messages,
                has_vision=has_vision and not effective_knowledge_base_ids,
                file_repo=self.file_repo,
                user_system_prompt=user_system_prompt,
                user_id=user_id,
                conversation_id=conversation.id,
            )
            if file_ids and not effective_knowledge_base_ids:
                image_ids = [fid for fid in file_ids if is_image_file(fid, self.file_repo)]
                non_image_ids = [fid for fid in file_ids if fid not in image_ids]
                if image_ids and not has_vision:
                    lm_messages = inject_no_vision_file_boundary(lm_messages)
                if non_image_ids:
                    file_contents = self.file_repo.get_parsed_file_content(non_image_ids)
                    if file_contents:
                        lm_messages = inject_file_content(lm_messages, message, file_contents)
            initial_content_blocks: list[Any] = []
            if effective_knowledge_base_ids:
                grounding = await prepare_knowledge_grounding(
                    db=self.db,
                    user_id=user_id,
                    query=message,
                    knowledge_base_ids=effective_knowledge_base_ids,
                )
                initial_content_blocks.append(grounding.evidence_block)
                if grounding.no_evidence:
                    return self._persist_non_stream_grounding_answer(
                        conversation_id=conversation.id,
                        model_id=model_id,
                        assistant_message_id=assistant_message_id,
                        assistant_message_sequence=assistant_sequence,
                        evidence_block=grounding.evidence_block,
                        answer=grounding.deterministic_answer or "未在所选知识库中找到足够依据",
                        replace_existing=retry_assistant_message is not None,
                        generation_task_id=retry_generation_task_id,
                        retry_user_message_id=(
                            user_message.id
                            if retry_assistant_message is None and retry_generation_task_id is not None
                            else None
                        ),
                    )
                lm_messages = inject_knowledge_grounding_messages(lm_messages, grounding)
            lm_messages = inject_no_tool_network_boundary(lm_messages, call_kwargs={})
            return await self._handle_non_stream(
                litellm_model,
                model_id,
                litellm_kwargs,
                lm_messages,
                conversation.id,
                options,
                assistant_message_id,
                assistant_sequence,
                initial_content_blocks=initial_content_blocks,
                replace_existing=retry_assistant_message is not None,
                generation_task_id=retry_generation_task_id,
                retry_user_message_id=(
                    user_message.id
                    if retry_assistant_message is None and retry_generation_task_id is not None
                    else None
                ),
            )

    async def continue_agent_run(
        self,
        *,
        conversation_id: str,
        assistant_message_id: str,
        user_id: str,
        previous_run_id: str | None = None,
        trace_id: str | None = None,
    ) -> StreamingResponse:
        """基于最近一次 limit_reached run 续写同一条 assistant 消息。"""
        conversation = self.conversation_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ApiException.not_found("会话不存在或无权访问")

        meta = await get_stream_meta(conversation_id)
        if meta and meta.get("status") == "streaming":
            raise ApiException.conflict("当前会话已有回答正在生成，请结束后再继续")

        model_id = conversation.model_id
        litellm_model, provider, litellm_kwargs = llm_manager.resolve_model(model_id)
        capabilities = _get_model_capabilities(model_id)
        has_vision = capabilities.get("vision", False)

        continuation = build_continuation_context(
            self.db,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            previous_run_id=previous_run_id,
            default_limits=_agent_loop_limits(),
        )
        if any(
            (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "knowledge_evidence"
            for block in continuation.initial_content_blocks
        ):
            raise ApiException.bad_request("知识库回答暂不支持继续生成，请重新提问")
        stored_task_policy = getattr(continuation, "task_policy", None)
        continuation_options = (
            stored_task_policy.apply_to_options()
            if stored_task_policy is not None
            else {"plan_mode": continuation.plan_mode}
        )
        continuation_policy = resolve_agent_task_policy(
            options=continuation_options,
            capabilities=capabilities,
        )
        original_user_text = _continuation_original_user_text(
            conversation.messages,
            assistant_message_id=assistant_message_id,
        )

        task_id = str(uuid_mod.uuid4())
        self.conversation_service.claim_assistant_message_generation(
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            task_id=task_id,
        )
        try:
            init_result = await init_stream(
                conversation_id,
                str(user_id),
                model_id,
                assistant_message_id,
                task_id,
                stream_mode="continuation",
                message_sequence=continuation.assistant_message.sequence,
            )
        except Exception:
            self.db.rollback()
            raise
        if not init_result.ok:
            self.db.rollback()
        _require_stream_initialized(init_result)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            await finalize_stream(
                conversation_id,
                success=False,
                error_msg="消息持久化失败",
                task_id=task_id,
                error_code="generation_init_failed",
            )
            raise

        task = asyncio.create_task(
            self.stream_handler.generate_to_redis(
                conversation_id=conversation_id,
                user_id=user_id,
                model_id=model_id,
                litellm_model=litellm_model,
                litellm_kwargs=litellm_kwargs,
                provider=provider,
                raw_messages=conversation.messages,
                has_vision=has_vision,
                file_ids=None,
                original_message=original_user_text,
                assistant_message_id=assistant_message_id,
                assistant_message_sequence=continuation.assistant_message.sequence,
                task_id=task_id,
                options=continuation_policy.apply_to_options(),
                capabilities=capabilities,
                knowledge_base_ids=[],
                trace_id=trace_id,
                initial_content_blocks=continuation.initial_content_blocks,
                extra_system_prompts=[get_continuation_system_prompt()],
                preprocess_user_input=False,
                limits=continuation.limits,
            )
        )
        register_task(conversation_id, task, task_id)

        return StreamingResponse(
            stream_redis_as_sse(
                conversation_id=conversation_id,
                message_id=assistant_message_id,
                task_id=task_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def submit_agent_context_result(
        self,
        *,
        conversation_id: str,
        run_id: str,
        request_id: str,
        user_id: str,
        status: str,
        location: dict[str, Any] | None,
        reason: str | None,
    ) -> dict[str, Any]:
        """向仍在等待的同一 Agent run 提交短期上下文，不回显精确位置。"""
        conversation = self.conversation_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ApiException.not_found("会话不存在或无权访问")

        submission = await submit_context_result(
            request_id=request_id,
            user_id=str(user_id),
            conversation_id=conversation_id,
            run_id=run_id,
            status=status,
            location=location,
            reason=reason,
        )
        if submission.outcome in {"accepted", "idempotent"}:
            return submission.model_dump(mode="json")
        if submission.outcome == "expired":
            raise ApiException.gone("上下文请求已过期")
        if submission.outcome in {"conflict", "stale"}:
            raise ApiException.conflict("上下文请求已失效或结果冲突")
        if submission.outcome in {"not_found", "forbidden"}:
            raise ApiException.not_found("上下文请求不存在或无权访问")
        raise ApiException.conflict("上下文请求当前不可提交")

    def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: str,
        model_id: str,
        message: str,
    ) -> tuple:
        """获取已有会话，或初始化新会话对象。返回 (conversation, is_new)"""
        if conversation_id:
            existing = self.conversation_service.get_conversation(conversation_id, user_id)
            if existing:
                return existing, False

        return Conversation(
            id=conversation_id or str(uuid_mod.uuid4()),
            user_id=user_id,
            model_id=model_id,
            title=message[:30] + "..." if len(message) > 30 else message,
            messages=[],
        ), True

    async def _handle_non_stream(
        self,
        litellm_model: str,
        model_id: str,
        litellm_kwargs: dict,
        messages: List[dict],
        conversation_id: str,
        options: dict,
        assistant_message_id: str | None = None,
        assistant_message_sequence: int | None = None,
        initial_content_blocks: list[Any] | None = None,
        replace_existing: bool = False,
        generation_task_id: str | None = None,
        retry_user_message_id: str | None = None,
    ) -> ChatResponse:
        """处理非流式响应（LiteLLM Proxy 自己管 health / 重试）。"""
        controlled_call_kwargs = dict(litellm_kwargs)
        max_tokens = normalize_controlled_max_tokens(options.get("max_tokens"))
        if max_tokens is not None:
            controlled_call_kwargs["max_tokens"] = max_tokens
        final_call_kwargs = merge_litellm_kwargs("chat_non_stream", controlled_call_kwargs)
        finalized_messages = finalize_model_call_language_policy(messages)
        try:
            context_plan = await prepare_context(
                messages=finalized_messages,
                model_id=model_id,
                litellm_model=litellm_model,
                call_kwargs=final_call_kwargs,
            )
        except ContextBudgetExceededError as error:
            raise ApiException.bad_request(str(error)) from error
        except ContextEstimationUnavailableError as error:
            raise ApiException.service_unavailable(str(error)) from error
        response = await litellm.acompletion(
            model=litellm_model,
            messages=context_plan.messages,
            stream=False,
            **final_call_kwargs,
        )

        content_text = response.choices[0].message.content or ""
        evidence_block = next(
            (block for block in initial_content_blocks or [] if isinstance(block, KnowledgeEvidenceBlock)),
            None,
        )
        if evidence_block is not None and not validate_grounded_answer(content_text, evidence_block):
            content_text = KNOWLEDGE_UNVERIFIABLE_ANSWER_TEXT
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
        usage_data = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context=context_plan.to_usage_context(
                actual_prompt_tokens=input_tokens if input_tokens > 0 else None,
                round_index=1,
            ),
        )

        assistant_message_kwargs: dict[str, Any] = {
            "sequence": assistant_message_sequence,
            "role": "assistant",
            "content": [*(initial_content_blocks or []), TextBlock(type="text", text=content_text)],
            "model_id": model_id,
            "usage": usage_data,
        }
        if assistant_message_id is not None:
            assistant_message_kwargs["id"] = assistant_message_id
        assistant_message = Message(**assistant_message_kwargs)
        if replace_existing:
            self.conversation_service.replace_assistant_message(
                assistant_message,
                conversation_id,
                generation_task_id=generation_task_id,
            )
        elif retry_user_message_id is not None and generation_task_id is not None:
            self.conversation_service.create_retry_assistant_message(
                assistant_message,
                conversation_id,
                retry_user_message_id=retry_user_message_id,
                generation_task_id=generation_task_id,
            )
        else:
            self.conversation_service.create_message(assistant_message, conversation_id)
        self.db.commit()

        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
        )

    def _persist_non_stream_grounding_answer(
        self,
        *,
        conversation_id: str,
        model_id: str,
        assistant_message_id: str,
        assistant_message_sequence: int,
        evidence_block: KnowledgeEvidenceBlock,
        answer: str,
        replace_existing: bool = False,
        generation_task_id: str | None = None,
        retry_user_message_id: str | None = None,
    ) -> ChatResponse:
        """无检索命中时仍按普通 assistant 消息完成，确保刷新可恢复。"""

        assistant_message = Message(
            id=assistant_message_id,
            sequence=assistant_message_sequence,
            role="assistant",
            content=[evidence_block, TextBlock(type="text", text=answer)],
            model_id=model_id,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        if replace_existing:
            self.conversation_service.replace_assistant_message(
                assistant_message,
                conversation_id,
                generation_task_id=generation_task_id,
            )
        elif retry_user_message_id is not None and generation_task_id is not None:
            self.conversation_service.create_retry_assistant_message(
                assistant_message,
                conversation_id,
                retry_user_message_id=retry_user_message_id,
                generation_task_id=generation_task_id,
            )
        else:
            self.conversation_service.create_message(assistant_message, conversation_id)
        self.db.commit()
        return ChatResponse(conversation_id=conversation_id, message=assistant_message)

    # 辅助功能（标题、推荐问题）固定使用的轻量快速模型，不跟随对话模型，
    # 避免对话用的是慢/贵的 thinking 模型时拖累这些"锦上添花"的小活。
    # 注意：qwen-max-latest 是旗舰重模型，经 LiteLLM Proxy → dashscope 实测约 20s，
    # 会撞 main.py 的 TimeoutMiddleware(10s) 直接 408，故固定用快速的 deepseek-chat（实测约 3s）。
    UTILITY_MODEL_ID = "deepseek-chat"
    # 辅助 LLM 调用的内部超时（秒），必须 < TimeoutMiddleware 的 10s。
    # 这样即便将来换的辅助模型偏慢，也能在中间件掐断前自己抛错走 fallback，而不是把 408 吐给前端。
    UTILITY_LLM_TIMEOUT = 8
    # 标题最终会截断到 30 字，但 deepseek-chat 会先消耗 reasoning token；
    # 128 在真实回归中仍可能只返回 reasoning、正文为空，因此与推荐问题统一留足 512。
    TITLE_MAX_TOKENS = 512

    def _resolve_utility_model(self, conversation_model_id: str) -> tuple:
        """解析辅助功能模型，固定用轻量模型，找不到则回退对话模型"""
        try:
            return llm_manager.resolve_model(self.UTILITY_MODEL_ID)
        except ValueError:
            return llm_manager.resolve_model(conversation_model_id)

    async def generate_title(
        self,
        user_id: str,
        conversation_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """基于会话最后一条用户消息生成标题，并写回数据库"""
        conversation = self.conversation_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ApiException.not_found(f"找不到会话: {conversation_id}")

        # 提取最后一条用户消息文本
        seed_text = ""
        for msg in reversed(conversation.messages):
            if msg.role == "user":
                parts = [b.text for b in msg.content if b.type == "text"]
                seed_text = "\n".join(parts)
                if seed_text:
                    break

        if not seed_text:
            raise ApiException.bad_request("会话中没有可用的用户消息")

        # 生成失败时的回退标题
        fallback_title = seed_text[:30] + "..." if len(seed_text) > 30 else seed_text

        try:
            prompt, prompt_metadata = prompt_manager.format_prompt_with_metadata(
                "generate_title",
                content=seed_text,
            )
            litellm_model, _, litellm_kwargs = self._resolve_utility_model(conversation.model_id)
            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=self.TITLE_MAX_TOKENS,
                timeout=self.UTILITY_LLM_TIMEOUT,
                **merge_litellm_kwargs(
                    "generate_title",
                    litellm_kwargs,
                    prompt_metadata=prompt_metadata,
                ),
            )
            raw = response.choices[0].message.content or ""

            # 标题清理：去除引号、常见前缀、控制长度
            title = raw.strip().strip('"').strip("'")
            for prefix in ["标题：", "标题:", "Title:", "Title："]:
                if title.startswith(prefix):
                    title = title[len(prefix) :].strip()
            title = title[:30] if len(title) > 30 else title
            title = title or fallback_title

        except Exception as e:
            logger.error(f"生成标题失败，使用回退标题: {e}")
            title = fallback_title

        # 写回数据库
        self.conversation_service.repo.update_title(conversation_id, title)
        self.db.commit()

        return title

    async def generate_suggested_questions(
        self,
        user_id: str,
        conversation_id: str,
        assistant_message_id: str | None = None,
        force_refresh: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """兼容旧调用：生成并返回问题列表。"""
        result = await self.generate_suggested_questions_result(
            user_id=user_id,
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            force_refresh=force_refresh,
            options=options,
        )
        return result.questions

    async def generate_suggested_questions_result(
        self,
        *,
        user_id: str,
        conversation_id: str,
        assistant_message_id: str | None = None,
        force_refresh: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> SuggestedQuestionGenerationResult:
        """锁定 assistant message 生成推荐问题，并返回持久化状态。"""
        request_claim = self.suggested_question_service.claim_request_generation(
            conversation_id=conversation_id,
            user_id=user_id,
            assistant_message_id=assistant_message_id,
            force_refresh=force_refresh,
        )
        if request_claim.claim is None:
            return SuggestedQuestionGenerationResult(
                questions=request_claim.questions,
                message_id=request_claim.message_id,
                revision=request_claim.revision,
                status=request_claim.status,
                applied=False,
            )
        return await self.suggested_question_service.generate_claimed_questions(
            request_claim.claim,
            options=options,
        )

    @staticmethod
    def _build_recent_dialog_content(conversation: Conversation) -> str:
        """保留旧扩展兼容；正式生成路径由目标 message ID 锁定上下文。"""
        latest_user = ""
        latest_ai = ""
        for message in reversed(conversation.messages):
            text = "\n".join(block.text for block in message.content if block.type == "text")
            if not text:
                continue
            if not latest_ai and message.role == "assistant":
                latest_ai = text
            elif not latest_user and message.role == "user":
                latest_user = text
            if latest_user and latest_ai:
                break
        lines = []
        if latest_user:
            lines.append(f"用户: {latest_user}")
        if latest_ai:
            lines.append(f"助手: {latest_ai}")
        return "\n".join(lines)

    # ==================== CRUD 代理方法 ====================

    def get_conversation(self, conversation_id: str, user_id: str):
        return self.conversation_service.get_conversation(conversation_id, user_id)

    def get_all_conversations(self, user_id: str):
        return self.conversation_service.get_all_conversations(user_id)

    def get_conversations_paginated(self, user_id: str, page: int = 1, page_size: int = 20):
        return self.conversation_service.get_conversations_paginated(user_id, page, page_size)

    def get_conversations_metadata(self, user_id: str, conversation_ids: List[str]) -> List[Dict[str, Any]]:
        """按 ID 列表返回对话元数据（前端用于刷新已显示对话的标题等）。"""
        repo = ConversationRepository(self.db)
        conversations = repo.get_metadata_by_ids(user_id, conversation_ids)
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "model_id": conv.model_id,
                "knowledge_base_ids": conv.knowledge_base_ids,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            }
            for conv in conversations
        ]

    def search_conversations_by_title(self, user_id: str, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """按标题模糊搜索当前用户的对话。"""
        repo = ConversationRepository(self.db)
        conversations = repo.search_by_title(user_id, query, limit)
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "model_id": conv.model_id,
                "knowledge_base_ids": conv.knowledge_base_ids,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            }
            for conv in conversations
        ]

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        updated = self.conversation_service.update_message(message_id, update_data)
        if updated:
            self.db.commit()
        return updated

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        return self.conversation_service.delete_conversation(conversation_id, user_id)
