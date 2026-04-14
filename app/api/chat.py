# app/api/chat.py

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, get_current_user
from app.core.redis import get_redis_pool, stream_chunks_key
from app.db.models import User
from app.schemas.chat import (
    ChatRequest,
    MessageUpdateRequest,
    SuggestedQuestionsRequest,
    TitleGenerationRequest,
)
from app.schemas.response import ApiException, success
from app.services.chat_service import ChatService
from app.services.stream_handler import stream_redis_as_sse
from app.services.stream_state_service import cancel_stream, get_stream_meta
from app.services.task_manager import cancel_task

router = APIRouter()


@router.post("/send")
async def send_message(
    chat_request: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """发送消息，返回流式或非流式响应"""
    result = await chat_service.process_message(
        model_id=chat_request.model_id,
        message=chat_request.message,
        user_id=current_user.id,
        conversation_id=chat_request.conversation_id,
        stream=chat_request.stream,
        options=chat_request.options,
        file_ids=chat_request.file_ids,
        trace_id=request.state.request_id,
    )
    if isinstance(result, StreamingResponse):
        return result
    return success(data=result, request_id=request.state.request_id)


@router.get("/conversations")
def get_conversations(
    request: Request,
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """分页获取会话列表（不含消息内容）"""
    data = chat_service.get_conversations_paginated(current_user.id, page, page_size)
    return success(data=data, request_id=request.state.request_id)


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """获取会话详情，含完整消息列表"""
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise ApiException.not_found("会话不存在或无权访问")
    return success(data=conversation, request_id=request.state.request_id)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """删除会话"""
    result = chat_service.delete_conversation(conversation_id, user_id=current_user.id)
    if not result:
        raise ApiException.not_found("会话不存在或无权访问")
    return success(message="会话已删除", request_id=request.state.request_id)


@router.put("/conversations/{conversation_id}/messages/{message_id}")
def update_message(
    conversation_id: str,
    message_id: str,
    update_request: MessageUpdateRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """更新消息内容（如前端编辑重发场景）"""
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise ApiException.not_found("会话不存在或无权访问")
    message_ids = {msg.id for msg in conversation.messages}
    if message_id not in message_ids:
        raise ApiException.not_found("消息不属于此会话")
    updated = chat_service.update_message(message_id, update_request.model_dump(exclude_unset=True))
    if not updated:
        raise ApiException.not_found("消息不存在或更新失败")
    return success(data=updated, request_id=request.state.request_id)


@router.post("/generate-title")
async def generate_title(
    title_request: TitleGenerationRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """基于会话内容生成标题并写回"""
    conversation = chat_service.get_conversation(title_request.conversation_id, user_id=current_user.id)
    if not conversation:
        raise ApiException.not_found("会话不存在或无权访问")
    title = await chat_service.generate_title(
        user_id=current_user.id,
        conversation_id=title_request.conversation_id,
        options=title_request.options,
    )
    return success(
        data={"title": title, "conversation_id": title_request.conversation_id},
        request_id=request.state.request_id,
    )


@router.post("/suggest-questions")
async def suggest_questions(
    sq_request: SuggestedQuestionsRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: User = Depends(get_current_user),
):
    """基于会话内容生成推荐问题"""
    conversation = chat_service.get_conversation(sq_request.conversation_id, user_id=current_user.id)
    if not conversation:
        raise ApiException.not_found("会话不存在或无权访问")
    questions = await chat_service.generate_suggested_questions(
        user_id=current_user.id,
        conversation_id=sq_request.conversation_id,
        options=sq_request.options,
    )
    return success(
        data={"questions": questions, "conversation_id": sq_request.conversation_id},
        request_id=request.state.request_id,
    )


@router.get("/stream-status/{conv_id}")
async def get_stream_status_endpoint(
    conv_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """查询流状态"""
    meta = await get_stream_meta(conv_id)
    if not meta:
        return success(data={"status": "not_found"}, request_id=request.state.request_id)
    if meta.get("user_id") != str(current_user.id):
        return success(data={"status": "not_found"}, request_id=request.state.request_id)
    if meta["status"] == "streaming":
        redis = get_redis_pool()
        last_entry_id = "0"
        if redis:
            try:
                entries = await redis.xrevrange(stream_chunks_key(conv_id), count=1)
                if entries:
                    last_entry_id = entries[0][0]
            except Exception:
                pass
        return success(
            data={"status": "streaming", "last_entry_id": last_entry_id, "message_id": meta.get("message_id")},
            request_id=request.state.request_id,
        )
    return success(data={"status": meta["status"]}, request_id=request.state.request_id)


@router.get("/stream/{conv_id}")
async def reconnect_stream(
    conv_id: str,
    last_entry_id: str = "0",
    current_user: User = Depends(get_current_user),
):
    """断线重连端点：从 Redis Stream 的断点续读 SSE"""
    meta = await get_stream_meta(conv_id)
    if not meta or meta.get("user_id") != str(current_user.id):
        raise ApiException.not_found("无进行中的流")

    message_id = meta.get("message_id", "")

    return StreamingResponse(
        stream_redis_as_sse(conv_id, message_id, last_entry_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/stop/{conv_id}")
async def stop_stream(
    conv_id: str,
    request: Request,
    message_id: str = Query(default="", description="要取消的消息 ID，防止误杀新一轮流"),
    current_user: User = Depends(get_current_user),
):
    """用户手动停止流生成"""
    local_cancelled = cancel_task(conv_id)
    redis_cancelled = await cancel_stream(conv_id, message_id)
    return success(data={"cancelled": local_cancelled or redis_cancelled}, request_id=request.state.request_id)
