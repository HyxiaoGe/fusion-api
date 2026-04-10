# app/api/chat.py

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.logger import app_logger
from app.core.redis import get_redis_pool, stream_chunks_key
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.chat import (
    ChatRequest,
    Conversation,
    Message,
    MessageUpdateRequest,
    SuggestedQuestionsRequest,
    SuggestedQuestionsResponse,
    TitleGenerationRequest,
    TitleGenerationResponse,
)
from app.services.chat_service import ChatService
from app.services.stream_handler import stream_redis_as_sse
from app.services.stream_state_service import cancel_stream, get_stream_meta
from app.services.task_manager import cancel_task

router = APIRouter()


@router.post("/send")
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """发送消息，返回流式或非流式响应"""
    chat_service = ChatService(db)
    try:
        return await chat_service.process_message(
            model_id=request.model_id,
            message=request.message,
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            stream=request.stream,
            options=request.options,
            file_ids=request.file_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        app_logger.exception("处理聊天请求失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/conversations", response_model=dict)
def get_conversations(
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """分页获取会话列表（不含消息内容）"""
    chat_service = ChatService(db)
    return chat_service.get_conversations_paginated(current_user.id, page, page_size)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取会话详情，含完整消息列表"""
    chat_service = ChatService(db)
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return conversation


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除会话"""
    chat_service = ChatService(db)
    success = chat_service.delete_conversation(conversation_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return {"status": "success"}


@router.put(
    "/conversations/{conversation_id}/messages/{message_id}",
    response_model=Message,
)
def update_message(
    conversation_id: str,
    message_id: str,
    update_request: MessageUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新消息内容（如前端编辑重发场景）"""
    chat_service = ChatService(db)

    # 校验会话归属
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    message_ids = {msg.id for msg in conversation.messages}
    if message_id not in message_ids:
        raise HTTPException(status_code=404, detail="消息不属于此会话")

    updated = chat_service.update_message(
        message_id,
        update_request.model_dump(exclude_unset=True),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="消息不存在或更新失败")
    return updated


@router.post("/generate-title", response_model=TitleGenerationResponse)
async def generate_title(
    request: TitleGenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """基于会话内容生成标题并写回"""
    chat_service = ChatService(db)

    # 校验会话归属
    conversation = chat_service.get_conversation(request.conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    try:
        title = await chat_service.generate_title(
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            options=request.options,
        )
        return TitleGenerationResponse(
            title=title,
            conversation_id=request.conversation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        app_logger.exception("生成标题失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/suggest-questions", response_model=SuggestedQuestionsResponse)
async def suggest_questions(
    request: SuggestedQuestionsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """基于会话内容生成推荐问题"""
    chat_service = ChatService(db)

    # 校验会话归属
    conversation = chat_service.get_conversation(request.conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    try:
        questions = await chat_service.generate_suggested_questions(
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            options=request.options,
        )
        return SuggestedQuestionsResponse(
            questions=questions,
            conversation_id=request.conversation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        app_logger.exception("生成推荐问题失败")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/stream-status/{conv_id}")
async def get_stream_status_endpoint(
    conv_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    查询流状态。前端 mount 时调用，判断是否有未完成的流。
    返回 streaming 时附带 last_entry_id 和 message_id 供重连使用。
    """
    meta = await get_stream_meta(conv_id)
    if not meta:
        return {"status": "not_found"}

    if meta.get("user_id") != str(current_user.id):
        return {"status": "not_found"}

    if meta["status"] == "streaming":
        # 查询 Redis Stream 最新 entry ID
        redis = get_redis_pool()
        last_entry_id = "0"
        if redis:
            try:
                entries = await redis.xrevrange(stream_chunks_key(conv_id), count=1)
                if entries:
                    last_entry_id = entries[0][0]
            except Exception:
                pass

        return {
            "status": "streaming",
            "last_entry_id": last_entry_id,
            "message_id": meta.get("message_id"),
        }

    return {"status": meta["status"]}


@router.get("/stream/{conv_id}")
async def reconnect_stream(
    conv_id: str,
    last_entry_id: str = "0",
    current_user: User = Depends(get_current_user),
):
    """
    断线重连端点：从 Redis Stream 的断点续读 SSE。
    独立于 /send，不创建新消息、不调 LLM。
    """
    meta = await get_stream_meta(conv_id)
    if not meta or meta.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=404, detail="无进行中的流")

    message_id = meta.get("message_id", "")

    return StreamingResponse(
        stream_redis_as_sse(conv_id, message_id, last_entry_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/stop/{conv_id}")
async def stop_stream(
    conv_id: str,
    message_id: str = Query(default="", description="要取消的消息 ID，防止误杀新一轮流"),
    current_user: User = Depends(get_current_user),
):
    """
    用户手动停止流生成。

    双通道取消：
    1. cancel_task — 同进程内取消 asyncio 任务（即时生效）
    2. cancel_stream — 通过 Redis 原子取消（跨 worker，校验 message_id 防误杀）
    """
    local_cancelled = cancel_task(conv_id)
    redis_cancelled = await cancel_stream(conv_id, message_id)
    return {"cancelled": local_cancelled or redis_cancelled}
