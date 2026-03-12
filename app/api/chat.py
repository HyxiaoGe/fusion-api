from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.core.logger import app_logger
from app.schemas.chat import ChatRequest, Conversation, TitleGenerationRequest, TitleGenerationResponse, SuggestedQuestionsRequest, SuggestedQuestionsResponse, MessageUpdateRequest, Message
from app.services.chat_service import ChatService
from app.core.security import get_current_user

router = APIRouter()


@router.post("/send")
async def send_message(request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """发送消息到指定的AI模型并获取响应"""

    chat_service = ChatService(db)
    try:
        response = await chat_service.process_message(
            user_id=current_user.id,
            provider=request.provider,
            model=request.model,
            message=request.message,  # 保持用户原始完整消息
            conversation_id=request.conversation_id,
            stream=request.stream,
            options=request.options,
            file_ids=request.file_ids,
        )
        return response
    except Exception as e:
        app_logger.exception("处理聊天请求失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
def get_conversations(
    page: int = Query(default=1, ge=1, description="页码，从1开始"), 
    page_size: int = Query(default=10, ge=1, le=100, description="每页数量，最大100"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """分页获取对话列表"""
    chat_service = ChatService(db)
    return chat_service.get_conversations_paginated(current_user.id, page, page_size)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取特定对话的详细信息"""
    chat_service = ChatService(db)
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    return conversation


@router.put("/conversations/{conversation_id}/messages/{message_id}", response_model=Message)
def update_message(
    conversation_id: str,
    message_id: str,
    update_request: MessageUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新特定消息的信息，如内容、类型或持续时间"""
    chat_service = ChatService(db)

    # 验证消息是否属于该会话
    conversation = chat_service.get_conversation(conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    
    message_ids = [msg.id for msg in conversation.messages]
    if message_id not in message_ids:
        raise HTTPException(status_code=404, detail="消息不属于此对话")

    updated_message = chat_service.update_message(
        message_id, update_request.model_dump(exclude_unset=True)
    )
    if not updated_message:
        raise HTTPException(status_code=404, detail="消息不存在或更新失败")
        
    db.commit()
    return updated_message


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """删除特定对话"""
    chat_service = ChatService(db)
    success = chat_service.delete_conversation(conversation_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    return {"status": "success"}


@router.post("/generate-title", response_model=TitleGenerationResponse)
async def generate_title(request: TitleGenerationRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """生成与消息或会话相关的标题"""
    chat_service = ChatService(db)
    # 验证权限
    if request.conversation_id:
        conversation = chat_service.get_conversation(request.conversation_id, user_id=current_user.id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在或无权访问")
            
    try:
        title = await chat_service.generate_title(
            user_id=current_user.id,
            message=request.message,
            conversation_id=request.conversation_id,
            options=request.options
        )
        return TitleGenerationResponse(title=title, conversation_id=request.conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest-questions", response_model=SuggestedQuestionsResponse)
async def suggest_questions(request: SuggestedQuestionsRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """基于对话内容生成推荐问题"""
    chat_service = ChatService(db)
    # 验证权限
    conversation = chat_service.get_conversation(request.conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
        
    try:
        questions = await chat_service.generate_suggested_questions(
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            options=request.options
        )
        return SuggestedQuestionsResponse(questions=questions, conversation_id=request.conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
