from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.ai.prompts import prompt_manager
from app.db.database import get_db
from app.schemas.chat import ChatRequest, Conversation, TitleGenerationRequest, TitleGenerationResponse, SuggestedQuestionsRequest, SuggestedQuestionsResponse
from app.services.chat_service import ChatService
router = APIRouter()


@router.post("/send")
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """发送消息到指定的AI模型并获取响应"""
    print(f"收到聊天请求: conversation_id={request.conversation_id}")
    print(f"接收到的消息: {request.message}")
    print(f"话题ID: {request.topic_id}")

    # 处理话题相关逻辑
    topic_info = None
    original_message = request.message  # 保存原始消息
    
    if request.topic_id:
        from app.services.hot_topic_service import HotTopicService
        hot_topic_service = HotTopicService(db)
        topic = hot_topic_service.get_topic_by_id(request.topic_id)
        if topic:
            print(f"找到话题: {topic.title}")
            # 保存话题信息，传递给ChatService处理
            topic_info = {
                "title": topic.title,
                "description": topic.description or "",
                "additional_content": ""
            }
            
            # 如果用户消息就是话题标题，自动构造完整的分析请求
            if request.message.strip() == topic.title.strip():
                request.message = f"请帮我分析以下热点话题： {topic.title}"
                print(f"自动构造用户消息: {request.message}")
            
            # 增加浏览计数
            hot_topic_service.increment_view_count(request.topic_id)

    chat_service = ChatService(db)
    try:
        response = await chat_service.process_message(
            provider=request.provider,
            model=request.model,
            message=request.message,  # 保持用户原始完整消息
            conversation_id=request.conversation_id,
            stream=request.stream,
            options=request.options,
            file_ids=request.file_ids,
            topic_info=topic_info  # 传递话题信息用于LLM处理
        )
        print(f"请求处理完成，返回响应")
        return response
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
def get_conversations(
    page: int = Query(default=1, ge=1, description="页码，从1开始"), 
    page_size: int = Query(default=10, ge=1, le=100, description="每页数量，最大100"),
    db: Session = Depends(get_db)
):
    """分页获取对话列表"""
    chat_service = ChatService(db)
    return chat_service.get_conversations_paginated(page, page_size)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """获取特定对话的详细信息"""
    chat_service = ChatService(db)
    conversation = chat_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """删除特定对话"""
    chat_service = ChatService(db)
    success = chat_service.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"status": "success"}


@router.post("/generate-title", response_model=TitleGenerationResponse)
async def generate_title(request: TitleGenerationRequest, db: Session = Depends(get_db)):
    """生成与消息或会话相关的标题"""
    chat_service = ChatService(db)
    try:
        title = await chat_service.generate_title(
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
async def suggest_questions(request: SuggestedQuestionsRequest, db: Session = Depends(get_db)):
    """基于对话内容生成推荐问题"""
    chat_service = ChatService(db)
    try:
        questions = await chat_service.generate_suggested_questions(
            conversation_id=request.conversation_id,
            options=request.options
        )
        return SuggestedQuestionsResponse(questions=questions, conversation_id=request.conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))