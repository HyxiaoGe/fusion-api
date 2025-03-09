from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.chat import ChatRequest, Conversation, TitleGenerationRequest, TitleGenerationResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/send")
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """发送消息到指定的AI模型并获取响应"""
    print(f"收到聊天请求: conversation_id={request.conversation_id}")
    chat_service = ChatService(db)
    try:
        response = await chat_service.process_message(
            model=request.model,
            message=request.message,
            conversation_id=request.conversation_id,
            stream=request.stream,
            options=request.options
        )
        print(f"请求处理完成，返回响应")
        return response
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[Conversation])
def get_conversations(db: Session = Depends(get_db)):
    """获取所有对话列表"""
    chat_service = ChatService(db)
    return chat_service.get_all_conversations()


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
            model=request.model,
            message=request.message,
            conversation_id=request.conversation_id,
            options=request.options
        )
        return TitleGenerationResponse(title=title, conversation_id=request.conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
