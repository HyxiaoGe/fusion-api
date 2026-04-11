# app/api/memories.py
# 用户记忆管理 API 路由

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.response import success
from app.services.user_memory_service import UserMemoryService

router = APIRouter()


class MemoryCreateRequest(BaseModel):
    content: str


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryToggleRequest(BaseModel):
    is_active: bool


@router.get("")
def get_memories(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的所有记忆"""
    service = UserMemoryService(db)
    memories = service.get_all_memories(current_user.id)
    return success(
        data=[
            {
                "id": m.id,
                "content": m.content,
                "source": m.source,
                "conversation_id": m.conversation_id,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in memories
        ],
        request_id=request.state.request_id,
    )


@router.post("")
def create_memory(
    body: MemoryCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """手动添加记忆"""
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="记忆内容不能为空")
    service = UserMemoryService(db)
    memory = service.create_memory(current_user.id, body.content.strip())
    return success(
        data={
            "id": memory.id,
            "content": memory.content,
            "source": memory.source,
            "is_active": memory.is_active,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        },
        request_id=request.state.request_id,
    )


@router.put("/{memory_id}")
def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """编辑记忆内容"""
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="记忆内容不能为空")
    service = UserMemoryService(db)
    memory = service.update_memory(memory_id, current_user.id, body.content.strip())
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return success(
        data={
            "id": memory.id,
            "content": memory.content,
            "source": memory.source,
            "is_active": memory.is_active,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        },
        request_id=request.state.request_id,
    )


@router.patch("/{memory_id}")
def toggle_memory(
    memory_id: str,
    body: MemoryToggleRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """启用/停用记忆"""
    service = UserMemoryService(db)
    memory = service.toggle_memory(memory_id, current_user.id, body.is_active)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return success(
        data={"id": memory.id, "is_active": memory.is_active},
        request_id=request.state.request_id,
    )


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """软删除记忆"""
    service = UserMemoryService(db)
    result = service.delete_memory(memory_id, current_user.id)
    if not result:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return success(message="记忆已删除", request_id=request.state.request_id)
