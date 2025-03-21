from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.vector_service import VectorService

router = APIRouter()


class SearchResult(BaseModel):
    results: List[Dict[str, Any]]


@router.get("/conversations", response_model=SearchResult)
def search_conversations(
        query: str = Query(..., description="搜索查询文本"),
        limit: int = Query(5, description="返回结果数量限制"),
        db: Session = Depends(get_db)
):
    """基于语义搜索相关对话"""
    try:
        vector_service = VectorService.get_instance(db)
        results = vector_service.search_conversations(query, limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索对话失败: {str(e)}")


@router.get("/messages", response_model=SearchResult)
def search_messages(
        query: str = Query(..., description="搜索查询文本"),
        conversation_id: Optional[str] = Query(None, description="限制在特定对话中搜索"),
        limit: int = Query(5, description="返回结果数量限制"),
        db: Session = Depends(get_db)
):
    """基于语义搜索相关消息"""
    try:
        vector_service = VectorService.get_instance(db)
        results = vector_service.search_messages(query, conversation_id, limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索消息失败: {str(e)}")


@router.get("/context", response_model=Dict[str, Any])
def enhance_context(
        query: str = Query(..., description="用户查询"),
        conversation_id: Optional[str] = Query(None, description="当前对话ID"),
        db: Session = Depends(get_db)
):
    """获取增强查询的上下文"""
    from app.services.context_service import ContextEnhancer

    try:
        context_enhancer = ContextEnhancer(db)
        enhanced_context = context_enhancer.enhance_prompt(query, conversation_id)
        return enhanced_context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增强上下文失败: {str(e)}")
