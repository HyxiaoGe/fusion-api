from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.database import get_db
from app.services.hot_topic_service import HotTopicService
from app.schemas.hot_topics import HotTopicResponse

router = APIRouter()

@router.get("/hot", response_model=List[HotTopicResponse])
def get_hot_topics(
    limit: int = Query(10, description="返回结果数量限制", ge=1, le=50),
    db: Session = Depends(get_db)
):
    """获取热点话题列表"""
    service = HotTopicService(db)
    topics = service.get_hot_topics(limit=limit)
    return topics

@router.get("/{topic_id}", response_model=HotTopicResponse)
def get_hot_topic(
    topic_id: str,
    db: Session = Depends(get_db)
):
    """获取单个热点话题详情"""
    service = HotTopicService(db)
    topic = service.get_topic_by_id(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="热点话题不存在")
    return topic

@router.post("/refresh")
async def refresh_hot_topics(
    force: bool = Query(True, description="是否强制更新，忽略时间间隔限制"),
    db: Session = Depends(get_db)
):
    """手动刷新热点话题数据（同步方式）"""
    service = HotTopicService(db)
    # 等待更新完成并获取新增数量
    new_count = await service.update_hot_topics(force=force)
    
    return {
        "status": "success", 
        "message": "热点话题刷新操作已完成",
        "new_count": new_count,  # 返回新增的数量
        "timestamp": datetime.now().isoformat()
    }