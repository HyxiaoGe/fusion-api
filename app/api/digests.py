"""
话题聚合摘要API
"""

from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, DailyTopicDigest
from app.core.security import get_current_user
from app.services.digest_service import DigestService
from app.schemas.digest import DigestResponse, DigestListResponse

router = APIRouter()


@router.get("/daily", response_model=DigestListResponse)
def get_daily_digests(
    target_date: Optional[date] = Query(None, description="目标日期，默认为今天"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取指定日期的话题摘要"""
    digest_service = DigestService(db)
    digests = digest_service.get_daily_digests(target_date)
    
    return DigestListResponse(
        date=target_date or date.today(),
        total=len(digests),
        digests=[
            DigestResponse(
                id=digest.id,
                category=digest.category,
                cluster_title=digest.cluster_title,
                cluster_summary=digest.cluster_summary,
                key_points=digest.key_points,
                topic_count=digest.topic_count,
                heat_score=digest.heat_score,
                view_count=digest.view_count
            )
            for digest in digests
        ]
    )


@router.post("/{digest_id}/view")
def increment_view_count(
    digest_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """增加摘要的查看次数"""
    digest_service = DigestService(db)
    success = digest_service.increment_digest_view_count(digest_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="摘要不存在")
    
    return {"status": "success"}