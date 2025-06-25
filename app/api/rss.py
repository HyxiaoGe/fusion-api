from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.db.repositories import RssSourceRepository
from app.schemas.rss import RssSourceCreate, RssSourceUpdate, RssSourceResponse
from app.core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=RssSourceResponse, status_code=201)
def create_rss_source(
    rss_source: RssSourceCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建一个新的RSS源"""
    repo = RssSourceRepository(db)
    try:
        created_source = repo.create(rss_source, user_id=current_user.id)
        return created_source
    except Exception as e:
        # 可能是因为name或url重复
        raise HTTPException(status_code=409, detail=f"无法创建RSS源: {e}")

@router.get("/{source_id}", response_model=RssSourceResponse)
def get_rss_source(
    source_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取指定ID的RSS源"""
    repo = RssSourceRepository(db)
    db_source = repo.get_by_id(source_id, user_id=current_user.id)
    if db_source is None:
        raise HTTPException(status_code=404, detail="RSS源未找到或无权访问")
    return db_source

@router.get("/", response_model=List[RssSourceResponse])
def get_all_rss_sources(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取所有RSS源的列表（分页），包括用户的和全局的"""
    repo = RssSourceRepository(db)
    sources = repo.get_all(user_id=current_user.id, skip=skip, limit=limit)
    return sources

@router.put("/{source_id}", response_model=RssSourceResponse)
def update_rss_source(
    source_id: str,
    rss_source: RssSourceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新指定ID的RSS源"""
    repo = RssSourceRepository(db)
    updated_source = repo.update(source_id, rss_source, user_id=current_user.id)
    if updated_source is None:
        raise HTTPException(status_code=404, detail="RSS源未找到或无权访问")
    return updated_source

@router.delete("/{source_id}", status_code=204)
def delete_rss_source(
    source_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除指定ID的RSS源"""
    repo = RssSourceRepository(db)
    if not repo.delete(source_id, user_id=current_user.id):
        raise HTTPException(status_code=404, detail="RSS源未找到或无权访问")
    return None 