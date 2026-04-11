from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User as UserModel
from app.db.repositories import ProviderRepository
from app.schemas.models import ProviderCreate, ProviderUpdate
from app.schemas.response import success

router = APIRouter()


@router.get("/")
async def get_providers(
    request: Request,
    enabled: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    """获取所有提供商列表"""
    repo = ProviderRepository(db)
    providers = repo.get_all(enabled=enabled)
    provider_list = [repo.to_schema(p, order=idx) for idx, p in enumerate(providers, start=1)]
    return success(data={"providers": provider_list}, request_id=request.state.request_id)


@router.get("/{provider_id}")
async def get_provider(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """获取提供商详情"""
    repo = ProviderRepository(db)
    provider = repo.get_by_id(provider_id)
    if not provider:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"提供商 {provider_id} 不存在")
    return success(data=repo.to_schema(provider), request_id=request.state.request_id)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_provider(
    provider: ProviderCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """创建提供商"""
    repo = ProviderRepository(db)
    existing = repo.get_by_id(provider.id)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"提供商 {provider.id} 已存在")
    data = provider.dict()
    if "auth_config" in data and data["auth_config"]:
        data["auth_config"] = provider.auth_config.dict()
    new_provider = repo.create(data)
    return success(data=repo.to_schema(new_provider), message="提供商创建成功", request_id=request.state.request_id)


@router.put("/{provider_id}")
async def update_provider(
    provider_id: str,
    provider: ProviderUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """更新提供商"""
    repo = ProviderRepository(db)
    existing = repo.get_by_id(provider_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"提供商 {provider_id} 不存在")
    update_data = provider.dict(exclude_unset=True)
    if "auth_config" in update_data and update_data["auth_config"]:
        update_data["auth_config"] = provider.auth_config.dict()
    updated = repo.update(provider_id, update_data)
    return success(data=repo.to_schema(updated), request_id=request.state.request_id)


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """删除提供商"""
    repo = ProviderRepository(db)
    existing = repo.get_by_id(provider_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"提供商 {provider_id} 不存在")
    if existing.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"提供商 {provider_id} 下仍有 {len(existing.models)} 个模型，请先删除模型",
        )
    repo.delete(provider_id)
    return success(message="提供商已删除", request_id=request.state.request_id)
