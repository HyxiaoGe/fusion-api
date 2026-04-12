from typing import Optional

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import get_current_user, get_provider_repo
from app.db.models import User as UserModel
from app.db.repositories import ProviderRepository
from app.schemas.models import ProviderCreate, ProviderUpdate
from app.schemas.response import ApiException, success

router = APIRouter()


@router.get("/")
async def get_providers(
    request: Request,
    enabled: Optional[bool] = None,
    repo: ProviderRepository = Depends(get_provider_repo),
):
    """获取所有提供商列表"""
    providers = repo.get_all(enabled=enabled)
    provider_list = [repo.to_schema(p, order=idx) for idx, p in enumerate(providers, start=1)]
    return success(data={"providers": provider_list}, request_id=request.state.request_id)


@router.get("/{provider_id}")
async def get_provider(
    provider_id: str,
    request: Request,
    repo: ProviderRepository = Depends(get_provider_repo),
):
    """获取提供商详情"""
    provider = repo.get_by_id(provider_id)
    if not provider:
        raise ApiException.not_found(f"提供商 {provider_id} 不存在")
    return success(data=repo.to_schema(provider), request_id=request.state.request_id)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_provider(
    provider: ProviderCreate,
    request: Request,
    repo: ProviderRepository = Depends(get_provider_repo),
    current_user: UserModel = Depends(get_current_user),
):
    """创建提供商"""
    existing = repo.get_by_id(provider.id)
    if existing:
        raise ApiException.conflict(f"提供商 {provider.id} 已存在")
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
    repo: ProviderRepository = Depends(get_provider_repo),
    current_user: UserModel = Depends(get_current_user),
):
    """更新提供商"""
    existing = repo.get_by_id(provider_id)
    if not existing:
        raise ApiException.not_found(f"提供商 {provider_id} 不存在")
    update_data = provider.dict(exclude_unset=True)
    if "auth_config" in update_data and update_data["auth_config"]:
        update_data["auth_config"] = provider.auth_config.dict()
    updated = repo.update(provider_id, update_data)
    return success(data=repo.to_schema(updated), request_id=request.state.request_id)


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    request: Request,
    repo: ProviderRepository = Depends(get_provider_repo),
    current_user: UserModel = Depends(get_current_user),
):
    """删除提供商"""
    existing = repo.get_by_id(provider_id)
    if not existing:
        raise ApiException.not_found(f"提供商 {provider_id} 不存在")
    if existing.models:
        raise ApiException.bad_request(f"提供商 {provider_id} 下仍有 {len(existing.models)} 个模型，请先删除模型")
    repo.delete(provider_id)
    return success(message="提供商已删除", request_id=request.state.request_id)
