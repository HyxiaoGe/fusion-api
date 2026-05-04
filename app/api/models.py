from typing import Optional

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import get_current_user, get_model_source_repo
from app.db.models import User as UserModel
from app.db.repositories import ModelSourceRepository
from app.schemas.models import (
    ModelCreate,
    ModelUpdate,
    ProviderBasicInfo,
)
from app.schemas.response import ApiException, success

router = APIRouter()


@router.get("/")
async def get_models(
    request: Request,
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
    capability: Optional[str] = None,
    repository: ModelSourceRepository = Depends(get_model_source_repo),
):
    """获取所有可用的模型列表，支持筛选"""
    model_sources = repository.get_all(provider, enabled, capability)
    models = [repository.to_basic_schema(model) for model in model_sources]
    providers = [ProviderBasicInfo(**p) for p in repository.get_providers()]
    return success(data={"models": models, "providers": providers}, request_id=request.state.request_id)


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    request: Request,
    repository: ModelSourceRepository = Depends(get_model_source_repo),
):
    """根据ID获取模型详情"""
    model_source = repository.get_by_id(model_id)
    if not model_source:
        raise ApiException.not_found(f"模型 {model_id} 不存在")
    return success(data=repository.to_full_schema(model_source), request_id=request.state.request_id)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repository: ModelSourceRepository = Depends(get_model_source_repo),
):
    """创建新模型"""
    existing_model = repository.get_by_id(model.modelId)
    if existing_model:
        raise ApiException.conflict(f"模型ID {model.modelId} 已存在")
    model_data = model.dict()
    model_source = repository.create(model_data)
    return success(
        data=repository.to_full_schema(model_source), message="模型创建成功", request_id=request.state.request_id
    )


@router.put("/{model_id}")
async def update_model(
    model_id: str,
    model: ModelUpdate,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repository: ModelSourceRepository = Depends(get_model_source_repo),
):
    """更新模型信息"""
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise ApiException.not_found(f"模型 {model_id} 不存在")
    update_data = model.dict(exclude_unset=True)
    updated_model = repository.update(model_id, update_data)
    return success(data=repository.to_full_schema(updated_model), request_id=request.state.request_id)


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    repository: ModelSourceRepository = Depends(get_model_source_repo),
):
    """删除模型"""
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise ApiException.not_found(f"模型 {model_id} 不存在")
    repository.delete(model_id)
    return success(message="模型已删除", request_id=request.state.request_id)
