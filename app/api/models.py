from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User as UserModel
from app.db.repositories import ModelCredentialRepository, ModelSourceRepository
from app.schemas.models import (
    CredentialTestRequest,
    ModelCreate,
    ModelCredentialCreate,
    ModelCredentialUpdate,
    ModelUpdate,
    ProviderInfo,
)
from app.schemas.response import success

router = APIRouter()


@router.get("/")
async def get_models(
    request: Request,
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
    capability: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """获取所有可用的模型列表，支持筛选"""
    repository = ModelSourceRepository(db)
    model_sources = repository.get_all(provider, enabled, capability)
    models = [repository.to_basic_schema(model) for model in model_sources]
    providers = [ProviderInfo(**p) for p in repository.get_providers()]
    return success(data={"models": models, "providers": providers}, request_id=request.state.request_id)


@router.get("/{model_id}/credentials")
async def get_model_credentials(
    model_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    credentials = repository.get_all(model_id)
    credential_infos = [repository.to_schema(cred) for cred in credentials]
    return success(data={"credentials": credential_infos}, request_id=request.state.request_id)


@router.post("/{model_id}/credentials", status_code=status.HTTP_201_CREATED)
async def create_model_credential(
    model_id: str,
    credential: ModelCredentialCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    model_repo = ModelSourceRepository(db)
    model = model_repo.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")
    credential_repo = ModelCredentialRepository(db)
    credential_data = credential.dict()
    credential_data["model_id"] = model_id
    new_credential = credential_repo.create(credential_data)
    return success(
        data=credential_repo.to_schema(new_credential), message="凭证创建成功", request_id=request.state.request_id
    )


@router.put("/credentials/{credential_id}")
async def update_model_credential(
    credential_id: int,
    credential: ModelCredentialUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"凭证 {credential_id} 不存在")
    credential_data = credential.dict(exclude_unset=True)
    updated = repository.update(credential_id, credential_data)
    return success(data=repository.to_schema(updated), request_id=request.state.request_id)


@router.delete("/credentials/{credential_id}")
async def delete_model_credential(
    credential_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"凭证 {credential_id} 不存在")
    repository.delete(credential_id)
    return success(message="凭证已删除", request_id=request.state.request_id)


@router.post("/credentials/test")
async def test_model_credential(
    test_request: CredentialTestRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        model_repo = ModelSourceRepository(db)
        model = model_repo.get_by_id(test_request.model_id)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {test_request.model_id} 不存在")
        result = await llm_manager.test_credentials(model.provider, test_request.model_id, test_request.credentials)
        if result:
            return success(data={"success": True, "message": "凭证有效"}, request_id=request.state.request_id)
        return success(data={"success": False, "message": "凭证无效"}, request_id=request.state.request_id)
    except HTTPException:
        raise
    except Exception as e:
        return success(data={"success": False, "message": f"测试失败: {str(e)}"}, request_id=request.state.request_id)


@router.get("/{model_id}")
async def get_model(model_id: str, request: Request, db: Session = Depends(get_db)):
    """根据ID获取模型详情"""
    repository = ModelSourceRepository(db)
    model_source = repository.get_by_id(model_id)
    if not model_source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")
    return success(data=repository.to_full_schema(model_source), request_id=request.state.request_id)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """创建新模型"""
    repository = ModelSourceRepository(db)
    existing_model = repository.get_by_id(model.modelId)
    if existing_model:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"模型ID {model.modelId} 已存在")
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
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """更新模型信息"""
    repository = ModelSourceRepository(db)
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")
    update_data = model.dict(exclude_unset=True)
    updated_model = repository.update(model_id, update_data)
    return success(data=repository.to_full_schema(updated_model), request_id=request.state.request_id)


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """删除模型"""
    repository = ModelSourceRepository(db)
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")
    repository.delete(model_id)
    return success(message="模型已删除", request_id=request.state.request_id)
