from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User as UserModel
from app.db.repositories import ModelCredentialRepository, ModelSourceRepository
from app.schemas.models import (
    CredentialsResponse,
    CredentialTestRequest,
    CredentialTestResponse,
    ModelCreate,
    ModelCredentialCreate,
    ModelCredentialInfo,
    ModelCredentialUpdate,
    ModelInfo,
    ModelsResponse,
    ModelUpdate,
    ProviderInfo,
)

router = APIRouter()


@router.get("/", response_model=ModelsResponse)
async def get_models(
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
    capability: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """获取所有可用的模型列表，支持筛选"""
    repository = ModelSourceRepository(db)
    model_sources = repository.get_all(provider, enabled, capability)

    # 将数据库模型转换为Pydantic模型
    models = [repository.to_basic_schema(model) for model in model_sources]
    providers = [ProviderInfo(**p) for p in repository.get_providers()]
    return ModelsResponse(models=models, providers=providers)


@router.get("/{model_id}/credentials", response_model=CredentialsResponse)
async def get_model_credentials(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    credentials = repository.get_all(model_id)
    credential_infos = [repository.to_schema(cred) for cred in credentials]
    return CredentialsResponse(credentials=credential_infos)


@router.post("/{model_id}/credentials", response_model=ModelCredentialInfo, status_code=status.HTTP_201_CREATED)
async def create_model_credential(
    model_id: str,
    credential: ModelCredentialCreate,
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
    return credential_repo.to_schema(new_credential)


@router.put("/credentials/{credential_id}", response_model=ModelCredentialInfo)
async def update_model_credential(
    credential_id: int,
    credential: ModelCredentialUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"凭证 {credential_id} 不存在")

    credential_data = credential.dict(exclude_unset=True)
    updated = repository.update(credential_id, credential_data)
    return repository.to_schema(updated)


@router.delete("/credentials/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_credential(
    credential_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    repository = ModelCredentialRepository(db)
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"凭证 {credential_id} 不存在")

    repository.delete(credential_id)
    return None


@router.post("/credentials/test", response_model=CredentialTestResponse)
async def test_model_credential(
    request: CredentialTestRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        model_repo = ModelSourceRepository(db)
        model = model_repo.get_by_id(request.model_id)
        if not model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {request.model_id} 不存在")

        result = await llm_manager.test_credentials(model.provider, request.model_id, request.credentials)
        if result:
            return CredentialTestResponse(success=True, message="凭证有效")
        return CredentialTestResponse(success=False, message="凭证无效")
    except Exception as e:
        return CredentialTestResponse(success=False, message=f"测试失败: {str(e)}")


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """根据ID获取模型详情"""
    repository = ModelSourceRepository(db)
    model_source = repository.get_by_id(model_id)

    if not model_source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")

    return repository.to_full_schema(model_source)


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """创建新模型"""
    repository = ModelSourceRepository(db)

    # 检查模型ID是否已存在
    existing_model = repository.get_by_id(model.modelId)
    if existing_model:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"模型ID {model.modelId} 已存在")

    # 创建新模型
    model_data = model.dict()
    model_source = repository.create(model_data)
    return repository.to_full_schema(model_source)


@router.put("/{model_id}", response_model=ModelInfo)
async def update_model(
    model_id: str,
    model: ModelUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """更新模型信息"""
    repository = ModelSourceRepository(db)

    # 检查模型是否存在
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")

    # 更新模型
    update_data = model.dict(exclude_unset=True)
    updated_model = repository.update(model_id, update_data)
    return repository.to_full_schema(updated_model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """删除模型"""
    repository = ModelSourceRepository(db)

    # 检查模型是否存在
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型 {model_id} 不存在")

    # 删除模型
    repository.delete(model_id)
    return None
