from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import ModelCredentialRepository, ModelSourceRepository
from app.schemas.models import (
    ModelCredentialInfo, CredentialsResponse,
    ModelCredentialCreate, ModelCredentialUpdate,
    CredentialTestRequest, CredentialTestResponse
)
from app.ai.llm_manager import llm_manager

router = APIRouter()

@router.get("/", response_model=CredentialsResponse)
async def get_credentials(
    model_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取所有凭证或特定模型的凭证"""
    repository = ModelCredentialRepository(db)
    credentials = repository.get_all(model_id)
    
    # 转换为响应模型并返回
    credential_infos = [repository.to_schema(cred) for cred in credentials]
    return CredentialsResponse(credentials=credential_infos)

@router.get("/{credential_id}", response_model=ModelCredentialInfo)
async def get_credential(
    credential_id: int,
    db: Session = Depends(get_db)
):
    """获取特定凭证详情"""
    repository = ModelCredentialRepository(db)
    credential = repository.get_by_id(credential_id)
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"凭证 {credential_id} 不存在"
        )
    
    return repository.to_schema(credential)

@router.post("/", response_model=ModelCredentialInfo, status_code=status.HTTP_201_CREATED)
async def create_credential(
    credential: ModelCredentialCreate,
    db: Session = Depends(get_db)
):
    """创建新的凭证"""
    # 验证模型是否存在
    model_repo = ModelSourceRepository(db)
    model = model_repo.get_by_id(credential.model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"模型 {credential.model_id} 不存在"
        )
    
    # 创建凭证
    credential_repo = ModelCredentialRepository(db)
    credential_data = credential.dict()
    new_credential = credential_repo.create(credential_data)
    
    return credential_repo.to_schema(new_credential)

@router.put("/{credential_id}", response_model=ModelCredentialInfo)
async def update_credential(
    credential_id: int,
    credential: ModelCredentialUpdate,
    db: Session = Depends(get_db)
):
    """更新凭证"""
    repository = ModelCredentialRepository(db)
    
    # 检查凭证是否存在
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"凭证 {credential_id} 不存在"
        )
    
    # 更新凭证
    credential_data = credential.dict(exclude_unset=True)
    updated = repository.update(credential_id, credential_data)
    
    return repository.to_schema(updated)

@router.delete("/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credential(
    credential_id: int,
    db: Session = Depends(get_db)
):
    """删除凭证"""
    repository = ModelCredentialRepository(db)
    
    # 检查凭证是否存在
    existing = repository.get_by_id(credential_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"凭证 {credential_id} 不存在"
        )
    
    # 删除凭证
    repository.delete(credential_id)
    return None

@router.post("/test", response_model=CredentialTestResponse)
async def test_credential(
    request: CredentialTestRequest,
    db: Session = Depends(get_db)
):
    """测试凭证是否有效"""
    try:
        # 验证模型是否存在
        model_repo = ModelSourceRepository(db)
        model = model_repo.get_by_id(request.model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型 {request.model_id} 不存在"
            )
        
        # 调用LLM管理器测试凭证
        result = await llm_manager.test_credentials(model.provider, request.credentials)
        
        if result:
            return CredentialTestResponse(success=True, message="凭证有效")
        else:
            return CredentialTestResponse(success=False, message="凭证无效")
    except Exception as e:
        return CredentialTestResponse(success=False, message=f"测试失败: {str(e)}")