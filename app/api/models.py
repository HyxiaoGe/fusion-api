from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import ModelSourceRepository
from app.schemas.models import ModelInfo, ModelsResponse, ModelCreate, ModelUpdate

router = APIRouter()


@router.get("/", response_model=ModelsResponse)
async def get_models(
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
    capability: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取所有可用的模型列表，支持筛选"""
    repository = ModelSourceRepository(db)
    model_sources = repository.get_all(provider, enabled, capability)
    
    # 将数据库模型转换为Pydantic模型
    models = [repository.to_basic_schema(model) for model in model_sources]
    return ModelsResponse(models=models)


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str, 
    db: Session = Depends(get_db)
):
    """根据ID获取模型详情"""
    repository = ModelSourceRepository(db)
    model_source = repository.get_by_id(model_id)
    
    if not model_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"模型 {model_id} 不存在"
        )
    
    return repository.to_full_schema(model_source)


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate, 
    db: Session = Depends(get_db)
):
    """创建新模型"""
    repository = ModelSourceRepository(db)
    
    # 检查模型ID是否已存在
    existing_model = repository.get_by_id(model.modelId)
    if existing_model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"模型ID {model.modelId} 已存在"
        )
    
    # 创建新模型
    model_data = model.dict()
    model_source = repository.create(model_data)
    return repository.to_full_schema(model_source)


@router.put("/{model_id}", response_model=ModelInfo)
async def update_model(
    model_id: str, 
    model: ModelUpdate, 
    db: Session = Depends(get_db)
):
    """更新模型信息"""
    repository = ModelSourceRepository(db)
    
    # 检查模型是否存在
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"模型 {model_id} 不存在"
        )
    
    # 更新模型
    update_data = model.dict(exclude_unset=True)
    updated_model = repository.update(model_id, update_data)
    return repository.to_full_schema(updated_model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str, 
    db: Session = Depends(get_db)
):
    """删除模型"""
    repository = ModelSourceRepository(db)
    
    # 检查模型是否存在
    existing_model = repository.get_by_id(model_id)
    if not existing_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"模型 {model_id} 不存在"
        )
    
    # 删除模型
    repository.delete(model_id)
    return None