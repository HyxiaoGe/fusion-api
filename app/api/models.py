from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import ModelSourceRepository
from app.schemas.models import ModelInfo, ModelsResponse, ModelCreate, ModelUpdate

router = APIRouter()


@router.get("/", response_model=ModelsResponse)
async def get_models(db: Session = Depends(get_db)):
    """获取所有可用的模型列表"""
    repository = ModelSourceRepository(db)
    model_sources = repository.get_all()
    
    # 如果数据库中没有模型数据，则返回默认的模型列表
    if not model_sources:
        return init_default_models(db)
    
    # 将数据库模型转换为Pydantic模型
    models = [repository.to_schema(model) for model in model_sources]
    return ModelsResponse(models=models)


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """根据ID获取模型"""
    repository = ModelSourceRepository(db)
    model_source = repository.get_by_id(model_id)
    
    if not model_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"模型 {model_id} 不存在"
        )
    
    return repository.to_schema(model_source)


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def create_model(model: ModelCreate, db: Session = Depends(get_db)):
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
    model_source = repository.create(model)
    return repository.to_schema(model_source)


@router.put("/{model_id}", response_model=ModelInfo)
async def update_model(model_id: str, model: ModelUpdate, db: Session = Depends(get_db)):
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
    updated_model = repository.update(model_id, model.dict(exclude_unset=True))
    return repository.to_schema(updated_model)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str, db: Session = Depends(get_db)):
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


def init_default_models(db: Session) -> ModelsResponse:
    """初始化默认模型数据"""
    from app.schemas.models import ModelCapabilities, ModelPricing
    
    repository = ModelSourceRepository(db)
    
    # 默认模型列表
    default_models = [
        ModelInfo(
            name="GPT-3.5 Turbo",
            modelId="gpt-3.5-turbo",
            provider="openai",
            knowledgeCutoff="2023-09",
            capabilities=ModelCapabilities(
                vision=True,
                deepThinking=False,
                fileSupport=True,
            ),
            pricing=ModelPricing(
                input=0.0015,
                output=0.002,
                unit="USD"
            ),
            enabled=True,
            description="OpenAI高效且经济的模型，适合日常对话和一般性任务，反应速度快。",
        ),
        ModelInfo(
            name="GPT-4o",
            modelId="gpt-4o",
            provider="openai",
            knowledgeCutoff="2023-12",
            capabilities=ModelCapabilities(
                vision=True,
                deepThinking=True,
                fileSupport=True,
            ),
            pricing=ModelPricing(
                input=0.01,
                output=0.03,
                unit="USD"
            ),
            enabled=True,
            description="OpenAI最新的多模态大型语言模型，具有卓越的理解能力和创造力，支持图像识别。",
        ),
        ModelInfo(
            name="Claude 3 Opus",
            modelId="claude-3-opus-20240229",
            provider="anthropic",
            knowledgeCutoff="2023-12",
            capabilities=ModelCapabilities(
                vision=True,
                deepThinking=True,
                fileSupport=True,
            ),
            pricing=ModelPricing(
                input=0.015,
                output=0.075,
                unit="USD"
            ),
            enabled=True,
            description="Anthropic的顶级模型，具有极强的推理能力和更新的知识，支持图像理解和文件处理。",
        ),
        ModelInfo(
            name="通义千问Max",
            modelId="qwen-max",
            provider="qwen",
            knowledgeCutoff="2023-12",
            capabilities=ModelCapabilities(
                vision=True,
                deepThinking=True,
                fileSupport=True,
            ),
            pricing=ModelPricing(
                input=0.001,
                output=0.002,
                unit="USD"
            ),
            enabled=True,
            description="阿里云推出的大型语言模型，在中文理解和创作上表现出色，支持多模态输入。",
        ),
    ]
    
    # 将默认模型添加到数据库
    models = []
    for model in default_models:
        # 检查模型是否已存在
        existing_model = repository.get_by_id(model.modelId)
        if not existing_model:
            model_source = repository.create(model)
            models.append(repository.to_schema(model_source))
        else:
            models.append(repository.to_schema(existing_model))
    
    return ModelsResponse(models=models) 