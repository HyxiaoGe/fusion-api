from typing import Dict, List, Optional
from pydantic import BaseModel


class ModelCapabilities(BaseModel):
    """模型能力配置"""
    vision: bool = False
    deepThinking: bool = False
    fileSupport: bool = False


class ModelPricing(BaseModel):
    """模型定价信息"""
    input: float
    output: float
    unit: str = "USD"


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    modelId: str
    provider: str
    knowledgeCutoff: Optional[str] = None
    capabilities: ModelCapabilities
    pricing: ModelPricing
    enabled: bool = True
    description: str = ""


class ModelsResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelInfo]


class ModelCreate(ModelInfo):
    """创建模型请求体"""
    pass


class ModelUpdate(BaseModel):
    """更新模型请求体"""
    name: Optional[str] = None
    provider: Optional[str] = None
    knowledgeCutoff: Optional[str] = None
    capabilities: Optional[ModelCapabilities] = None
    pricing: Optional[ModelPricing] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None 