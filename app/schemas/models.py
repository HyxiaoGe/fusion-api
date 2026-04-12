from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelCapabilities(BaseModel):
    """模型能力配置"""

    imageGen: bool = False
    deepThinking: bool = False
    fileSupport: bool = False
    functionCalling: bool = False  # 工具调用（含联网搜索）
    vision: bool = False  # 是否支持图片理解（多模态视觉输入）


class ModelPricing(BaseModel):
    """模型定价信息"""

    input: float
    output: float
    unit: str = "USD"


class AuthConfigField(BaseModel):
    """认证配置字段定义"""

    name: str
    display_name: str
    type: str  # 如 "password", "text" 等
    required: bool = True
    default: Optional[str] = None
    description: Optional[str] = None


class AuthConfig(BaseModel):
    """认证配置模板"""

    fields: List[AuthConfigField]
    auth_type: str = "api_key"  # 如 "api_key", "dual_key", "oauth" 等


class ModelConfigParam(BaseModel):
    """模型参数配置字段定义"""

    name: str
    display_name: str
    type: str  # 如 "number", "text", "boolean" 等
    required: bool = False
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None


class ModelConfiguration(BaseModel):
    """模型参数配置模板"""

    params: List[ModelConfigParam]


class ModelBasicInfo(BaseModel):
    """模型基础信息（用于列表展示）"""

    name: str
    modelId: str
    provider: str
    knowledgeCutoff: Optional[str] = None
    capabilities: ModelCapabilities
    priority: int = 100  # 添加优先级字段，默认为100
    enabled: bool = True
    description: str = ""

    class Config:
        from_attributes = True


class ModelInfo(ModelBasicInfo):
    """模型信息"""

    pricing: ModelPricing
    auth_config: Optional[AuthConfig] = None
    model_configuration: Optional[ModelConfiguration] = None

    class Config:
        from_attributes = True


class ModelCreate(BaseModel):
    """创建模型请求"""

    modelId: str
    name: str
    provider: str
    knowledgeCutoff: Optional[str] = None
    capabilities: ModelCapabilities
    pricing: ModelPricing
    model_configuration: Optional[ModelConfiguration] = None
    priority: int = 100
    enabled: bool = True
    description: str = ""


class ModelUpdate(BaseModel):
    """更新模型请求"""

    name: Optional[str] = None
    provider: Optional[str] = None
    knowledgeCutoff: Optional[str] = None
    capabilities: Optional[ModelCapabilities] = None
    pricing: Optional[ModelPricing] = None
    model_configuration: Optional[ModelConfiguration] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None


class ProviderBasicInfo(BaseModel):
    """提供商基础信息（用于模型列表的筛选下拉）"""

    id: str
    name: str
    order: int = 0


class ProviderInfo(BaseModel):
    """提供商完整信息"""

    id: str
    name: str
    auth_config: Optional[AuthConfig] = None
    litellm_prefix: str = ""
    custom_base_url: bool = False
    priority: int = 100
    enabled: bool = True
    order: int = 0

    class Config:
        from_attributes = True


class ProviderCreate(BaseModel):
    """创建提供商"""

    id: str
    name: str
    auth_config: AuthConfig
    litellm_prefix: str
    custom_base_url: bool = False
    priority: int = 100
    enabled: bool = True


class ProviderUpdate(BaseModel):
    """更新提供商"""

    name: Optional[str] = None
    auth_config: Optional[AuthConfig] = None
    litellm_prefix: Optional[str] = None
    custom_base_url: Optional[bool] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None


# 模型列表响应
class ModelsResponse(BaseModel):
    """模型列表响应"""

    models: List[ModelBasicInfo]
    providers: List[ProviderBasicInfo] = []


class ModelCredentialInfo(BaseModel):
    """模型凭证信息"""

    id: int
    model_id: str
    name: str
    is_default: bool = False
    credentials: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelCredentialCreate(BaseModel):
    """创建模型凭证请求"""

    model_id: str
    name: str
    is_default: bool = False
    credentials: Dict[str, Any]


class ModelCredentialUpdate(BaseModel):
    """更新凭证请求"""

    name: Optional[str] = None
    is_default: Optional[bool] = None
    credentials: Optional[Dict[str, Any]] = None


# 凭证列表响应
class CredentialsResponse(BaseModel):
    """凭证列表响应"""

    credentials: List[ModelCredentialInfo]


# 凭证测试请求
class CredentialTestRequest(BaseModel):
    """凭证测试请求"""

    model_id: str
    credentials: Dict[str, Any]


# 凭证测试响应
class CredentialTestResponse(BaseModel):
    """凭证测试响应"""

    success: bool
    message: str
