from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

McpTransport = Literal["streamable_http"]
McpAuthType = Literal["none", "bearer", "header", "query"]
McpHealthStatus = Literal["unknown", "healthy", "unhealthy", "disabled"]


class McpServerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=80)
    endpoint_url: str = Field(min_length=1, max_length=2_048)
    transport: McpTransport = "streamable_http"
    auth_type: McpAuthType = "none"
    auth_name: str | None = Field(default=None, max_length=128)
    credential_ref: str | None = Field(default=None, max_length=128)
    allowed_tools: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 128 for value in normalized):
            raise ValueError("allowed_tools 包含无效工具名")
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_tools 不得包含重复工具")
        return normalized

    @model_validator(mode="after")
    def validate_auth_shape(self):
        if self.auth_type == "none" and (self.auth_name or self.credential_ref):
            raise ValueError("none 鉴权不得配置 auth_name 或 credential_ref")
        if self.auth_type == "bearer" and (self.auth_name or not self.credential_ref):
            raise ValueError("bearer 鉴权必须仅配置 credential_ref")
        if self.auth_type in {"header", "query"} and (not self.auth_name or not self.credential_ref):
            raise ValueError(f"{self.auth_type} 鉴权必须配置 auth_name 和 credential_ref")
        return self


class McpServerCreate(McpServerPayload):
    pass


class McpServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=120)
    provider: str | None = Field(default=None, min_length=1, max_length=80)
    endpoint_url: str | None = Field(default=None, min_length=1, max_length=2_048)
    transport: McpTransport | None = None
    auth_type: McpAuthType | None = None
    auth_name: str | None = Field(default=None, max_length=128)
    credential_ref: str | None = Field(default=None, max_length=128)
    allowed_tools: list[str] | None = Field(default=None, max_length=50)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return McpServerPayload.validate_allowed_tools(values)


class McpServerStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_enabled: bool


class McpToolSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None
    input_schema: dict[str, Any]


class McpServerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    provider: str
    endpoint_url: str
    transport: McpTransport
    auth_type: McpAuthType
    auth_name: str | None
    credential_ref: str | None
    is_enabled: bool
    allowed_tools: list[str]
    discovered_tools: list[McpToolSnapshot]
    health_status: McpHealthStatus
    last_checked_at: datetime | None
    last_error_code: str | None
    last_error_message: str | None
    created_at: datetime
    updated_at: datetime
