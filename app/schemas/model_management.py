from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelVisibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=200)
    selectable: bool
    reason: str = Field(min_length=1, max_length=300)
    expected_revision: int | None = Field(default=None, ge=1)

    @field_validator("model_id", "reason")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized


class CandidateAdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=200)
    expected_run_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=300)

    @field_validator("model_id", "expected_run_id", "reason")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized


class AdmissionCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    phase: str = Field(min_length=1, max_length=80)
    error_code: str | None = Field(default=None, max_length=100)
    completed_phases: list[str] = Field(default_factory=list, max_length=20)
    writes_performed: bool
    compensation: dict[str, Any] = Field(default_factory=dict)
