import uuid
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorCode(str, Enum):
    """业务状态码枚举"""

    # 通用
    SUCCESS = "SUCCESS"
    INVALID_PARAM = "INVALID_PARAM"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"

    # 业务相关
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CREDENTIAL_INVALID = "CREDENTIAL_INVALID"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_TYPE_NOT_ALLOWED = "FILE_TYPE_NOT_ALLOWED"
    STREAM_NOT_FOUND = "STREAM_NOT_FOUND"
    GENERATION_FAILED = "GENERATION_FAILED"


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应结构"""

    code: str = "SUCCESS"
    message: str = "ok"
    data: Optional[T] = None
    request_id: str


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def success(data: Any = None, message: str = "ok", request_id: str = "") -> ApiResponse:
    """构造成功响应"""
    return ApiResponse(code="SUCCESS", message=message, data=data, request_id=request_id)


class ApiException(Exception):
    """自定义业务异常，由全局异常处理器捕获并格式化"""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)
