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
    return uuid.uuid4().hex


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

    @classmethod
    def bad_request(cls, message: str = "参数错误") -> "ApiException":
        return cls(ErrorCode.INVALID_PARAM, message, 400)

    @classmethod
    def not_found(cls, message: str = "资源不存在") -> "ApiException":
        return cls(ErrorCode.NOT_FOUND, message, 404)

    @classmethod
    def unauthorized(cls, message: str = "未授权") -> "ApiException":
        return cls(ErrorCode.UNAUTHORIZED, message, 401)

    @classmethod
    def forbidden(cls, message: str = "无权限") -> "ApiException":
        return cls(ErrorCode.FORBIDDEN, message, 403)

    @classmethod
    def conflict(cls, message: str = "资源冲突") -> "ApiException":
        return cls(ErrorCode.CONFLICT, message, 409)

    @classmethod
    def internal_error(cls, message: str = "服务器内部错误") -> "ApiException":
        return cls(ErrorCode.INTERNAL_ERROR, message, 500)
