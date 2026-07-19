"""性能工具访问受控账密认证接口所需的内部门禁。"""

from __future__ import annotations

INTERNAL_AUTH_ENV_VAR = "FUSION_PERF_INTERNAL_AUTH_TOKEN"
INTERNAL_AUTH_HEADER = "X-Fusion-Internal-Auth"


class InternalAuthGateError(ValueError):
    """受控账密认证门禁未满足。"""


def require_internal_auth_token(value: str | None) -> str:
    """返回有效内部 token；缺失时在认证网络请求前 fail closed。"""

    if not isinstance(value, str) or len(value) < 32 or value != value.strip():
        raise InternalAuthGateError(
            f"{INTERNAL_AUTH_ENV_VAR} 必须至少 32 字符且不能包含首尾空白，禁止调用受控账密认证接口"
        )
    return value
