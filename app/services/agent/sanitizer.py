"""arguments 脱敏 hook + result_summary 截断."""
from __future__ import annotations

import json
from typing import Any


def sanitize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """对 tool arguments 做脱敏。

    v1 默认透传；未来加敏感工具时按 tool_name 派发到具体规则。
    """
    return arguments


def cap_and_truncate(payload: dict[str, Any], max_bytes: int = 1024) -> dict[str, Any]:
    """把 result_summary 控制在 max_bytes 之内。

    超出时截断字符串字段（保留非字符串字段不动），并置 truncated=True。

    注：v1 仅截断顶层 string 字段；嵌套 dict / list 不参与截断。
    若调用方写入嵌套字段且总体超限，结果可能仍超过 max_bytes（truncated=True 仍会被设）。
    """
    serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(serialized) <= max_bytes:
        return payload

    out = dict(payload)
    out["truncated"] = True

    # 估算可保留预算：先把所有字符串字段缩一半，循环到达标
    while len(json.dumps(out, ensure_ascii=False).encode("utf-8")) > max_bytes:
        shrunk = False
        for k, v in list(out.items()):
            if isinstance(v, str) and len(v) > 9:
                out[k] = v[: max(8, len(v) // 2)] + "…"
                shrunk = True
        if not shrunk:
            break  # 字符串字段都已最小，无法再缩
    return out
