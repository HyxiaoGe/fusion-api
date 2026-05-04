"""arguments 脱敏 hook + result_summary 截断."""
from __future__ import annotations

import copy
import json
from typing import Any


def sanitize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """对 tool arguments 做脱敏。

    v1 默认透传；未来加敏感工具时按 tool_name 派发到具体规则。
    """
    return arguments


def _utf8_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _shrink_strings(node: Any) -> bool:
    """递归把 node 中所有 len>9 的 str 字段减半 + "…"。

    支持 dict / list 任意嵌套；in-place 修改。
    返回 True 表示本次至少有一个字段被缩短。
    """
    shrunk = False
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str) and len(v) > 9:
                node[k] = v[: max(8, len(v) // 2)] + "…"
                shrunk = True
            elif isinstance(v, (dict, list)):
                if _shrink_strings(v):
                    shrunk = True
    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, str) and len(item) > 9:
                node[i] = item[: max(8, len(item) // 2)] + "…"
                shrunk = True
            elif isinstance(item, (dict, list)):
                if _shrink_strings(item):
                    shrunk = True
    return shrunk


def cap_and_truncate(payload: dict[str, Any], max_bytes: int = 1024) -> dict[str, Any]:
    """把 result_summary 控制在 max_bytes 之内（UTF-8 JSON 字节，硬上限）。

    分三阶段：
      Phase 1 — 递归把任意层级的 str 字段反复减半 + "…"，置 truncated=True
      Phase 2 — 仍超限时，按序列化体积从大到小删除嵌套 dict/list 字段
      Phase 3 — 仍超限时回退到最小 {kind?, truncated:True}；极端情况回退 {}

    强保证：返回的 dict 序列化为 UTF-8 JSON 后字节数 ≤ max_bytes。
    """
    if _utf8_size(payload) <= max_bytes:
        return payload

    out = copy.deepcopy(payload)
    out["truncated"] = True

    # Phase 1: 递归裁字符串
    while _utf8_size(out) > max_bytes:
        if not _shrink_strings(out):
            break
    if _utf8_size(out) <= max_bytes:
        return out

    # Phase 2: 按体积从大到小删嵌套 container
    container_keys = [
        k for k, v in out.items()
        if isinstance(v, (dict, list)) and k != "truncated"
    ]
    container_keys.sort(key=lambda k: _utf8_size({k: out[k]}), reverse=True)
    for k in container_keys:
        del out[k]
        if _utf8_size(out) <= max_bytes:
            return out

    # Phase 3: 最小回退 {kind?, truncated:True}
    kind = payload.get("kind")
    if isinstance(kind, str):
        kind_short = kind if len(kind) <= 32 else kind[:32] + "…"
        candidate = {"kind": kind_short, "truncated": True}
        if _utf8_size(candidate) <= max_bytes:
            return candidate

    minimal = {"truncated": True}
    if _utf8_size(minimal) <= max_bytes:
        return minimal

    # 极端：max_bytes 极小到连 {"truncated": true} 都装不下
    return {}
