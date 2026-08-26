"""只计算提示词内容标识，不保存或记录提示词正文。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def fingerprint_system_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """保留有效 system 消息的顺序、边界和内容结构，生成确定性 SHA-256。"""
    system_messages = [dict(message) for message in messages if message.get("role") == "system"]
    serialized = json.dumps(system_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
