"""Redis Stream → SSE 协议层。

从 stream_handler.py 抽出（spec §4.5）。每条 SSE 事件形如
`{chunk_type, data}` envelope，包含 `id:` 行供断线重连使用。
"""

import json
from typing import AsyncGenerator

from app.services.stream_state_service import read_stream_chunks


def entry_to_sse_envelope(entry_fields: dict) -> dict:
    """把 Redis Stream entry 的 hash 字段转成 {chunk_type, data} envelope。

    spec §4.6 SSE 顶层契约：每条 SSE message 形如 {"chunk_type": <type>, "data": {...}}。
    本函数不负责 SSE 包装（id: 行、data: 前缀、[DONE]）— 这由 stream_redis_as_sse 处理。
    """
    chunk_type = entry_fields.get("type", "")
    content = entry_fields.get("content", "")
    block_id = entry_fields.get("block_id", "")

    if chunk_type == "agent_event":
        # agent_event 的 content 由 emitter 序列化为 JSON dict
        data = json.loads(content) if content else {}
    elif chunk_type in ("reasoning", "answering"):
        data = {"block_id": block_id, "delta": content}
        # 透传可选关联字段（emitter 通过 append_chunk 的 **extras 写入）
        for k in ("run_id", "step_id"):
            if k in entry_fields:
                data[k] = entry_fields[k]
    elif chunk_type == "thinking_pending":
        # 思考中占位事件：FE 用来显示脉冲动画
        data = {"block_id": block_id}
    elif chunk_type == "error":
        # error chunk: BYOK 结构化 error_code (JSON object) 升入 data；
        # 普通字符串 error_msg 兜底为 {message, code='stream_error'}，避免 FE
        # 收到 {data: {}} 丢失 "用户中止" / "被新请求取代" 等错误文本。
        if not content:
            data = {}
        else:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    data = parsed  # BYOK 结构化路径（自带 code 字段，不被覆盖）
                else:
                    data = {"code": "stream_error", "message": str(parsed)}
            except (ValueError, TypeError):
                data = {"code": "stream_error", "message": content}
    else:
        # done / preparing / 其它已知 type 用空 data
        data = {}

    return {"chunk_type": chunk_type, "data": data}


async def stream_redis_as_sse(
    conversation_id: str,
    message_id: str,
    last_entry_id: str = "0",
) -> AsyncGenerator[str, None]:
    """SSE 读取器：从 Redis Stream 读 chunk，按 spec §4.6 顶层 envelope 输出。

    每条 SSE 事件包含 id: 行（Redis entry ID），供断线重连使用。
    Redis 不可用时立即返回 error 帧 + [DONE]。
    """
    from app.core.redis import get_redis_pool

    if not get_redis_pool():
        # 维持新外层 envelope 形态
        error_envelope = {
            "chunk_type": "error",
            "data": {
                "code": "redis_unavailable",
                "message": "Redis 不可用，无法读取流",
            },
        }
        yield f"data: {json.dumps(error_envelope, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    async for chunk in read_stream_chunks(conversation_id, last_entry_id):
        entry_id = chunk.pop("entry_id")
        chunk_type = chunk.get("type", "")

        # 跳过内部 start 标记
        if chunk_type == "start":
            continue

        envelope = entry_to_sse_envelope(chunk)
        yield f"id: {entry_id}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
