"""流式架构子包。

模块组成：
- runner: StreamHandler.generate_to_redis 编排
- llm_stream: LLM SSE 消费 + 调用重试
- tool_executor: 并行工具执行 + emitter 适配
- persistence: 消息落库 + URL 路径 A 预处理
- sse_encoder: Redis Stream → SSE 协议层

对外公开符号：StreamHandler / stream_redis_as_sse。
"""

from app.services.stream.runner import StreamHandler
from app.services.stream.sse_encoder import stream_redis_as_sse

__all__ = ["StreamHandler", "stream_redis_as_sse"]
