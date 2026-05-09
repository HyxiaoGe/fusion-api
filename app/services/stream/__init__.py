"""流式架构子包。

模块组成：
- runner: StreamHandler.generate_to_redis 编排
- llm_stream: LLM SSE 消费 + 调用重试
- tool_executor: 并行工具执行 + emitter 适配
- persistence: 消息落库 + URL 路径 A 预处理
- sse_encoder: Redis Stream → SSE 协议层

过渡期：本 __init__ 暂时从 app.services.stream_handler 重导出，
所有抽取完成后切换到从子模块直接导出（Task 9）。
"""

from app.services.stream_handler import StreamHandler, stream_redis_as_sse

__all__ = ["StreamHandler", "stream_redis_as_sse"]
