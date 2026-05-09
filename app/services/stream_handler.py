"""[已弃用] 兼容性 shim — 真正的实现在 app.services.stream 子包中。

外部代码请改用：
    from app.services.stream import StreamHandler, stream_redis_as_sse

本文件将在 Task 8 删除。
"""

# ruff: noqa: F401
from app.services.stream.runner import StreamHandler
from app.services.stream.sse_encoder import (
    entry_to_sse_envelope as _entry_to_sse_envelope,
)
from app.services.stream.sse_encoder import (
    stream_redis_as_sse,
)
