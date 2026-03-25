"""
事件相关常量定义
"""


class FinishReasons:
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


class EventTypes:
    """事件类型常量类"""
    REASONING_START = "reasoning_start"
    REASONING_CONTENT = "reasoning_content"
    REASONING_COMPLETE = "reasoning_complete"
    CONTENT = "content"
    DONE = "done"
    ERROR = "error"
