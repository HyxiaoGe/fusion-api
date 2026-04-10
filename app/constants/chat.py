"""
聊天相关常量定义
"""


# 消息角色常量
class MessageRoles:
    """消息角色常量类"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# FinishReasons 从 events.py 移入，stream_handler 使用
class FinishReasons:
    """流式响应结束原因"""

    STOP = "stop"
    ERROR = "error"
