"""
聊天相关常量定义
"""


# 消息角色常量
class MessageRoles:
    """消息角色常量类"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# 消息类型常量
class MessageTypes:
    """消息类型常量类"""
    USER_QUERY = "user_query"                    # 用户提问
    ASSISTANT_CONTENT = "assistant_content"      # AI正常回复
    REASONING_CONTENT = "reasoning_content"      # AI推理过程
