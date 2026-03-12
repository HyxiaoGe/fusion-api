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
    FUNCTION_CALL = "function_call"              # 函数调用
    FUNCTION_RESULT = "function_result"          # 函数调用结果


# 函数名称常量
class FunctionNames:
    """函数名称常量类"""
    WEB_SEARCH = "web_search"


# 消息文本常量
class MessageTexts:
    """消息文本常量类"""
    OPTIMIZING_SEARCH_QUERY = "正在优化搜索查询..."
    SEARCH_QUERY_PREFIX = "搜索查询: "
    USER_PREVIOUS_QUESTION = "用户的先前问题"
    PROCESSING_ERROR_PREFIX = "处理出错: "
    FUNCTION_CALL_ERROR_PREFIX = "函数调用流处理出错: "


# 用户友好的函数调用描述常量
USER_FRIENDLY_FUNCTION_DESCRIPTIONS = {
    FunctionNames.WEB_SEARCH: "我需要搜索网络获取最新信息...",
}

# 函数描述常量（内部使用）
FUNCTION_DESCRIPTIONS = {
    FunctionNames.WEB_SEARCH: "我需要搜索网络获取更多信息...",
}
