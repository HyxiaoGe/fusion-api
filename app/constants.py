class MessageRoles:
    """消息角色常量"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageTypes:
    """消息类型常量"""
    USER_QUERY = "user_query"                    # 用户提问
    ASSISTANT_CONTENT = "assistant_content"      # AI正常回复
    REASONING_CONTENT = "reasoning_content"      # AI推理过程
    FUNCTION_CALL = "function_call"              # 函数调用
    FUNCTION_RESULT = "function_result"          # 函数调用结果
    WEB_SEARCH = "web_search"                    # 网络搜索结果
    HOT_TOPICS = "hot_topics"                    # 热点话题信息 