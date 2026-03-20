"""
事件相关常量定义
"""


class FinishReasons:
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"


# 保留旧事件枚举，避免未纳入本次改造范围的 function_call 逻辑被打断。
class EventTypes:
    """事件类型常量类"""
    FUNCTION_STREAM_START = "function_stream_start"
    REASONING_START = "reasoning_start"
    REASONING_CONTENT = "reasoning_content"
    REASONING_COMPLETE = "reasoning_complete"
    CONTENT = "content"
    DONE = "done"
    ERROR = "error"
    FUNCTION_CALL_DETECTED = "function_call_detected"
    FUNCTION_RESULT = "function_result"
    GENERATING_QUERY = "generating_query"
    QUERY_GENERATED = "query_generated"
    USER_SEARCH_START = "user_search_start"
    PERFORMING_SEARCH = "performing_search"
    SYNTHESIZING_ANSWER = "synthesizing_answer"
