"""
LLM Tool 定义 — web_search
"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "搜索互联网获取实时信息。以下情况应该使用此工具：\n"
            "- 用户询问最新新闻、时事、实时数据（天气、股价、赛事比分）\n"
            "- 用户的问题包含时间敏感词（'最新'、'今天'、'目前'、'2025年'、'2026年'）\n"
            "- 用户询问你不确定或可能已过时的事实\n"
            "- 用户询问特定产品、价格、上市日期等可能变化的信息\n\n"
            "以下情况不应使用此工具：\n"
            "- 通用知识问答（数学公式、科学原理、历史常识）\n"
            "- 代码编写、翻译、创意写作\n"
            "- 用户明确要求基于你自身知识回答\n"
            "- 纯闲聊或情感交流"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索关键词。规则：\n"
                        "- 使用与用户消息相同的语言（用户用中文就搜中文）\n"
                        "- 简洁精准，3-8 个词\n"
                        "- 包含时间限定词（如 '2026年'、'最新'、'今日'）以获取最新结果\n"
                        "- 示例：'2026年3月缅甸地震最新伤亡' 而不是 'Myanmar earthquake'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

URL_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "url_read",
        "description": (
            "读取指定 URL 的网页内容。以下情况应该使用此工具：\n"
            "- 用户要求你查看、分析或总结某个网页链接\n"
            "- 对话中提到了一个具体的 URL 需要获取内容\n"
            "- 搜索结果中的某个链接需要深入阅读\n\n"
            "以下情况不应使用此工具：\n"
            "- 用户的当前消息中已经包含 URL（系统会自动预读取）\n"
            "- 不确定具体的 URL 地址\n"
            "- 需要搜索信息而非读取特定网页（应使用 web_search）"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要读取的完整 URL 地址（包含 http:// 或 https://）",
                }
            },
            "required": ["url"],
        },
    },
}
