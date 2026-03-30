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
                    "description": "搜索关键词，简洁精准，1-6 个词效果最佳",
                }
            },
            "required": ["query"],
        },
    },
}
