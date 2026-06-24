"""
LLM Tool 定义 — web_search / url_read

WEB_SEARCH_TOOL 是函数（不是 const），每次调用时重算当前年份，避免硬编码
过期。message_builder 同时注入"当前日期"system prompt 双重约束。
"""

from datetime import datetime, timedelta, timezone

_CHINA_TZ = timezone(timedelta(hours=8))


def build_web_search_tool() -> dict:
    """运行时构造 web_search tool definition，当前年份动态注入到 description 里。"""
    now = datetime.now(_CHINA_TZ)
    year = now.year
    month = now.month
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取实时信息。以下情况应该使用此工具：\n"
                "- 用户询问最新新闻、时事、实时数据（天气、股价、赛事比分）\n"
                f"- 用户的问题包含时间敏感词（'最新'、'今天'、'目前'、'{year}年'）\n"
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
                            f"- **必须用 {year} 或更晚的年份，禁止使用 {year - 1} 或更早的年份**\n"
                            f"- 包含时间限定词（如 '{year}年'、'最新'、'今日'）以获取最新结果\n"
                            f"- 示例：'{year}年{month}月AI 视频生成最新进展' 而不是 'AI video generation'"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "期望返回的搜索结果数量，后端会限制在 3 到 10 条之间，默认 5 条。",
                    },
                    "intent": {
                        "type": "string",
                        "description": "搜索意图，例如 lookup、news、comparison、official、research、verification。",
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选域名过滤列表，最多 5 个合法域名，例如 openai.com。",
                    },
                    "recency_days": {
                        "type": "integer",
                        "description": "可选时间范围天数，后端会限制在 1 到 365 天之间。",
                    },
                },
                "required": ["query"],
            },
        },
    }


# 向后兼容：保留旧的常量名（同名 alias），第一次 import 时实例化。
# 严格意义上不是动态的（启动后年份固定），但启动重启频繁，约等于 dynamic。
# 推荐新代码用 build_web_search_tool()。
WEB_SEARCH_TOOL = build_web_search_tool()

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
                },
                "reason": {
                    "type": "string",
                    "description": "读取该 URL 的原因，用于联网诊断展示，后端最多保留 160 个字符。",
                },
            },
            "required": ["url"],
        },
    },
}
