"""在首个 LLM Round 前解析并冻结 Run 级能力包。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from app.services.agent.plan_coordinator import PlanMode
from app.services.stream.agent_plan_tool_policy import (
    INTERCITY_LOCATION_NAMES,
    resolve_product_capability_signals,
)
from app.services.stream.agent_task_policy import AgentTaskPolicy
from app.utils.run_capability_contract import (
    CAPABILITY_AUTO_PLAN_PACKAGES,
    CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER,
    CAPABILITY_CONTROL_TOOL_NAMES,
    CAPABILITY_PACKAGE_EXTERNAL_TOOL_NAMES,
    CAPABILITY_REASON_CODES,
    is_authorized_mcp_tool_alias,
    validate_capability_resolution_semantics,
)

Confidence = Literal["high", "medium", "low"]
ResolutionMode = Literal["routed", "degraded", "clarification"]

SCHEMA_VERSION = 1
ROUTER_VERSION = "2026-08-27.2"

_CANONICAL_EXTERNAL_TOOL_ORDER = CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER
_CONTROL_TOOL_NAMES = CAPABILITY_CONTROL_TOOL_NAMES

_TRANSFORM_RE = re.compile(
    r"翻译|译成|改写|重写|润色|措辞|"
    r"(?:概括|摘要|总结)(?:这|以下|上述|给定|已给|后面|内容|文本|[:：])|"
    r"(?:对|将|把)(?:这|以下|上述|给定|已给|后面).{0,24}(?:概括|摘要|总结)|"
    r"\b(?:translate|rewrite|rephrase|proofread|polish)\b",
    re.IGNORECASE,
)
_CURRENT_DATE_ONLY_RE = re.compile(
    r"^(?:请问|请告诉我|帮我看下|帮我看看)?(?:今天|现在)"
    r"(?:是)?(?:几月几日|几号|星期几|周几|日期|什么日子)"
    r"(?:[、，,和及](?:星期几|周几|几月几日|几号|日期))?[？?。！!]*$|"
    r"^(?:(?:what(?:'s| is) )?(?:today(?:'s)? date|the date today)|"
    r"what day is it today)\??$",
    re.IGNORECASE,
)
_RELATIVE_DATE_RE = re.compile(
    r"今天|今日|明天|后天|昨天|本周|下周|这个月|本月|下个月|当前|现在|"
    r"\b(?:today|tomorrow|yesterday|this week|next week|this month|next month|currently|now)\b",
    re.IGNORECASE,
)
_FRESH_EXTERNAL_RE = re.compile(
    r"最新|新闻|开市|收盘|股价|汇率|比分|发布了什么|刚刚发布|公开发布|现任|目前的|"
    r"\b(?:latest|breaking news|most recent|"
    r"current (?:price|score|exchange rate|ceo|president|release|version))\b",
    re.IGNORECASE,
)
_VERIFIED_SOURCE_RE = re.compile(
    r"官方(?:公告|原文|资料|来源)|一手来源|可靠来源|权威来源|"
    r"(?:查证|核验|验证|交叉验证)|只依据(?:该|这个)页面|"
    r"\b(?:official.{0,32}(?:announcement|source|documentation|release notes)|"
    r"primary source|verify|fact-check|cross-check)\b",
    re.IGNORECASE,
)
_URL_LOCAL_SOURCE_ONLY_RE = re.compile(
    r"只依据(?:该|这个)页面|\b(?:using only|based on) (?:that|this|the) page\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"https?://[^\s\"'”’」`<>。！)）\]】}—–、]+?"
    r"(?=(?:[>）)\]】}]|[—–、]|"
    r"\.{2,}(?=\s*(?:and\b|then\b|search\b(?!\s*=)|read\b(?!\s*=)|open\b(?!\s*=)))|"
    r"[，,；;：:](?=\s*(?:并|然后|and\b|then\b|搜索(?!\s*=)|检索(?!\s*=)|"
    r"search\b(?!\s*=)|read\b(?!\s*=)|open\b(?!\s*=)))|\s|$))",
    re.IGNORECASE,
)
_URL_READ_ACTION_RE = re.compile(
    r"总结|摘要|读取|阅读|分析|概括|只依据|基于|"
    r"打开|看看|看下|翻译|"
    r"\b(?:summarize|read|analyze|review|open|translate|take a look|using only|based on)\b|"
    r"\b(?:call|use|run|invoke|execute)\s+(?:the\s+)?url_read\b|"
    r"(?:调用|使用|运行|执行)\s*url_read",
    re.IGNORECASE,
)
_POSITIVE_WEB_SEARCH_ACTION_RE = re.compile(
    r"(?:联网|上网|网上).{0,8}(?:查|搜索|检索)|"
    r"(?:并|然后)(?:(?:联网|上网|网上)\s*)?(?:搜索|检索|查找).{1,}|"
    r"(?:请|帮我|要)(?:搜索|检索|查找)\s*.{1,}|"
    r"(?:^|[；;，,。]\s*)(?:搜索|检索|查找)(?!算法|功能|结果|框|引擎|组件|模块).{1,}|"
    r"(?:^|[；;，,。]\s*)查询"
    r"(?=[^，,。；;：:！？!?]{0,64}(?:最新|新闻|公告|动态|资讯|官网|官方|现任|目前的|"
    r"一手来源|可靠来源|权威来源|股价|汇率|比分))|"
    r"(?:^|[;:,.!?：]\s*|\b(?:and|then|please|can you|could you|i want you to)\s+)"
    r"(?:search(?: the web| online)?|look up|find online|"
    r"browse (?:(?:the )?(?:public )?web|the internet|online)|"
    r"(?:call|use|run|invoke|execute)\s+(?:the\s+)?web_search|"
    r"(?:do|perform|run|conduct) (?:a )?(?:(?:quick|brief|targeted) )?(?:web|online) search)\b",
    re.IGNORECASE,
)
_NEGATED_ALL_NETWORK_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再)?\s*"
    r"(?:在|于)\s*(?:本次|此次|当前|目前|这个|该|本轮|这次|本条|这条|本|整个)"
    r"(?:请求|任务|问题|对话|轮次|消息|回答|回复|答复|响应|查询)"
    r"(?:中|里|内|范围内|期间)?\s*"
    r"(?:联网|上网|互联网|使用网络|用网络|接入网络|访问(?:互联网|网络)|连接(?:互联网|网络))|"
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行|使用)?\s*"
    r"(?:联网|上网|互联网|使用网络|用网络|接入网络|访问(?:互联网|网络)|连接(?:互联网|网络))"
    r"(?!\s*(?:搜索|检索|查))|"
    r"(?:在|处于)?不联网(?:的情况下|时)?|"
    r"\b(?:do not|don['’]t|dont|never|without)\s+(?:use|using|access|accessing)\s+"
    r"(?:the\s+)?(?:internet|network)\b|"
    r"\b(?:avoid|refrain\s+from)\s+(?:using|accessing)\s+(?:the\s+)?(?:internet|network)\b|"
    r"\b(?:do not|don['’]t|dont|never)\s+go\s+online\b|"
    r"\bavoid\s+going\s+online\b|"
    r"\b(?:please\s+)?(?:work|answer|respond)\s+(?:(?:entirely|fully|completely)\s+)?offline\b|"
    r"\b(?:please\s+)?(?:stay|remain)\s+(?:(?:entirely|fully|completely)\s+)?offline\b|"
    r"\b(?:please\s+)?keep\s+(?:this|it|the answer|the response)?\s*offline\b|"
    r"\b(?:(?:i['’]d|i would|i) prefer|my preference is) (?:an )?offline (?:answer|response)\b|"
    r"\bwithout (?:going|getting) online\b|"
    r"\bwithout (?:connecting|connect|accessing|access) to (?:the )?(?:internet|network)\b|"
    r"\bwithout (?:the )?(?:web|internet|network)(?:\s+access)?\b|"
    r"\b(?:use|using|answer from|answer with|based on)\s+"
    r"(?:local knowledge|offline knowledge)\s+only\b|"
    r"\b(?:use|using)\s+only\s+(?:local knowledge|offline knowledge)\b|"
    r"\b(?:rely|relying)\s+only\s+on\s+(?:local knowledge|offline knowledge)\b|"
    r"(?:请)?(?:保持|维持)?(?:离线|在离线(?:模式|情况下)?)(?:并|地)?(?:回答|处理|工作)|"
    r"\bno\s+(?:internet|network)(?:\s+access)?\b|"
    r"^(?:in\s+)?offline(?:\s+mode)?\b|^离线(?:模式|情况下|状态下)?",
    re.IGNORECASE,
)
_NEGATED_WEB_SEARCH_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行|使用)?\s*"
    r"(?:联网|上网|网上)\s*(?:搜索|检索|查询|查找|查)|"
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行|使用)?\s*"
    r"(?:搜索|检索|查找|查(?!询|找))|"
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行|使用)?\s*"
    r"查询|"
    r"\b(?:do not|don['’]t|dont|never|without)\s+"
    r"(?:search(?:ing)?(?: the)? web|look(?:ing)? up|find(?:ing)? online|"
    r"brows(?:e|ing)(?: the)? web)\b|"
    r"\b(?:do not|don['’]t|dont|never)\s+search\b|"
    r"\b(?:do not|don['’]t|dont|never)\s+look\s+(?:it|this|that|them)\s+up\b|"
    r"\b(?:avoid|refrain\s+from|skip)\s+(?:searching|search|browsing|brows(?:e|ing))\s+"
    r"(?:the\s+)?(?:web|internet|online)\b|"
    r"\bno\s+(?:web|online|internet)\s+(?:search|lookup)\b",
    re.IGNORECASE,
)
_NEGATED_URL_READ_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行|使用)?\s*"
    r"(?:打开|读取|阅读|访问|浏览)(?:网页|网站|页面|链接|url)?|"
    r"\b(?:do not|don['’]t|dont|never|without)\s+"
    r"(?:open(?:ing)?|read(?:ing)?|access(?:ing)?|brows(?:e|ing))\b|"
    r"\b(?:avoid|refrain\s+from|skip)\s+"
    r"(?:opening|reading|accessing|browsing)\b",
    re.IGNORECASE,
)
_NEGATED_VERIFIED_WEB_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可)(?:再|去|进行)?\s*"
    r"(?:查证|核验|验证|交叉验证)[^，,。；;]*|"
    r"\b(?:do not|don['’]t|dont|never|without)\s+"
    r"(?:verify|verifying|fact[- ]check(?:ing)?|cross[- ]check(?:ing)?)\b[^,.;!?]*|"
    r"\b(?:avoid|refrain\s+from|skip)\s+"
    r"(?:verifying|fact[- ]checking|cross[- ]checking)\b[^,.;!?]*|"
    r"\b(?:do not|don['’]t|dont|never)\s+consult\s+(?:the\s+)?"
    r"(?:official|primary|authoritative)\s+sources?\b[^,.;!?]*|"
    r"\b(?:do not|don['’]t|dont|never)\s+(?:use|check)\s+(?:the\s+)?"
    r"(?:official|primary|authoritative)\s+sources?\b[^,.;!?]*|"
    r"\b(?:avoid|refrain\s+from|skip)\s+(?:checking|using)\s+(?:the\s+)?"
    r"(?:official|primary|authoritative)\s+sources?\b[^,.;!?]*|"
    r"\b(?:use|check|consult)\s+no\s+(?:official|primary|authoritative)\s+sources?\b[^,.;!?]*|"
    r"\b(?:exclude|excluding|omit|omitting|skip|skipping)\s+(?:the\s+)?"
    r"(?:official|primary|authoritative)\s+sources?\b[^,.;!?]*",
    re.IGNORECASE,
)
_IN_DOCUMENT_SEARCH_RE = re.compile(
    r"\bsearch\s+(?:within|inside)\s+(?:the\s+)?(?:page|document)\b|"
    r"\bsearch\s+(?:within|inside)\s+(?:this|that|it)\b|"
    r"\bsearch\s+(?:this|that|the)\s+(?:page|document)\b|"
    r"\bsearch\s+(?:it|its\s+contents?)\s+for\b|"
    r"\bsearch\s+(?:the\s+)?(?:page|document)\s+for\b|"
    r"(?:搜索|检索|查找)(?:该|这个|此)?(?:页面|文档)(?:中|内|里|中的|里的)",
    re.IGNORECASE,
)
_NEGATED_WEB_TOOL_NAME_RE = re.compile(
    r"(?:不要|不用|别|请勿|禁止|严禁|不得|不可).{0,16}?\bweb_search\b|"
    r"\b(?:do not|don['’]t|dont|never)\s+(?:call|use|invoke|run|execute)\s+"
    r"(?:the\s+)?(?:tool\s+)?web_search\b(?:\s+tool\b)?|"
    r"\b(?:without|avoid(?:ing)?|refrain\s+from|skip(?:ping)?)\s+"
    r"(?:call(?:ing)?|us(?:e|ing)|invok(?:e|ing)|run(?:ning)?|execut(?:e|ing))\s+"
    r"(?:the\s+)?(?:tool\s+)?web_search\b(?:\s+tool\b)?",
    re.IGNORECASE,
)
_NEGATED_URL_TOOL_NAME_RE = re.compile(
    r"(?:不要|不用|别|请勿|禁止|严禁|不得|不可).{0,16}?\burl_read\b|"
    r"\b(?:do not|don['’]t|dont|never)\s+(?:call|use|invoke|run|execute)\s+"
    r"(?:the\s+)?(?:tool\s+)?url_read\b(?:\s+tool\b)?|"
    r"\b(?:without|avoid(?:ing)?|refrain\s+from|skip(?:ping)?)\s+"
    r"(?:call(?:ing)?|us(?:e|ing)|invok(?:e|ing)|run(?:ning)?|execut(?:e|ing))\s+"
    r"(?:the\s+)?(?:tool\s+)?url_read\b(?:\s+tool\b)?",
    re.IGNORECASE,
)
_POSITIVE_WEB_TOOL_NAME_RE = re.compile(
    r"(?:调用|使用|运行|执行)\s*web_search\b|"
    r"\b(?:call|use|invoke|run|execute)\s+(?:the\s+)?web_search\b",
    re.IGNORECASE,
)
_POSITIVE_URL_TOOL_NAME_RE = re.compile(
    r"(?:调用|使用|运行|执行)\s*url_read\b|"
    r"\b(?:call|use|invoke|run|execute)\s+(?:the\s+)?url_read\b",
    re.IGNORECASE,
)
_QUOTED_LITERAL_RE = re.compile(
    r"(?:“[^”]{1,240}”|‘[^’]{1,240}’|\"[^\"]{1,240}\"|(?<!\w)'[^']{1,240}'(?!\w)|"
    r"「[^」]{1,240}」|`[^`]{1,240}`)"
)
_QUOTED_RESOURCE_RE = re.compile(
    r"https?://[^\s\"'”’」`]+|\bmcp_[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)
_QUOTED_LITERAL_TRANSFORM_RE = re.compile(
    r"(?:翻译|译成|改写|重写|润色|\b(?:translate|rewrite|rephrase|proofread|polish))"
    r"(?:\s+(?:the\s+)?(?:phrase|text|words?))?\s*:?[：]?\s*"
    r"(?:“[^”]{1,240}”|‘[^’]{1,240}’|\"[^\"]{1,240}\"|'[^']{1,240}')|"
    r"(?:“[^”]{1,240}”|‘[^’]{1,240}’|\"[^\"]{1,240}\"|'[^']{1,240}')\s*"
    r"(?:翻译|译成|改写|重写|润色|\b(?:translate|rewrite|rephrase|proofread|polish))",
    re.IGNORECASE,
)
_GIVEN_TEXT_TRANSFORM_RE = re.compile(
    r"(?:把|将)(?:这|以下|上述|给定|已给|后面).{1,240}"
    r"(?:翻译|译成|改写|重写|润色|概括|摘要|总结)|"
    r"\b(?:translate|rewrite|rephrase|proofread|polish)\s+"
    r"(?:this|that|the following|(?:the\s+)?(?:words?|phrase))\b|"
    r"(?:把|将).{1,120}这(?:一|两|三|四|五|几)个字.{0,24}(?:翻译|译成|改写|重写)",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^(?:(?:你?好|嗨)(?:[，,\s]*很高兴见到你)?|hi|hello|早上好|下午好|晚上好|很高兴见到你)"
    r"[呀啊！!。\s]*$",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(r"你是谁|你叫什么|介绍一下你自己|你能做什么")
_STABLE_KNOWLEDGE_RE = re.compile(
    r"^(?:为什么|为何|什么是|解释(?:一下)?|介绍一下|讲讲|"
    r"why\b|what (?:is|are|does)\b|how (?:does|do|is|are|can)\b|explain\b)|"
    r"^(?!从.{1,64}(?:到|至)).{1,80}(?:是什么|原理是什么|怎么工作)[？?]?$",
    re.IGNORECASE,
)
_NOUN_DEFINITION_RE = re.compile(
    r"^what (?:is|are)\s+(?P<noun>(?:an?\s+)?(?:weather\s+forecasts?|current\s+price|"
    r"(?:primary|official|authoritative)\s+sources?)|breaking\s+news|"
    r"official\s+(?:documentation|announcements?))(?P<tail>.*?)[?.!]*$",
    re.IGNORECASE,
)
_SAFE_DEFINITION_TAIL_RE = re.compile(
    r"^(?:|\s+in\s+[a-z][a-z0-9 &'/-]{0,63}|"
    r",\s*in\s+(?:simple|plain|everyday)\s+terms|"
    r"\s+for\s+(?:beginners?|a\s+beginner)|"
    r",\s*(?:simply|briefly))$",
    re.IGNORECASE,
)
_SAFE_BASIC_DEFINITION_TAIL_RE = re.compile(
    r"^(?:|,?\s*in\s+(?:simple|plain|everyday)\s+terms|"
    r"\s+for\s+(?:beginners?|a\s+beginner)|"
    r",\s*(?:simply|briefly))$",
    re.IGNORECASE,
)
_SAFE_CURRENT_PRICE_DEFINITION_TAIL_RE = re.compile(
    r"^(?:|\s+in\s+(?:finance|economics|accounting|financial\s+markets?|market\s+analysis)|"
    r",?\s*in\s+(?:simple|plain|everyday)\s+terms|"
    r"\s+for\s+(?:beginners?|a\s+beginner)|"
    r",\s*(?:simply|briefly))$",
    re.IGNORECASE,
)
_SAFE_WEATHER_DEFINITION_TAIL_RE = re.compile(
    r"^(?:|\s+in\s+(?:meteorology|forecasting|weather\s+science|climate\s+science)|"
    r",?\s*in\s+(?:simple|plain|everyday)\s+terms|"
    r"\s+for\s+(?:beginners?|a\s+beginner)|"
    r",\s*(?:simply|briefly))$",
    re.IGNORECASE,
)
_EXTERNAL_QUERY_DEFINITION_TAIL_RE = re.compile(
    r"\b(?:about|regarding|latest|newest|today|tomorrow|yesterday|now|"
    r"announcement|release|price\s+of)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_KNOWLEDGE_RE = re.compile(
    r"^(?:what (?:is|are)\s+the\s+difference\s+between\s+.+|"
    r"what does\s+.{1,96}\b(?:mean|usually\s+contain)\b|"
    r"how do\s+.{1,96}\bwork\b|"
    r"explain what\s+.{1,96}\bmeans?\b|"
    r"(?:什么是|为什么|为何).+|"
    r"解释(?:一下)?.*(?:区别|含义|意思|概念|原理))",
    re.IGNORECASE,
)
_SIMPLE_CALC_RE = re.compile(
    r"^(?:请)?(?:计算|算一下|算算)?\s*[\d\s()+\-*/.%]+(?:等于多少|是多少)?[？?]?$", re.IGNORECASE
)
_EN_PRODUCT_SEQUENCE_ACTION = (
    r"(?:do not|don['’]t|dont|never|not|avoid|without|find|search|show|book|compare|"
    r"check|look for|get|recommend|give|provide|plan|route|directions?)"
)
_EN_PRODUCT_SEQUENCE_BOUNDARY = (
    r"(?:\s[—–]\s|[—–]{2}|\s+(?:and\s+then|then|afterwards|later|finally)\b"
    rf"(?=\s+{_EN_PRODUCT_SEQUENCE_ACTION}))"
)
_EN_PRODUCT_CLAUSE_TEXT_ATOM = rf"(?:(?!{_EN_PRODUCT_SEQUENCE_BOUNDARY})[^,;:.!?])"
_ZH_PRODUCT_SEQUENCE_ACTION = (
    r"(?:不要|不用|无需|不需要|不必|没必要|没有必要|用不着|别|请勿|禁止|严禁|不得|不可|"
    r"查|查询|搜索|找|预订|订|购买|买|比较|对比|推荐|给出|提供|查看|获取|规划)"
)
_ZH_PRODUCT_SEQUENCE_BOUNDARY = (
    r"(?:\s[—–]\s|[—–]{2}|"
    rf"(?:并且|并|然后|随后|最后|但(?:是)?|不过|而(?:是|要)?)(?={_ZH_PRODUCT_SEQUENCE_ACTION}))"
)
_ZH_PRODUCT_CLAUSE_TEXT_ATOM = rf"(?:(?!{_ZH_PRODUCT_SEQUENCE_BOUNDARY})[^，,。；;：:！？!?])"
_PRODUCT_DIRECTIVE_BOUNDARY_RE = re.compile(
    r"(?:[，,。；;：:！？!?]|\s+[—–]\s+|[—–]{2}|"
    r"\b(?:and\s+then|then|afterwards|later|finally)\b|"
    r"(?:并且|并|然后|随后|最后|但(?:是)?|不过|而(?:是|要)?))",
    re.IGNORECASE,
)
_EN_WEATHER_TASK_RE = re.compile(
    r"\b(?:weather|temperature|rain|snow|wind) forecast\b|"
    r"\b(?:check|show|find|get|look up)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,24}}\bweather\b|"
    r"\b(?:weather|temperature|rain(?:ing)?|snow(?:ing)?|wind)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,48}}"
    r"\b(?:in|for|at|today|tomorrow|this week|next week)\b|"
    r"\b(?:will it|is it going to) (?:rain|snow)\b|"
    r"\bis it (?:raining|snowing)\b",
    re.IGNORECASE,
)
_EN_FLIGHT_TASK_RE = re.compile(
    r"\b(?:find|search|show|book|compare|check|look for)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,48}}"
    r"\b(?:flights?|airfare|plane tickets?)\b|"
    r"\b(?:flights?|airfare|plane tickets?)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,64}}\bfrom\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{1,48}}\bto\b",
    re.IGNORECASE,
)
_EN_TRAIN_TASK_RE = re.compile(
    r"\b(?:find|search|show|book|compare|check|look for)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,48}}"
    r"\b(?:trains?|rail tickets?)\b|"
    r"\b(?:trains?|rail tickets?)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,64}}\bfrom\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{1,48}}\bto\b",
    re.IGNORECASE,
)
_ZH_FLIGHT_TASK_RE = re.compile(
    r"(?:查|查询|搜索|找|预订|订|购买|买|比较|对比|推荐)"
    rf"{_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,24}}(?:航班|机票|飞机)|"
    rf"(?:航班|机票|飞机){_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,24}}"
    r"(?:查|查询|搜索|找|预订|订|购买|买|时间|价格|多少钱)|"
    r"(?:坐|乘).{0,4}飞机",
)
_ZH_TRAIN_TASK_RE = re.compile(
    r"(?:查|查询|搜索|找|预订|订|购买|买|比较|对比|推荐)"
    rf"{_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,24}}(?:高铁|动车|火车|列车|车次)|"
    rf"(?:高铁|动车|火车|列车|车次){_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,24}}"
    r"(?:查|查询|搜索|找|预订|订|购买|买|时间|价格|多少钱)|"
    r"(?:坐|乘).{0,4}(?:高铁|动车|火车|列车)",
)
_ZH_AIR_RAIL_COMPARISON_RE = re.compile(
    r"(?:飞机|航班|机票).{0,20}(?:高铁|动车|火车|列车).{0,12}(?:还是|比较|对比|哪个好|更好)|"
    r"(?:高铁|动车|火车|列车).{0,20}(?:飞机|航班|机票).{0,12}(?:还是|比较|对比|哪个好|更好)|"
    r"(?:飞机|航班|机票).{0,8}(?:还是|比较|对比).{0,8}(?:高铁|动车|火车|列车)|"
    r"(?:高铁|动车|火车|列车).{0,8}(?:还是|比较|对比).{0,8}(?:飞机|航班|机票)",
)
_EN_PLACE_TASK_RE = re.compile(
    r"\b(?:find|search|show|recommend|look for)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,64}}"
    r"\b(?:nearby|near|coffee shops?|cafes?|restaurants?|hotels?|attractions?|"
    r"places? to (?:eat|visit|stay)|things to do)\b",
    re.IGNORECASE,
)
_EN_ROUTE_RELATION_RE = re.compile(
    r"\b(?:how (?:do|can|should) i get|i (?:need|want|plan) to (?:travel|go)|"
    r"directions?|route|public transit|driving|walking|cycling)\b"
    rf"{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,48}}\bfrom\s+(?P<origin>[a-z][a-z0-9 .'-]{{0,64}}?)\s+"
    rf"to\s+(?P<destination>[a-z]{_EN_PRODUCT_CLAUSE_TEXT_ATOM}{{0,63}}?)"
    rf"(?=\s+(?:by|via)\s+|[?.!,;:]|{_EN_PRODUCT_SEQUENCE_BOUNDARY}|$)",
    re.IGNORECASE,
)
_EN_STRONG_MOBILITY_ACTION_RE = re.compile(
    r"\b(?:how (?:do|can|should) i get|i (?:need|want|plan) to (?:travel|go)|"
    r"directions?|public transit|driving|walking|cycling)\b",
    re.IGNORECASE,
)
_EN_ROUTE_MODE_SEGMENT_RE = re.compile(
    r"\b(?:by|via)\s+(?P<modes>[^?.!]{1,96})(?=[?.!]|$)",
    re.IGNORECASE,
)
_EN_NATURAL_INTERCITY_RE = re.compile(
    r"\bi(?:'m| am)(?: currently)? in (?P<origin>[a-z][a-z .'-]{1,40}?)"
    r"(?:,\s*(?:and\s+)?|\s+and\s+)i (?:want|need|plan) to (?:go|travel) to "
    r"(?P<destination>[a-z][a-z .'-]{1,40}?)(?:[.?!]|$)",
    re.IGNORECASE,
)
_EN_INTERCITY_LOCATION_NAMES = (
    "beijing",
    "shanghai",
    "tianjin",
    "chongqing",
    "guangzhou",
    "shenzhen",
    "hangzhou",
    "nanjing",
    "suzhou",
    "chengdu",
    "wuhan",
    "xi'an",
    "changsha",
    "zhengzhou",
    "qingdao",
    "xiamen",
    "fuzhou",
    "kunming",
    "shenyang",
    "dalian",
    "jinan",
    "hefei",
    "ningbo",
    "wuxi",
    "hong kong",
    "macau",
    "harbin",
    "shijiazhuang",
    "taiyuan",
    "hohhot",
    "changchun",
    "nanchang",
    "nanning",
    "haikou",
    "sanya",
    "guiyang",
    "lhasa",
    "lanzhou",
    "xining",
    "yinchuan",
    "urumqi",
)
_EN_PHYSICAL_LOCATION_NAMES = frozenset(
    {
        *_EN_INTERCITY_LOCATION_NAMES,
        "the bund",
        "people's square",
        "hongqiao station",
        "shanghai hongqiao station",
        "hongqiao airport",
        "pudong airport",
        "disneyland",
        "tiananmen",
        "the forbidden city",
        "forbidden city",
        "the summer palace",
        "summer palace",
        "grand central station",
        "city hall",
        "downtown",
        "city center",
    }
)
_EN_PHYSICAL_LOCATION_SUFFIX_RE = re.compile(
    r"\b(?:station|airport|terminal|square|park|museum|hospital|hotel|mall|"
    r"road|street|bridge|tower)\b$",
    re.IGNORECASE,
)
_EN_ABSTRACT_ROUTE_CONTEXT_RE = re.compile(
    r"\b(?:process|workflow|career|promotion|business|technical|development|growth)\b",
    re.IGNORECASE,
)
_EN_ABSTRACT_ROUTE_ENDPOINT_RE = re.compile(
    r"\b(?:draft|publication|requirement|review|production|deployment|release|"
    r"engineer|architect|manager|director|career|role|rank|level|cost|profit|loss|"
    r"awareness|conversion|familiar|unfamiliar|beginner|expert|idea|implementation|"
    r"cold start|scale|scaling)\b",
    re.IGNORECASE,
)
_PACKAGE_TOOLS = CAPABILITY_PACKAGE_EXTERNAL_TOOL_NAMES
_AUTO_PLAN_PACKAGES = CAPABILITY_AUTO_PLAN_PACKAGES
_REASON_CODES = CAPABILITY_REASON_CODES


@dataclass(frozen=True)
class RunCapabilityResolution:
    schema_version: int
    router_version: str
    package_id: str
    confidence: Confidence
    resolution_mode: ResolutionMode
    reason_codes: tuple[str, ...]
    external_tool_names: tuple[str, ...]
    effective_plan_mode: PlanMode
    include_current_date: bool
    network_boundary_required: bool


@dataclass(frozen=True)
class _CandidateRoute:
    package_id: str
    confidence: Confidence
    reason_codes: tuple[str, ...]
    include_current_date: bool
    resolution_mode: ResolutionMode = "routed"
    explicit_tool_names: tuple[str, ...] | None = None


def resolve_run_capability_route(
    *,
    original_message: str | None,
    task_context_messages: list[object] | None,
    available_tool_names: list[str],
    requested_plan_mode: PlanMode,
    task_policy: AgentTaskPolicy,
    capabilities: dict,
    tools_disabled: bool,
    knowledge_grounded: bool,
    unavailable_tool_names: list[str] | None = None,
) -> RunCapabilityResolution:
    """根据受信运行态与当前用户消息解析最小能力包。"""

    message = _normalize_message(original_message)
    function_calling = capabilities.get("functionCalling") is True
    search_capable = capabilities.get("searchCapable") is True

    if knowledge_grounded:
        blocked_candidate = (
            _CandidateRoute(
                package_id="deep_research",
                confidence="high",
                reason_codes=("deep_research_mode",),
                include_current_date=True,
            )
            if task_policy.task_mode == "deep_research"
            else _classify_standard_request(
                message=message,
                task_context_messages=task_context_messages,
                available_tool_names=available_tool_names,
            )
        )
        blocked_tool_names = blocked_candidate.explicit_tool_names or _PACKAGE_TOOLS.get(
            blocked_candidate.package_id,
            (),
        )
        return _validated_resolution(
            _resolution(
                candidate=_CandidateRoute(
                    package_id="knowledge_grounded",
                    confidence="high",
                    reason_codes=("knowledge_grounded_mode",),
                    include_current_date=blocked_candidate.include_current_date,
                ),
                available_tool_names=available_tool_names,
                requested_plan_mode="off",
                function_calling=function_calling,
                tools_disabled=True,
                network_boundary_required=bool(blocked_tool_names),
            )
        )

    if task_policy.task_mode == "deep_research":
        candidate = _CandidateRoute(
            package_id="deep_research",
            confidence="high",
            reason_codes=("deep_research_mode",),
            include_current_date=True,
        )
    else:
        candidate = _classify_standard_request(
            message=message,
            task_context_messages=task_context_messages,
            available_tool_names=available_tool_names,
        )

    requested_tools = candidate.explicit_tool_names or _PACKAGE_TOOLS.get(
        candidate.package_id,
        (),
    )
    unavailable_tools = frozenset(unavailable_tool_names or ())
    requires_search = any(name in {"web_search", "url_read"} for name in requested_tools)
    needs_external_capability = bool(requested_tools)
    degraded_reason: str | None = None
    if needs_external_capability and tools_disabled:
        degraded_reason = "tools_disabled"
    elif needs_external_capability and not function_calling:
        degraded_reason = "function_calling_unavailable"
    elif any(name in unavailable_tools for name in requested_tools):
        degraded_reason = "required_tools_unavailable"
    elif requires_search and not search_capable:
        degraded_reason = "search_capability_unavailable"

    if degraded_reason is not None:
        return _validated_resolution(
            _resolution(
                candidate=_CandidateRoute(
                    package_id="tools_unavailable",
                    confidence=candidate.confidence,
                    reason_codes=(degraded_reason,),
                    include_current_date=candidate.include_current_date,
                    resolution_mode="degraded",
                ),
                available_tool_names=available_tool_names,
                requested_plan_mode="off",
                function_calling=function_calling,
                tools_disabled=True,
                network_boundary_required=True,
            )
        )

    resolution = _resolution(
        candidate=candidate,
        available_tool_names=available_tool_names,
        requested_plan_mode=requested_plan_mode,
        function_calling=function_calling,
        tools_disabled=tools_disabled,
    )
    if candidate.package_id == "deep_research" and not frozenset(requested_tools).issubset(
        resolution.external_tool_names
    ):
        return _validated_resolution(
            _resolution(
                candidate=_CandidateRoute(
                    package_id="tools_unavailable",
                    confidence=candidate.confidence,
                    reason_codes=("required_tools_unavailable",),
                    include_current_date=candidate.include_current_date,
                    resolution_mode="degraded",
                ),
                available_tool_names=available_tool_names,
                requested_plan_mode="off",
                function_calling=function_calling,
                tools_disabled=True,
                network_boundary_required=True,
            )
        )
    if needs_external_capability and not resolution.external_tool_names:
        return _validated_resolution(
            _resolution(
                candidate=_CandidateRoute(
                    package_id="tools_unavailable",
                    confidence=candidate.confidence,
                    reason_codes=("required_tools_unavailable",),
                    include_current_date=candidate.include_current_date,
                    resolution_mode="degraded",
                ),
                available_tool_names=available_tool_names,
                requested_plan_mode="off",
                function_calling=function_calling,
                tools_disabled=True,
                network_boundary_required=True,
            )
        )
    return _validated_resolution(resolution)


def serialize_capability_resolution(resolution: RunCapabilityResolution) -> dict:
    """转换为可持久化的安全协议，不包含原文或自由文本。"""

    payload = asdict(resolution)
    payload["reason_codes"] = list(resolution.reason_codes)
    payload["external_tool_names"] = list(resolution.external_tool_names)
    return payload


def _classify_standard_request(
    *,
    message: str,
    task_context_messages: list[object] | None,
    available_tool_names: list[str],
) -> _CandidateRoute:
    original_transform_request = bool(_TRANSFORM_RE.search(message))
    control_message = _mask_quoted_literals(message)
    routing_message, web_search_denied, url_read_denied, all_network_denied = _resolve_network_scope(control_message)
    external_signal_message = _URL_RE.sub(" ", routing_message)
    external_web_action_message = _IN_DOCUMENT_SEARCH_RE.sub(" ", external_signal_message)
    include_current_date = _needs_current_date(routing_message)
    explicit_web_search_request = bool(
        _POSITIVE_WEB_SEARCH_ACTION_RE.search(external_web_action_message)
        or _POSITIVE_WEB_TOOL_NAME_RE.search(external_web_action_message)
    )
    url_read_request = bool(_URL_RE.search(routing_message) and _URL_READ_ACTION_RE.search(routing_message))
    verified_web_request = bool(_VERIFIED_SOURCE_RE.search(external_signal_message))
    independent_verified_web_request = verified_web_request and not _URL_LOCAL_SOURCE_ONLY_RE.search(
        external_signal_message
    )
    fresh_web_request = bool(_FRESH_EXTERNAL_RE.search(external_signal_message))

    if _is_definitional_knowledge_request(routing_message):
        return _CandidateRoute(
            "direct",
            "high",
            ("stable_knowledge_question",),
            False,
        )

    if _CURRENT_DATE_ONLY_RE.search(routing_message):
        return _CandidateRoute("date", "high", ("current_date_question",), True)

    if (
        url_read_request
        and not url_read_denied
        and not web_search_denied
        and (independent_verified_web_request or explicit_web_search_request or fresh_web_request)
    ):
        return _CandidateRoute(
            "verified_web",
            "high",
            ("verified_source_request",),
            True,
        )
    if not url_read_denied and url_read_request:
        return _CandidateRoute("url_read", "high", ("explicit_url_read",), False)
    if not web_search_denied and not url_read_denied and verified_web_request:
        return _CandidateRoute(
            "verified_web",
            "high",
            ("verified_source_request",),
            True,
        )
    if not web_search_denied and explicit_web_search_request and original_transform_request:
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )
    if original_transform_request and _has_explicit_given_text(message):
        return _CandidateRoute(
            "transform",
            "high",
            ("text_transform_request",),
            False,
        )
    if not web_search_denied and fresh_web_request:
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )
    if original_transform_request:
        return _CandidateRoute(
            "transform",
            "high",
            ("text_transform_request",),
            False,
        )
    denied_search_request = web_search_denied and bool(
        verified_web_request or fresh_web_request or explicit_web_search_request
    )
    denied_url_request = url_read_denied and url_read_request
    if denied_search_request or denied_url_request:
        return _CandidateRoute(
            "clarification_only",
            "low",
            ("insufficient_capability_signal",),
            False,
            resolution_mode="clarification",
        )

    signals = resolve_product_capability_signals(
        original_message=routing_message,
        task_context_messages=task_context_messages,
    )
    if signals.adjacent_route_followup and not all_network_denied:
        return _CandidateRoute(
            "mobility_route",
            "high",
            ("adjacent_route_followup",),
            include_current_date,
        )

    english_relation = _extract_english_route_relation(routing_message)
    english_abstract_route = bool(
        english_relation
        and _is_english_abstract_route_relation(
            routing_message,
            english_relation[0],
            english_relation[1],
        )
    )
    english_known_pair = bool(
        english_relation
        and english_relation[0] in _EN_PHYSICAL_LOCATION_NAMES
        and english_relation[1] in _EN_PHYSICAL_LOCATION_NAMES
    )
    english_strong_physical_pair = bool(
        english_relation
        and _EN_STRONG_MOBILITY_ACTION_RE.search(routing_message)
        and _is_english_physical_location(english_relation[0])
        and _is_english_physical_location(english_relation[1])
    )
    english_route = bool(
        english_relation and not english_abstract_route and (english_known_pair or english_strong_physical_pair)
    )
    english_intercity = bool(
        english_relation
        and not english_abstract_route
        and english_relation[0] in _EN_INTERCITY_LOCATION_NAMES
        and english_relation[1] in _EN_INTERCITY_LOCATION_NAMES
    )
    requested_english_route_modes = _extract_english_route_modes(routing_message) if english_relation else frozenset()
    english_route_mode_directive_present = bool(english_relation and _EN_ROUTE_MODE_SEGMENT_RE.search(routing_message))
    trusted_english_endpoints = english_route or english_intercity
    authorized_english_route_modes = requested_english_route_modes if trusted_english_endpoints else frozenset()
    english_route_mode_flight = "flight" in authorized_english_route_modes
    english_route_mode_train = "train" in authorized_english_route_modes
    english_route_mode_route = "route" in authorized_english_route_modes
    intercity_relation = (
        signals.endpoint_relation and signals.intercity_mobility and _has_two_intercity_locations(routing_message)
    ) or english_intercity
    english_local_route = bool(
        english_route
        and not english_intercity
        and (not requested_english_route_modes or english_route_mode_route or english_route_mode_train)
    )
    english_intercity_route = bool(english_intercity and english_route_mode_route)
    reauthorized_route_clause = _final_reauthorized_natural_product_clause(control_message, "route_compare")
    reauthorized_route = bool(
        reauthorized_route_clause
        and resolve_product_capability_signals(
            original_message=reauthorized_route_clause,
            task_context_messages=None,
        ).explicit_route
    )
    explicit_route_requested = bool(
        not all_network_denied
        and (
            (
                signals.explicit_route
                and not (english_intercity and english_route_mode_directive_present and not english_route_mode_route)
            )
            or english_local_route
            or english_intercity_route
            or _has_final_positive_product_tool_directive(control_message, "route_compare")
            or reauthorized_route
        )
    )
    flight_requested = bool(
        not all_network_denied
        and (
            _ZH_FLIGHT_TASK_RE.search(routing_message)
            or _EN_FLIGHT_TASK_RE.search(routing_message)
            or _ZH_AIR_RAIL_COMPARISON_RE.search(routing_message)
            or (english_intercity and english_route_mode_flight)
            or _has_final_positive_product_tool_directive(control_message, "search_flights")
        )
    )
    train_requested = bool(
        not all_network_denied
        and (
            _ZH_TRAIN_TASK_RE.search(routing_message)
            or _EN_TRAIN_TASK_RE.search(routing_message)
            or _ZH_AIR_RAIL_COMPARISON_RE.search(routing_message)
            or (english_intercity and english_route_mode_train)
            or _has_final_positive_product_tool_directive(control_message, "search_trains")
        )
    )
    weather_requested = bool(
        not all_network_denied
        and (
            signals.weather
            or _EN_WEATHER_TASK_RE.search(routing_message)
            or _has_final_positive_product_tool_directive(control_message, "weather_forecast")
        )
    )
    place_requested = bool(
        not all_network_denied
        and (
            signals.place
            or _EN_PLACE_TASK_RE.search(routing_message)
            or _has_final_positive_product_tool_directive(control_message, "local_place_search")
        )
    )
    denied_product_requests = {
        tool_name
        for requested, tool_name in (
            (weather_requested, "weather_forecast"),
            (place_requested, "local_place_search"),
            (explicit_route_requested, "route_compare"),
            (flight_requested, "search_flights"),
            (train_requested, "search_trains"),
        )
        if requested and _is_product_tool_finally_denied(control_message, tool_name)
    }
    if _is_product_tool_finally_denied(control_message, "route_compare"):
        denied_product_requests.add("route_compare")
    explicit_route = explicit_route_requested and "route_compare" not in denied_product_requests
    flight = flight_requested and "search_flights" not in denied_product_requests
    train = train_requested and "search_trains" not in denied_product_requests
    weather = weather_requested and "weather_forecast" not in denied_product_requests
    place = place_requested and "local_place_search" not in denied_product_requests
    if all_network_denied:
        intercity_relation = False

    if (
        intercity_relation
        and explicit_route
        and not english_route_mode_directive_present
        and not flight
        and not train
        and not weather
        and not place
    ):
        return _CandidateRoute(
            "mobility_intercity",
            "medium",
            ("origin_destination_relation", "intercity_locations"),
            True,
        )

    product_tools = tuple(
        name
        for enabled, name in (
            (weather, "weather_forecast"),
            (place, "local_place_search"),
            (explicit_route, "route_compare"),
            (flight, "search_flights"),
            (train, "search_trains"),
        )
        if enabled
    )
    if denied_product_requests and not product_tools:
        return _CandidateRoute(
            "clarification_only",
            "low",
            ("insufficient_capability_signal",),
            False,
            resolution_mode="clarification",
        )
    if len(product_tools) > 3:
        return _CandidateRoute(
            "clarification_only",
            "low",
            ("insufficient_capability_signal",),
            False,
            resolution_mode="clarification",
        )
    if len(product_tools) >= 2 and frozenset(product_tools) != frozenset({"search_flights", "search_trains"}):
        return _CandidateRoute(
            "mixed_itinerary",
            "high",
            ("mixed_itinerary_request",),
            True,
            explicit_tool_names=product_tools,
        )
    if flight and train:
        return _CandidateRoute(
            "travel_air_rail",
            "high",
            ("air_rail_comparison",),
            True,
        )
    if flight:
        return _CandidateRoute("flight", "high", ("explicit_flight_request",), True)
    if train:
        return _CandidateRoute("train", "high", ("explicit_train_request",), True)
    if weather:
        return _CandidateRoute(
            "weather",
            "high",
            ("explicit_weather_request",),
            True,
        )
    if place:
        return _CandidateRoute(
            "place_discovery",
            "high",
            ("explicit_place_discovery",),
            False,
        )
    if explicit_route:
        return _CandidateRoute(
            "mobility_route",
            "high",
            ("explicit_route_task",),
            include_current_date,
        )
    if intercity_relation and english_route_mode_directive_present:
        return _CandidateRoute(
            "clarification_only",
            "low",
            ("insufficient_capability_signal",),
            False,
            resolution_mode="clarification",
        )
    if intercity_relation:
        return _CandidateRoute(
            "mobility_intercity",
            "medium",
            ("origin_destination_relation", "intercity_locations"),
            True,
        )

    if english_relation is not None:
        if english_abstract_route:
            return _CandidateRoute(
                "direct",
                "high",
                ("stable_knowledge_question",),
                False,
            )
        return _CandidateRoute(
            "clarification_only",
            "low",
            ("insufficient_capability_signal",),
            False,
            resolution_mode="clarification",
        )

    if not web_search_denied and explicit_web_search_request:
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )

    if not web_search_denied and include_current_date and re.search(r"查|查询|多少|是否|吗[？?]?$", routing_message):
        return _CandidateRoute(
            "fresh_web",
            "high",
            ("fresh_external_fact",),
            True,
        )

    explicit_alias = (
        None if all_network_denied else _resolve_explicit_authorized_alias(routing_message, available_tool_names)
    )
    if explicit_alias is not None:
        return _CandidateRoute(
            "mcp_explicit",
            "high",
            ("explicit_authorized_tool_alias",),
            include_current_date,
            explicit_tool_names=(explicit_alias,),
        )

    if _GREETING_RE.search(routing_message):
        return _CandidateRoute("direct", "high", ("direct_greeting",), False)
    if _IDENTITY_RE.search(routing_message):
        return _CandidateRoute(
            "direct",
            "high",
            ("assistant_identity_question",),
            False,
        )
    if _SIMPLE_CALC_RE.search(routing_message):
        return _CandidateRoute("direct", "high", ("simple_calculation",), False)
    if _STABLE_KNOWLEDGE_RE.search(routing_message):
        return _CandidateRoute(
            "direct",
            "high",
            ("stable_knowledge_question",),
            False,
        )
    return _CandidateRoute(
        "clarification_only",
        "low",
        ("insufficient_capability_signal",),
        False,
        resolution_mode="clarification",
    )


def _resolution(
    *,
    candidate: _CandidateRoute,
    available_tool_names: list[str],
    requested_plan_mode: PlanMode,
    function_calling: bool,
    tools_disabled: bool,
    network_boundary_required: bool = False,
) -> RunCapabilityResolution:
    requested_tools = candidate.explicit_tool_names or _PACKAGE_TOOLS.get(candidate.package_id, ())
    available = frozenset(name for name in available_tool_names if isinstance(name, str) and name)
    tools = tuple(
        name
        for name in _canonicalize_tool_names(requested_tools)
        if name in available and name not in _CONTROL_TOOL_NAMES
    )
    reason_codes = tuple(code for code in candidate.reason_codes if code in _REASON_CODES)
    if reason_codes != candidate.reason_codes:
        raise ValueError("能力路由包含未注册的 reason code")
    return RunCapabilityResolution(
        schema_version=SCHEMA_VERSION,
        router_version=ROUTER_VERSION,
        package_id=candidate.package_id,
        confidence=candidate.confidence,
        resolution_mode=candidate.resolution_mode,
        reason_codes=reason_codes,
        external_tool_names=tools,
        effective_plan_mode=_effective_plan_mode(
            package_id=candidate.package_id,
            requested_plan_mode=requested_plan_mode,
            function_calling=function_calling,
            tools_disabled=tools_disabled,
        ),
        include_current_date=candidate.include_current_date,
        network_boundary_required=network_boundary_required,
    )


def _validated_resolution(resolution: RunCapabilityResolution) -> RunCapabilityResolution:
    validate_capability_resolution_semantics(
        package_id=resolution.package_id,
        confidence=resolution.confidence,
        resolution_mode=resolution.resolution_mode,
        reason_codes=resolution.reason_codes,
        external_tool_names=resolution.external_tool_names,
        effective_plan_mode=resolution.effective_plan_mode,
        include_current_date=resolution.include_current_date,
        network_boundary_required=resolution.network_boundary_required,
    )
    return resolution


def _effective_plan_mode(
    *,
    package_id: str,
    requested_plan_mode: PlanMode,
    function_calling: bool,
    tools_disabled: bool,
) -> PlanMode:
    if not function_calling or tools_disabled:
        return "off"
    if package_id == "deep_research":
        return "on"
    if requested_plan_mode in {"on", "off"}:
        return requested_plan_mode
    return "auto" if package_id in _AUTO_PLAN_PACKAGES else "off"


def _canonicalize_tool_names(tool_names: tuple[str, ...]) -> tuple[str, ...]:
    known_order = {name: index for index, name in enumerate(_CANONICAL_EXTERNAL_TOOL_ORDER)}
    return tuple(sorted(set(tool_names), key=lambda name: (known_order.get(name, 10_000), name)))


def _resolve_explicit_authorized_alias(
    message: str,
    available_tool_names: list[str],
) -> str | None:
    product_names = frozenset(_CANONICAL_EXTERNAL_TOOL_ORDER) | _CONTROL_TOOL_NAMES
    aliases = sorted(
        {name for name in available_tool_names if is_authorized_mcp_tool_alias(name) and name not in product_names}
    )
    matched: list[str] = []
    for alias in aliases:
        alias_pattern = rf"(?<![\w]){re.escape(alias.lower())}(?![\w])"
        directives: list[tuple[int, bool]] = []
        for alias_match in re.finditer(alias_pattern, message):
            prefix = message[max(0, alias_match.start() - 48) : alias_match.start()]
            if re.search(
                r"(?:不要|不用|别|请勿|禁止|严禁|不得|不可)\s*"
                r"(?:再|随后)?(?:调用|使用|运行|执行)?\s*$",
                prefix,
            ) or re.search(
                r"\b(?:(?:do not|don['’]t|dont|never)\s+(?:call|use|run|invoke)|"
                r"(?:do not|don['’]t|dont|never)\s+execute|"
                r"(?:without|avoid(?:ing)?|refrain\s+from|skip(?:ping)?)\s+"
                r"(?:call(?:ing)?|us(?:e|ing)|run(?:ning)?|invok(?:e|ing)|execut(?:e|ing)))\s+"
                r"(?:the\s+)?(?:mcp\s+)?(?:tool\s+)?$",
                prefix,
                re.IGNORECASE,
            ):
                directives.append((alias_match.start(), False))
            elif re.search(
                r"(?:调用|使用|运行|执行)(?:工具)?\s*$|"
                r"\b(?:call|use|run|invoke|execute)\s+(?:the\s+)?(?:mcp\s+)?(?:tool\s+)?$",
                prefix,
            ):
                directives.append((alias_match.start(), True))
        if directives and directives[-1][1]:
            matched.append(alias)
    return matched[0] if len(matched) == 1 else None


def _normalize_message(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _mask_quoted_literals(message: str) -> str:
    def mask_match(match: re.Match[str]) -> str:
        masked = [" "] * len(match.group(0))
        for resource_match in _QUOTED_RESOURCE_RE.finditer(match.group(0)):
            masked[resource_match.start() : resource_match.end()] = resource_match.group(0)
        return "".join(masked)

    return _QUOTED_LITERAL_RE.sub(mask_match, message)


def _resolve_network_scope(message: str) -> tuple[str, bool, bool, bool]:
    network_denial_patterns = (
        _NEGATED_ALL_NETWORK_RE,
        _NEGATED_WEB_SEARCH_RE,
        _NEGATED_URL_READ_RE,
        _NEGATED_VERIFIED_WEB_RE,
        _NEGATED_WEB_TOOL_NAME_RE,
        _NEGATED_URL_TOOL_NAME_RE,
    )
    if not any(pattern.search(message) for pattern in network_denial_patterns):
        return message, False, False, False

    clauses = re.split(
        r"(?:[，,。；;]|[.!:：](?=\s)|\s+[—–]\s+|[—–]{2}|但(?:是)?|不过|而(?:是|要)?|\bbut\b|"
        r"\band\s+(?=(?:do not|don['’]t|dont|never|search|read|open|find|verify|"
        r"cross-check|call|use|browse|summarize|analyze|translate|do\s+a|perform)\b)|"
        r"\b(?:then|afterwards|later|finally)\s+(?=(?:search|read|open|find|verify|"
        r"cross-check|call|use|browse|summarize|analyze|translate|do\s+a|perform)\b)|"
        r"(?:并且|并|然后|随后|最后)(?=(?:不要|请勿|禁止|严禁|不得|不可|搜索|检索|查询|读取|打开|查找|核验|验证|调用|使用|"
        r"总结|摘要|分析|翻译)))\s*",
        message,
        flags=re.IGNORECASE,
    )
    web_search_denied = False
    url_read_denied = False
    all_network_denied = False
    hard_current_network_denied = any(
        _is_current_request_network_denial(message, match) for match in _NEGATED_ALL_NETWORK_RE.finditer(message)
    )
    hard_web_tool_denied = False
    hard_url_tool_denied = False
    routing_clauses: list[str] = []
    allowed_web_objects: set[str] = set()
    allowed_url_targets: set[str] = set()

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        negative_matches = [(match.start(), match.end(), "all") for match in _NEGATED_ALL_NETWORK_RE.finditer(clause)]
        negative_matches.extend(
            (match.start(), match.end(), "web") for match in _NEGATED_WEB_SEARCH_RE.finditer(clause)
        )
        negative_matches.extend((match.start(), match.end(), "url") for match in _NEGATED_URL_READ_RE.finditer(clause))
        negative_matches.extend(
            (match.start(), match.end(), "web") for match in _NEGATED_VERIFIED_WEB_RE.finditer(clause)
        )
        explicit_tool_events: list[tuple[int, str, bool, bool]] = []
        for match in _NEGATED_WEB_TOOL_NAME_RE.finditer(clause):
            negative_matches.append((match.start(), match.end(), "web"))
            explicit_tool_events.append((match.start(), "web", False, _has_scoped_denial_tail(clause, match.end())))
        for match in _NEGATED_URL_TOOL_NAME_RE.finditer(clause):
            negative_matches.append((match.start(), match.end(), "url"))
            explicit_tool_events.append((match.start(), "url", False, _has_scoped_denial_tail(clause, match.end())))
        explicit_negative_spans = [(start, end) for start, end, _capability in negative_matches]
        explicit_tool_events.extend(
            (match.start(), "web", True, False)
            for match in _POSITIVE_WEB_TOOL_NAME_RE.finditer(clause)
            if not _overlaps_any(match.span(), explicit_negative_spans)
        )
        explicit_tool_events.extend(
            (match.start(), "url", True, False)
            for match in _POSITIVE_URL_TOOL_NAME_RE.finditer(clause)
            if not _overlaps_any(match.span(), explicit_negative_spans)
        )
        routing_negative_matches = tuple(negative_matches)
        negative_web_object = _extract_web_request_object(clause)
        negative_url_targets = _extract_url_targets(clause)
        if (
            negative_web_object
            and allowed_web_objects
            and not _is_anaphoric_request_object(negative_web_object)
            and any(existing != negative_web_object for existing in allowed_web_objects)
        ):
            negative_matches = [match for match in negative_matches if match[2] != "web"]
        if negative_url_targets and allowed_url_targets and not negative_url_targets.issuperset(allowed_url_targets):
            negative_matches = [match for match in negative_matches if match[2] != "url"]
        events: list[tuple[int, str, bool]] = []
        for start, _end, capability in negative_matches:
            if capability == "all":
                if _is_current_request_scope_tail(clause, _end):
                    hard_current_network_denied = True
                if _has_scoped_denial_tail(clause, _end):
                    continue
                events.extend(((start, "web", False), (start, "url", False)))
                all_network_denied = True
            else:
                events.append((start, capability, False))

        events.extend(
            (position, capability, True) for position, capability, allowed, _scoped in explicit_tool_events if allowed
        )

        negative_spans = [(start, end) for start, end, _ in routing_negative_matches]
        for match in _POSITIVE_WEB_SEARCH_ACTION_RE.finditer(clause):
            if not _overlaps_any(match.span(), negative_spans):
                events.append((match.start(), "web", True))
                web_object = _extract_web_request_object(clause)
                if web_object:
                    allowed_web_objects.add(web_object)
        if _URL_RE.search(clause):
            for match in _URL_READ_ACTION_RE.finditer(clause):
                if not _overlaps_any(match.span(), negative_spans):
                    events.append((match.start(), "url", True))
                    allowed_url_targets.update(_extract_url_targets(clause))
        for match in _VERIFIED_SOURCE_RE.finditer(_URL_RE.sub(" ", clause)):
            if not negative_matches and not _overlaps_any(match.span(), negative_spans):
                events.extend(((match.start(), "web", True), (match.start(), "url", True)))

        for _position, capability, allowed in sorted(events, key=lambda event: event[0]):
            if capability == "web":
                web_search_denied = not allowed
            else:
                url_read_denied = not allowed
            if allowed and not hard_current_network_denied:
                all_network_denied = False

        for _position, capability, allowed, scoped in sorted(explicit_tool_events, key=lambda event: event[0]):
            if scoped:
                continue
            if capability == "web":
                hard_web_tool_denied = not allowed
            else:
                hard_url_tool_denied = not allowed

        if not routing_negative_matches or any(allowed for _, _, allowed in events):
            routing_clauses.append(clause)

    return (
        "; ".join(routing_clauses),
        web_search_denied or hard_current_network_denied or hard_web_tool_denied,
        url_read_denied or hard_current_network_denied or hard_url_tool_denied,
        all_network_denied,
    )


def _has_scoped_denial_tail(clause: str, match_end: int) -> bool:
    return _extract_directive_scope(clause, match_end) is not None


def _is_current_request_network_denial(message: str, match: re.Match[str]) -> bool:
    if _extract_directive_scope(message, match.end()) is not None:
        return False
    matched_text = match.group(0)
    has_embedded_current_scope = bool(
        re.search(
            r"(?:本次|此次|当前|目前|这个|该|本轮|这次|本条|这条|本|整个)"
            r"(?:请求|任务|问题|对话|轮次|消息|回答|回复|答复|响应|查询)"
            r"(?:中|里|内|范围内|期间)?",
            matched_text,
        )
    )
    return bool(
        has_embedded_current_scope
        or _has_current_request_scope_prefix(message, match.start())
        or _is_current_request_scope_tail(message, match.end())
    )


def _has_current_request_scope_prefix(message: str, match_start: int) -> bool:
    prefix = message[:match_start]
    return bool(
        re.search(
            r"(?:\b(?:for|in|within|during|regarding|on)\s+"
            r"(?:(?:this|my|your|the)\s+)?"
            r"(?:(?:current|present|specific|entire|whole|full)\s+)?"
            r"(?:request|task|question|conversation|chat|turn|message|answer|reply|response|query)|"
            r"(?:针对|关于|在|就)?(?:本次|此次|当前|目前|这个|该|本轮|这次|本条|这条|本|整个)"
            r"(?:请求|任务|问题|对话|轮次|消息|回答|回复|答复|响应|查询)"
            r"(?:中|里|内|范围内|期间|而言)?)"
            r"\s*[,，:：]?\s*(?:(?:please|kindly)\s+|(?:请|麻烦)?\s*务必\s*|(?:请|麻烦)\s*)?$",
            prefix,
            re.IGNORECASE,
        )
    )


def _is_current_request_scope_tail(clause: str, match_end: int) -> bool:
    tail = clause[match_end:]
    return bool(
        re.match(
            r"\s*(?:for|about|in|within|during|regarding|on)\b\s+"
            r"(?:this(?:\s+one)?|"
            r"(?:(?:this|the|my|your|any)\s+)?"
            r"(?:(?:very|current|present|specific|entire|whole|full)\s+)?"
            r"(?:request|task|question|conversation|chat|turn|message|answer|reply|response|query)"
            r"(?:\s+at\s+hand)?)\b|"
            r"\s*(?:用于|针对|关于|在|于)(?:本次|此次|当前|目前|这个|该|本轮|这次|本条|这条|本|整个)"
            r"(?:请求|任务|问题|对话|轮次|消息|回答|回复|答复|响应|查询)"
            r"(?:中|里|内|范围内|期间)?",
            tail,
            re.IGNORECASE,
        )
    )


def _extract_directive_scope(clause: str, match_end: int) -> str | None:
    tail = clause[match_end:]
    scoped_match = re.match(
        r"\s*(?:for|about|in|within|during|regarding|on)\b\s+"
        r"(?P<english>[^\s,.;!?，。；：！？]+(?:\s+[^\s,.;!?，。；：！？]+){0,3})|"
        r"\s*(?:用于|针对|关于)(?P<chinese>[^,.;!?，。；：！？]+)",
        tail,
        re.IGNORECASE,
    )
    if scoped_match is None:
        return None
    scope = (scoped_match.group("english") or scoped_match.group("chinese") or "").strip(".,;:!?，。；：！？")
    if _is_current_request_scope_tail(clause, match_end):
        return None
    return scope.lower()


def _extract_url_targets(clause: str) -> set[str]:
    return {match.group(0).rstrip(".,;:!?，。；：！？") for match in _URL_RE.finditer(clause)}


def _extract_web_request_object(clause: str) -> str:
    normalized = clause.lower()
    normalized = re.sub(r"\b(?:newest|most\s+recent)\b", "latest", normalized)
    normalized = re.sub(r"\b(?:do not|don['’]t|dont|without|never)\b", " ", normalized)
    normalized = re.sub(
        r"\b(?:call|use|invoke|run)\s+(?:the\s+)?(?:tool\s+)?"
        r"(?:web_search|url_read)\b(?:\s+tool\b)?",
        " ",
        normalized,
    )
    normalized = re.sub(r"(?:调用|使用|运行|执行)\s*(?:web_search|url_read)", " ", normalized)
    normalized = re.sub(
        r"(?:不要|不用|无需|不需要|不必|别|请勿|禁止|严禁|不得|不可|请|帮我|要|再|随后)",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:search(?:ing)?(?: the)? web(?: for)?|search|look(?:ing)? up|find(?:ing)? online|"
        r"brows(?:e|ing)(?: the)? web|look\s+(?:it|this|that|them)\s+up)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"(?:联网|上网|网上)?(?:搜索|检索|查询|查找|查)", " ", normalized)
    normalized = re.sub(r"(?:用于|针对|关于)", " ", normalized)
    normalized = re.sub(r"\b(?:the|a|an|for|about|regarding|on|please)\b", " ", normalized)
    normalized = re.sub(r"\b(?:again|anymore|any longer|further)\b", " ", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_anaphoric_request_object(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:it|this|that|them|this topic|that topic|this subject|that subject|"
            r"this content|that content|the above content|above content|the same topic|same topic|"
            r"它|这个|那个|这些|那些|该主题|这个主题|那个主题|同一主题|相同主题|"
            r"这个话题|那个话题|这个内容|那个内容|上述内容|以上内容|该内容)",
            value,
            re.IGNORECASE,
        )
    )


def _overlaps_any(span: tuple[int, int], other_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in other_spans)


def _is_product_tool_finally_denied(message: str, tool_name: str) -> bool:
    explicit_directives = _explicit_product_tool_directives(message, tool_name)
    if explicit_directives:
        final_explicit = max(explicit_directives, key=lambda event: event[0])
        if not final_explicit[1] and final_explicit[2] is None:
            return True
        if (
            not final_explicit[1]
            and final_explicit[2] is not None
            and _scope_matches_prior_product_request(
                final_explicit[2],
                message[: final_explicit[0]],
            )
        ):
            return True
    natural_directives = [
        (position, allowed, None) for position, allowed in _natural_product_tool_directives(message, tool_name)
    ]
    directives = explicit_directives + natural_directives
    positive_directives = [(position, scope) for position, allowed, scope in directives if allowed]
    filtered_directives = [
        event
        for event in directives
        if event[1]
        or not (
            event[2] is not None
            and any(
                positive_position < event[0] and (positive_scope is None or positive_scope != event[2])
                for positive_position, positive_scope in positive_directives
            )
        )
    ]
    events = [(position, allowed) for position, allowed, _scope in filtered_directives]
    return bool(events and not max(events, key=lambda event: event[0])[1])


def _natural_product_tool_directives(message: str, tool_name: str) -> list[tuple[int, bool]]:
    return [
        (match.start(), not _is_natural_product_match_negated(message, match.start()))
        for pattern in _product_tool_positive_patterns(tool_name)
        for match in pattern.finditer(message)
    ]


def _final_reauthorized_natural_product_clause(message: str, tool_name: str) -> str | None:
    directives = _natural_product_tool_directives(message, tool_name)
    if not directives:
        return None
    final_position, final_allowed = max(directives, key=lambda event: event[0])
    if not final_allowed or not any(position < final_position and not allowed for position, allowed in directives):
        return None
    prior_boundaries = list(_PRODUCT_DIRECTIVE_BOUNDARY_RE.finditer(message[:final_position]))
    clause_start = prior_boundaries[-1].end() if prior_boundaries else 0
    return message[clause_start:].strip()


def _scope_matches_prior_product_request(scope: str, prior_text: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", scope):
        compact_scope = re.sub(r"[^\w\u4e00-\u9fff]+", "", scope.lower())
        compact_prior = re.sub(r"[^\w\u4e00-\u9fff]+", "", prior_text.lower())
        return len(compact_scope) >= 2 and compact_scope in compact_prior

    def significant_tokens(value: str) -> set[str]:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value.lower())
        ignored = {
            "a",
            "an",
            "for",
            "from",
            "on",
            "the",
            "to",
            "about",
            "regarding",
            "route",
            "routes",
            "request",
            "task",
            "用于",
            "关于",
            "针对",
        }
        return {token for token in normalized.split() if token not in ignored}

    scope_tokens = significant_tokens(scope)
    if not scope_tokens:
        return False
    prior_tokens = significant_tokens(prior_text)
    return scope_tokens.issubset(prior_tokens)


def _is_natural_product_match_negated(message: str, match_start: int) -> bool:
    prefix = message[max(0, match_start - 48) : match_start]
    prefix = _PRODUCT_DIRECTIVE_BOUNDARY_RE.split(prefix)[-1]
    if re.search(
        r"\bnot\s+(?:only|merely|solely|exclusively)\s+$|"
        r"\b(?:do not|don['’]t|dont)\s+(?:forget\s+to|fail\s+to|just|only|simply)\s+$",
        prefix,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:are|is|were|was)\s+there\s+no(?:\s+[\w-]+){0,3}\s+$", prefix, re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:exclude|excluding|avoid|avoiding|skip|skipping|no)\s+"
        r"(?:[\w-]+\s+){1,4}$",
        prefix,
        re.IGNORECASE,
    ) and re.search(
        r"(?:\band\b|\bthen\b|[;；]).{0,96}\b(?:show|find|list|include|prefer|compare)\b"
        r".{0,64}\b(?:options?|alternatives?|connections?|itineraries?)\b",
        message[match_start:],
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:do not|don['’]t|dont|never|not)\s+"
            r"(?:(?!(?:hesitate|forget|fail|just|only|merely|simply)\b)\w+\s+){0,3}$|"
            r"\b(?:avoid|avoiding|without|refrain\s+from|exclude|excluding|skip|skipping)\s+"
            r"(?:\w+\s+){0,3}$|"
            r"\bno(?:\s+need\s+for)?\s+(?:\w+\s+){0,2}$|"
            r"(?:不要|不用|无需|不需要|不必|没必要|没有必要|用不着|别|请勿|禁止|严禁|不得|不可|"
            r"避免|跳过|排除)(?:再)?(?:"
            r"(?:给我|帮我|去|给出|提供|查|查询|搜索|检索|查找|找|查看|获取|推荐|规划|"
            r"比较|对比|预订|订|购买|买)[\w\u4e00-\u9fff\s-]{0,24}"
            r")?\s*$",
            prefix,
            re.IGNORECASE,
        )
    )


def _has_final_positive_product_tool_directive(message: str, tool_name: str) -> bool:
    directives = _explicit_product_tool_directives(message, tool_name)
    positive_directives = [(position, scope) for position, allowed, scope in directives if allowed]
    effective_directives = [
        event
        for event in directives
        if event[1]
        or not (
            event[2] is not None
            and any(
                positive_position < event[0] and (positive_scope is None or positive_scope != event[2])
                for positive_position, positive_scope in positive_directives
            )
        )
    ]
    return bool(effective_directives and max(effective_directives, key=lambda event: event[0])[1])


def _has_natural_product_denial(message: str, tool_name: str) -> bool:
    return any(
        _is_natural_product_match_negated(message, match.start())
        for pattern in _product_tool_positive_patterns(tool_name)
        for match in pattern.finditer(message)
    )


def _explicit_product_tool_directives(message: str, tool_name: str) -> list[tuple[int, bool, str | None]]:
    escaped_tool_name = re.escape(tool_name)
    negative_pattern = re.compile(
        rf"(?:不要|不用|别|请勿|禁止|严禁|不得|不可).{{0,24}}?(?<![\w]){escaped_tool_name}(?![\w])|"
        rf"\b(?:do not|don['’]t|dont|never)\s+(?:call|use|run|invoke|execute)\s+"
        rf"(?:the\s+)?(?:tool\s+)?{escaped_tool_name}\b(?:\s+tool\b)?|"
        rf"\b(?:without|avoid(?:ing)?|refrain\s+from|skip(?:ping)?)\s+"
        rf"(?:call(?:ing)?|us(?:e|ing)|run(?:ning)?|invok(?:e|ing)|execut(?:e|ing))\s+"
        rf"(?:the\s+)?(?:tool\s+)?{escaped_tool_name}\b(?:\s+tool\b)?",
        re.IGNORECASE,
    )
    positive_tool_pattern = re.compile(
        rf"(?:调用|使用|运行|执行)\s*(?<![\w]){escaped_tool_name}(?![\w])|"
        rf"\b(?:call|use|run|invoke|execute)\s+(?:the\s+)?(?:tool\s+)?{escaped_tool_name}\b(?:\s+tool\b)?",
        re.IGNORECASE,
    )
    negative_matches = list(negative_pattern.finditer(message))
    negative_spans = [match.span() for match in negative_matches]
    events: list[tuple[int, bool, str | None]] = [
        (match.start(), False, _extract_directive_scope(message, match.end())) for match in negative_matches
    ]
    events.extend(
        (match.start(), True, _extract_directive_scope(message, match.end()))
        for match in positive_tool_pattern.finditer(message)
        if not _overlaps_any(match.span(), negative_spans)
    )
    return events


def _product_tool_positive_patterns(tool_name: str) -> tuple[re.Pattern[str], ...]:
    if tool_name == "search_flights":
        return (
            _ZH_FLIGHT_TASK_RE,
            _EN_FLIGHT_TASK_RE,
            re.compile(r"\b(?:by|via)\s+[^?.!,]{0,36}\b(?:plane|air|airplane|flight)\b", re.IGNORECASE),
        )
    if tool_name == "search_trains":
        return (
            _ZH_TRAIN_TASK_RE,
            _EN_TRAIN_TASK_RE,
            re.compile(
                r"\b(?:by|via)\s+[^?.!,]{0,36}\b(?:train|rail|railway|high[- ]speed (?:train|rail))\b",
                re.IGNORECASE,
            ),
        )
    if tool_name == "route_compare":
        return (
            _EN_ROUTE_RELATION_RE,
            re.compile(
                rf"从{_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{1,64}}(?:到|至)"
                rf"{_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{1,64}}(?:怎么|路线|公共交通|地铁|驾车|开车)"
            ),
        )
    if tool_name == "weather_forecast":
        return (_EN_WEATHER_TASK_RE, re.compile(r"天气|气温|温度|下雨|下雪|降雨|降雪|刮风"))
    if tool_name == "local_place_search":
        return (
            _EN_PLACE_TASK_RE,
            re.compile(
                rf"(?:附近|周边){_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,32}}(?:咖啡|餐厅|酒店|景点|店)|"
                rf"(?:找|推荐){_ZH_PRODUCT_CLAUSE_TEXT_ATOM}{{0,32}}(?:咖啡|餐厅|酒店|景点)"
            ),
        )
    return ()


def _has_explicit_given_text(message: str) -> bool:
    return bool(_QUOTED_LITERAL_TRANSFORM_RE.search(message) or _GIVEN_TEXT_TRANSFORM_RE.search(message))


def _is_definitional_knowledge_request(message: str) -> bool:
    noun_definition = _NOUN_DEFINITION_RE.fullmatch(message)
    noun_definition_tail = noun_definition.group("tail").rstrip("?.!") if noun_definition else ""
    noun_definition_term = noun_definition.group("noun").lower() if noun_definition else ""
    if "weather forecast" in noun_definition_term:
        safe_tail_pattern = _SAFE_WEATHER_DEFINITION_TAIL_RE
    elif "current price" in noun_definition_term:
        safe_tail_pattern = _SAFE_CURRENT_PRICE_DEFINITION_TAIL_RE
    elif "source" in noun_definition_term:
        safe_tail_pattern = _SAFE_DEFINITION_TAIL_RE
    else:
        safe_tail_pattern = _SAFE_BASIC_DEFINITION_TAIL_RE
    is_safe_noun_definition = bool(
        noun_definition
        and safe_tail_pattern.fullmatch(noun_definition_tail)
        and not _EXTERNAL_QUERY_DEFINITION_TAIL_RE.search(noun_definition_tail)
    )
    if not is_safe_noun_definition and not _DEFINITIONAL_KNOWLEDGE_RE.search(message):
        return False
    if _URL_RE.search(message):
        return False
    return not bool(
        _POSITIVE_WEB_SEARCH_ACTION_RE.search(message)
        or _POSITIVE_WEB_TOOL_NAME_RE.search(message)
        or _POSITIVE_URL_TOOL_NAME_RE.search(message)
    )


def _needs_current_date(message: str) -> bool:
    return bool(_RELATIVE_DATE_RE.search(message))


def _has_two_intercity_locations(message: str) -> bool:
    institution_suffix = r"(?:大学|学院|公交|地铁|交通|公司|集团|医院|博物馆|体育馆)"
    return (
        len(
            {
                location
                for location in INTERCITY_LOCATION_NAMES
                if re.search(rf"{re.escape(location)}(?!{institution_suffix})", message)
            }
        )
        >= 2
    )


def _extract_english_route_relation(message: str) -> tuple[str, str] | None:
    match = _EN_NATURAL_INTERCITY_RE.search(message) or _EN_ROUTE_RELATION_RE.search(message)
    if match is None:
        return None
    origin = _normalize_english_location(match.group("origin"))
    destination = _normalize_english_location(match.group("destination"))
    if not origin or not destination:
        return None
    return origin, destination


def _normalize_english_location(value: str) -> str:
    normalized = value.strip(" ,.!?").lower()
    normalized = re.sub(
        r"\s+(?:by|via)\s+(?:public transit|train|rail|plane|air|flight|car|bus|taxi|"
        r"walking|cycling|high[- ]speed (?:train|rail))$",
        "",
        normalized,
    )
    return normalized.strip()


def _extract_english_route_modes(message: str) -> frozenset[str]:
    mode_pattern = re.compile(
        r"\b(?:high[- ]speed (?:train|rail|railway)|public transit|"
        r"plane|airplane|flight|"
        r"air(?![- ]condition(?:ed|ing)\b|\s+(?:quality|exposure|pollution|emissions?|pollutants?)\b)|"
        r"train|railway|rail|"
        r"coach|bus|car|taxi|walking|walk|cycling|bike|driving|drive)\b",
        re.IGNORECASE,
    )
    mode_categories = {
        "plane": "flight",
        "airplane": "flight",
        "flight": "flight",
        "air": "flight",
        "train": "train",
        "railway": "train",
        "rail": "train",
        "high-speed train": "train",
        "high speed train": "train",
        "high-speed rail": "train",
        "high speed rail": "train",
        "high-speed railway": "train",
        "high speed railway": "train",
        "public transit": "route",
        "coach": "route",
        "bus": "route",
        "car": "route",
        "taxi": "route",
        "walking": "route",
        "walk": "route",
        "cycling": "route",
        "bike": "route",
        "driving": "route",
        "drive": "route",
    }
    resolved: dict[str, tuple[str, bool]] = {}
    for match in _EN_ROUTE_MODE_SEGMENT_RE.finditer(message):
        context = message[max(0, match.start() - 32) : match.start()] + match.group(0)
        explicit_exclusion_spans = _english_route_mode_exclusion_spans(context)
        for mode_match in mode_pattern.finditer(context):
            prefix = context[: mode_match.start()]
            if mode_match.group(0).lower() == "air" and not re.search(
                r"(?:\b(?:by|via|or|and)|[,/])\s*$",
                prefix,
                re.IGNORECASE,
            ):
                continue
            reset_matches = list(re.finditer(r"\b(?:but|however|then)\b", prefix, re.IGNORECASE))
            scoped_prefix = prefix[reset_matches[-1].end() :] if reset_matches else prefix
            denied = (
                _overlaps_any(mode_match.span(), explicit_exclusion_spans)
                or _is_english_route_mode_denied(scoped_prefix, mode_pattern)
                or bool(
                    re.match(
                        r"\s*(?:\(\s*)?(?:is\s+)?"
                        r"(?:excluded|omitted|skipped|avoided|prohibited|not\s+allowed|off\s+limits)\b",
                        context[mode_match.end() :],
                        re.IGNORECASE,
                    )
                )
            )
            normalized_mode = mode_match.group(0).lower()
            category = mode_categories[normalized_mode]
            state_key = category if category in {"flight", "train"} else normalized_mode
            resolved[state_key] = (category, not denied)
    return frozenset(category for category, allowed in resolved.values() if allowed)


def _english_route_mode_exclusion_spans(context: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"\b(?:excluding|exclude|avoiding|avoid|skipping|skip|leaving out|leave out|"
        r"omitting|omit|without|except(?: for)?|with the exception of|all but|other than|"
        r"rather than|instead of|as opposed to)\b\s+"
        r"(?P<body>.*?)(?=,\s*by\b|\b(?:but|however|then)\b|$)",
        context,
        re.IGNORECASE,
    ):
        spans.append(match.span("body"))
    for match in re.finditer(
        r"\bwith\s+(?P<body>.*?\b(?:plane|airplane|flight|air|train|railway|rail|"
        r"bus|coach|car|taxi|walking|walk|cycling|bike|driving|drive)\b.*?)\s+"
        r"(?:are\s+)?(?:excluded|omitted|skipped|avoided|prohibited|not\s+allowed|off\s+limits)\b",
        context,
        re.IGNORECASE,
    ):
        spans.append(match.span("body"))
    return spans


def _is_english_route_mode_denied(prefix: str, mode_pattern: re.Pattern[str]) -> bool:
    negative_matches = list(
        re.finditer(
            r"\b(?:rather than|instead of|as opposed to|other than|avoiding|avoid|"
            r"excluding|exclude|skipping|skip|leaving out|leave out|omitting|omit|"
            r"without|except(?: for)?|with the exception of|all but|neither|nor|no)\b|"
            r"\bnot\b(?!\s+only\b)",
            prefix,
            re.IGNORECASE,
        )
    )
    if not negative_matches:
        return False
    tail = prefix[negative_matches[-1].end() :]
    if re.search(r",\s*(?:by|via)\b", tail, re.IGNORECASE):
        return False
    if re.search(r",\s+and\s*$", tail, re.IGNORECASE):
        return False
    tail = mode_pattern.sub(" ", tail)
    tail = re.sub(
        r"\b(?:by|via|or|and|either|both|also|the|a|an|taking|using|traveling|travelling)\b",
        " ",
        tail,
        flags=re.IGNORECASE,
    )
    tail = re.sub(r"[^a-z0-9]+", " ", tail, flags=re.IGNORECASE).strip()
    return not tail


def _is_english_physical_location(value: str) -> bool:
    return value in _EN_PHYSICAL_LOCATION_NAMES or bool(_EN_PHYSICAL_LOCATION_SUFFIX_RE.search(value))


def _is_english_abstract_route_relation(message: str, origin: str, destination: str) -> bool:
    return bool(
        _EN_ABSTRACT_ROUTE_CONTEXT_RE.search(message)
        or _EN_ABSTRACT_ROUTE_ENDPOINT_RE.search(origin)
        or _EN_ABSTRACT_ROUTE_ENDPOINT_RE.search(destination)
    )
