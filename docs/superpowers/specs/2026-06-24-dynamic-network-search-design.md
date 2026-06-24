# 动态联网搜索与读取预算设计

## 背景

Fusion 当前联网搜索链路把“搜索候选数量”“进入模型上下文的资料数量”“前端回答依据预览数量”混在一起。`web_search` 工具只允许模型传 `query`，后端固定向 search-service 请求 5 条结果；`url_read` 已存在，但缺少用于解释“为什么读取这个页面”的参数。

本设计把联网资料流拆成两层：

- `web_search`：返回候选来源，负责找资料。
- `url_read`：读取少量关键页面，负责核实原文细节。

模型决定是否继续找资料；后端负责硬预算、参数裁剪、日志和诊断；前端负责展示本轮使用了多少候选和读取了多少页面。

## 目标

- 允许模型为 `web_search` 传入 `count`、`intent`、`domains`、`recency_days`。
- 后端对模型传入的搜索参数做 clamp，避免延迟、成本和上下文失控。
- 允许模型为 `url_read` 传入 `reason`，便于诊断面板解释读取目的。
- 保持现有工具名：继续使用 `web_search` 和 `url_read`，不新增 `web_fetch`。
- 回答依据和联网诊断能展示 query、count、intent、返回数、使用数、读取数和触顶原因。

## 非目标

- 不实现复杂 rerank 模型。
- 不新增用户可配置的联网搜索设置。
- 不更换 search-service provider。
- 不改变 reader-service 的抓取实现。
- 不改变现有 SSE/Redis Stream 两段式架构。

## 工具契约

### `web_search`

新增可选参数：

```json
{
  "query": "搜索关键词",
  "count": 8,
  "intent": "comparison",
  "domains": ["openai.com"],
  "recency_days": 30
}
```

字段语义：

- `query`：必填，搜索关键词。
- `count`：可选，模型期望候选数量；后端限制为 3 到 10，默认 5。
- `intent`：可选，允许值为 `quick_fact`、`freshness`、`comparison`、`deep_research`、`official_source`。
- `domains`：可选，最多 5 个域名，只允许普通域名，不允许协议、路径、端口或通配符。
- `recency_days`：可选，允许 1 到 365；后端映射为 search-service 支持的 freshness 粒度。

### `url_read`

新增可选参数：

```json
{
  "url": "https://example.com/page",
  "reason": "需要核实官方原文细节"
}
```

字段语义：

- `url`：必填，完整 URL。
- `reason`：可选，最多 160 个字符；写入工具日志和诊断，不注入为可信事实。

## 后端预算

单次工具参数：

- `web_search.count` 默认 5，范围 3 到 10。
- `web_search.domains` 最多 5 个。
- `web_search.recency_days` 范围 1 到 365。
- `url_read.reason` 最多 160 个字符。

单轮回答预算：

- 最多 3 次 `web_search`。
- 最多 5 次 `url_read`。
- 累计搜索候选最多 30 条。
- 注入模型上下文的搜索候选最多 8 条。
- 注入模型上下文的已读取网页最多 5 个。

触顶策略：

- 单次 `count` 超过范围时静默 clamp，并在工具结果 metadata 里记录原始值和实际值。
- 超过 search/read 次数预算时，工具返回 `degraded`，错误信息说明本轮联网预算已用完。
- 超过累计候选预算时，本次搜索仍可返回已裁剪结果，但标记 `budget_limited=true`。
- Agent loop 原有 `max_steps`、`max_tool_calls`、`timeout` 仍作为全局兜底。

## intent 策略

第一版只做轻量规则，不做复杂排序模型：

- `quick_fact`：默认少量候选，通常不鼓励读取多个页面。
- `freshness`：优先近期结果，允许模型二次搜索。
- `comparison`：鼓励多来源，保留域名多样性。
- `deep_research`：允许更高 count，但仍受硬预算限制。
- `official_source`：鼓励使用 `domains` 或优先读取官网、文档、公告。

Prompt 需要明确：搜索结果只是候选；技术文档、新闻、价格、政策类问题优先读取关键页面后再引用。

## 数据流

1. 模型判断需要联网，调用 `web_search`，可带 `count`、`intent`、`domains`、`recency_days`。
2. `WebSearchHandler` 归一化参数，检查本轮预算，调用 search-service。
3. search-service 继续负责 provider、缓存、fallback 和 Firecrawl/Brave 适配。
4. `WebSearchHandler` 返回候选来源，并把请求参数、实际返回数、预算信息写入 `SearchBlock` 和工具日志。
5. 模型从候选里选择关键 URL，调用 `url_read`，可带 `reason`。
6. `UrlReadHandler` 复用现有 URL 安全策略和 reader-service，记录读取原因和结果。
7. 模型最终回答时，引用优先来自 `url_read` 正文；搜索摘要只作为候选或补充来源。
8. 前端回答依据继续聚合搜索和读取来源；联网诊断展示本轮联网预算和工具参数。

## 前端展示

回答依据主区域：

- 仍保持轻量预览。
- 搜索来源和读取来源继续聚合为一个“回答依据”入口。
- 如果返回候选超过预览数量，显示未预览数量。

联网诊断侧栏：

- 对 `web_search` 展示 query、intent、请求 count、实际 count、返回数、是否 fallback、是否触顶。
- 对 `url_read` 展示 URL、reason、读取状态、耗时。
- 汇总区展示 search 次数、url_read 次数、候选总数、上下文使用数。

## 测试策略

后端：

- `web_search` 工具 schema 包含新增参数。
- `WebSearchHandler` 对 count、intent、domains、recency_days 做归一化和 clamp。
- `WebSearchHandler` 调用 search-service 时透传 count、domains 和 freshness。
- 超过 search 次数或候选预算时返回 degraded 或裁剪结果。
- `url_read` 工具 schema 包含 reason。
- `UrlReadHandler` 截断 reason，并写入 `UrlBlock` / 工具日志数据。

前端：

- hydration 保留新增 SearchBlock/UrlBlock metadata。
- view model 能从多个 search/read block 派生诊断展示数据。
- 诊断面板展示 count、intent、返回数、读取 reason 和触顶信息。

集成验证：

- 不启动本地 Fusion 服务。
- 使用单元测试覆盖 schema、handler、hydration、诊断模型。
- 部署后通过 dev 环境实际联网问题验证工具调用、回答依据和诊断显示。

