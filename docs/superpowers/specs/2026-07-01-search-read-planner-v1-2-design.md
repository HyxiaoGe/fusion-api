# Search / Read Planner v1.2 评估与约束增强设计

## 背景

Search / Read Planner v1.1 已经完成了基础控制面：

- 后端接管搜索 `count`，不再让 LLM 自己决定固定 8 条或 10 条。
- `NetworkToolBudget` 能识别重复搜索、相似 follow-up 搜索和普通搜索轮次上限。
- `SourceCandidateRanker` 能对搜索候选做确定性排序，优先官方、原文、权威媒体和高相关来源。
- `search_read_planner` 已按 intent/budget 给出推荐深读数量。
- Evidence Ledger 已能承接 `candidate`、`selected`、`read_success`、`read_degraded`、`read_failed`、`used` 状态。

v1.1 的主要限制是：这些判断虽然已经影响 LLM 上下文和 UI 依据，但系统还没有把“为什么允许这次搜索”“为什么推荐这些来源”“为什么没有读剩余来源”统一成可评估的结构化决策。结果是线上真实对话里一旦出现“两次搜索、只深读 3 个网页、还有一些未读候选”，用户只能看到结果数字，很难理解系统的选择逻辑，开发侧也缺少稳定评估样本防止回退。

## 目标

v1.2 的目标是把 Search / Read Planner 从“有规则但解释分散”推进到“规则可解释、可离线评估、可回归验证”：

1. 建立最小 Search / Read Decision Ledger，记录搜索预算决策、候选推荐决策和未推荐原因。
2. 强化 SourceCandidateRanker 的输出结构，让推荐和未推荐都带稳定 reason code，而不是只给 LLM 一段自然语言。
3. 扩展离线 agent 行为评估集，让搜索轮次、搜索预算、推荐深读数量、低优先级未读来源都能被测试。
4. 保持 advisory 模式：LLM 仍可自主发起工具调用，但后端能收敛明显机械或重复的调用，并能解释收敛原因。
5. 为后续 PromptHub 迁移提供稳定策略文本和测试基线，但本次不做 PromptHub 接入。

## 非目标

- 不实现独立 LLM planner step。
- 不把 `url_read` 改成后端自动执行。
- 不要求所有搜索结果都深读。
- 不新增数据库表。
- 不新增前端大面积 UI。现阶段以后端结构和测试为主，必要时只复用已有 evidence/progress 字段。
- 不引入 LLM reranker、embedding reranker 或外部搜索评测服务。
- 不把策略迁移到 PromptHub；等规则稳定后再迁移。

## 设计原则

### 只在确定性边界做强约束

代码层面只负责确定性、可测试的事情：

- 搜索轮次上限。
- 重复 query 拦截。
- 相似 follow-up 缩小预算。
- 候选 URL 去重。
- 来源类型和域名启发式评分。
- 推荐深读数量上限。

代码层面不强行判断“最终答案事实是否正确”，也不替 LLM 选择所有阅读动作。

### 解释必须面向用户和测试双重可用

同一个决策要能同时服务两类消费者：

- 给 LLM 的 guidance：自然语言、短、能改变下一轮工具行为。
- 给测试和未来 UI 的 ledger：结构化字段、稳定 reason code、可断言。

### 不把未读候选当异常

搜索 10 条结果但只读 2-3 条是正常设计，不是失败。v1.2 要明确表达“未读”的原因，例如：

- `covered_by_recommended_source`：已有更高质量来源覆盖。
- `low_priority_source_type`：视频、论坛、社交来源默认低优先级。
- `duplicate_url`：同 URL 或 canonical URL 重复。
- `outside_read_limit`：超过当前意图的推荐深读上限。
- `search_summary_sufficient`：搜索摘要足够回答，深读收益低。

## 后端设计

### SearchBudgetDecision

新增轻量结构，用于描述每次 `web_search` 参数归一化后的预算决策。

字段：

```python
@dataclass(frozen=True)
class SearchBudgetDecision:
    query: str
    intent: str | None
    action: str
    budget_name: str
    requested_count: int
    context_source_limit: int
    reason_code: str
    previous_query_count: int
    planned_search_limit: int
```

`action` 取值：

- `execute`：允许真实 provider 搜索。
- `narrow_followup`：允许真实搜索，但使用 follow-up 小预算。
- `skip_duplicate`：跳过重复搜索。
- `limit_planner`：超过 planner 轮次上限，不调用 provider。
- `limit_budget`：达到硬预算上限，不调用 provider。

`reason_code` 取值：

- `initial_search`
- `complementary_search`
- `similar_followup`
- `duplicate_query`
- `planned_search_limit_reached`
- `hard_search_limit_reached`

实现上不需要大改 `NetworkToolBudget.prepare_web_search_args()` 的公开返回值，可以新增内部纯函数或把 decision 存进 `ToolResult.data["budget_decision"]`，保证现有调用方兼容。

### SourceReadDecision

扩展 `SourceSelectionPlan`，为所有去重后的候选提供读取决策。

字段：

```python
@dataclass(frozen=True)
class SourceReadDecision:
    candidate: RankedSourceCandidate
    action: str
    reason_code: str
```

`action` 取值：

- `recommend_read`
- `keep_candidate`
- `deprioritize`

`reason_code` 取值：

- `official_original`
- `official_document`
- `authority_media`
- `high_relevance`
- `covered_by_recommended_source`
- `low_priority_source_type`
- `outside_read_limit`

`SourceSelectionPlan` 增加：

```python
read_decisions: tuple[SourceReadDecision, ...]
decision_summary: dict[str, int]
```

兼容策略：

- 现有 `recommended`、`low_priority`、`not_recommended_count` 保留。
- guidance 仍用现有格式，但从 `read_decisions` 汇总未推荐原因。
- Evidence Ledger 最小只继续 emit `selected`；未推荐来源先进入测试和 LLM guidance，不新增前端列表。

### Search / Read Decision Ledger

新增 `app/services/search_read_decision_ledger.py`，只做纯函数聚合，不依赖数据库。

职责：

1. 从 `ToolExecutionRecord` 收集成功、降级、被 planner 限制或重复跳过的搜索决策。
2. 从 `SourceSelectionPlan` 收集推荐和未推荐来源决策。
3. 生成给离线评估使用的 summary。

输出示例：

```python
{
    "search_decisions": [
        {
            "query": "OpenAI 最新公告 2026年7月",
            "action": "execute",
            "budget_name": "freshness",
            "reason_code": "initial_search",
        }
    ],
    "read_decisions": [
        {
            "url": "https://openai.com/index/...",
            "domain": "openai.com",
            "action": "recommend_read",
            "reason_code": "official_original",
        }
    ],
    "summary": {
        "executed_search_count": 1,
        "recommended_read_count": 2,
        "deprioritized_count": 3,
    },
}
```

### LLM guidance 调整

`format_source_selection_guidance()` 保留当前结构，但补充两点：

1. “未建议深读”的原因不再只是一句笼统解释，而是按 reason code 汇总，例如：
   - `2 条为社交/视频来源，默认不读`
   - `3 条被更高质量来源覆盖`
   - `4 条超过本轮推荐深读上限`
2. 明确允许 LLM 在必要时读取未推荐来源，但要求有明确收益：
   - “只有当推荐来源无法回答关键事实，才读取未推荐来源。”

### 离线评估增强

扩展 `scripts/agent_behavior_eval.py`，让样本支持以下字段：

```json
{
  "max_search_calls": 2,
  "max_provider_search_calls": 2,
  "expected_search_budgets": ["freshness", "freshness_followup"],
  "max_recommended_reads": 2,
  "forbidden_read_domains": ["youtube.com", "threads.com"],
  "required_decision_reason_codes": ["official_original"]
}
```

评估器仍不调用外部服务。它只消费真实回归或单元测试构造的 observation。

### 真实回归约束

真实 Chrome 回归必须复用用户已打开且已登录的 `fusion.seanfield.org` 标签，不打开新 Chrome、新标签、新窗口、`about:blank` 或 isolated context。

回归输入必须避免“请你联网搜索”这种作弊提示，改用自然问题让模型自主判断：

- “OpenAI 最近发布了哪些产品更新？”
- “微信A2A互通怎么用？”
- “1+1等于几？”
- “你好，你是谁？”

## 测试矩阵

| Case | 场景 | 预期 |
| --- | --- | --- |
| SRP12-01 | 简单闲聊 | 不调用 `web_search`，不展示执行过程或回答依据 |
| SRP12-02 | 简单数学 | 不调用 `web_search`，直接回答 |
| SRP12-03 | 自然语言时效问题，用户没说联网 | LLM 应自主调用 `web_search` |
| SRP12-04 | 第一次 freshness 搜索 | `SearchBudgetDecision.action=execute`，`reason_code=initial_search` |
| SRP12-05 | 相似 follow-up 搜索 | 使用 follow-up 小预算，`reason_code=similar_followup` |
| SRP12-06 | 重复搜索 | 不调用 provider，`action=skip_duplicate` |
| SRP12-07 | 普通问题第三次搜索 | 不调用 provider，`action=limit_planner` |
| SRP12-08 | deep research 第三次搜索 | 允许 provider 搜索 |
| SRP12-09 | 官方原文 + 权威媒体候选 | 优先推荐官方原文和权威媒体 |
| SRP12-10 | 视频/社交候选 | 默认 `deprioritize`，不进入推荐深读 |
| SRP12-11 | 搜索返回 8-10 条候选 | 推荐深读不超过当前 intent 上限 |
| SRP12-12 | 两次搜索结果合并 | 未读候选有 reason code，不被当作异常 |
| SRP12-13 | 离线 eval observation 缺关键词 | 评估失败 |
| SRP12-14 | 离线 eval observation 推荐读数超限 | 评估失败 |
| SRP12-15 | 输出泄漏 `url_read` / `reader-service` | 评估失败 |
| SRP12-16 | 部署后刷新真实对话 | 搜索关键词、依据、执行过程无 running 残留 |

## 风险与回滚

- 风险：结构化决策字段过多，增加维护成本。缓解：只在纯函数和测试中使用，先不扩展前端展示。
- 风险：reason code 过细导致频繁改测试。缓解：保持少量稳定 reason code，把细节放在中文 guidance。
- 风险：LLM 仍可能读取未推荐来源。缓解：本版本只记录和评估，不强制禁止；后续如有必要再引入更强控制器。
- 风险：PromptHub 迁移时策略文本漂移。缓解：把 tool description 和 guidance 的关键规则纳入测试。

回滚方式：

- 移除 `SearchBudgetDecision` / `SourceReadDecision` 的新增字段和评估检查。
- 保留 v1.1 的预算、ranker、planner 逻辑不变。
- 不影响现有对话主链路和 Evidence Ledger 基础状态。
