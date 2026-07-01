# Search Failure Recovery / Budget v1.3 设计

## 背景

Search / Read Planner v1.2 已经能控制搜索条数、搜索轮次、重复 query、候选排序和读取推荐，但预算器仍只知道“已经搜过几次、query 是否重复、intent 是什么”。它不知道上一轮搜索是否真的产出了可用候选，也不知道网页读取失败后是否还有未读取的高价值候选。

因此线上会出现两类体验问题：

- 搜索失败或弱结果后，系统只能按普通 follow-up 搜索解释，无法明确这是一次 query repair。
- 读取网页失败后，LLM 可能马上再次搜索，而不是先读取已有候选里的替代来源。

v1.3 的目标是补齐失败恢复状态，让系统在不引入复杂 LLM planner 的情况下，对“继续搜还是先换源读”做可测试的确定性约束。

## 目标

1. 搜索结果为空或明显过弱时，允许一次 query repair，并用结构化 action/reason code 标记。
2. 网页读取失败后，如果已有未读取候选可用，阻止马上再次 provider search，引导 LLM 先读取替代候选。
3. 普通实时问题保持默认 1-2 次 provider search；第三次 provider search 只允许 deep_research 或明确失败恢复场景。
4. 离线 eval 能区分普通 provider search、repair search、重复搜索跳过、读取失败后的搜索重定向。
5. 保持现有 advisory 架构，不自动执行 `url_read`，不改前端 UI，不迁移 PromptHub。

## 非目标

- 不做全量自动阅读。
- 不引入 embedding reranker、LLM reranker 或多模型投票。
- 不把搜索策略扩展成长期 research pipeline。
- 不新增数据库表。
- 不启动本地 Fusion 服务做验证。
- 不打开新的 Chrome、新标签、新窗口或 isolated context。

## 核心设计

### 预算器新增反馈状态

`NetworkToolBudget` 继续作为单次 assistant run 的联网预算状态容器，新增两类反馈：

- 搜索质量反馈：记录上一轮 provider search 的状态、结果数和是否弱结果。
- 读取候选反馈：记录已推荐或保留的候选 URL、已尝试读取 URL、读取失败 URL 和剩余未读候选数。

反馈由 `tool_round` 在工具执行完成后调用预算器更新，预算器不直接依赖数据库或 emitter。

### SearchBudgetDecision 新增 action/reason

新增 action：

- `repair_search`：允许真实 provider search，但这是对上轮空/弱搜索的修复搜索。
- `redirect_to_read_alternative`：不调用 provider search，要求先读取已有替代候选。

新增 reason code：

- `previous_search_no_results`
- `previous_search_weak_results`
- `read_alternatives_available`

`repair_search` 计入 provider search；`redirect_to_read_alternative` 不计入 provider search，也不消耗搜索预算。

### 读取失败优先换源

当满足以下条件时，下一次 `web_search` 会被预算器降级为 `redirect_to_read_alternative`：

- 本轮已有至少一次 `url_read` 失败或降级。
- `SourceSelectionPlan` 中仍存在未尝试读取的推荐或保留候选。
- 当前请求不是硬性读取某个 URL，而是普通 `web_search`。

降级结果会进入 LLM tool context，提示“先读取已有候选中的替代来源，不要马上继续搜索”。

### Query repair 只允许一次

当上一轮 provider search 没有可用结果，或有效结果数低于弱结果阈值时，下一次非重复搜索可被标记为 `repair_search`。

约束：

- 每个 assistant run 最多一次 repair search。
- repair search 使用小预算，避免一次失败后扩大搜索面。
- repair search 后仍遵守普通 planner limit；不会无限重试。

## 测试矩阵

| 编号 | 场景 | 自动化断言 |
| --- | --- | --- |
| V13-01 | 首次搜索空结果后第二次搜索 | 第二次 action 为 `repair_search`，reason 为 `previous_search_no_results`，provider count 增加 |
| V13-02 | 首次搜索弱结果后第二次搜索 | 第二次 action 为 `repair_search`，reason 为 `previous_search_weak_results`，使用小预算 |
| V13-03 | 读取失败且还有未读候选时再次搜索 | 搜索被降级为 `redirect_to_read_alternative`，不消耗 provider budget |
| V13-04 | 读取失败但候选已耗尽时再次搜索 | 不触发 redirect，可按原 planner limit 判断 |
| V13-05 | repair 后继续第三次普通搜索 | 第三次被 `limit_planner` 拦截 |
| V13-06 | 离线 eval observation | 能断言 `expected_search_actions`、`required_search_actions`、`max_repair_search_calls` |

## 验收

自动化验收：

- `test/services/stream/test_network_budget.py`
- `test/services/stream/test_tool_round.py`
- `test/test_agent_behavior_eval.py`
- `scripts/agent_behavior_eval.py --dry-run`
- 相关全量 pytest、ruff。

真实回归：

- 合并部署后，只复用用户已打开且登录的 `fusion.seanfield.org` 标签。
- 至少覆盖直接回答、普通实时搜索、搜索失败/弱结果恢复、读取失败换源路径。
- 记录 case id、输入、对话 URL、预期、实际、console error、刷新后结果和结论。
