# 2026-07-13 长对话 Context 管理调研基线

## 结论

- 当前生产数据**没有复现真实长对话把模型窗口塞满**：990 个会话中，用户轮数最大为 8，达到 5 轮的会话只有 2 个，样本不足以证明真实用户已经遭遇长对话超窗。
- 当前代码已经**确定性复现 Context 管理缺口**：纯文本历史会被完整拼入模型请求，没有按模型窗口做 Token 预算、裁剪、滚动摘要或超窗降级。用 4,096 Token 的假设预算构造 5,641 Token 输入后，12/12 条历史仍全部保留。
- 生产库持久化的 `messages.usage.input_tokens` 不能直接当作“单次上下文长度”。Agent 回复会累计多个 LLM round 的 prompt token；现有高值主要由 Agent 多轮调用产生，而不是长对话历史。
- 当前可观测性不足：没有保存每个 LLM round 的 prompt token、模型窗口、预算占用比例和上下文组成，因而无法从生产数据准确回答“某次请求用了窗口的多少”。
- 当前模型目录也不足以直接驱动预算：生产 `/api/models/` 返回 14 个模型，只有 4 个带 `contextWindowTokens`，其余 10 个为空。
- 在设计 Context Manager 前，应先补最小请求级可观测性，再执行真实模型阶梯实验；否则容易用累计 Agent usage 误判单次 Context，或用理论窗口替代真实供应商行为。

本轮没有调用真实 LLM、没有写入生产数据、没有读取消息正文或用户身份，也没有修改业务行为。

## 调研范围与方法

本轮使用三条证据链：

1. **生产数据库只读盘点**：聚合 `messages`、`conversations`、`agent_sessions`，只读取数量、Token、步骤、工具次数、状态和耗时；所有查询均在只读事务中执行并回滚。
2. **线上日志信号**：在 Loki 中查询最近 30 天的 Context 超限关键词和 LLM 调用失败信号，不展开用户消息正文。
3. **确定性小窗口复现**：不启动服务、不调用真实模型，直接调用 `build_llm_messages()` 构造不同轮数的纯文本历史，并用 `cl100k_base` 做量级估算。

生产数据时间范围为北京时间 **2026-03-26 14:11:06 至 2026-07-12 17:55:40**。

## 当前实现事实

### 同一会话历史会全量进入请求

- `Conversation.messages` 按创建时间加载全部消息。
- 新消息持久化后，`chat_service` 把完整 `conversation.messages` 交给生成任务。
- `build_llm_messages()` 逐条遍历消息；文本块直接追加，只有空内容以及 `thinking/search` 等非文本块被过滤。
- `llm_call_with_retry()` 将构造后的 `messages` 原样交给 LiteLLM。

当前唯一明确的历史轮次限制是图片：历史图片仅保留最近 3 个用户轮次；该限制不作用于文本。

### `limit_summary` 不是历史压缩

`limit_summary` 只在 Agent 达到步骤、工具或时间上限后追加一个 system prompt，再执行一次无工具最终回答。它不会替换旧消息、保存滚动摘要或降低后续请求的历史 Token。

### 当前长上下文验收不是真实长输入

`long_context_contract` 场景只要求模型在同一条短问题里复述 `FUSION-LONG-CONTEXT-V1`，并检查模型带 `longContext` capability。它没有构造长文本、多轮历史或窗口边界；场景的 `60_000` 是超时毫秒，不是 Token 数。

### 代码证据定位

- 会话消息关系：`app/db/models.py:55-70`。
- 全量历史进入生成任务：`app/services/chat_service.py:268-289`。
- 历史消息构造与最近 3 轮图片边界：`app/services/chat/message_builder.py:19-20,61-162`。
- LiteLLM 请求边界：`app/services/stream/llm_stream.py:328-352`。
- Agent 触顶总结：`app/services/stream/limit_summary.py:68-85,194-230`。
- 当前长上下文契约场景：`scripts/model_catalog_eval_baseline.py:191-197`。

## 生产数据库基线

### 数据覆盖

| 指标 | 结果 |
|---|---:|
| 消息总数 | 2,197 |
| 会话总数 | 990 |
| Assistant 消息 | 1,086 |
| 带数值 `input_tokens` 的 Assistant 消息 | 1,063（97.9%） |
| 有可测 Assistant usage 的会话 | 968 |

### 真实会话轮数

| 指标 | 用户轮数 |
|---|---:|
| P50 | 1 |
| P90 | 1 |
| P95 | 2 |
| P99 | 3 |
| 最大值 | 8 |

分桶结果：

| 用户轮数 | 会话数 |
|---|---:|
| 1–2 | 965 |
| 3–5 | 23 |
| 6–10 | 2 |
| 10 轮以上 | 0 |

结论：生产库目前主要是单轮或极短多轮会话，不能用它直接验证 20、40、80 轮历史的行为。

### 持久化 Input Token

| 指标 | `input_tokens` |
|---|---:|
| P50 | 1,978 |
| P90 | 19,928 |
| P95 | 33,044 |
| P99 | 68,593 |
| 最大值 | 140,395 |
| 平均值 | 7,094 |

这些数值不能直接解释为单次 Context 占用，原因是 Agent 多个 LLM round 的 usage 会累加进同一条 Assistant 消息。

在 1,063 条可测回复中，791 条能明确匹配 Agent session。高 Token 样本大部分属于 Agent 累计：

- `input_tokens >= 10K` 的 189 条中，162 条匹配 Agent；171 条来自只有 1 个用户轮次的会话。
- `input_tokens >= 40K` 的 41 条中，34 条匹配 Agent；39 条来自只有 1 个用户轮次的会话。
- 达到 100K 的 3 条不能据此认定单次 Prompt 已到 100K，因为其中包含 Agent 累计或旧历史口径不完整的数据。

### Token 与 Agent 总耗时的支持性信号

下表只用于说明“累计工作量越大，总耗时通常越高”，不能证明单次 Context 导致延迟；工具次数、输出长度和供应商也会影响结果。

| 持久化累计 Input Token | 样本 | 平均总耗时 | P50 | P95 |
|---|---:|---:|---:|---:|
| <2K | 453 | 8.9 s | 6.2 s | 24.5 s |
| 2K–5K | 132 | 14.0 s | 11.2 s | 35.7 s |
| 5K–10K | 44 | 23.2 s | 18.3 s | 54.2 s |
| 10K–20K | 70 | 31.7 s | 29.9 s | 56.1 s |
| >=20K | 92 | 57.5 s | 47.9 s | 108.7 s |

## 模型窗口元数据

生产 `/api/models/` 当前返回 14 个模型：

- `deepseek-chat`、`deepseek-reasoner`：`contextWindowTokens=1,000,000`。
- `gemini-3.1-pro-preview`：`1,048,576`。
- `kimi-k2.5`：`262,144`。
- 其余 10 个模型未返回 `contextWindowTokens`。

已知的 4 个窗口确实都很大，因此“不容易靠普通对话塞满”的担心对这些模型成立；但其余 10 个模型窗口未知，不能据此推断整个平台的窗口都足够大。这里同时暴露出另一个前置缺口：当前无法为大多数模型建立可信预算。目录元数据需要先明确来源、缺失回退和可信度，不能直接把 UI 的“长上下文”标签当作运行时硬限制。

## 线上错误信号

Loki 与 `agent_sessions` 最近 30 天聚合结果（北京时间 2026-06-13 09:22 至 2026-07-13 09:22）：

| 信号 | 数量 |
|---|---:|
| `fusion-api` / `litellm-proxy` 精确 Context 超限关键词 | 0 |
| Prompt/Input 过长、Request/Payload 过大或 HTTP 413 | 0 |
| Agent completed | 811 |
| Agent error | 3 |
| Agent interrupted | 2 |
| Agent limit_reached | 3 |
| Loki `BadRequestError` 日志 | 6（均为参数校验，无 Context 超限） |

3 个 Agent error 的脱敏分类也没有命中 Context overflow、请求过大、超时、限流、鉴权或连接错误，暂归类为“其他”。数据库与 Loki 数量并不完全一致，因此失败事实以 `agent_sessions` 为主，日志只作辅助。

这说明当前线上没有发现明确的 `context_length_exceeded`、`maximum context`、`too many tokens` 等错误，但不能据此证明不存在风险：真实长对话样本很少，且供应商错误文案可能未命中现有关键词。

当前也不能把 Token 与 TTFT 关联：消息只保存累计 input/output token，不保存 TTFT；Loki 最近 30 天没有 `PERF_JSON` 样本；历史压测只保存 TTFT 汇总，不包含每条请求的 Context 大小。用 user/assistant 创建时间差估算总完成耗时会被模型、输出长度、工具调用和排队严重混杂，不能替代 TTFT。

## 确定性复现

### 历史规模阶梯

测试为每轮构造一条用户消息和一条 Assistant 消息，并在首尾放置 marker。结果如下：

| 用户轮数 | 历史消息数 | 保留结果 | 估算 Token |
|---:|---:|---:|---:|
| 5 | 10 | 10/10 | 4,607 |
| 10 | 20 | 20/20 | 9,197 |
| 20 | 40 | 40/40 | 18,377 |
| 40 | 80 | 80/80 | 36,737 |
| 80 | 160 | 160/160 | 73,457 |

所有档位的首尾 marker 都存在，证明文本历史没有被裁剪或摘要。

### 4K 小窗口复现

主 Agent 独立复跑结果：

| 指标 | 结果 |
|---|---:|
| 假设输入预算 | 4,096 Token |
| 历史消息 | 12 |
| 加入固定 system 消息后的输出消息 | 14 |
| 估算输入 | 5,641 Token |
| 超出预算 | 1,545 Token |
| 历史保留 | 12/12 |

当前函数不知道模型预算，即使明确超出 4K，仍会完整返回消息。这个用例可以直接转成后续 Context Manager 的首条失败测试。

### 复现边界

- `cl100k_base` 只用于量级估算，不是所有供应商的精确 tokenizer。
- 测试固定了日期和身份 prompt；生产 prompt 更长，实际输入只会更多。
- 消息转换本身非常快，5–80 轮的纯 Python 构建中位数约 0.002–0.049 ms。真正的风险在 Token 化、请求体、网络、供应商预填充、重复 Agent round 和费用，不在 Python 循环本身。
- 本轮没有验证模型在长上下文中的召回质量，也没有制造真实供应商超窗错误。

## 已证明与尚未证明

### 已证明

1. 当前代码没有文本历史 Token 预算、裁剪或滚动摘要。
2. 在任意较小模型窗口下，当前构造器可以确定性地产生超预算请求。
3. 生产 telemetry 不能区分“单次 Prompt Context”与“多个 Agent round 累计 Input Token”。
4. 当前生产数据缺少足够长的真实多轮会话。
5. 大多数生产模型缺少可用的 Context Window 元数据。

### 尚未证明

1. 当前真实用户已经因为长对话遇到 Context 超限。
2. 在 20K、40K、80K 单次 Prompt 下，各供应商 TTFT、费用和召回质量如何变化。
3. 不同供应商对超窗的真实错误格式和降级行为。
4. Context 压缩后对回答质量的净收益。

## 实施前建议门禁

### OBS-01：先补请求级可观测性

每个 LLM round 记录以下脱敏指标，不记录 Prompt 正文：

- `conversation_id`、`message_id`、`run_id`、`round_index`。
- `model_id`、供应商、模型窗口来源和可信状态。
- 本轮 `prompt_tokens`、`completion_tokens`，而不是只存 run 累计值。
- 请求消息数、用户/Assistant/System/Tool 消息数。
- 文本、图片、文件、工具 Context 的估算 Token 分桶。
- `context_window_tokens`、预留输出预算、占用比例。
- TTFT、总耗时、finish reason、错误类型。

### REP-01：把 4K 缺口固化成失败测试

在不调用真实 LLM 的情况下，注入 `context_window_tokens=4096`，构造约 5.6K 的历史。Context Manager 实现前，测试应证明当前超预算；实现后应断言：

- System Prompt 与最新用户消息必保留。
- 输入稳定落入预算。
- 不拆坏 user/assistant 轮次。
- 裁剪、摘要或降级结果可观测。

### REP-02：可观测性上线后再跑真实阶梯

使用专用评估会话，建议按 5K、10K、20K、40K 和模型窗口 60%–80% 逐档执行：

- 在早期、中间、最近位置加入随机 canary。
- 每档只要求短回答，降低输出成本。
- 记录真实 per-round prompt token、TTFT、总耗时、召回结果和错误。
- Agent 与无工具普通对话分开测试，避免累计 usage 混淆。

该实验会创建生产会话并产生真实模型费用，本轮报告阶段没有执行。

## 决策建议

Context 管理问题已经在代码层确定性复现，但生产用户影响尚未被真实长对话数据证明。建议按以下顺序推进：

1. 请求级 Context 可观测性。
2. 4K 确定性失败测试。
3. 真实模型无工具阶梯与 Agent 阶梯。
4. 根据结果设计 Token-aware Context Manager。
5. 最后再评估 rolling summary、工具/文件摘要恢复和跨会话 Memory。

不要先恢复历史自动 Memory。当前最明确的问题是单会话 Context 缺少预算和治理，而不是跨会话召回能力不足。
