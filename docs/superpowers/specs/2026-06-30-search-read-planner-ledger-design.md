# Search / Read Planner + Evidence Ledger 最小版设计

## 背景

Fusion 现有联网 agent 已具备 `web_search`、`url_read`、动态搜索预算和 `SourceCandidateRanker`。当前仍有两个结构性缺口：

1. 搜索结果选择建议主要以文本形式注入给 LLM，系统本身没有把“候选、推荐、读取成功/失败”统一成可追踪对象。
2. 前端“执行过程”和“回答依据”分别从 tool call、content block、agent evidence 等来源派生，容易出现数字、状态和来源列表不一致。

本设计的目标不是一次性实现完整 agent control plane，而是落地最小可验证版本：LLM 仍可自由判断，后端提供结构化建议和证据账本，前端消费统一 evidence 生命周期。

## 目标

- 建立最小 Evidence Ledger：同一 URL 在搜索候选、推荐深读、读取成功/失败之间使用同一个稳定 evidence id。
- 将 `SourceCandidateRanker` 的推荐结果从纯文本建议升级为结构化 evidence upsert 事件。
- 让 `url_read` 的成功、降级、失败也写入同一 evidence ledger。
- 保持 LLM advisory 模式：系统只建议优先读哪些来源，不禁止 LLM 读取其他来源。
- 保持历史恢复：刷新后仍能从 `agent_progress_snapshots.state.evidence` 还原 evidence 状态。
- 不新增数据库表，不做 PromptHub 迁移，不强制改造 `url_read` 调用方式。

## 非目标

- 不实现独立 LLM planner step。
- 不把 `url_read` 从 LLM tool call 改成系统自动执行。
- 不实现 citation audit 或最终答案事实核验。
- 不新增持久化 ledger 表。
- 不重做回答依据 UI 视觉结构，只补齐统一数据模型和最小展示语义。

## 术语

- `candidate`：搜索返回的候选来源。
- `selected`：系统建议优先深读的来源。
- `read_success`：已成功读取网页内容。
- `read_degraded`：读取降级，部分或全部内容不可用。
- `read_failed`：读取失败。
- `used`：未来最终答案引用审计使用，本次不主动生成。
- `discarded`：低优先级或不可用来源，本次只保留兼容能力，不主动大规模生成。

## 后端设计

### Evidence ID

新增稳定 evidence id 规则：

```text
ev-web-{sha1(canonical_url)[:12]}
```

URL canonical 规则：

- scheme/host 小写。
- 去除 `www.`。
- 去除 fragment。
- 去除常见 tracking query 参数。
- 保留非 tracking query 参数，排序后重组。
- URL 非法时回退到旧的 `tool_call_id + index` 形式。

这样同一 URL 来自多个搜索或后续 `url_read` 时，会更新同一条 evidence，而不是生成多个重复卡片。

### Evidence 状态扩展

后端 `AgentEvidenceItem.status` 从当前：

```text
candidate | used | discarded
```

扩展为：

```text
candidate | selected | read_success | read_degraded | read_failed | used | discarded
```

兼容策略：

- 前端旧逻辑只把 `discarded` 当作隐藏，其他状态都可展示。
- snapshot reducer 继续按 `id` upsert。
- `used_by_final_answer` 保留，未来 citation audit 再使用。

### Search Candidate Evidence

`web_search` 成功后，每个搜索结果生成 `candidate` evidence：

```json
{
  "id": "ev-web-...",
  "kind": "web",
  "status": "candidate",
  "title": "...",
  "url": "...",
  "domain": "...",
  "claim": "搜索摘要",
  "snippet": "搜索正文摘要",
  "used_by_final_answer": false
}
```

### Source Selection Evidence

同一 tool round 内，`SourceCandidateRanker` 对成功搜索结果排序后，对 `recommended` 候选额外发出 `selected` upsert：

```json
{
  "id": "ev-web-...",
  "kind": "web",
  "status": "selected",
  "title": "...",
  "url": "...",
  "domain": "...",
  "claim": "建议深读：官方来源 / 权威媒体 / 高相关",
  "snippet": "来自搜索 query: ...",
  "used_by_final_answer": false
}
```

这不会阻止 LLM 读取其他来源，只提供结构化推荐。

### URL Read Evidence

`url_read` 完成后按 URL upsert：

- success -> `read_success`
- degraded -> `read_degraded`
- failed -> `read_failed`

同一 URL 如果之前是 `selected`，读取完成后覆盖成读取状态；如果之前不是候选，也会作为新 evidence 记录，避免 LLM 自主读取的 URL 丢失。

### LLM Context

继续保留当前文本版 `结构化来源选择建议`，因为它直接影响 LLM 决策。新增 ledger 事件只服务 UI、历史恢复和后续评估。

### Progress Snapshot

`agent_progress_snapshots.state.evidence` 继续作为历史恢复来源。需要更新 reducer：

- 允许新 evidence status。
- cap 规则优先保留 `used`、`read_success`、`selected`，再保留普通 candidate。

## 前端设计

### Type 扩展

`AgentEvidenceItem.status` 扩展为：

```ts
'candidate' | 'selected' | 'read_success' | 'read_degraded' | 'read_failed' | 'used' | 'discarded'
```

### 展示规则

最小版不重做 UI，只做语义兼容：

- 执行过程侧栏仍展示搜索关键词、读取过程和“查看依据”入口。
- 回答依据侧栏继续展示实际来源列表。
- EvidenceDigest 可以基于 tool digest 保持现状，不新增大面积信息。
- 如果后续要展示 ledger 状态，优先映射：
  - `selected`：建议深读
  - `read_success`：已深读
  - `read_degraded/read_failed`：默认折叠为不可用来源

## 测试策略

### 后端单元测试

- 同一 URL 的 search candidate 和 url_read 使用同一个 evidence id。
- `web_search` evidence 默认为 `candidate`。
- ranker 推荐候选会 emit `selected` evidence upsert。
- `url_read success/degraded/failed` 分别 upsert `read_success/read_degraded/read_failed`。
- progress snapshot reducer 接受并保留新状态。
- cap evidence 时优先保留 `read_success/selected`，不被普通 candidate 挤掉。

### 前端单元测试

- stream event handler 能接收新 evidence status。
- hydration 能恢复新 evidence status。
- execution process model 不把新状态误判为不可展示。

### 真实回归

部署后复用现有登录 Chrome 标签，新建真实对话覆盖：

1. 官方公告 + 权威媒体对照。
2. 预期至少出现 search candidate evidence。
3. 被 ranker 推荐的来源进入 selected/read_success/read_degraded 生命周期。
4. 执行过程和回答依据数字不互相矛盾。
5. 刷新后 evidence、执行过程、回答依据恢复正常。
6. Console error 为空。

## 风险与回滚

- 风险：前端未覆盖新 status 导致类型或展示异常。缓解：前端类型和 hydration 测试同步更新。
- 风险：稳定 evidence id 改变 digest `source_refs`，可能影响执行过程来源补齐。缓解：测试 `tool_result_digest.source_refs` 和 `run.evidence` 能按新 id 关联。
- 风险：selected 后被 read_failed 覆盖，用户看不到曾被推荐。最小版接受，未来完整 ledger 可引入 `stage_history`。
- 回滚：移除新状态 emit 和稳定 id，恢复旧 `ev-{tool_call_id}-{index}`，不会影响工具主链路。
