# Evidence Ledger 来源使用判定增强设计

## 背景

Search / Read Planner v1.1 已经控制了搜索次数和深读数量，但回答依据仍主要表达“候选来源”和“深读来源”。当前 `used_by_final_answer` 字段几乎一直是 `false`，前端也没有用它区分“最终回答真的用了哪些来源”。这会让用户看到很多候选来源，却不清楚哪些来源实际支撑了正文结论。

## 目标

让 Evidence Ledger 在最终回答完成时，对本轮可用来源做一次保守的“是否被最终回答使用”判定，并把判定结果通过现有 `evidence_item_upserted` 协议写回。前端优先展示 `status=used` 或 `usedByFinalAnswer=true` 的来源，把其他候选/深读来源作为“候选来源”而不是“已使用来源”处理。

## 非目标

- 不引入额外 LLM 二次判定。
- 不做自然语言语义相似度归因。
- 不改变搜索、读取、候选排序策略。
- 不改数据库 schema；继续使用现有 agent progress snapshot。

## 后端规则

新增 `final_answer_evidence` 纯函数模块：

1. 从当前 assistant 消息 content blocks 收集 `search.source_refs` 和 `url_read.source_refs` 中的成功来源。
2. 用 `stable_web_evidence_id(url)` 复用现有 evidence id，保证 upsert 能覆盖 candidate/selected/read_success。
3. 判定 used 的优先级：
   - 正文出现 `[n]` 或 `⟦n⟧` 时，按搜索来源的展示顺序把第 n 个搜索来源标记为 used。
   - 正文出现来源 URL 或 canonical URL 时，标记对应来源。
   - 正文出现域名时，只有该域名在候选中唯一对应一个来源，才标记该来源。
   - 如果没有显式命中，但本轮只有一个成功 `url_read` 来源，且正文非空，则把该深读来源标记为 used。
4. 命中的来源通过 `evidence_item_upserted` 发送 `status="used"`、`used_by_final_answer=true`。
5. 判定失败不阻断主回答完成，只记录 warning。

## 前端规则

1. `AnswerEvidenceModel` 增加 `usedItems` 和 `candidateItems`。
2. `deriveAnswerEvidence` 优先读取 `currentRun.evidence`：
   - `usedByFinalAnswer=true` 或 `status="used"` -> used。
   - `candidate/selected/read_success` 且未 used -> candidate。
   - `read_degraded/read_failed/discarded` 仍走 issue 展示。
3. 回答依据摘要仍保持低干扰：主卡片显示 `已使用 N 条`、`候选 M 条`、`深读 K 个网页`。
4. 侧边栏把 used 来源标题显示为“已使用来源”，候选来源单独显示，避免把所有搜索候选都当成正文依据。
5. 没有 agent evidence 的历史数据继续走旧的 source_refs fallback。

## 验收用例

| Case | 输入 | 预期 |
| --- | --- | --- |
| ELS-01 | 最终回答含 `[1]`，有 3 条搜索来源 | 只把第 1 条搜索来源标记 used |
| ELS-02 | 最终回答含 `⟦2⟧` | 第 2 条搜索来源标记 used |
| ELS-03 | 最终回答含唯一来源 URL | 对应来源标记 used |
| ELS-04 | 最终回答含唯一域名 | 对应来源标记 used |
| ELS-05 | 同域名多个来源且只出现域名 | 不用域名做 used 判定 |
| ELS-06 | 没有显式引用但只有 1 个成功深读来源 | 该深读来源作为 used 兜底 |
| ELS-07 | 没有显式引用且多个候选来源 | 不标记 used，只展示候选 |
| ELS-08 | 前端收到 used + candidate evidence | 侧边栏分成“已使用来源”和“候选来源” |
| ELS-09 | 历史消息没有 agent evidence | 继续用 source_refs fallback 展示回答依据 |
