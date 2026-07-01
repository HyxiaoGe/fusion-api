# Search / Read Planner v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Fusion 的联网链路从“LLM 自己决定搜索/深读，后端只给提示”推进到“后端提供可测试的搜索预算、重复搜索收敛、读源选择和证据解释”。

**Architecture:** 保留 LLM 自主 tool-call 模式，不引入自动读网页的强控制器。后端新增轻量 Search / Read Planner 边界：`NetworkToolBudget` 负责搜索次数和重复搜索拦截，`SourceCandidateRanker` 负责候选评分，`search_read_planner` 负责按意图决定推荐深读数量和生成 LLM guidance，`Evidence Ledger` 继续承接 UI 和历史恢复。

**Tech Stack:** FastAPI service layer, pytest, ruff, existing agent_event/evidence protocol, deployed Fusion regression with existing logged-in Chrome tab only.

---

## Acceptance Matrix

| Case | Input Shape | Expected |
| --- | --- | --- |
| SRP-01 | 简单闲聊/数学/身份问题 | 不调用 `web_search` / `url_read`，不展示执行过程或回答依据 |
| SRP-02 | 自然语言实时问题，用户没有说“联网” | LLM 自主调用 `web_search`，展示搜索关键词和来源 |
| SRP-03 | 中文 query 含 `2026年` | 后端能推断为 freshness，不落到机械 standard 预算 |
| SRP-04 | 第二次搜索与第一次高度重复 | 后端跳过真实搜索，返回用户不可见的重复搜索降级上下文，不消耗 provider |
| SRP-05 | 第二次搜索与第一次相似但不是完全重复 | 后端使用 follow-up 小预算，不再出现两个机械 `5 + 5` |
| SRP-06 | 官方来源 + 权威媒体互补搜索 | 允许第二次搜索，保留更大的互补预算 |
| SRP-07 | 搜索返回 8-10 条候选 | planner 只推荐少量高价值来源深读，并说明“不是全部读取” |
| SRP-08 | quick fact 搜索 | 最多推荐 1 个深读来源 |
| SRP-09 | freshness / standard 搜索 | 最多推荐 2 个深读来源 |
| SRP-10 | official / comparison / deep research 搜索 | 最多推荐 3 个深读来源 |
| SRP-11 | url_read 失败或降级 | evidence 记录读取状态，但 LLM 上下文和用户展示不泄漏 `reader-service` / `url_read` |
| SRP-12 | 刷新已完成对话 | 搜索关键词、回答依据、执行过程摘要保持一致，无 running 残留 |
| SRP-13 | 普通实时问题被模型机械扩成 3+ 次搜索 | 后端默认最多执行 2 次真实搜索；第三次返回 `search_plan_limited`，不消耗 provider，不进入回答依据 |
| SRP-14 | deep_research 深度研究 | 最多执行 3 次真实搜索；第四次返回 `search_plan_limited` |

## File Map

- Modify: `app/services/search_budget.py`
  - 修正中文年份识别。
  - 提供重复搜索判定和相似 follow-up 判定。
- Modify: `app/services/stream/network_budget.py`
  - 对重复搜索直接返回 degraded `ToolResult`，不调用搜索 provider。
  - 对相似 follow-up 使用更小预算。
- Create: `app/services/search_read_planner.py`
  - 根据搜索结果 intent / budget 计算推荐深读上限。
  - 调用 ranker 生成 `SourceSelectionPlan`。
  - 生成面向 LLM 的读源 guidance。
- Modify: `app/services/source_candidate_ranker.py`
  - 让 `SearchResultForRanking` 携带 `intent` / `search_budget`。
  - `SourceSelectionPlan` 记录推荐上限和未推荐数量。
  - guidance 增强为“推荐读哪些”和“为什么不用读全部”。
- Modify: `app/services/stream/tool_round.py`
  - 使用 `search_read_planner` 构造本轮 plan。
  - selected evidence 数量跟随 planner 上限。
- Modify: `app/services/tool_handlers/web_search.py`
  - 对 duplicate-skipped 搜索返回明确的 LLM 上下文，避免模型误以为搜索失败。
- Modify: `app/ai/tools.py`
  - 收紧描述：默认一次搜索；第二次必须是官方/媒体/地区/时间范围等互补维度。
- Modify: `scripts/agent_behavior_eval.py`
  - 增加搜索次数、搜索关键词去重、推荐深读数量的离线打分字段。
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
  - 增加 Search / Read Planner v1.1 样本。
- Tests:
  - `test/services/stream/test_network_budget.py`
  - `test/test_source_candidate_ranker.py`
  - `test/services/stream/test_tool_round.py`
  - `test/test_ai_tools.py`
  - `test/test_agent_behavior_eval.py`

## Task 1: Search Budget And Duplicate Control

- [ ] Add failing tests:
  - `test_chinese_year_query_infers_freshness_intent`
  - `test_second_similar_chinese_year_query_uses_followup_budget`
  - `test_duplicate_web_search_returns_degraded_without_consuming_provider_budget`
- [ ] Run:
  - `.venv311/bin/python -m pytest test/services/stream/test_network_budget.py -q`
  - Expected before implementation: new tests fail.
- [ ] Implement:
  - Replace `\b20\d{2}\b` with digit-boundary regex that matches `2026年`.
  - Add duplicate detection for exact or very-high-similarity same-intent queries.
  - Return degraded `ToolResult` with `duplicate_search_skipped=True`, `requested_count=0`, `context_source_limit=0`.
- [ ] Verify targeted tests pass.

## Task 2: Read Planner Boundary

- [ ] Add failing tests:
  - quick fact recommends 1 source.
  - freshness recommends 2 sources.
  - official/comparison/deep research recommends 3 sources.
  - guidance includes search keywords, recommended read limit, and “do not read all results” semantics.
- [ ] Run:
  - `.venv311/bin/python -m pytest test/test_source_candidate_ranker.py -q`
  - Expected before implementation: new planner imports or fields fail.
- [ ] Implement `app/services/search_read_planner.py`.
- [ ] Extend ranker dataclasses without breaking existing callers.
- [ ] Verify targeted tests pass.

## Task 3: Tool Round Integration

- [ ] Add failing tests:
  - `tool_round` passes `intent` / `search_budget` into planner.
  - selected evidence count follows planner limit.
  - duplicate-skipped search guidance is not treated as a normal evidence source.
- [ ] Run:
  - `.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q`
  - Expected before implementation: planner integration assertions fail.
- [ ] Implement:
  - Replace direct `rank_search_sources()` calls in `tool_round.py` with `build_search_read_plan()`.
  - Use planner guidance for LLM context.
- [ ] Verify targeted tests pass.

## Task 4: Tool Description And Offline Eval

- [ ] Add failing tests:
  - web search description mentions complementary second search dimensions.
  - eval scorer flags duplicate search keywords when a sample sets `max_duplicate_search_keywords=0`.
  - eval scorer flags too many recommended reads when a sample sets `max_recommended_reads`.
- [ ] Run:
  - `.venv311/bin/python -m pytest test/test_ai_tools.py test/test_agent_behavior_eval.py -q`
  - Expected before implementation: new assertions fail.
- [ ] Implement schema/fixture/scorer updates.
- [ ] Verify targeted tests pass.

## Task 5: Full Verification, Commit, CI/CD, Real Regression

- [ ] Run backend checks:
  - `.venv311/bin/python -m pytest -q`
  - `.venv/bin/ruff check .`
  - `.venv311/bin/python scripts/agent_behavior_eval.py --dry-run`
- [ ] Commit with structured Chinese message and `Co-Authored-By`.
- [ ] Push once after code and plan are together.
- [ ] Monitor GitHub Actions until deploy completes.
- [ ] Real Chrome regression uses existing `fusion.seanfield.org` tab only:
  - SRP-01 simple no-search case.
  - SRP-02 autonomous realtime-search case.
  - SRP-05/SRP-07 multi-search/read-selection case.
  - Refresh completed conversation and check no running residue.
  - Record URL, expected, actual, console errors, refresh result.

## Task 6: Post-Deploy Regression Tightening

- [x] Add failing tests:
  - normal realtime/product-update questions execute at most 2 provider searches.
  - `deep_research` may execute the 3rd provider search, but the 4th is plan-limited.
  - plan-limited search does not create answer-evidence content blocks.
  - plan-limited search gives the LLM a reuse-existing-results instruction, not a generic search failure.
- [x] Implement:
  - `NetworkToolBudget` returns degraded `ToolResult` with `search_plan_limited=True`, `requested_count=0`, `context_source_limit=0`.
  - `WebSearchHandler` hides duplicate/plan-limited control results from persisted answer-evidence blocks.
  - Tool description states the third search is only for `deep_research`.
- [ ] Verify targeted tests, full backend tests, CI/CD, and real Chrome regression again.
