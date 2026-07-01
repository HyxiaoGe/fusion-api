# Search / Read Planner v1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Search / Read Planner 的搜索预算、读取推荐和未读候选原因变成结构化、可测试、可回归的决策链路。

**Architecture:** 保留 v1.1 advisory 模式，不自动执行 `url_read`，不引入 LLM reranker。后端新增确定性 decision ledger：`NetworkToolBudget` 暴露搜索预算决策，`SourceCandidateRanker` 输出读取推荐/未推荐 reason code，离线评估器消费这些 observation 字段做回归断言。

**Tech Stack:** FastAPI service layer, Python dataclass, pytest, ruff, existing `ToolResult.data`, existing agent behavior eval script, deployed Fusion regression with existing logged-in Chrome tab only.

---

## Scope And Commit Policy

- 本计划只改 `fusion-api`。
- 本计划不新增前端 UI。
- 文档和代码可以分 commit，但不单独 push 文档 commit；实现、验证完成后统一 push，触发一次 CI/CD。
- 真实 Chrome 回归只复用用户已经打开且登录的 `fusion.seanfield.org` 标签；禁止新 Chrome、新标签、新窗口、`about:blank` 和 isolated context。

## File Map

- Create: `app/services/search_read_decision_ledger.py`
  - 聚合搜索预算决策和读取来源决策，输出离线评估可用 summary。
- Modify: `app/services/search_budget.py`
  - 新增 `SearchBudgetDecision` dataclass 和预算决策 reason code。
- Modify: `app/services/stream/network_budget.py`
  - 把搜索预算 decision 写入 normalized args 和 degraded `ToolResult.data`。
- Modify: `app/services/source_candidate_ranker.py`
  - 新增 `SourceReadDecision` dataclass。
  - `SourceSelectionPlan` 增加 `read_decisions` 和 `decision_summary`。
  - guidance 增加未推荐原因汇总。
- Modify: `app/services/search_read_planner.py`
  - 保持读源上限逻辑，返回增强后的 `SourceSelectionPlan`。
- Modify: `app/services/stream/tool_round.py`
  - 使用增强 plan，不改变现有 selected evidence 行为。
- Modify: `scripts/agent_behavior_eval.py`
  - 增加搜索次数、provider 搜索次数、预算名称、禁止读取域名、必需 reason code 检查。
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
  - 增加 v1.2 评估样本字段。
- Test: `test/services/stream/test_network_budget.py`
- Test: `test/test_source_candidate_ranker.py`
- Test: `test/services/stream/test_tool_round.py`
- Test: `test/test_agent_behavior_eval.py`
- Test: `test/test_ai_tools.py`

## Task 1: Search Budget Decision

**Files:**
- Modify: `app/services/search_budget.py`
- Modify: `app/services/stream/network_budget.py`
- Test: `test/services/stream/test_network_budget.py`

- [x] Add failing tests:
  - `test_initial_search_records_budget_decision`
  - `test_similar_followup_records_narrow_followup_decision`
  - `test_duplicate_search_records_skip_duplicate_decision`
  - `test_planner_limited_search_records_limit_decision`

Expected assertions:

```python
args, degraded = budget.prepare_web_search_args({"query": "OpenAI 最近发布了哪些产品更新？"})
self.assertIsNone(degraded)
self.assertEqual(args["budget_decision"]["action"], "execute")
self.assertEqual(args["budget_decision"]["reason_code"], "initial_search")
self.assertEqual(args["budget_decision"]["budget_name"], args["search_budget"])
```

```python
_first_args, _ = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})
second_args, degraded = budget.prepare_web_search_args({"query": "OpenAI 最新产品更新 2026年7月"})
self.assertIsNone(degraded)
self.assertEqual(second_args["budget_decision"]["action"], "narrow_followup")
self.assertEqual(second_args["budget_decision"]["reason_code"], "similar_followup")
self.assertEqual(second_args["count"], 3)
```

```python
_first_args, _ = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})
second_args, degraded = budget.prepare_web_search_args({"query": "OpenAI 最新公告 2026年7月"})
self.assertIsNotNone(degraded)
self.assertEqual(second_args["budget_decision"]["action"], "skip_duplicate")
self.assertEqual(degraded.data["budget_decision"]["reason_code"], "duplicate_query")
self.assertEqual(budget.web_search_calls, 1)
```

```python
budget.prepare_web_search_args({"query": "OpenAI 2026年产品更新 最新发布"})
budget.prepare_web_search_args({"query": "OpenAI 2026年最新新闻 媒体报道"})
third_args, degraded = budget.prepare_web_search_args({"query": "OpenAI GPT-5.6 预览 2026年7月"})
self.assertIsNotNone(degraded)
self.assertEqual(third_args["budget_decision"]["action"], "limit_planner")
self.assertEqual(degraded.data["budget_decision"]["reason_code"], "planned_search_limit_reached")
```

- [x] Run targeted test and confirm failure:

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py -q
```

- [x] Implement:
  - Add `SearchBudgetDecision` to `app/services/search_budget.py`.
  - Add helper `build_search_budget_decision(...) -> SearchBudgetDecision`.
  - In `NetworkToolBudget.prepare_web_search_args()`, attach `budget_decision` via `decision.__dict__`.
  - For degraded duplicate/planner/hard-budget results, include the same `budget_decision` in `ToolResult.data`.

- [x] Verify:

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py -q
```

Expected: all tests in file pass.

## Task 2: Source Read Decision

**Files:**
- Modify: `app/services/source_candidate_ranker.py`
- Modify: `app/services/search_read_planner.py`
- Test: `test/test_source_candidate_ranker.py`

- [x] Add failing tests:
  - `test_source_selection_plan_records_read_decisions_for_all_candidates`
  - `test_low_priority_sources_are_deprioritized_with_reason_code`
  - `test_guidance_summarizes_not_recommended_reason_codes`

Expected assertions:

```python
plan = rank_search_sources(search_results, max_recommended=2)
self.assertEqual(len(plan.read_decisions), plan.unique_source_count)
self.assertEqual(
    [decision.action for decision in plan.read_decisions[:2]],
    ["recommend_read", "recommend_read"],
)
self.assertIn("recommended_read", plan.decision_summary)
```

```python
low = next(decision for decision in plan.read_decisions if decision.candidate.domain == "youtube.com")
self.assertEqual(low.action, "deprioritize")
self.assertEqual(low.reason_code, "low_priority_source_type")
```

```python
guidance = format_source_selection_guidance(plan)
self.assertIn("未建议深读原因", guidance)
self.assertIn("低优先级来源", guidance)
self.assertIn("超过本轮推荐深读上限", guidance)
```

- [x] Run targeted test and confirm failure:

```bash
.venv311/bin/python -m pytest test/test_source_candidate_ranker.py -q
```

- [x] Implement:
  - Add `SourceReadDecision`.
  - Extend `SourceSelectionPlan` with `read_decisions` and `decision_summary`.
  - Build read decisions after ranking:
    - recommended candidates -> `recommend_read`
    - low priority candidates -> `deprioritize / low_priority_source_type`
    - non-low candidates outside read limit -> `keep_candidate / outside_read_limit`
  - Keep `recommended`, `low_priority`, `not_recommended_count` unchanged for compatibility.
  - Update guidance to summarize reason counts.

- [x] Verify:

```bash
.venv311/bin/python -m pytest test/test_source_candidate_ranker.py -q
```

Expected: all tests in file pass.

## Task 3: Search / Read Decision Ledger

**Files:**
- Create: `app/services/search_read_decision_ledger.py`
- Modify: `app/services/stream/tool_round.py`
- Test: `test/services/stream/test_tool_round.py`

- [x] Add failing tests:
  - `test_build_search_read_decision_ledger_summarizes_budget_and_read_decisions`
  - `test_tool_round_guidance_uses_enhanced_read_decision_summary`

Expected ledger shape:

```python
ledger = build_search_read_decision_ledger(results, source_plan=plan)
self.assertEqual(ledger["summary"]["executed_search_count"], 2)
self.assertEqual(ledger["summary"]["recommended_read_count"], 3)
self.assertIn("search_decisions", ledger)
self.assertIn("read_decisions", ledger)
```

Expected guidance assertion:

```python
self.assertIn("未建议深读原因", messages[3]["content"])
self.assertIn("只有当推荐来源无法回答关键事实", messages[3]["content"])
```

- [x] Run targeted test and confirm failure:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q
```

- [x] Implement:
  - Create pure function `build_search_read_decision_ledger(results, source_plan=None)`.
  - Extract `budget_decision` from `ToolExecutionRecord.result.data`.
  - Extract `read_decisions` from `SourceSelectionPlan`.
  - Do not emit new SSE event in v1.2; keep ledger usable by tests and future UI.
  - Keep current selected evidence emission unchanged.

- [x] Verify:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q
```

Expected: all tests in file pass.

## Task 4: Offline Behavior Eval V1.2

**Files:**
- Modify: `scripts/agent_behavior_eval.py`
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
- Test: `test/test_agent_behavior_eval.py`

- [x] Add failing tests:
  - `test_load_samples_accepts_v1_2_planner_fields`
  - `test_score_observation_flags_excess_search_calls_and_wrong_budgets`
  - `test_score_observation_flags_forbidden_read_domains`
  - `test_score_observation_requires_decision_reason_codes`

Expected scoring assertions:

```python
sample = {
    "id": "planner-v1-2",
    "expected_tool_policy": "search",
    "expected_surface": "evidence",
    "max_search_calls": 2,
    "max_provider_search_calls": 2,
    "expected_search_budgets": ["freshness", "freshness_followup"],
    "forbidden_read_domains": ["youtube.com"],
    "required_decision_reason_codes": ["official_original"],
}
observation = {
    "tool_calls": ["web_search", "web_search", "web_search"],
    "surfaces": ["execution_process", "answer_evidence"],
    "search_call_count": 3,
    "provider_search_call_count": 3,
    "search_budgets": ["freshness", "standard", "standard"],
    "read_domains": ["youtube.com"],
    "decision_reason_codes": [],
}
score = score_observation(sample, observation)
self.assertFalse(score["passed"])
self.assertIn("搜索调用次数过多", "\n".join(score["issues"]))
self.assertIn("provider 搜索次数过多", "\n".join(score["issues"]))
self.assertIn("搜索预算不符合预期", "\n".join(score["issues"]))
self.assertIn("读取了禁止深读的域名", "\n".join(score["issues"]))
self.assertIn("缺少必需决策原因", "\n".join(score["issues"]))
```

- [x] Run targeted test and confirm failure:

```bash
.venv311/bin/python -m pytest test/test_agent_behavior_eval.py -q
```

- [x] Implement:
  - Validate optional fields:
    - `max_search_calls`
    - `max_provider_search_calls`
    - `expected_search_budgets`
    - `forbidden_read_domains`
    - `required_decision_reason_codes`
  - Add scoring checks using observation fields:
    - `search_call_count`
    - `provider_search_call_count`
    - `search_budgets`
    - `read_domains`
    - `decision_reason_codes`
  - Extend fixture with one v1.2 planner sample.

- [x] Verify:

```bash
.venv311/bin/python -m pytest test/test_agent_behavior_eval.py -q
```

Expected: all tests in file pass.

## Task 5: Tool Prompt Guardrail Test

**Files:**
- Modify: `app/ai/tools.py`
- Test: `test/test_ai_tools.py`

- [x] Add failing or strengthening assertions:
  - tool description says second search requires complementary dimension.
  - tool description says third search only for `deep_research`.
  - tool description says not to repeat by translation or synonym rewrite.

Expected assertions:

```python
description = build_web_search_tool()["function"]["description"]
self.assertIn("第二次搜索", description)
self.assertIn("互补维度", description)
self.assertIn("第三次搜索只适用于 deep_research", description)
self.assertIn("同义改写重复搜索", description)
```

- [x] Run targeted test:

```bash
.venv311/bin/python -m pytest test/test_ai_tools.py -q
```

- [x] Implement only if current prompt misses an assertion. If all assertions already pass, keep production prompt unchanged.

- [x] Verify:

```bash
.venv311/bin/python -m pytest test/test_ai_tools.py -q
```

Expected: all tests in file pass.

## Task 6: Full Backend Verification And Commit

**Files:**
- No new production file unless earlier tasks require it.

- [x] Run targeted tests:

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py test/test_source_candidate_ranker.py test/services/stream/test_tool_round.py test/test_agent_behavior_eval.py test/test_ai_tools.py -q
```

Expected: pass.

- [x] Run full backend tests:

```bash
.venv311/bin/python -m pytest -q
```

Expected: pass.

- [x] Run lint:

```bash
.venv/bin/ruff check .
```

Expected: pass.

- [x] Run dry-run behavior eval:

```bash
.venv311/bin/python scripts/agent_behavior_eval.py --dry-run
```

Expected: JSONL output without exception.

- [x] Commit with structured Chinese commit message:

```bash
git add app/services/search_budget.py app/services/stream/network_budget.py app/services/source_candidate_ranker.py app/services/search_read_planner.py app/services/search_read_decision_ledger.py app/services/stream/tool_round.py scripts/agent_behavior_eval.py test/services/stream/test_network_budget.py test/test_source_candidate_ranker.py test/services/stream/test_tool_round.py test/test_agent_behavior_eval.py test/test_ai_tools.py test/fixtures/agent_behavior_eval_samples.json docs/superpowers/specs/2026-07-01-search-read-planner-v1-2-design.md docs/superpowers/plans/2026-07-01-search-read-planner-v1-2.md
git commit -m "feat: 增强搜索读取规划评估约束" -m "背景：
Search / Read Planner v1.1 已经完成搜索预算、重复搜索收敛和深读推荐上限，但搜索/读取决策缺少结构化解释和可回归评估，真实对话中难以判断为什么搜、为什么读这些、为什么不读剩余候选。

改动：
- 增加搜索预算决策和读取来源决策结构，输出稳定 action 与 reason code
- 增强候选来源推荐与未推荐原因汇总，避免只依赖自然语言 prompt
- 扩展离线行为评估字段，覆盖搜索次数、provider 搜索次数、预算、读取域名和决策原因
- 补充 Search / Read Planner v1.2 设计与实施计划

Co-Authored-By: Codex <noreply@anthropic.com>"
```

## Task 7: CI/CD And Deployed Regression

**Files:**
- No file edits expected unless CI exposes real issues.

- [ ] Push once after implementation commit:

```bash
git push
```

- [ ] Monitor GitHub Actions until backend deploy finishes.

- [ ] Real Chrome regression, using existing logged-in `fusion.seanfield.org` tab only:

| Case | Input | Expected |
| --- | --- | --- |
| SRP12-R01 | `你好，你是谁？` | 不搜索，不展示执行过程/回答依据，无 console error |
| SRP12-R02 | `1+1等于几？` | 不搜索，不展示执行过程/回答依据，无 console error |
| SRP12-R03 | `OpenAI 最近发布了哪些产品更新？` | 自主搜索，搜索关键词可见，依据可见，不泄漏 `url_read` / `reader-service` |
| SRP12-R04 | `微信A2A互通怎么用？` | 自主搜索，搜索次数不过度机械，依据和执行过程数字一致 |

- [ ] For each regression case, record:
  - case id
  - input
  - conversation URL
  - expected
  - actual
  - console error count
  - refresh result
  - conclusion

- [ ] If regression fails, inspect current deployed behavior and add a failing automated test before fixing.
