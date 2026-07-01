# Search Failure Recovery / Budget v1.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让搜索预算器能基于上一轮搜索质量和读取失败状态做最小失败恢复决策，减少机械重复搜索。

**Architecture:** 继续保留现有 advisory 模式。`NetworkToolBudget` 增加纯内存反馈状态，`tool_round` 在每轮工具结果后回填状态，`agent_behavior_eval` 增加 action 级断言。系统不自动读取网页，不新增前端 UI，不启动本地服务。

**Tech Stack:** FastAPI service layer, Python dataclass, pytest, ruff, existing `ToolResult.data`, existing `ToolExecutionRecord`, existing agent behavior eval script.

---

## Scope And Commit Policy

- 本计划只改 `fusion-api`。
- 文档和代码本地一起提交；不单独 push 文档 commit。
- Chrome 回归只复用已有登录态 Fusion 标签；无可复用标签时停止并报告阻塞。

## File Map

- Modify: `app/services/search_budget.py`
  - 增加 repair 小预算 helper。
  - 扩展 `SearchBudgetDecision` 的可选恢复字段。
- Modify: `app/services/stream/network_budget.py`
  - 增加搜索质量和读取失败反馈状态。
  - 新增 `record_tool_results(...)`。
  - 新增 `repair_search` 和 `redirect_to_read_alternative` 决策。
- Modify: `app/services/stream/tool_round.py`
  - 工具执行后构造一次 `SourceSelectionPlan`。
  - 回填 `NetworkToolBudget.record_tool_results(...)`。
  - 复用同一个 plan 生成 evidence 和 LLM guidance。
- Modify: `app/services/search_read_decision_ledger.py`
  - 把 `repair_search` 计入 provider search。
- Modify: `app/services/tool_handlers/web_search.py`
  - 为 `redirect_to_read_alternative` 输出安全 LLM context 和摘要。
- Modify: `scripts/agent_behavior_eval.py`
  - 增加 `expected_search_actions`、`required_search_actions`、`forbidden_search_actions`、`max_repair_search_calls`。
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
  - 增加 v1.3 样本字段。
- Test: `test/services/stream/test_network_budget.py`
- Test: `test/services/stream/test_tool_round.py`
- Test: `test/test_agent_behavior_eval.py`

## Task 1: Network Budget Failure Feedback

**Files:**
- Modify: `app/services/stream/network_budget.py`
- Modify: `app/services/search_budget.py`
- Test: `test/services/stream/test_network_budget.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_empty_first_search_marks_next_search_as_repair(self):
    budget = NetworkToolBudget()
    first_args, first_degraded = budget.prepare_web_search_args({"query": "OpenAI 2026 最新产品"})
    budget.record_tool_results([_search_record(first_args, status="degraded", sources=[])])

    second_args, second_degraded = budget.prepare_web_search_args({"query": "OpenAI 官方公告 2026 最新"})

    self.assertIsNone(first_degraded)
    self.assertIsNone(second_degraded)
    self.assertEqual(second_args["budget_decision"]["action"], "repair_search")
    self.assertEqual(second_args["budget_decision"]["reason_code"], "previous_search_no_results")
    self.assertLessEqual(second_args["count"], 3)
    self.assertEqual(budget.web_search_calls, 2)
```

```python
def test_read_failure_redirects_search_to_unread_candidate(self):
    budget = NetworkToolBudget()
    plan = _source_plan(["https://a.example/post", "https://b.example/post"])
    budget.record_tool_results(
        [_url_read_record("https://a.example/post", status="degraded")],
        source_plan=plan,
    )

    args, degraded = budget.prepare_web_search_args({"query": "继续搜索同一问题"})

    self.assertIsNotNone(degraded)
    self.assertEqual(args["budget_decision"]["action"], "redirect_to_read_alternative")
    self.assertEqual(args["budget_decision"]["reason_code"], "read_alternatives_available")
    self.assertEqual(args["count"], 0)
    self.assertEqual(budget.web_search_calls, 0)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py -q
```

Expected: new tests fail because `record_tool_results` and recovery actions do not exist.

- [ ] **Step 3: Implement minimal feedback state**

Implement:

- `NetworkToolBudget.record_tool_results(results, source_plan=None)`.
- Search result feedback from `web_search` result status and `result_count`.
- Candidate/read feedback from `SourceSelectionPlan.read_decisions` and `url_read` results.
- One repair search maximum.
- Redirect to unread candidate before new provider search.

- [ ] **Step 4: Verify Task 1**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py -q
```

Expected: all tests in file pass.

## Task 2: Tool Round Feedback Wiring

**Files:**
- Modify: `app/services/stream/tool_round.py`
- Test: `test/services/stream/test_tool_round.py`

- [ ] **Step 1: Write failing test**

Add:

```python
async def test_tool_round_records_network_budget_feedback_with_source_plan(self):
    budget = Mock()
    budget.record_tool_results = Mock()
    # execute a web_search record with sources
    await handle_tool_calls_round(...)
    budget.record_tool_results.assert_called_once()
    self.assertIsNotNone(budget.record_tool_results.call_args.kwargs["source_plan"])
```

- [ ] **Step 2: Verify test fails**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q
```

- [ ] **Step 3: Implement wiring**

Implement:

- Build source plan once after `execute_tool_round_tools`.
- Pass plan to `record_tool_results`.
- Reuse the same plan for selected evidence and source guidance.
- Keep existing public behavior unchanged when no search result exists.

- [ ] **Step 4: Verify Task 2**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py -q
```

Expected: all tests in file pass.

## Task 3: Eval And Ledger Support

**Files:**
- Modify: `app/services/search_read_decision_ledger.py`
- Modify: `scripts/agent_behavior_eval.py`
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
- Test: `test/test_agent_behavior_eval.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- `repair_search` counts as provider search.
- `expected_search_actions` exact list mismatch is flagged.
- `required_search_actions` missing action is flagged.
- `max_repair_search_calls` flags too many repairs.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
.venv311/bin/python -m pytest test/test_agent_behavior_eval.py test/services/stream/test_tool_round.py::ToolRoundTests::test_build_search_read_decision_ledger_summarizes_budget_and_read_decisions -q
```

- [ ] **Step 3: Implement eval fields**

Implement:

- `OPTIONAL_STRING_LIST_FIELDS` adds `expected_search_actions`, `required_search_actions`, `forbidden_search_actions`.
- `OPTIONAL_NON_NEGATIVE_INT_FIELDS` adds `max_repair_search_calls`.
- `_check_search_context` reads `observation["search_actions"]`.
- Ledger `PROVIDER_SEARCH_ACTIONS` includes `repair_search`.

- [ ] **Step 4: Verify Task 3**

Run:

```bash
.venv311/bin/python -m pytest test/test_agent_behavior_eval.py test/services/stream/test_tool_round.py -q
```

Expected: all tests pass.

## Task 4: Full Verification And Release

**Files:**
- All touched files.

- [ ] **Step 1: Run targeted tests**

```bash
.venv311/bin/python -m pytest \
  test/services/stream/test_network_budget.py \
  test/services/stream/test_tool_round.py \
  test/test_agent_behavior_eval.py \
  -q
```

- [ ] **Step 2: Run behavior eval dry-run**

```bash
.venv311/bin/python scripts/agent_behavior_eval.py --dry-run
```

- [ ] **Step 3: Run full backend verification**

```bash
.venv311/bin/python -m pytest test/ -q
.venv311/bin/python -m ruff check .
.venv311/bin/python -m ruff format --check .
```

- [ ] **Step 4: Commit and push**

Use a structured Chinese commit message with `背景：` and `改动：`, including:

```text
Co-Authored-By: Codex <noreply@anthropic.com>
```

- [ ] **Step 5: CI/CD and real regression**

After push:

- Watch GitHub Actions until build/test/deploy complete.
- Reuse existing logged-in `fusion.seanfield.org` Chrome tab only.
- Run real cases and record URL/input/expected/actual/console/refresh/conclusion.
