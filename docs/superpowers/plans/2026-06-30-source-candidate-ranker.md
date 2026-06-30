# Source Candidate Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将搜索候选来源选择从纯 prompt 约束升级为可测试的结构化排序建议，降低“搜了但不读高价值来源”的概率。

**Architecture:** 新增纯函数 ranker 对同一工具回合内的多个 `web_search` 结果做跨搜索合并、去重、排序和推荐；`tool_round` 在把工具结果注入下一轮 LLM 前追加一段结构化来源选择建议。后端不自动替 LLM 调用 `url_read`，避免破坏 agent loop 的工具决策权。

**Tech Stack:** FastAPI 后端、Pydantic 搜索来源模型、unittest/pytest、现有 agent loop tool round。

---

### Task 1: SourceCandidateRanker 纯函数

**Files:**
- Create: `app/services/source_candidate_ranker.py`
- Test: `test/test_source_candidate_ranker.py`

- [x] **Step 1: Write failing tests**

覆盖以下行为：
- 同一轮多个搜索结果按 canonical URL 去重。
- 官方/原文公告/PDF/权威媒体优先。
- 视频、社交、论坛来源默认降权。
- 最多推荐 3 个深读候选。

- [x] **Step 2: Verify RED**

Run:

```bash
.venv311/bin/python -m pytest test/test_source_candidate_ranker.py -q
```

Expected: FAIL because `app.services.source_candidate_ranker` does not exist.

- [x] **Step 3: Implement ranker**

新增 `rank_search_sources(search_results, max_recommended=3)`，返回：
- `candidates`: 全量去重候选，带 `rank`、`priority`、`reasons`
- `recommended`: top N 高价值候选

评分规则保持启发式、可解释：
- 查询品牌与域名匹配、官方站点、文档/新闻室/公告、PDF/System Card 加分。
- Reuters/AP/Axios/TechCrunch/CNBC 等权威媒体加分。
- YouTube、Threads、X/Twitter、Facebook、Reddit、论坛/视频默认降权。
- 标题/摘要与 query 词重合越高，相关性越高。

- [x] **Step 4: Verify GREEN**

Run:

```bash
.venv311/bin/python -m pytest test/test_source_candidate_ranker.py -q
```

Expected: PASS.

### Task 2: Tool Round 跨搜索聚合注入

**Files:**
- Modify: `app/services/stream/tool_round.py`
- Test: `test/services/stream/test_tool_round.py`

- [x] **Step 1: Write failing tests**

新增测试：同一轮两个 `web_search` record 返回 10 条候选时，`append_tool_round_messages` 应只在第一个搜索 tool message 中追加结构化来源选择建议，并推荐 OpenAI 官方公告、Axios、System Card；第二个 tool message 保持原 handler 上下文，避免重复注入。

- [x] **Step 2: Verify RED**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py::ToolRoundTests::test_append_tool_round_messages_adds_round_level_source_selection_guidance -q
```

Expected: FAIL because no selection guidance is appended.

- [x] **Step 3: Implement injection**

在 `append_tool_round_messages` 中：
- 先构造 assistant tool-call message。
- 对本轮成功的 `web_search` results 调用 ranker。
- 第一个成功 `web_search` 的 tool message 追加 `format_source_selection_guidance(plan)`。
- 其余 tool message 保持原工具上下文。
- content block 不变，避免前端 UI schema 变更。

- [x] **Step 4: Verify GREEN**

Run:

```bash
.venv311/bin/python -m pytest test/services/stream/test_tool_round.py::ToolRoundTests::test_append_tool_round_messages_adds_round_level_source_selection_guidance -q
```

Expected: PASS.

### Task 3: Existing Search Context Regression

**Files:**
- Modify if needed: `app/services/tool_handlers/web_search.py`
- Test: `test/test_tool_handlers.py`

- [x] **Step 1: Ensure existing prompt behavior remains**

`web_search.format_llm_context` 仍保留：
- 搜索摘要足够可直接回答。
- 关键事实/官方公告/原文细节需要少量高价值来源 `url_read`。
- 不要求读满全部搜索结果。

- [x] **Step 2: Run related regressions**

Run:

```bash
.venv311/bin/python -m pytest test/test_source_candidate_ranker.py test/test_tool_handlers.py test/services/stream/test_tool_round.py -q
```

Expected: PASS.

### Task 4: Full Verification, CI/CD, Real Regression

**Files:**
- No additional code files expected.

- [x] **Step 1: Run backend verification**

Run:

```bash
.venv311/bin/python -m pytest test/ -q
ruff check app test
ruff format --check app/services/source_candidate_ranker.py app/services/stream/tool_round.py test/test_source_candidate_ranker.py test/services/stream/test_tool_round.py
git diff --check
```

- [ ] **Step 2: Commit and push**

Use structured Chinese commit message with `背景：`、`改动：`、`验证：` and `Co-Authored-By: Codex <noreply@anthropic.com>`.

- [ ] **Step 3: CI/CD follow-through**

Watch GitHub Actions until deployment succeeds.

- [ ] **Step 4: Real Chrome regression**

Use the existing logged-in Chrome tab only. Create a new Fusion conversation on the deployed site, ask for a search task with official + media cross-check, and record:
- case id
- input
- conversation URL
- expected
- actual
- console errors
- refresh result
- conclusion
