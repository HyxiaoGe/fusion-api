# 动态搜索预算 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让联网搜索从机械的“多次搜索每次固定 5 条”变成基于意图、查询相似度和轮次的确定性预算策略。

**Architecture:** 工具描述负责减少模型发起重复搜索的概率；后端 `NetworkToolBudget` 负责最终兜底，统一推断 intent、忽略模型传入 count，并对相似 follow-up 搜索自动缩窄。`WebSearchHandler` 继续只消费规范化后的参数，不承担策略判断。

**Tech Stack:** FastAPI service layer, Python dataclass, pytest/unittest, ruff.

---

### Task 1: 搜索预算策略测试

**Files:**
- Modify: `test/services/stream/test_network_budget.py`
- Create: `test/test_ai_tools.py`

- [x] **Step 1: 写失败测试**
  - 覆盖无 intent 但 query 明确是官方公告时自动推断 `official_source`。
  - 覆盖同一轮第二个相似官方查询自动缩窄到 3 条。
  - 覆盖官方查询后追加媒体/对照查询时不被误判为重复缩窄。
  - 覆盖工具描述明确要求默认一次搜索、避免中英文同义重复搜索。

- [x] **Step 2: 运行测试确认失败**

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py test/test_ai_tools.py -q
```

Expected: 新增测试失败，证明当前策略仍机械。

### Task 2: 动态搜索预算实现

**Files:**
- Modify: `app/services/search_budget.py`
- Modify: `app/services/stream/network_budget.py`
- Modify: `app/ai/tools.py`

- [x] **Step 1: 实现 intent 推断**
  - 显式合法 intent 优先。
  - 无 intent 时从 query 推断 `official_source` / `comparison` / `deep_research` / `freshness` / `quick_fact`。

- [x] **Step 2: 实现相似 follow-up 缩窄**
  - `NetworkToolBudget` 记录本轮已消费的搜索 query 和 intent。
  - 只有“query 相似且 intent 相同”的 follow-up 才缩窄。
  - `standard/freshness/official_source` follow-up 缩到 3 条上下文 3 条。
  - `comparison/deep_research` follow-up 缩到 5 条，避免深度任务失去覆盖面。

- [x] **Step 3: 优化工具描述**
  - 默认一次搜索。
  - 只有官方+媒体、对比、多来源核验等场景才发起第二个互补 query。
  - 禁止同一轮用中英文翻译/同义改写重复搜索同一意图。

### Task 3: 验证和收尾

**Files:**
- No extra files expected.

- [x] **Step 1: 运行精确测试**

```bash
.venv311/bin/python -m pytest test/services/stream/test_network_budget.py test/test_ai_tools.py test/test_tool_executor.py -q
```

- [x] **Step 2: 扩大后端回归**

```bash
.venv311/bin/python -m pytest test/ -q
ruff check app test
ruff format --check app/services/search_budget.py app/services/stream/network_budget.py app/ai/tools.py test/services/stream/test_network_budget.py test/test_ai_tools.py
git diff --check
```

- [ ] **Step 3: 提交、push、CI/CD 和真实 Chrome 回归**
  - 按中文结构化 commit message 提交。
  - push 后跟 GitHub Actions 和 dev 部署。
  - 部署完成后复用现有已登录 Chrome 标签，新建真实对话验证搜索次数/候选条数/深读来源/刷新恢复。
