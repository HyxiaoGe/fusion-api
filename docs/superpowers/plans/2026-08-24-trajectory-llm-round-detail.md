# Fusion Trajectory LLM Round Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为新产生的 Fusion Agent Run 按 `llm_round_id` 持久化模型显式返回的 Reasoning/Output，并在 Trajectory 中以 DSH 风格的 `ASSISTANT · Request #N` 行和 Thinking/Output/Timing 详情展示，同时保持聊天视图不变。

**Architecture:** `agent_events` 继续只保存 LLM 生命周期和指标；新增 `agent_llm_round_details` 保存净化后的正文与有界 preview。每个新 Run 通过 `run_trajectory_meta.llm_detail_schema_version=1` 声明能力，历史 Run 不走兼容或猜测路径。前端使用快照中的 `llm_round_summaries` 构建 `LlmRoundCell`，完整正文通过 LLM Node Detail API 懒加载，实时完成后只补取单节点详情。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、PostgreSQL、Pydantic v2、pytest、Next.js、React、Redux Toolkit、TypeScript、Vitest、Testing Library。

**Spec:** `docs/TRAJECTORY_DESIGN.md` v0.22，重点 §7.4.11、§8.4、§8.5、§10。

## Global Constraints

- 只处理新契约 Run；不回填旧 Reasoning，不按 step/name/time/message 猜测关联。
- 仅保存 `sanitize_user_visible_reasoning(..., final=True)` 处理后的 Reasoning；`raw_reasoning_buf` 不进入数据库、API 或日志。
- 聊天视图的 Thinking 隐藏策略不变；Trajectory 是独立诊断投影。
- `agent_events` 不保存 Reasoning/Output 正文；完整正文仅存在于 `agent_llm_round_details`。
- LLM 详情写入按 `(run_id, llm_round_id)` 幂等；正文写入失败不能中断 Agent 主链路。
- P3 不实现逐 Token Thinking；Round 终态后展示完整内容。
- 实时补齐不得重拉整个快照，不修改既有 SSE envelope。
- 不在旧 `feat/trajectory-ledger-p0` worktree 实施；API 使用 `feat/trajectory-p3-contract-guard`，UI 使用 `feat/trajectory-p3-ui`。
- 保留 UI worktree 中现有未提交 UX 修改，不覆盖、不回滚。

---

### Task 1: 迁入 v0.22 并收口实施契约

**Files:**
- Modify: `docs/TRAJECTORY_DESIGN.md`
- Create: `docs/superpowers/plans/2026-08-24-trajectory-llm-round-detail.md`

**Interfaces:**
- Consumes: 旧 P0 worktree 中的 v0.22 文档内容。
- Produces: 正式 P3 API 分支内唯一设计依据与本实施计划。

- [ ] **Step 1: 将 §7.4.11、§8.4、§8.5 与 P3 交付物迁入正式分支**

只迁移设计文本，不复制旧 worktree 的代码或 Git 状态。

- [ ] **Step 2: 修正文档残留**

将 `v0.21`、`P3 定稿待实施`、`全量会话账本` 等残留改为当前事实；拆分 Tool 与 LLM status；明确 `reasoning_tokens` 若实现则属于 `llm_round_completed` 可选生命周期字段，否则不作为 P3 阻塞。

- [ ] **Step 3: 校验文档差异**

Run:

```bash
rg -n "v0\.21 位于|将 v0\.21|全量会话账本|只承诺 Tool|待定稿" docs/TRAJECTORY_DESIGN.md
git diff --check
```

Expected: 无版本/范围残留，`git diff --check` 退出码为 0。

### Task 2: 数据模型、迁移与新 Run 能力声明

**Files:**
- Create: `alembic/versions/<revision>_add_llm_round_details.py`
- Modify: `app/db/models.py`
- Modify: `app/services/agent/trajectory_recorder.py`
- Test: `test/test_trajectory_llm_round_detail_migration.py`
- Test: `test/services/agent/test_trajectory_recorder.py`

**Interfaces:**
- Consumes: `RunTrajectoryMeta`、`AgentSession`、`Conversation`、P0 幂等 meta 创建。
- Produces: `AgentLlmRoundDetail` ORM 模型、`run_trajectory_meta.llm_detail_schema_version`、新 Run 的 version=1 声明。

- [ ] **Step 1: 写迁移失败测试**

测试升级后存在 `agent_llm_round_details`、唯一约束、级联 FK 和 `llm_detail_schema_version`；降级后全部移除。

- [ ] **Step 2: 运行测试确认 RED**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q test/test_trajectory_llm_round_detail_migration.py
```

Expected: 因迁移或模型尚不存在而失败。

- [ ] **Step 3: 实现最小迁移与 ORM**

详情表字段固定为：`id/conversation_id/run_id/message_id/llm_round_id/reasoning_text/content_text/reasoning_preview/output_preview/redacted_fields/truncated_fields/recorded_at`；唯一键为 `(run_id, llm_round_id)`。

- [ ] **Step 4: 写新 Run 能力声明失败测试并实现**

`TrajectoryRecorder._insert_meta_if_missing()` 对新建 meta 写入 `llm_detail_schema_version=1`，冲突时不覆盖历史 `NULL`。

- [ ] **Step 5: 运行目标测试确认 GREEN**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q \
  test/test_trajectory_llm_round_detail_migration.py \
  test/services/agent/test_trajectory_recorder.py
```

### Task 3: LLM Round Detail 幂等后台写入

**Files:**
- Create: `app/services/agent/llm_round_detail_recorder.py`
- Modify: `main.py`
- Test: `test/services/agent/test_llm_round_detail_recorder.py`

**Interfaces:**
- Consumes: `AgentLlmRoundDetail`、`SessionLocal`、`sanitize_user_visible_reasoning`。
- Produces: `schedule_llm_round_detail(draft) -> asyncio.Task`、`stop_llm_round_detail_workers() -> Awaitable[None]`。

- [ ] **Step 1: 写净化、preview、幂等和 shutdown 失败测试**

断言原始协议不入库、preview 不超过 200 字符、重复 `(run_id,llm_round_id)` 只保留一行、任务集合在完成后释放、shutdown 可观察未完成任务。

- [ ] **Step 2: 运行测试确认 RED**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q test/services/agent/test_llm_round_detail_recorder.py
```

- [ ] **Step 3: 实现最小后台记录器**

使用模块级受控 task 集合跟踪 `asyncio.to_thread()` 数据库写入；创建任务失败时关闭 coroutine；写入使用 PostgreSQL/SQLite `ON CONFLICT DO NOTHING`；日志只记录标识和异常类型，不记录正文。

- [ ] **Step 4: 接入 FastAPI shutdown**

在关闭外部客户端前调用 `stop_llm_round_detail_workers()`；取消未完成任务并等待 `gather(return_exceptions=True)`。

- [ ] **Step 5: 运行测试确认 GREEN**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q test/services/agent/test_llm_round_detail_recorder.py
```

### Task 4: LLM 生命周期精确提交正文

**Files:**
- Modify: `app/services/stream/llm_round_lifecycle.py`
- Modify: `app/services/stream/agent_round.py`
- Modify: `app/services/stream/limit_summary.py`
- Modify: `app/services/stream/agent_loop_runtime.py`
- Modify: `app/services/stream/agent_loop_execution.py`
- Modify: `app/services/stream/agent_loop_driver.py`
- Modify: `app/services/stream/agent_loop_step_requests.py`
- Test: `test/services/stream/test_llm_round_lifecycle.py`
- Test: `test/services/stream/test_agent_round.py`
- Test: `test/services/stream/test_limit_summary.py`
- Test: `test/services/stream/test_agent_loop_wiring.py`

**Interfaces:**
- Consumes: `schedule_llm_round_detail` 和流式结果中的 `reasoning_buf/content_buf/partial_output`。
- Produces: 每个 completed/failed/cancelled `llm_round_id` 最多提交一条详情草稿。

- [ ] **Step 1: 写成功、失败、取消和延迟收口失败测试**

成功 Round 提交完整正文；流中失败/取消提交 `partial_output`；deferred lifecycle 在最终收口时仍使用该 Round 已捕获正文；重复 terminal 不重复提交。

- [ ] **Step 2: 运行测试确认 RED**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q \
  test/services/stream/test_llm_round_lifecycle.py \
  test/services/stream/test_agent_round.py \
  test/services/stream/test_limit_summary.py
```

- [ ] **Step 3: 实现详情草稿注入和 partial capture**

`LLMRoundLifecycle` 只持有受控 detail scheduler；`record_detail()` 保存净化前的可见缓冲区，terminal 事件成功发出后调度详情。`collect_agent_round_stream()` 向支持 `partial_output` 的 stream 函数传入字典，异常路径把已有部分交给 lifecycle。

- [ ] **Step 4: 运行目标测试确认 GREEN**

运行 Step 2 命令并确认全部通过。

### Task 5: 快照摘要、Run 计数与 LLM Node Detail API

**Files:**
- Modify: `app/schemas/trajectory.py`
- Modify: `app/db/trajectory_repository.py`
- Modify: `app/services/trajectory_query_service.py`
- Modify: `app/api/trajectory.py`
- Modify: `app/api/admin_audit.py`
- Test: `test/services/test_trajectory_query_service.py`
- Test: `test/services/test_trajectory_node_detail_service.py`
- Test: `test/test_trajectory_api.py`

**Interfaces:**
- Consumes: 生命周期事件、`AgentLlmRoundDetail`、`llm_detail_schema_version`。
- Produces: `TrajectoryLlmRoundSummary`、`LlmNodeDetail`、`TrajectorySnapshot.llm_round_summaries`、`TrajectoryRunSummary.llm_round_count` 和用户/管理员 LLM Detail 端点。

- [ ] **Step 1: 写快照与详情契约失败测试**

覆盖：批量 preview；Reasoning 为空仍 available；运行中 pending；version=1 终态 grace 后缺详情 degraded；version=NULL 不暴露正文；越权 404；跨 Run 计数稳定。

- [ ] **Step 2: 运行测试确认 RED**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q \
  test/services/test_trajectory_query_service.py \
  test/services/test_trajectory_node_detail_service.py \
  test/test_trajectory_api.py
```

- [ ] **Step 3: 实现 repository 批量查询和 DTO**

快照一次查询当前 Run 的所有 LLM summaries；Run 列表一次聚合各 Run 的 distinct `llm_round_started.llm_round_id` 数量；不得逐行查询。

- [ ] **Step 4: 实现普通/管理员端点**

普通端点严格按 conversation owner；管理员端点沿用访问原因和审计依赖；二者共享查询服务但不共享权限判断。

- [ ] **Step 5: 运行测试确认 GREEN**

运行 Step 2 命令并确认全部通过。

### Task 6: 前端类型、缓存与 LlmRoundCell 投影

**Files:**
- Modify: `src/types/trajectory.ts`
- Modify: `src/redux/slices/trajectorySlice.ts`
- Modify: `src/lib/trajectory/TrajectoryCellProjection.ts`
- Modify: `src/components/chat/trajectory/TrajectoryCell.tsx`
- Test: `src/redux/slices/trajectorySlice.test.ts`
- Test: `src/lib/trajectory/TrajectoryCellProjection.test.ts`
- Test: `src/components/chat/trajectory/TrajectoryTable.test.tsx`

**Interfaces:**
- Consumes: `llm_round_summaries`、`llm_round_count`、`llm_round_*` events。
- Produces: `LlmRoundCell`，类型为 `assistant_request`，带 `llmRoundId/requestIndex/model/status/preview/tokens/duration/ttft`。

- [ ] **Step 1: 写 Cell 顺序、会话编号和能力缺席失败测试**

断言 `USER → assistant_request → TOOL → assistant_request → MESSAGE`；跨 Run 编号使用前序 `llm_round_count`；version=NULL 的旧 Run 仍有 Summary/Timing，但无正文 preview。

- [ ] **Step 2: 运行测试确认 RED**

```bash
npm test -- --run \
  src/redux/slices/trajectorySlice.test.ts \
  src/lib/trajectory/TrajectoryCellProjection.test.ts \
  src/components/chat/trajectory/TrajectoryTable.test.tsx
```

- [ ] **Step 3: 实现最小类型与投影**

按事件 sequence 投影一个逻辑 Round 一行；preview 优先 reasoning，其次 output；不得为 started/completed 分别产生两行。

- [ ] **Step 4: 运行测试确认 GREEN**

运行 Step 2 命令并确认全部通过。

### Task 7: LLM Detail client、页签与实时单节点补齐

**Files:**
- Modify: `src/lib/api/trajectory.ts`
- Create: `src/hooks/useTrajectoryLlmNodeDetail.ts`
- Modify: `src/components/chat/trajectory/TrajectoryNodeDetailPanel.tsx`
- Modify: `src/components/chat/trajectory/TrajectoryTabView.tsx`
- Test: `src/lib/api/trajectory.test.ts`
- Test: `src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx`
- Test: `src/components/chat/trajectory/TrajectoryTabView.test.tsx`

**Interfaces:**
- Consumes: LLM Node Detail API、completed live event、`LlmRoundCell`。
- Produces: `Summary/Thinking/Output/Timing` 页签、7 秒内有界 settle、available/degraded 缓存更新。

- [ ] **Step 1: 写 API 路径、懒加载和 settle 失败测试**

Thinking/Output 首次点击才加载；pending 每秒最多重试一次且不超过既有 7 秒窗口；切换节点取消旧结果；completed 只请求该节点，不重拉 snapshot。

- [ ] **Step 2: 运行测试确认 RED**

```bash
npm test -- --run \
  src/lib/api/trajectory.test.ts \
  src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx \
  src/components/chat/trajectory/TrajectoryTabView.test.tsx
```

- [ ] **Step 3: 实现最小 client/hook/UI**

Tool 与 LLM 使用同一 Envelope、不同 discriminated detail；正文使用可滚动、保留换行的文本区域；`redacted_fields/truncated_fields` 只在存在时提示。

- [ ] **Step 4: 运行测试确认 GREEN**

运行 Step 2 命令并确认全部通过。

### Task 8: 完整验证与真实链路验收

**Files:**
- Modify only if tests expose a contract defect.

**Interfaces:**
- Consumes: Tasks 1–7 的完整实现。
- Produces: 代码证据和本地真实浏览器证据；不推送、不部署。

- [ ] **Step 1: 后端目标测试、静态检查与迁移检查**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/pytest -q \
  test/test_trajectory_llm_round_detail_migration.py \
  test/services/agent/test_llm_round_detail_recorder.py \
  test/services/stream/test_llm_round_lifecycle.py \
  test/services/test_trajectory_query_service.py \
  test/services/test_trajectory_node_detail_service.py \
  test/test_trajectory_api.py
/Users/sean/code/fusion/fusion-api/.venv/bin/ruff check <modified-python-files>
```

- [ ] **Step 2: 前端目标测试、类型检查与构建**

```bash
npm test -- --run <modified-trajectory-tests>
npm run typecheck
npm run build
```

- [ ] **Step 3: 差异和工作区检查**

```bash
git diff --check
git status --short
```

分别在 API/UI worktree 执行，确认没有覆盖用户已有修改。

- [ ] **Step 4: 本地真实新会话验收**

复用用户已打开的 Fusion Chrome 标签页；使用真实 dev API、认证、数据库和模型触发包含至少两次 LLM Round 与一次 Tool 的新 Run。验证聊天视图不显示工具轮次中间 Reasoning，Trajectory 显示 `Request #N`，点击后可见 Thinking/Output/Timing，刷新后仍可回放。

- [ ] **Step 5: 记录未执行的发布边界**

不得推送、创建 PR、合并或部署 dev；这些动作需要后续明确授权。
