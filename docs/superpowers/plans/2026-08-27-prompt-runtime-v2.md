# 主聊天 Prompt Runtime v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 缩小 Run 初始系统提示词、保留主 LLM 的自然语言工具召回，并让 Trajectory 准确说明 Run 初始 prompt 与每轮有效 system 指纹的区别。

**Architecture:** 固定核心和运行规则先组装，动态日期及用户偏好后置；产品工具 schema 继续由现有权限和模型能力链提供，产品事实边界随真实 ToolMessage 结果进入上下文。前端只修正现有 Run 初始快照和 LLM Round 指纹的语义，不新增状态服务。

**Tech Stack:** Python 3.11、FastAPI、pytest；TypeScript、React、Vitest。

**Spec:** `docs/superpowers/specs/2026-08-27-prompt-runtime-v2.md`

## Global Constraints

- 自然语言语义判断由主 LLM 完成；服务端正则未命中不得移除低置信请求的已授权工具。
- 不实现 Skills runtime、独立路由模型、数据库迁移或每轮完整请求正文存储。
- 不改变工具权限、Deep Research 阶段调度、计划门禁、上下文裁剪或 SSE 重连。
- 不启动本地 Fusion 服务，不提交、推送、合并或发布。
- API 工作区 `/Users/sean/code/fusion/.worktrees/prompt-runtime-api`；UI 工作区 `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui`。

---

### Task 1: 固定前缀与动态上下文分层

**Files:**
- Modify: `app/ai/prompts/system_prompt.py`
- Test: `test/test_system_prompt_assembly.py`
- Test: `test/services/chat/test_message_builder.py`

**Interfaces:**
- Consumes: `SystemPromptSection(section_id, content)` 和既有 `sections()` 回调。
- Produces: `assemble_system_prompt()` 按 `app_identity → trusted sections → current_date → user_preferences` 返回消息和 metadata。

- [x] **Step 1: 写失败测试，断言固定段落位于动态段落之前**

```python
result = assemble_system_prompt(
    user_system_prompt="请简洁回答",
    sections=lambda: [SystemPromptSection("tool", "固定工具规则")],
)
assert result.metadata["section_ids"] == [
    "app_identity",
    "tool",
    "current_date",
    "user_preferences",
]
```

- [x] **Step 2: 运行测试并确认因当前日期位于首段而失败**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_system_prompt_assembly.py test/services/chat/test_message_builder.py -q`

- [x] **Step 3: 最小实现稳定段落和动态段落分层，并更新模板版本**

```python
stable = [SystemPromptSection("app_identity", get_app_identity_prompt())]
if sections is not None:
    stable.extend(sections())
dynamic = [SystemPromptSection("current_date", build_current_date_system_prompt())]
```

- [x] **Step 4: 重跑 Task 1 测试并确认通过**

### Task 2: 解除产品工具可见性与初始产品 Prompt 的绑定

**Files:**
- Modify: `app/services/stream/agent_loop_request_prep.py`
- Test: `test/services/stream/test_agent_loop_request_prep.py`

**Interfaces:**
- Consumes: 现有 `AgentLoopCallConfig.call_kwargs` 和公告工具列表。
- Produces: 初始 `selected_sections()` 不再产生 `amap_fact_boundary`、`flyai_travel_fact_boundary`，工具 schema 和 handler 保持原样。

- [x] **Step 1: 写失败测试，覆盖默认工具集与自然语言路线表达**

```python
assert "amap_fact_boundary" not in prepared.prompt_assembly["section_ids"]
assert "flyai_travel_fact_boundary" not in prepared.prompt_assembly["section_ids"]
assert "route_compare" in announced_tool_names_from_call_kwargs(config.call_kwargs)
```

- [x] **Step 2: 运行测试并确认当前实现仍注入两段产品 Prompt**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py -q`

- [x] **Step 3: 从初始 selector 移除两段产品 Prompt，删除不再可达的注入 helper 和 import**

- [x] **Step 4: 重跑请求准备、计划策略和 Deep Research 测试**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_plan_tool_policy.py -q`

### Task 3: 真实产品结果携带完整事实边界

**Files:**
- Modify: `app/services/mcp/flyai_travel_tools.py`
- Test: `test/test_flyai_travel_tools.py`
- Test: `test/test_amap_product_tools.py`

**Interfaces:**
- Consumes: `ToolResult` 中经过服务端验证的结构化产品结果。
- Produces: FlyAI `format_llm_context()` 的本地可信包装包含 `FLYAI_TRAVEL_FACT_BOUNDARY_SYSTEM_PROMPT`；高德既有 usage contract 保持不变。

- [x] **Step 1: 写失败测试，断言 FlyAI 成功结果包含完整事实边界**

```python
context = handler.format_llm_context(success_result)
assert FLYAI_TRAVEL_FACT_BOUNDARY_SYSTEM_PROMPT in context
```

- [x] **Step 2: 运行产品工具测试并确认 FlyAI 断言失败**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_flyai_travel_tools.py test/test_amap_product_tools.py -q`

- [x] **Step 3: 在 FlyAI ToolMessage 本地包装中加入事实边界，保持外部结果为不可信数据**

- [x] **Step 4: 重跑产品工具及工具执行器测试**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_flyai_travel_tools.py test/test_amap_product_tools.py test/services/stream/test_tool_executor.py -q`

### Task 4: Trajectory 明确 Run 初始与 Round 有效语义

**Files:**
- Modify: `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui/src/lib/i18n/locales/zh-CN.json`
- Modify: `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui/src/lib/i18n/locales/en-US.json`
- Modify: `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui/src/components/chat/trajectory/TrajectoryNodeDetailPanel.tsx`
- Test: `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui/src/lib/trajectory/TrajectoryCellProjection.test.ts`
- Test: `/Users/sean/code/fusion/.worktrees/prompt-runtime-ui/src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx`

**Interfaces:**
- Consumes: 现有 `system_prompt_prepared` 与 `llm_round_started.system_prompt_fingerprint`。
- Produces: “Run 初始系统提示词”和“当轮实际系统消息指纹”文案，不改变 API 类型。

- [x] **Step 1: 写失败测试，断言节点、正文说明和 Round 指纹采用新语义**

```typescript
expect(presentation.summary).toBe('Run 初始系统提示词已组装');
expect(screen.getByText(/不包含后续 LLM Round 追加的规则/)).toBeInTheDocument();
```

- [x] **Step 2: 安装或复用锁文件依赖，运行目标测试并确认旧文案导致失败**

Run: `npm test -- --run src/lib/trajectory/TrajectoryCellProjection.test.ts src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx`

- [x] **Step 3: 最小修改中英文文案并在正文详情增加范围说明**

- [x] **Step 4: 重跑目标 Vitest**

### Task 5: 集成验证与文档事实同步

**Files:**
- Modify: `docs/EXECUTION_LEDGER.md`
- Modify: `docs/superpowers/plans/2026-08-27-prompt-runtime-v2.md`

**Interfaces:**
- Consumes: Tasks 1-4 的实现及测试结果。
- Produces: 本地实现证据、未发布边界和剩余真实模型验收项。

- [x] **Step 1: 运行 API 相关回归、Ruff 和格式检查**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_system_prompt_assembly.py test/test_prompt_runtime_templates.py test/test_amap_product_tools.py test/test_flyai_travel_tools.py test/services/chat/test_message_builder.py test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_plan_tool_policy.py test/services/stream/test_agent_round.py test/services/stream/test_tool_round.py test/services/stream/test_tool_executor.py -q`

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/ruff check app/ai/prompts/system_prompt.py app/services/stream/agent_loop_request_prep.py app/services/mcp/flyai_travel_tools.py test/test_system_prompt_assembly.py test/test_flyai_travel_tools.py test/services/stream/test_agent_loop_request_prep.py`

- [x] **Step 2: 运行 UI 相关回归、ESLint 和生产构建**

Run: `npm test -- --run src/lib/trajectory/TrajectoryCellProjection.test.ts src/lib/trajectory/trajectoryNodeDetailModel.test.ts src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx`

Run: `npx eslint src/components/chat/trajectory/TrajectoryNodeDetailPanel.tsx src/lib/trajectory/TrajectoryCellProjection.test.ts src/components/chat/trajectory/TrajectoryNodeDetailPanel.test.tsx`

Run: `npm run build`

- [x] **Step 3: 运行两仓 `git diff --check` 并复核没有意外文件**

- [x] **Step 4: 更新执行台账和本计划勾选状态，如实记录真实模型、浏览器、CI 和发布均未执行**

## 本地执行记录（2026-08-27，Asia/Shanghai）

- Task 1：失败阶段 `5 failed, 10 passed`；实现后 `15 passed`。
- Task 2：失败阶段确认默认产品工具仍产生 `amap_fact_boundary` 与 `flyai_travel_fact_boundary`；实现后请求准备测试 `46 passed, 13 subtests`。默认问候配置仍公告 Web、高德、FlyAI 工具，初始 section IDs 收敛为 `app_identity`、`tool_usage_contract`、`agent_plan_control`、`current_date`。
- Task 3：失败阶段确认 FlyAI 成功结果缺少完整事实边界及组合行程后续规则；另用超长结果复现新增包装后总上下文达到 `13,523 bytes`。实现后 FlyAI 目标测试 `15 passed, 4 subtests`，包装与 payload 合计不超过既有 `12,000 bytes` 预算。
- Task 4：失败阶段目标 Vitest `6 failed, 49 passed`；实现后 `55 passed`。
- 集成验证：API 全量 `2918 passed, 2 skipped, 828 subtests`；Ruff check 与 format check 通过。UI 全量 `207` 个文件、`2371` 个测试通过；目标 ESLint 与 `next build` 通过。
- 未执行：本地服务、真实 LLM、浏览器、CI、提交、推送、PR、合并和发布。
