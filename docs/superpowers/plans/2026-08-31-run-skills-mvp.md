# Run 级 Skills MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fusion 增加首个代码托管 `verified-research` Skill，使其在首次 LLM 前被 Run 路由冻结，并在 Trajectory 中展示真实状态和历史正文。

**Architecture:** `build_agent_loop_call_config()` 消费严格 Skill registry 并冻结全文；能力包与 Skill `allowed-tools` 原子收窄同一工具集合；Prompt 组装只消费冻结内容。安全元数据进入 capability resolution 和 `skills_resolved`，正文复用已有 owner-scoped system prompt snapshot，并通过独立 Skills detail 接口按需读取。

**Tech Stack:** Python 3.11、FastAPI、Pydantic、SQLAlchemy、pytest、Next.js 15、React 19、TypeScript、Vitest。

**Spec:** `docs/superpowers/specs/2026-08-31-run-skills-mvp.md`

## Global Constraints

- 首期只支持 `verified-research@1.0.0`，只由 `verified_web` 包激活。
- 不新增数据库表、PromptHub、在线管理、动态发现、Slash 激活或运行中 Skill 晋升。
- Skill 正文不得进入 Run config、SSE、账本、Redux、Dexie、日志或普通会话响应。
- 不启动本地 Fusion 服务；发布后只复用已有已登录 Chrome 标签验收。
- 所有行为变更先观察测试因功能缺失而失败，再写最小生产实现。

---

### Task 1: 严格 Skill registry 与冻结快照

**Files:**
- Create: `app/ai/skills/__init__.py`
- Create: `app/ai/skills/registry.py`
- Create: `app/ai/skills/verified-research/1.0.0/SKILL.md`
- Test: `test/ai/skills/test_registry.py`

**Interfaces:**
- Produces: `LoadedSkillSnapshot`、`RunSkillResolution`、`load_skills_for_package(package_id, routed_tool_names)`。
- `LoadedSkillSnapshot` 含安全元数据和仅内存 `content`；`RunSkillResolution` 可序列化但不含正文。

- [ ] 写失败测试：合法文件稳定解析；未选择不读取文件；非法 UTF-8/frontmatter/版本/allowed-tools/超限/路径逃逸返回 `load_failed`。
- [ ] 运行 `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/ai/skills/test_registry.py -q`，确认因模块或接口不存在失败。
- [ ] 实现受控 frontmatter 解析、32 KiB 上限、SHA-256、版本目录校验、固定 package 映射和 fail-closed 结果。
- [ ] 重跑单文件测试至通过，并运行目标 Ruff。

### Task 2: Run 能力包、工具权限与 Prompt 原子组装

**Files:**
- Modify: `app/services/stream/run_capability_router.py`
- Modify: `app/utils/run_capability_contract.py`
- Modify: `app/services/stream/agent_loop_request_prep.py`
- Modify: `app/services/stream/agent_loop_wiring.py`
- Modify: `app/services/stream/runner.py`
- Modify: `app/ai/prompts/system_prompt.py`
- Test: `test/services/stream/test_run_capability_router.py`
- Test: `test/services/stream/test_agent_loop_request_prep.py`
- Test: `test/services/stream/test_agent_loop_wiring.py`

**Interfaces:**
- Consumes: Task 1 的冻结快照和安全 Skill resolution。
- Produces: schema v2 `RunCapabilityResolution.skill_resolution`、`AgentLoopCallConfig.loaded_skills`、稳定 Skill Prompt section。

- [ ] 写失败测试：verified_web 精确加载 Skill；问候/稳定知识/普通 fresh_web/URL 不加载；Skill 权限不能扩大工具；加载失败降级为 tools_unavailable；正文只注入一次；旧研究 Prompt 不再出现。
- [ ] 运行三个目标测试文件，确认新增断言失败且基线断言仍可解释。
- [ ] 实现 schema/router/template 版本递增、loader 依赖注入、工具交集、失败降级、call config 全文冻结、Prompt section 注入和旧文本移除。
- [ ] 验证 schemas、announced tools、handlers、bindings、plan enum、final tools 完全一致；重跑目标测试和 Ruff。

### Task 3: API Trajectory 状态、正文详情与兼容读取

**Files:**
- Modify: `app/services/agent/events.py`
- Modify: `app/services/agent/emitter.py`
- Modify: `app/services/agent/trajectory_payload.py`
- Modify: `app/services/stream/agent_loop_lifecycle.py`
- Modify: `app/schemas/trajectory.py`
- Modify: `app/db/trajectory_repository.py`
- Modify: `app/services/trajectory_query_service.py`
- Modify: `app/api/trajectory.py`
- Test: `test/services/agent/test_trajectory_payload.py`
- Test: `test/services/stream/test_agent_loop_lifecycle.py`
- Test: `test/services/test_trajectory_query_service.py`
- Test: `test/test_trajectory_api.py`

**Interfaces:**
- Produces: 每 Run 一条 `skills_resolved`；`GET /api/conversations/{conversation_id}/runs/{run_id}/node-detail/skills`。
- Detail 响应只返回当次 Prompt snapshot 中经过事件元数据交叉校验的 Skill sections。

- [ ] 写失败测试：loaded/not_selected/load_failed；事件严格脱敏；旧 v1 可读；详情所有权、刷新历史、哈希/字符数/section 交叉校验、pending/degraded/not_recorded。
- [ ] 运行目标测试确认新协议缺失导致失败。
- [ ] 实现事件、sanitizer、生命周期 detail_status、读侧 DTO/仓库/服务/API；禁止从磁盘重建正文。
- [ ] 重跑目标测试、Ruff、架构检查并确认无迁移文件。

### Task 4: UI Skills 节点、详情、复制与刷新恢复

**Files:**
- Modify: `src/types/trajectory.ts`
- Modify: `src/lib/trajectory/normalizeTrajectoryEvent.ts`
- Modify: `src/lib/trajectory/TrajectoryCellProjection.ts`
- Modify: `src/lib/trajectory/trajectoryCellPresentation.ts`
- Modify: `src/lib/trajectory/trajectoryNodeDetailModel.ts`
- Modify: `src/lib/api/trajectory.ts`
- Create: `src/hooks/useTrajectorySkillsNodeDetail.ts`
- Modify: `src/components/chat/trajectory/TrajectoryNodeDetailPanel.tsx`
- Modify: locale files under `src/locales/`
- Test matching existing `*.test.ts(x)` files beside these modules.

**Interfaces:**
- Consumes: Task 3 的安全事件与 `node_type=skills` detail。
- Produces: 聚合 Skills ContextCell，真实状态、元数据、正文、复制、重试和切换隔离。

- [ ] 写失败测试：实时/历史归一化一致；正文未知字段丢弃；单节点投影；旧 Run 与 not_selected 区分；no-store/AbortSignal；正文复制；切换 Run 取消和防迟到覆盖。
- [ ] 运行目标 Vitest 确认失败。
- [ ] 实现严格 normalizer、specialized projection、文案/摘要、API/hook/detail panel；正文只留组件 state。
- [ ] 重跑目标 Vitest、ESLint 和 production build。

### Task 5: 跨仓回归、对抗审查与发布验收

**Files:**
- Modify: `docs/EXECUTION_LEDGER.md` in both repositories after evidence exists.
- Update: this plan checklist and release evidence.

- [ ] API 运行 Registry、router、request prep、wiring、trajectory、API 目标集，再运行权威全量 pytest、Ruff、architecture 和 `git diff --check`。
- [ ] UI 运行全部相关 Vitest、全量 Vitest、目标 ESLint、production build 和 `git diff --check`。
- [ ] 独立审查检查 TOCTOU、权限原子性、正文泄漏、旧版兼容、失败降级和 UI 迟到响应；先关闭所有当前可达 P0/P1。
- [ ] 精确暂存、中文提交并包含 `Co-Authored-By: Codex <noreply@anthropic.com>`，push 两仓特性分支并监督 CI。
- [ ] CI 成功后合并 master，监督 API/UI dev 部署到精确 merge SHA。
- [ ] 复用现有已登录 Chrome 标签，按规格矩阵创建真实新 Run，检查 Skill 节点、Prompt 正文、复制、工具集合、话题切换、刷新历史、console/network。
- [ ] 只有本地、CI、合并、部署和真实页面证据分别成立后，更新两仓执行台账并报告交付完成。
