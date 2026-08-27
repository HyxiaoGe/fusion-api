# Run Capability Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在每个 Fusion Agent Run 首轮 LLM 前冻结最小能力包，使系统提示词、工具 schema、handler/binding、计划模式和 Trajectory 展示完全一致。

**Architecture:** 新增无 IO 的确定性路由纯函数，先从当前消息、受信 options、模型能力和紧邻结构化上下文选择代码固定能力包，再由现有 call-config 原子物化实际工具和 Prompt sections。安全 route resolution 持久化到 `AgentSession.run_config`，通过 Trajectory Run summary 返回并由 UI 展示；不增加 LLM Router、Skills runtime 或数据库迁移。

**Tech Stack:** Python 3.11、FastAPI、Pydantic、SQLAlchemy JSONB、pytest、Next.js/React、TypeScript、Vitest、Testing Library。

**Spec:** `docs/superpowers/specs/2026-08-27-run-capability-router.md`

## Global Constraints

- 所有代码注释、文档与提交信息使用中文；提交必须包含 `Co-Authored-By: Codex <noreply@anthropic.com>`。
- 不启动本地 Fusion API、UI、Docker 或其他 Fusion 服务。
- 路由不增加 LLM 调用，不在 Run 中途晋升工具，不实现 Skills runtime。
- 低置信能力包最多 3 个代码固定外部工具；完全无族时零工具澄清。
- 用户 `system_prompt` 不参与路由；route resolution 不含用户原文、Prompt 正文、schema、凭据或 endpoint。
- 同一 route 必须原子过滤 definitions、handlers、bindings、announced tools 和计划工具。
- `app_identity` 始终存在；`current_date` 与 `no_tool_network_boundary` 只能按规格条件加入。
- 旧 Run 无 route resolution 时兼容为“未记录”，禁止反推。
- API 与 UI 各自在独立 `feat/run-capability-router` 工作树实现；不得编辑 detached 根工作区。

---

### Task 1: 纯函数 Run 能力路由

**Files:**
- Create: `app/services/stream/run_capability_router.py`
- Create: `test/services/stream/test_run_capability_router.py`
- Modify: `app/services/stream/agent_plan_tool_policy.py`
- Test: `test/services/stream/test_agent_plan_tool_policy.py`

**Interfaces:**
- Consumes: `AgentTaskPolicy`、`PlanMode`、当前 `original_message`、`task_context_messages`、available tool names、模型 capabilities 与受信 options。
- Produces: `RunCapabilityResolution`、`resolve_run_capability_route(...)`、`serialize_capability_resolution(...)`；字段固定为 spec 的安全协议。

- [ ] **Step 1: 写 RED 路由矩阵测试**

参数化覆盖中英文 direct、transform、date、fresh/verified web、URL、天气、地点、路线、航班、高铁、飞机高铁比较、自然跨城、多能力并集、clarification、Deep Research、禁用工具、无 FC、显式 plan on/off、紧邻路线追问和话题切换。每个 case 精确断言：

```python
assert route.package_id == expected_package
assert route.external_tool_names == expected_external_tools
assert route.effective_plan_mode == expected_plan_mode
assert route.include_current_date is expected_include_date
assert route.network_boundary_required is expected_network_boundary
assert len(route.external_tool_names) <= 3 or route.package_id == "deep_research"
```

另加反向样本：`把北京到上海翻译成英文` 不得触发出行，`帮我查一下这个` 不得公开 Web/MCP/产品工具；机场/火车站路线不得被航班/铁路关键词抢占，明确 URL/联网搜索与外部公告翻译不得被本地 transform 或 clarification 截断。

- [ ] **Step 2: 运行 RED 测试并保存失败证据**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_run_capability_router.py test/services/stream/test_agent_plan_tool_policy.py -q`

Expected: FAIL，原因是新模块/接口不存在或旧策略仍全量回退；不得以 collection error 之外的无关环境问题冒充 RED。

- [ ] **Step 3: 实现不可变路由类型与固定能力包**

实现：

```python
@dataclass(frozen=True)
class RunCapabilityResolution:
    schema_version: int
    router_version: str
    package_id: str
    confidence: Literal["high", "medium", "low"]
    resolution_mode: Literal["routed", "degraded", "clarification"]
    reason_codes: tuple[str, ...]
    external_tool_names: tuple[str, ...]
    effective_plan_mode: PlanMode
    include_current_date: bool
    network_boundary_required: bool


def resolve_run_capability_route(
    *, original_message: str | None, task_context_messages: list[object] | None,
    available_tool_names: list[str], requested_plan_mode: PlanMode,
    task_policy: AgentTaskPolicy, capabilities: dict, tools_disabled: bool,
    knowledge_grounded: bool,
) -> RunCapabilityResolution: ...
```

正则与结构解析只使用代码固定白名单。将现有产品正则抽成可复用 resolver，Plan policy 和 Run router 共用，避免两套规则漂移。最终工具只取固定包与 available names 的交集；普通 MCP 只允许消息精确提及已授权 alias 时选择一个。

- [ ] **Step 4: 运行 GREEN 路由测试**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_run_capability_router.py test/services/stream/test_agent_plan_tool_policy.py -q`

Expected: PASS，且矩阵中低置信包的外部工具数均不超过 3。

- [ ] **Step 5: 提交 Task 1**

```bash
git add app/services/stream/run_capability_router.py app/services/stream/agent_plan_tool_policy.py test/services/stream/test_run_capability_router.py test/services/stream/test_agent_plan_tool_policy.py
git commit -m "feat: 新增Run能力路由" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

### Task 2: 原子物化工具与条件 Prompt

**Files:**
- Modify: `app/services/stream/agent_plan_tool_policy.py`
- Modify: `test/services/stream/test_agent_plan_tool_policy.py`
- Modify: `test/services/stream/test_run_capability_router.py`
- Modify: `app/services/stream/agent_loop_request_prep.py`
- Modify: `app/ai/prompts/system_prompt.py`
- Modify: `test/services/stream/test_agent_loop_request_prep.py`
- Modify: `test/ai/prompts/test_system_prompt.py`
- Test: `test/services/stream/test_agent_plan_tool_policy.py`

**Interfaces:**
- Consumes: Task 1 的 `resolve_run_capability_route(...)` 与 `RunCapabilityResolution`。
- Produces: `AgentLoopCallConfig.capability_resolution`；route 驱动的 `call_kwargs.tools`、handlers、bindings、plan mode、Prompt sections 和 snapshot。

- [ ] **Step 1: 翻转现有错误契约并写 RED 集成测试**

先补 Task 1 breaker 携带的默认拒绝回归：`从亏损到盈利怎么走？` 与 `请规划从冷启动到规模化的路线` 必须 `clarification_only`，`从故宫到颐和园怎么走？` 与 `请给我从故宫到颐和园的路线` 必须 `mobility_route`。随后把 `original_message="你好"` 的旧断言改为：`call_kwargs.tools`、handlers、bindings、final tools 都为空；Prompt sections 精确等于 `app_identity`。新增自然跨城精确三工具、实时 Web、URL、天气、显式 Plan、禁用工具、无 FC、恶意用户偏好和 provider reasoning 组合测试。

- [ ] **Step 2: 运行 RED 测试并保存失败证据**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py test/ai/prompts/test_system_prompt.py -q`

Expected: 至少“你好仍携带工具/日期/契约”相关断言 FAIL。

- [ ] **Step 3: 用 route 原子物化 call config**

先将明确路线授权收紧为默认拒绝：未知 `BOUNDED` 端点不能仅凭“怎么走/路线”提升为物理地点；必须命中代码固定的城市、POI、交通/机构地点类型或其他确认物理地点证据。保留故宫/颐和园、迪士尼/东方明珠、大学之间明确路线正例，抽象流程与商业演进保持澄清。再在 `build_agent_loop_call_config()` 中先构造 available tools，解析 route，只保留 `route.external_tool_names`；从同一 final-name set 同步过滤 handlers 和 bindings。按 `effective_plan_mode` 决定 `update_plan` 和 `_plan_item_id`，最后再调用现有 provider reasoning 适配。

给 `AgentLoopCallConfig` 增加：

```python
capability_resolution: RunCapabilityResolution
```

`announced_tools` 继续排除 `update_plan`，`final_tool_names` 与外部工具一致。

- [ ] **Step 4: 让 Prompt 动态段落服从 route**

把组装接口改为：

```python
def assemble_system_prompt(*, user_system_prompt: str | None = None,
                           include_current_date: bool = True,
                           sections: Callable[[], Iterable[SystemPromptSection]] | None = None) -> SystemPromptAssembly: ...
```

`prepare_agent_loop_messages()` 传 `call_config.capability_resolution.include_current_date`。`no_tool_network_boundary` 仅由 `network_boundary_required` 选择，普通零工具请求不注入。递增 `TEMPLATE_VERSION`。

- [ ] **Step 5: 运行 GREEN 与相关回归**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_run_capability_router.py test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_plan_tool_policy.py test/services/stream/test_agent_task_policy.py test/ai/prompts/test_system_prompt.py -q`

Expected: PASS；Prompt section 顺序与规格矩阵一致。

- [ ] **Step 6: 提交 Task 2**

```bash
git add app/services/stream/agent_plan_tool_policy.py app/services/stream/agent_loop_request_prep.py app/ai/prompts/system_prompt.py test/services/stream/test_agent_plan_tool_policy.py test/services/stream/test_run_capability_router.py test/services/stream/test_agent_loop_request_prep.py test/ai/prompts/test_system_prompt.py
git commit -m "refactor: 按Run能力组装提示词和工具" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

### Task 3: 持久化 Route resolution 并加入 Trajectory 协议

**Files:**
- Modify: `app/services/stream/agent_loop_lifecycle.py`
- Modify: `app/services/agent/events.py`
- Modify: `app/services/agent/emitter.py`
- Modify: `app/services/agent/trajectory_payload.py`
- Modify: `app/services/trajectory_query_service.py`
- Modify: `app/schemas/trajectory.py`
- Modify: `test/services/stream/test_agent_loop_lifecycle.py`
- Modify: `test/services/agent/test_events.py`
- Modify: `test/services/agent/test_emitter.py`
- Modify: `test/services/agent/test_trajectory_payload.py`
- Modify: `test/services/test_trajectory_query_service.py`
- Modify: `test/test_trajectory_api.py`

**Interfaces:**
- Consumes: `AgentLoopCallConfig.capability_resolution` 与最终 Prompt assembly metadata。
- Produces: `run_config.capability_resolution`、`TrajectoryRunSummary.capability_resolution` 和实时 `run_started.capability_resolution` 的有界安全字段。

- [ ] **Step 1: 写 RED 持久化与查询测试**

断言新 Run 的 config 与 Run summary 返回同一 resolution；`external_tool_names` 等于 `run_started.tools`；旧 Run 返回 `null`；payload 白名单拒绝用户原文、完整 config、Prompt/schema/凭据。刷新查询不得重新调用 router。

- [ ] **Step 2: 运行 RED 测试**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_lifecycle.py test/services/agent/test_trajectory_payload.py test/services/test_trajectory_query_service.py test/test_trajectory_api.py -q`

Expected: FAIL，原因是 route resolution 尚未进入 run config/summary/event 白名单。

- [ ] **Step 3: 实现安全序列化与 bundle fingerprint**

在 Run 启动前用稳定 JSON 计算 `sha256:` 指纹，输入仅包含 router/template version、package、最终工具名、effective plan/task/evidence。实际 Prompt section IDs 与正文继续由随后持久化的 Prompt snapshot/fingerprint 单独证明。`run_config` 只保存规格允许字段；`TrajectoryRunSummary` 使用显式 DTO，不透传整个 config。

- [ ] **Step 4: 扩展实时与刷新协议**

给 `run_started` 增加顶层 `capability_resolution` 并加入 durable payload 白名单；Run summary 直接从 `AgentSession.run_config` 安全解析。历史 Run 缺失时返回 `None`，非法值也降为 `None` 而不是泄漏任意 JSON。

- [ ] **Step 5: 运行 GREEN 与协议回归**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_lifecycle.py test/services/agent/test_trajectory_payload.py test/services/test_trajectory_query_service.py test/test_trajectory_api.py test/services/agent/test_events.py test/services/agent/test_emitter.py -q`

Expected: PASS；实时与刷新协议一致，旧 Run 兼容。

- [ ] **Step 6: 提交 Task 3**

```bash
git add app/services/stream/agent_loop_lifecycle.py app/services/agent/events.py app/services/agent/emitter.py app/services/agent/trajectory_payload.py app/services/trajectory_query_service.py app/schemas/trajectory.py test/services/stream/test_agent_loop_lifecycle.py test/services/agent/test_events.py test/services/agent/test_emitter.py test/services/agent/test_trajectory_payload.py test/services/test_trajectory_query_service.py test/test_trajectory_api.py
git commit -m "feat: 记录Run能力路由状态" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

### Task 4: Trajectory UI 展示能力包

**Files:**
- Modify: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/types/trajectory.ts`
- Modify: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/normalizeTrajectoryEvent.ts`
- Modify: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/TrajectoryCellProjection.ts`
- Modify: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/trajectoryCellPresentation.ts`
- Modify: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/trajectoryNodeDetailModel.ts`
- Test: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/normalizeTrajectoryEvent.test.ts`
- Test: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/TrajectoryCellProjection.test.ts`
- Test: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/trajectoryCellPresentation.test.ts`
- Test: `/Users/sean/code/fusion/.worktrees/run-capability-router-ui/src/lib/trajectory/trajectoryNodeDetailModel.test.ts`

**Interfaces:**
- Consumes: Task 3 的 `capability_resolution` wire object，字段与 spec 完全一致。
- Produces: Run 卡片/详情中的能力包、置信度、初始工具、计划模式、版本和指纹摘要；旧 Run 显示“未记录能力路由”。

- [ ] **Step 1: 写 RED UI 协议与投影测试**

覆盖 SSE 实时事件、Run summary 刷新、旧 Run、非法/多余字段剥离、切换 Run 不串数据。断言 UI 文案区分“Run 初始外部工具”和实际工具调用。

- [ ] **Step 2: 运行 RED 测试**

Run: `npm test -- src/lib/trajectory/normalizeTrajectoryEvent.test.ts src/lib/trajectory/TrajectoryCellProjection.test.ts src/lib/trajectory/trajectoryCellPresentation.test.ts src/lib/trajectory/trajectoryNodeDetailModel.test.ts`

Expected: FAIL，原因是新字段未归一化/投影/展示。

- [ ] **Step 3: 实现有界归一化与 Run 展示**

新增显式 `CapabilityResolution` 类型和逐字段 validator；禁止 `...rawConfig`。实时事件与 summary 均归一化为同一 UI 类型。Run 详情展示 package、confidence、resolution mode、effective plan、工具名、router version 与短 fingerprint；旧 Run 展示“该历史运行未记录能力路由”。

- [ ] **Step 4: 运行 GREEN 与前端回归**

Run: `npm test -- src/lib/trajectory/normalizeTrajectoryEvent.test.ts src/lib/trajectory/TrajectoryCellProjection.test.ts src/lib/trajectory/trajectoryCellPresentation.test.ts src/lib/trajectory/trajectoryNodeDetailModel.test.ts`

Run: `npx eslint src/types/trajectory.ts src/lib/trajectory/normalizeTrajectoryEvent.ts src/lib/trajectory/TrajectoryCellProjection.ts src/lib/trajectory/trajectoryCellPresentation.ts src/lib/trajectory/trajectoryNodeDetailModel.ts`

Run: `npm run build`

Expected: 全部 PASS；不得启动 dev server。

- [ ] **Step 5: 提交 Task 4**

```bash
git add src/types/trajectory.ts src/lib/trajectory/normalizeTrajectoryEvent.ts src/lib/trajectory/TrajectoryCellProjection.ts src/lib/trajectory/trajectoryCellPresentation.ts src/lib/trajectory/trajectoryNodeDetailModel.ts src/lib/trajectory/normalizeTrajectoryEvent.test.ts src/lib/trajectory/TrajectoryCellProjection.test.ts src/lib/trajectory/trajectoryCellPresentation.test.ts src/lib/trajectory/trajectoryNodeDetailModel.test.ts
git commit -m "feat: 展示Run能力路由" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```

### Task 5: 行为评测、回归与文档收口

**Files:**
- Modify: `scripts/agent_behavior_eval.py`
- Modify: `test/fixtures/agent_behavior_eval_samples.json`
- Modify: `test/test_agent_behavior_eval.py`
- Modify: `docs/superpowers/specs/2026-08-27-run-capability-router.md`
- Modify: `docs/EXECUTION_LEDGER.md`

**Interfaces:**
- Consumes: route resolution、实际 announced tools、Prompt section IDs 和现有回答/工具观测。
- Produces: 能区分“工具已公告”“工具已调用”“Prompt 已加载”的离线验收结果和完整执行记录。

- [ ] **Step 1: 写 RED 评测 schema 测试**

fixture 新增并严格校验：

```json
{
  "expected_package_id": "weather",
  "expected_section_ids": ["app_identity", "current_date"],
  "expected_announced_tool_names": ["weather_forecast"],
  "forbidden_announced_tool_names": ["web_search", "search_flights"],
  "required_reason_codes": ["explicit_weather"]
}
```

至少覆盖规格矩阵的 22 类输入；同一 case 分别记录 announced、called、final answer，禁止从最终正文反推路由。

- [ ] **Step 2: 运行 RED 并实现最小评测扩展**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_agent_behavior_eval.py -q`

Expected: 先因新字段未支持 FAIL；实现后 PASS。实现只扩展已有脚本，不新增在线模型调用。

- [ ] **Step 3: 运行 API 全量相关回归与静态检查**

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_request_prep.py test/services/stream/test_agent_plan_tool_policy.py test/services/stream/test_agent_task_policy.py test/services/stream/test_agent_loop_lifecycle.py test/services/agent/test_continuation.py test/services/agent/test_events.py test/services/agent/test_emitter.py test/services/test_trajectory_query_service.py test/test_trajectory_api.py -q`

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app test scripts`

Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff format --check app test scripts`

Run: `git diff --check origin/master...HEAD`

Expected: 全部 PASS。

- [ ] **Step 4: 运行 UI 全量相关回归与静态检查**

在 UI 工作树运行：

Run: `npm test`

Run: `npx eslint src/types/trajectory.ts src/lib/trajectory/normalizeTrajectoryEvent.ts src/lib/trajectory/TrajectoryCellProjection.ts src/lib/trajectory/trajectoryCellPresentation.ts src/lib/trajectory/trajectoryNodeDetailModel.ts`

Run: `npm run build`

Run: `git diff --check origin/master...HEAD`

Expected: 全部 PASS；不得启动 dev server。

- [ ] **Step 5: 更新规格与执行台账**

在 spec 追加“本地验证证据”，逐条写实际命令、通过数、已覆盖边界和未验证边界。`docs/EXECUTION_LEDGER.md` 记录 API/UI commit、分支、工作树、测试结果；明确本地检查不等于 CI、部署或真实用户验收。

- [ ] **Step 6: 提交 Task 5**

```bash
git add scripts/agent_behavior_eval.py test/fixtures/agent_behavior_eval_samples.json test/test_agent_behavior_eval.py docs/superpowers/specs/2026-08-27-run-capability-router.md docs/superpowers/plans/2026-08-27-run-capability-router.md docs/EXECUTION_LEDGER.md
git commit -m "test: 扩展Run能力路由验收" -m "Co-Authored-By: Codex <noreply@anthropic.com>"
```
