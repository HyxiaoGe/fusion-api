# Agent 计划模式 v1 设计

## 背景

Fusion 已有 Agent Progress Protocol v2，能够通过 Redis Stream / SSE 传输
`plan_snapshot`、`plan_step_updated` 和 `run_progress_updated`，并把 compact
snapshot 持久化到 `agent_progress_snapshots`。当前计划由工具生命周期在首次
工具调用时推断生成，只能解释已经发生的执行，不能在复杂任务开始前表达模型
准备怎样完成任务。

本阶段在现有 Agent loop 内增加通用计划控制能力，不新增独立 Planner 请求，
不解析 reasoning 文本，也不先引入 DAG 调度器。模型通过内部控制工具
`update_plan` 提交或修订计划，PlanCoordinator 负责验证、revision、状态迁移
和事件发送。

## 目标

- 简单问答默认不展示计划，也不增加额外模型调用。
- 复杂任务可在首次外部工具调用前产生语义计划。
- 计划修订、工具执行、断线重连和历史刷新使用同一个计划状态。
- 控制工具不计入外部工具额度，计划修订使用独立的小额度。
- 已有 observed 计划和旧客户端保持兼容。
- 计划内容只包含安全的任务摘要，不持久化 reasoning、原始工具参数或结果。

## 非目标

- 不做独立 Planner 模型或额外强制 LLM planning call。
- 不做完整 DAG 调度、并发编排、用户编辑、暂停确认或预算调整。
- 不让前端根据标题、reasoning 或工具参数猜测计划状态。
- 不把 `update_plan` 暴露成 MCP Server 或产品工具。
- 第一阶段不实现环形图最终视觉。

## 模式

`plan_mode` 取值：

- `auto`：默认。模型可为复杂、多步骤任务调用 `update_plan`；简单问答可直接
  回答。
- `on`：首次外部工具调用前必须存在有效的 model plan。没有计划时，外部工具
  不解锁；模型直接回答、返回空工具调用或协议异常时同样进入有界结构修复。
  外部工具必须显式提交所属计划项 ID，不能依赖服务端猜测。
- `off`：不向模型提供 `update_plan`，保留当前 observed plan 行为。

前端提供计划模式开关，默认使用 `auto`，用户可显式开启 `on`。模型不支持
function calling 时，前端和后端都会把请求安全降级为 `off`，不会展示一个实际
无法执行的计划模式。

## 协议扩展

沿用 `protocol_version: 2`，只增加可选字段，旧客户端可以忽略。

### 计划快照

`plan_snapshot` 增加：

- `source`: `observed | model`
- `mode`: `auto | on | off`
- `reason`: 服务端生成的安全状态原因，如 `model_update`、`tool_progress`、
  `terminal_stop`。

### 计划项

在现有字段上增加：

- `depends_on`: 稳定计划项 ID 数组。
- `planned_tools`: 模型计划使用的公开工具别名数组。

控制工具对模型采用 `explanation + plan[]` 结构；每个计划项必须提交 `id`、
`step`、非终态 `status`（`pending | in_progress`）和 `planned_tools`，可选提交
`kind` 与 `depends_on`。服务端
归一化为内部 `reason + items[]` 状态。`id` 接受数字、字母、下划线和连字符，
必须在同一 plan 内稳定。模型修订标题或状态时不得无故更换 ID。
时间戳、可选步骤和结果摘要留到计划交互阶段再扩展，v1 不提前增加未消费字段。

### 工具关联

`tool_call_started`、`tool_call_completed` 和 `tool_result_digest` 可增加
`plan_item_id`。它只表达本次调用主要服务哪个计划项，不改变现有
`step_id/tool_call_id` 语义。

模型调用外部工具时使用保留参数 `_plan_item_id` 指定所属步骤；Fusion 在真实
工具执行前移除该参数。`on` 模式必须显式提交并与当前未终态步骤及
`planned_tools` 精确匹配，不满足时拒绝执行且不消耗外部工具额度。`auto`
模式保留兼容推断：只有目标工具在当前未终态计划中恰好对应一个候选项时才允许
自动关联。只要同名工具存在多个候选项，即使调用数量相同，也不得根据调用顺序
猜测对应关系。

## 内部控制工具

`update_plan` 是 Agent loop 内部控制工具：

- 由服务端构造 definition，只进入支持 function calling 的模型请求。
- 不进入 dynamic MCP handler，不经过外部 ToolExecutor。
- 不产生产品结果卡片，不计入 `max_tool_calls`。
- 控制调用本身仍保留最小安全审计，但不得向用户正文泄漏函数名或原始 JSON。
- 单个 run 最多接受 6 次有效修订；无效请求最多允许 2 次结构修复。
- 同一轮如果同时返回 `update_plan` 和外部工具，先应用计划，再判断外部工具
  是否可执行。
- 仅用于控制面的 reasoning、正文和回执不会写入 SSE、content blocks 或消息
  checkpoint；用户可见回答和思考流会定向净化计划控制字段。

PlanCoordinator 校验：

- 计划项数量 2 到 12。
- ID、标题、状态、依赖和 planned tools 均使用白名单与长度上限。
- 依赖只能引用同一快照内 ID，拒绝自依赖和环。
- 同一时刻最多一个 `running` 项。
- `completed/failed/skipped/blocked` 不得在模型修订中无依据回退为
  `pending`。
- 已尝试或终态项的标题、类型、依赖和预计工具由服务端锁定；模型后续修订出现
  轻微回声漂移时保留服务端 canonical 元数据，只接受安全状态更新。删除既有
  执行项或破坏稳定 ID 仍拒绝，并返回当前 canonical 计划帮助模型修复。
- revision 由服务端生成，忽略模型提交的 revision。

## 执行状态

PlanCoordinator 是 model plan 的 revision 和状态唯一所有者：

1. 有效 `update_plan` 生成全量 `plan_snapshot`。
2. 外部工具开始时，以显式或安全映射的 `plan_item_id` 把对应项推进到
   `running`。
3. 工具结束按同一计划项记录服务端执行证据，但计划项在 run 内保持
   `running`，允许同一计划项跨轮执行多个工具调用；复用的成功结果同样计入。
   被公告门禁、额度或上下文门禁拦截的调用不能先标 `running`。
4. run 正常完成前，仍为 `running` 的项必须由服务端终态化：
   - 有成功工具证据且没有尚未修复失败的工具项进入 `completed`。
   - 只有失败证据的工具项进入 `blocked`。
   - 即使 run 因限制或计划修复耗尽进入 `incomplete`，已有成功工具证据仍
     保持 `completed`；已经生成最终正文时，回答、整理和推理步骤可完成。
   - model plan 保留后端最终状态，不由前端推断。
   - 回答、整理和推理步骤可随正常回答完成。
   - 未执行的搜索、读取和工具步骤变为 `skipped`，仍在修复中的步骤变为
     `blocked`。
5. 失败时运行项变为 `failed`；取消或被取代时未完成项变为 `skipped`；
   触顶和不完整终态的未完成项变为 `blocked`。已有 terminal 状态不得被终态
   收口覆盖。

旧 observed plan 继续走现有工具生命周期逻辑，`source=observed`；它的固定
revision 规则只作为兼容路径，不能用于 model plan。

## 持久化与性能

- v1 不新增数据库表，复用 `agent_progress_snapshots.snapshot` JSONB。
- progress reducer 必须同时按 `plan_id + revision` 忽略倒退快照。
- 续跑从前一次 session 的 run config 继承 `plan_mode`，避免用户开启后被静默
  恢复成默认值。
- 高频 `plan_step_updated` 不应让每个事件同步 query + commit；第一阶段至少
  将同一次控制更新折叠为一次快照写入，终态与断线关键点强制 flush。
- Redis/SSE 写入仍是主链路；数据库 snapshot 失败不能阻断回答。

## 前端规则

- 按 `run_id + sequence` 防重放，按 `plan_id + revision` 防计划倒退。
- `source=model` 的终态完全信任服务端，不执行 `kind=other` 自动完成推断。
- `source=observed` 保留旧兼容归一化。
- live、Redis replay 和历史 hydration 使用同一个 `AgentPlanState`。
- 成功完成的 model plan 仍应可见；不能被 `ExecutionProcess` 直接替换。
- 计划面板使用环形进度总览，hover / focus 可查看步骤、依赖和状态；窄屏和
  `prefers-reduced-motion` 保留可访问降级。

## 分阶段

### 阶段 A：协议与状态基础（已完成）

- 后端 schema、PlanCoordinator、`update_plan` 控制路径与额度隔离。
- 前端类型、revision 防乱序、model plan 终态和历史恢复兼容。
- 不做最终视觉，不发布。

### 阶段 B：真实模型与执行关联（已完成）

- 接入工具 `plan_item_id`。
- 用真实任务验证计划、修订、降级、触顶和取消。
- 关闭跨模型结构差异与协议泄漏问题。

### 阶段 C：环形计划交互（已完成）

- 以计划项完成比例绘制环形总览。
- hover / focus 展示步骤、依赖和结果摘要。
- 兼容 prefers-reduced-motion、键盘操作和窄屏。

## 验收矩阵

| 场景 | 预期 |
| --- | --- |
| 简单事实问答，`auto` | 无计划、无控制调用、正常回答 |
| 简单事实问答，`off` | 与当前 Agent loop 一致 |
| 行程规划，`auto` | 首个外部工具前出现 model plan |
| 行程规划，`on` 且首轮直接调外部工具 | 阻止外部调用并要求结构化补计划 |
| 行程规划，`on` 且首轮直接回答/异常终止 | 隐藏未通过门禁的内容并有界修复 |
| 同轮计划 + 航班/高铁并行调用 | 先应用计划，两类工具分别关联计划项 |
| `on` 模式唯一工具候选但缺步骤 ID | 仍拒绝执行，不以唯一候选绕过显式绑定 |
| 工具参数不可用 | 不扣除未执行工具额度，计划进入修复或阻塞状态 |
| 工具调用失败/降级 | 工具摘要与计划状态一致，不伪装 completed |
| 模型修订计划 | revision 单调，旧快照/旧 step update 被忽略 |
| 无效 ID、依赖环、超长标题 | 拒绝并给模型结构化修复机会，不泄漏到正文 |
| 达到计划修订上限 | 停止接受控制更新，但外部工具和回答可继续 |
| 达到外部工具上限 | 不把 update_plan 计入提示给用户的工具数量 |
| SSE 重放乱序/重复 | UI 与数据库快照保持最新 revision |
| 刷新历史 | model plan、状态和关联信息恢复一致 |
| 用户取消/请求被取代 | 运行中计划项不会显示为 completed |
| 正常完成 | 必需计划项状态可信，成功页面仍展示计划 |
| 豆包、Kimi、DeepSeek | 不泄漏内部控制协议；数字 ID 和结构异常可恢复 |
| 通用研究任务 | 不依赖行程专用字段 |
| 文件分析任务 | 不依赖联网或 MCP 才能产生计划 |

## 发布边界

这是 Agent loop、流式协议和持久化的高风险改动。当前本地阶段已经过独立
对抗式审查、全量代码门禁，并在真实依赖环境完成简单问答、复杂行程、失败修复、
刷新恢复和多模型回归。未获得用户发布授权前，不推送或部署。
