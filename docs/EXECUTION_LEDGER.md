# Fusion 执行台账

> 本文件是 Fusion 项目的执行事实源，用来避免重复提出已经实施过的方向。回答“下一步”“还能怎么优化”“接下来做什么”之前，必须先读本文件，再核对两个子仓的 `git log` 和相关 `docs/superpowers/specs` 记录。

## 使用规则

- 不把 Codex memory 当执行记录；memory 只能作为偏好和约束提示。
- 每次重大功能、核心链路、发布门禁或真实回归完成后，在本文件补一条记录。
- 如果方向已经在“已完成基线”或“不要重复建议”中出现，不得作为下一步建议重新提出，除非用户明确要求返工或扩展。
- 如果当前文件和 `git log` 冲突，以当前 worktree 和 git 历史为准，并更新本文件。

## 已完成基线

| 领域 | 状态 | 关键证据 |
|---|---|---|
| Agent loop 基础拆分 | 已完成一轮 | `fusion-api` 2026-06-27 至 2026-06-28 相关 plan/spec；runner 状态、runtime、driver、summary 等拆分记录 |
| Agent 进度协议和前端状态 | 已完成一轮 | `fusion-api` / `fusion-ui` 2026-06-28 之后的 agent progress、执行过程、直接回答计划回归 |
| Search / Read Planner | 已完成 v1.1/v1.2/v1.3 一轮 | `fusion-api` commits `3933496`, `9bc7a9e`, `c4177ad`, `65bd446` |
| SourceCandidateRanker / Evidence Ledger | 已完成一轮 | `fusion-api` commits `78d3027`, `19423c3`；`fusion-ui` evidence 相关展示提交 |
| 工具过程 / 回答依据 UI | 已完成多轮收敛 | `fusion-ui` commits `1bc7fc5`, `0ea9aaa`, `fc8ede7`, `2391bee`, `cf65397` |
| 多模型验收矩阵 | 已完成并固化 | `fusion-api` commits `1068bf7`, `ab1d0c9`, `1c5364e`, `70f80e6`, `3b0b627`；`docs/MODEL_ACCEPTANCE_RUNBOOK.md` |
| 模型能力契约和展示 | 已完成一轮 | `fusion-api` commits `ed1da51`, `4ef734e`, `13c30ba`, `1e64334`；`fusion-ui` commits `f3a5033`, `272517e`, `9f29482`, `a33559a`, `bd1dbb5` |
| 小米 MiMo v2.5 模型更新 | 已完成 | `fusion-api` 模型目录治理和同步相关 commits `809b24b`, `4d29ae5`, `98ba0b9`, `855a39b` |
| CI / 发布门禁 | 已完成一轮 | `fusion-api` commit `9923fd0`；`fusion-ui` commits `bf9a112` 至 `68b7c9e`，以及 `014bb67` / `24601de` 指标修正 |
| Runtime Config 落库治理 | 已完成一轮 | `fusion-api` commits `092deb8`, `56ba600`, `d2af24f`, `6a69e55` |
| Runtime Config UI 观察面板 | 已完成一轮 | `fusion-ui` commits `fcff362`, `a27d3df`, `ea94879` |
| LiteLLM 观测标签透传 | 已完成修正 | `fusion-api` commits `aebf7a4`, `ab8eacc` |
| 图片文件解析链路修复 | 已完成 | `fusion-api` commit `21c2cf5` |
| PromptHub 正式迁移 | 已完成 | PromptHub published bundle / project-bound service token；Fusion 完整 LKG、`disabled -> shadow -> apply`、版本观测、管理保留域和部署持久化 smoke |
| Redis Stream 故障与并发隔离 | 已完成一轮 | `fusion-api` commit `363233f`；原子初始化/追加/检查/终态、task/message fencing、fail-fast 与 Redis 就绪检查 |
| 流式续传统一与页面恢复 | 已完成 P2 收尾 | `fusion-api` commits `70f391d`, `6d05512`, `18fb967` / `fusion-ui` commits `d4f66a9`, `765974c`；普通发送、Agent continuation 与页面恢复统一有限重连、安全游标、原子停止、部分输出持久化和取消终态收敛 |
| 生产性能基线 | 已完成首轮 | `docs/performance/2026-07-11-production-baseline.md`；API 单核约 145 RPS，公网 HTTP/2 P95 2.38 s，真实 SSE 1→3→5 并发 9/9 成功并完成零残留清理 |
| 生产完整性能矩阵 | 已完成 L1-L4 | `docs/performance/2026-07-12-production-full-matrix.md`；L1 600/600、L2 28/28、L3 恢复 9/9 + 停止/持久化 9/9、L4 30 分钟 60/60；0 重启/OOM，测试数据零残留，管理员页面导入并刷新持久化 |
| 长对话上下文预算管理 | 已完成第一阶段 | `fusion-api` commits `f5152be`, `b721edd`, `fc6ad22`；已知模型窗口按 85% 触发、75% 目标裁剪最旧完整 turn/工具事务，未知窗口保持兼容，估算失败 fail-closed；15 秒 SSE heartbeat；`docs/performance/2026-07-13-context-management-baseline.md`；生产 60%/80%/90% 阶梯与登录态 Chrome 刷新恢复通过 |
| 对话上下文状态展示 | 已完成 v1 | `fusion-api` / `fusion-ui` 本次提交；以最后一次 LLM round 为口径推送 `context_status_updated` v2，并在 assistant `usage.context` 中持久化可刷新快照；前端展示本轮实际/预计 Token、上下文窗口、剩余比例和自动裁剪明细，未知窗口、旧历史、失败状态与重连均安全降级 |
| 管理员审计中心 | 已完成 v1 | `fusion-api` commit `1ec0f01`、`fusion-ui` commits `f2517ca` / `5e47692`；全局用户/对话、消息、Agent/tool、文件元数据、压测留存和访问审计；`docs/acceptance/2026-07-11-admin-audit-center-v1.md` |
| 管理员压测审计详情 | 已完成 v1.1 | `fusion-api` commit `64d3452`、`fusion-ui` commit `2e3d857`；列表与详情安全契约分离、存量脏数据安全降级、L1-L4/资源/清理结果结构化详情、按需请求、响应式布局和管理页 no-store；生产登录态 Chrome 验收通过 |
| 管理员审计安全与身份展示 | 已完成 v1.2 | `fusion-api` commit `d036d89`、`fusion-ui` commit `9d1075d`；审计内容严格白名单投影、签名 URL/令牌脱敏、历史 schema 安全降级、用户昵称/用户名/ID 三元组、详情竞态隔离、Agent/tool 独立分页、压测指标语义与管理页 CSP 收紧；对抗式复审及生产登录态 Chrome 验收通过 |
| 管理员用户详情与对话联动 | 已完成 v1.3 | `fusion-ui` commit `d2d473b`；用户详情改为当前视口立即可见的弹窗，统一 loading/错误/重试并隔离迟到请求；可从详情直接进入该用户对话，用户筛选仅作用于本次关联导航；生产登录态 Chrome 验收通过 |
| 管理员时间展示与返回路由 | 已完成 v1.4 | `fusion-ui` commit `402644d`；对话列表展示创建/更新时间；Tab、用户详情、用户对话筛选、对话详情与压测详情以 URL 为事实源，支持手势返回、刷新恢复、深链接、旧压测记录跨页加载与 403 净 URL；生产登录态 Chrome 验收通过 |
| 管理员模型运营中心 | 已完成 v1 | `fusion-api` commits `5624241`, `1a2b456`、`fusion-ui` commit `3097c6b`；LiteLLM 当前目录与历史模型并集、健康/能力、持久化使用、Agent/压测摘要、模型到对话 URL 联动、目录降级和严格安全口径；`docs/acceptance/2026-07-12-admin-model-operations-v1.md` |

## 不要重复建议

除非用户明确要求扩展、返工或复盘，下列方向不要再作为“下一步”主动建议：

- “做多模型真实能力矩阵 / 多模型测验增强”。
- “做模型目录巡检/同步机制”。
- “做 Search / Read Planner v1.1/v1.2/v1.3”。
- “做 SourceCandidateRanker 或 Evidence Ledger 最小版”。
- “把 Prompt / Agent 策略 / 模型展示配置落库”。
- “做 CI / 发布门禁 v1”。
- “把 Runtime Config 页面做成配置编辑器”。当前产品定位是只读观察面板，写操作走 Agent + 测试 + CI/CD。

## 当前开放方向

当前没有已确认的 P0/P1 基础设施优化项。新的下一步应来自明确产品目标或线上问题证据，例如：

- 用户明确提出的新产品能力。
- 线上真实场景暴露的 bug、性能问题或回归。
- 已有验收报告中的慢响应、失败模型或质量风险进入产品策略调整。
- 知识库、项目空间等新方向，但必须先做现状确认和计划。

## 2026-08-26 主聊天系统提示词一期（代码与独立审查通过）

- 统一本地可信段落组装；用户偏好不会替代基础规则或因标题碰撞抑制工具/计划规则；修复历史查询年份限制。
- 新增 `system_prompt_prepared` 成功/失败结果，以及每次主聊天/收尾请求的有效 system 指纹；只保存安全元数据，无准备中状态、无新表、无 Skills。
- 迁移前已核对 dev 生效模板；收尾摘要差异保留现有解析路径，其他 PromptHub 消费者不变。
- 独立审查发现并修复空文本加非图片附件回归；API 22文件目标集合 `416 passed + 121 subtests`，ruff、架构及 diff 检查通过。
- 前端 27 文件 376 测试及隔离构建通过；37 个既有类型错误与干净同 SHA 基线逐字一致，无新增。跨仓独立审查通过，无未解决 P0/P1。
- 草稿 PR：API [#60](https://github.com/HyxiaoGe/fusion-api/pull/60)、UI [#48](https://github.com/HyxiaoGe/fusion-ui/pull/48)。UI 首轮 CI 全量 2350 测试及容器构建通过；API 首轮 CI 暴露两份旧集成测试的查询夹具/事件顺序不匹配，修正后主代理以 unittest 重跑 46 测试通过，生产代码未因此改动。最新分支 CI 结果以 PR Checks 为准。
- 本条记录本地实现、代码验证与独立审查；分支 CI 以配套 PR 为准，未合并部署，未完成新版本真实浏览器/模型验收。协议和范围见 [一期契约](superpowers/specs/2026-08-26-system-prompt-assembly.md)。

## 2026-08-27 主聊天 Prompt Runtime v2（仅本地开发与静态验证）

- Run 初始 Prompt 改为 `app_identity → 固定运行规则 → current_date → user_preferences`，模板版本更新为 `2026-08-27.1`；稳定前缀不再被动态日期和偏好截断。
- 默认 Web、高德、FlyAI、Plan 工具仍向主 LLM 公告 schema，但 Run 初始 Prompt 不再因工具可用而注入整段高德/FlyAI 领域 Prompt；自然表达“我现在在北京，我想去上海，你可以帮我吗”仍保留 `route_compare`。
- 工具调用前规则下沉到 description/schema 和后端校验；高德事实约束继续随实际结果进入 ToolMessage，FlyAI 实际结果新增完整事实边界及单班次接驳后续规则。旧高德整段 Prompt 常量和两条初始注入 helper 已删除。
- Trajectory 节点明确显示“Run 初始系统提示词”，正文说明后续 Round 可追加语言、修复、研究或总结规则；LLM 请求指纹改标“当轮实际系统消息指纹”。
- 本地证据：API 全量 `2918 passed, 2 skipped, 828 subtests`，Ruff check/format 通过；UI 全量 `2371 passed`，目标 ESLint 和 production build 通过；两仓 `git diff --check` 通过。
- 本条没有提交、推送、PR、CI、部署、本地服务或真实模型/浏览器验收。协议见 [v2 规格](superpowers/specs/2026-08-27-prompt-runtime-v2.md)。

## 2026-08-27 Run 级能力路由（本地实现与自动化回归）

- API 在首个 LLM Round 前以 `RunCapabilityResolution` 冻结最小能力包；同一 resolution 原子派生外部 tool definitions、handlers、bindings、announced/final tools、`update_plan` schema 和条件 Prompt sections。服务端共享 capability contract 同时约束当前 Run、实时事件、durable ledger 与历史 DTO；`AgentSession.run_config.capability_resolution` 是刷新和历史事实源。
- 权限边界默认收窄：普通 direct/transform/clarification 不公布工具；Deep Research 只开放完整 search/read；禁用工具、无 function calling、Knowledge Grounded 和未满足必需工具时不把 schema 发给模型。已授权 MCP alias 的只读元数据只参与禁用/无 FC 路由分类，不复用受执行预算裁剪的 ToolSet，也不授信未授权 alias。
- 自动化行为 fixture 共 501 条，其中 491 条路由记录通过真实 `build_agent_loop_call_config()` 与 `prepare_agent_loop_messages()` 核对 package、definitions、handlers、bindings、最终工具与 Prompt sections，不使用静态 JSON 自洽替身。矩阵覆盖问候、身份、日期、联网、URL、地图、航班/铁路、混合行程、Deep Research、Knowledge Grounded、模糊意图、抽象流程、按名词类型区分的定义类稳定知识与时效查询、URL query/自然动作边界、页面内搜索、中英文显式/自然内置、产品与 MCP 工具 hard deny 及最终再授权、当前请求及同对象中英文作用域、回指对象、`go online`/访问网络等全局中英文联网禁用、未知交通方式 fail-closed、交通方式否定、逗号/冒号/破折号子句边界及实体内部短横线保留、产品子集筛选、否定疑问、领域/解释补语、天气地点补语和中英文时序连接词。
- 本地证据：最新目标集（含生产 wiring 与 lifecycle 指纹契约）`654 passed + 1065 subtests`，其中路由单测 `506 passed`、真实组装 fixture `491 subtests`；API 权威全量 `3489 passed, 2 skipped, 1895 subtests`，Ruff、任务改动文件 format check 与 diff check 已通过。能力包指纹现覆盖完整 resolution、announced tools、安全 MCP bindings、task/network/evidence policy 与 Prompt 模板版本；实际 Prompt snapshot/fingerprint 继续单独证明 section/body。UI 全量 `2430 passed`、production build、目标 ESLint 与 diff check 已通过；最终替换式对抗审查结论为 CLEAN。全量首次发现 Trajectory 列表读取整个 `AgentSession.config`，共享路径以 `7e49f5f` 收窄为 capability resolution 轻量投影后，原失败单项与全量均通过。
- 当前状态仅为 API/UI 分支本地实现和静态/单元回归；没有推送、PR、CI、部署、真实模型或登录态浏览器验收。协议见 [Run 级能力路由规格](superpowers/specs/2026-08-27-run-capability-router.md)。

## 最近发布记录

| 日期 | 仓库 | commit | 内容 | 验证 |
|---|---|---|---|---|
| 2026-07-13 | `fusion-api` / `fusion-ui` | 本次提交 | 对话上下文状态 v1：单轮上下文安全事件、失败/停止快照与 JSONB 刷新恢复；输入区紧凑剩余比例入口、实际/预计 Token、裁剪说明、暗色/窄屏/键盘与中英文支持 | 对抗式复审关闭累计 Token 误用、历史回退、按钮闪烁和错误态误导；后端 Ruff、架构检查、全量 `996 tests`；前端全量 `1118 tests`、目标 ESLint、production build |
| 2026-07-13 | `fusion-api` | `f5152be+b721edd+fc6ad22` | 长对话上下文治理第一阶段：单轮上下文遥测、生成约束、Token 预算感知 Context Manager、完整 turn/工具事务原子裁剪、结构化错误、SSE heartbeat 与安全生产阶梯 runner | 对抗式复审无剩余 P0/P1；Ruff、架构检查、全量 `983 tests`；Actions `29218910325` / `29224101606` / `29225925681`；生产运行 `perf-20260713-051915-9512b94b` 四档全通过，90% 档从 232,305 裁到 192,280、移除 1 turn/2 messages，费用 `$0.564035`、资源 0 restart/OOM、会话/Token 清理成功且如实记录 1 个账号行残留；真实登录态 Chrome 新会话 `868bee65-2e50-4a3f-b3e5-eb538394c859` 即时渲染/流式完成/标题/刷新恢复通过，刷新记录 1 条非阻断 React `#418` hydration error |
| 2026-07-12 | `fusion-api` / `fusion-ui` | `api:5624241+1a2b456 / ui:3097c6b` | 管理员模型运营中心 v1：只读 current/history/unknown 模型视图、健康与能力、持久化用量、Agent/压测摘要、详情 URL、模型到对话联动；目录失败退避、脏 ID 降级和安全投影 | 独立对抗式复审无阻断；后端 `977 passed + 98 subtests`、Ruff、架构检查，前端 `1085 tests`、ESLint、build；Actions `29190231438` / `29190141564`；生产真实登录态 Chrome 验证 19 个模型、历史/当前详情、模型筛选 18 条对话、浏览器返回与刷新恢复，console 0；`%2F` 生产探测受 Browser Control 限制并已如实记录 |
| 2026-07-12 | `fusion-ui` | `402644d` | 管理员审计中心 v1.4：补齐对话创建/更新时间；将 Tab、用户详情、用户筛选对话、对话详情和压测详情接入 URL/history，并修复手动筛选 URL 漂移、旧压测深链、焦点恢复与约 1200px 宽度回退 | 独立对抗式复审无 P0-P2；目标 `38 tests`、全量 `1056 tests`、ESLint、build；Actions `29186419597`；生产镜像 commit 对齐且 0 重启/OOM，真实登录态 Chrome 验证用户详情/用户对话/对话详情 URL，后退序列恢复 `1` 条用户对话与用户列表，日期北京时间展示，用户详情和压测详情刷新恢复，深链关闭不退出管理中心，console 0 错误警告 |
| 2026-07-12 | `fusion-ui` | `d2d473b` | 管理员审计中心 v1.3：用户详情从表格底部改为可访问弹窗，新增按用户查看对话的跨 Tab 自动筛选，并清理权限失效和普通 Tab 切换时的关联状态 | 独立对抗式复审关闭 403 残留和旧筛选复用问题且无新增 P0-P3；目标 `17 tests`、全量 `1043 tests`、ESLint、build；Actions `29184669641`；生产镜像 commit 对齐且 0 重启/OOM，真实登录态 Chrome 验证即时 loading、详情弹窗完全位于视口、关联用户对话 `1` 条、普通对话恢复 `997` 条，console 0 错误警告 |
| 2026-07-12 | `fusion-api` / `fusion-ui` | `api:d036d89 / ui:9d1075d` | 管理员审计中心 v1.2 安全与展示收尾：严格投影消息/文件/工具/Agent 数据，统一敏感参数脱敏和历史 schema 降级；同名用户补充唯一用户名与用户 ID，修复详情竞态、独立分页、压测语义与 CSP 导航边界 | 独立对抗式复审关闭签名 URL、客户端导航 CSP、错误分类和刷新竞态问题且无新增 P0/P1；后端 `968 passed + 98 subtests`、Ruff、架构检查，前端 `1036 tests`、目标 ESLint、build；Actions `29179213390` / `29179194679`；生产镜像 commit 对齐且 0 重启/OOM，真实登录态 Chrome 验证用户三元组、用户详情、对话消息/Agent/tool、压测 v2、刷新清空详情和访问审计，console 0 错误警告 |
| 2026-07-12 | `fusion-api` / `fusion-ui` | `api:64d3452 / ui:2e3d857` | 管理员审计中心 v1.1：压测列表收敛为元数据安全投影，详情重新校验脱敏协议；前端按需展示 L1-L4、资源快照和清理结果，完善错误/空态、北京时间、窄屏布局与 no-store | 后端 `961 passed + 87 subtests`、Ruff、架构检查；前端 `1026 tests`、目标 ESLint、build；Actions `29177299252` / `29177299457`；生产镜像 commit 对齐且 0 重启/OOM，真实登录态 Chrome 验证详情展开/收起/刷新恢复、窄屏无横向溢出、console 0 错误警告，详情访问审计已落库 |
| 2026-07-12 | `fusion-api` / `fusion-ui` | `api:7b24bda+56699e2+18047c4+ad8252e+2852f80+d4733d6 / ui:ef6817c` | 生产 L1-L4 完整压测：安全 runner、HTTP/SSE/恢复/停止/30 分钟稳态、资源硬门禁、管理员导入协议；并修复并发首次鉴权、Prometheus 缺样本、prompt cache 假场景和 L4 回调契约 | 后端 `954 passed + 87 subtests` 及最终目标测试、前端 `1018 tests` + build；Actions `29163183361` / `29161977007`；生产最终运行 `perf-20260711-182155-ca9da746`，L1 600/600、L2 28/28、L3 18/18、L4 60/60，数据库恢复基线，真实登录态 Chrome 导入/刷新/console 0 错误 |
| 2026-07-11 | `fusion-api` / `fusion-ui` | `api:1ec0f01 / ui:f2517ca+5e47692` | 管理员审计中心 v1：跨用户只读检索、消息/Agent/tool/文件元数据安全投影、压测汇总留存、访问审计、点击劫持防护与 hydration 收敛 | 后端 `893 passed + 69 subtests`、Ruff、架构检查、Alembic 单 head；前端 `1017 tests`、ESLint、build；Actions `29153889249` / `29154128135` / `29154607305`；生产迁移、权限 200/403、真实新数据、文件/工具、压测导入与清理后保留、刷新恢复、console 0 新错误 |
| 2026-07-11 | `fusion-api` | 本次提交 | 生产性能首轮基线与可复用 HTTP/SSE runner：一次性认证、脱敏结果、阶梯门禁、会话/令牌/agent step 精确清理 | runner `12 passed`、Ruff、format、compileall、生产确认 guard；API/源站/公网分层压测；真实生产 SSE 1→3→5 并发 9/9 成功；关键容器无重启/OOM，数据库恢复测试前计数 |
| 2026-07-11 | `fusion-api` / `fusion-ui` | `api:70f391d+6d05512+18fb967 / ui:d4f66a9+765974c` | 流式可靠性 P2：显式 `initial/continuation` 模式、严格 Redis 状态判定、共享可恢复流执行器、页面刷新恢复、stop guard/task CAS、部分输出原子持久化与取消终态收敛 | 前端 `985 tests`、`npm run build`、目标文件 ESLint；后端 `857 tests + 69 subtests`、Ruff、架构检查；GitHub Actions 与部署后真实登录态 Chrome 新会话回归纳入发布门禁 |
| 2026-07-11 | `fusion-api` / `fusion-ui` | `api:363233f / ui:e7d596d` | 流式可靠性 P1：自适应追赶、发送自动重连、Redis fail-fast/就绪检查、孤儿流终态和 task/message fencing；真实回归补齐标题模型输出预算 | 前端 `944 tests`、`npm run build`、目标文件 ESLint；后端 `806 tests + 69 subtests`、Ruff、架构检查；GitHub Actions 与部署后真实登录态 Chrome 新会话回归 |
| 2026-07-10 | `prompthub` / `fusion-api` | `prompthub:70b371f / api:PromptHub 接入提交` | 11 个业务 Prompt 迁入 PromptHub：published bundle、只读服务令牌、完整本地 LKG、shadow/apply 切换、版本观测与回滚门禁 | PromptHub SDK `70 passed`、backend Ruff/架构/Alembic 单 head；Fusion `741 tests OK`、Ruff/架构；CI/CD、真实 dev shadow/apply 与登录态 Chrome 回归 |
| 2026-07-03 | `fusion-api` / `fusion-ui` | `api:aae8e87 / ui:c9d6eda` | 会话资料/文件体验 v1：同会话资料面板、资料复用、文件权限校验和历史附件元数据保真 | `.venv311/bin/python -m pytest test/test_file_service.py test/test_chat_service.py test/services/chat/test_message_builder.py -q`、`/opt/homebrew/bin/ruff check app test`、本次改动文件 `ruff format --check`、前端 `npm test`、`npm run build`；CI/CD 和真实 Chrome 回归待本次 push 后完成 |
| 2026-07-03 | `fusion-ui` | `ea94879` | 运行时配置页收敛为只读观察面板 | `npm test`、`npm run build`、CI/CD `28647885300`、真实 Chrome `/settings` 回归 |
| 2026-07-03 | `fusion-api` | `24601de` | CI 指标推送改走 nginx 9094 鉴权反代 | GitHub Actions / dev 发布门禁 |
| 2026-07-02 | `fusion-api` | `3b0b627` | 实现多模型真实验收矩阵 | `docs/MODEL_ACCEPTANCE_RUNBOOK.md`，`reports/model-acceptance/report-20260702-080341.md` |

## 下一步建议前检查清单

1. 读本文件。
2. 运行并阅读：
   - `git -C /Users/sean/code/fusion/fusion-api log --oneline -40`
   - `git -C /Users/sean/code/fusion/fusion-ui log --oneline -40`
3. 用 `rg` 搜索相关关键词，至少覆盖 `docs/superpowers/specs` 和 `docs/MODEL_ACCEPTANCE_RUNBOOK.md`。
4. 先列“已完成事实”，再列“不能重复建议”，最后才给新的建议。
5. 如果没有高置信下一步，直接说“当前不建议继续开基础设施优化坑”，不要硬凑方向。
