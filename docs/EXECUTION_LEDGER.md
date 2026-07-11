# Fusion 执行台账

> 本文件是 Fusion 项目的执行事实源，用来避免重复提出已经实施过的方向。回答“下一步”“还能怎么优化”“接下来做什么”之前，必须先读本文件，再核对两个子仓的 `git log` 和相关 `docs/superpowers` 记录。

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

## 最近发布记录

| 日期 | 仓库 | commit | 内容 | 验证 |
|---|---|---|---|---|
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
3. 用 `rg` 搜索相关关键词，至少覆盖 `docs/superpowers` 和 `docs/MODEL_ACCEPTANCE_RUNBOOK.md`。
4. 先列“已完成事实”，再列“不能重复建议”，最后才给新的建议。
5. 如果没有高置信下一步，直接说“当前不建议继续开基础设施优化坑”，不要硬凑方向。
