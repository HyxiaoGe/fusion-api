# SDD ledger — plan: docs/superpowers/plans/2026-08-27-run-capability-router.md

## 预检

| 任务/接口 | 生产者 | 消费者 | 检查结果 |
|---|---|---|---|
| Task 1 → Task 2 | `RunCapabilityResolution`、`resolve_run_capability_route` | call config 原子物化与 Prompt 组装 | 一致；Task 2 只能消费 Task 1 冻结结果，不得二次扩容 |
| Task 1 → Task 3 | 安全 route 字段与序列化 | run config、event、Run summary | 一致；bundle fingerprint 在 Task 3 补齐，不改变路由语义 |
| Task 2 → Task 3 | `AgentLoopCallConfig.capability_resolution` 与实际 tools | lifecycle `_start_run` | 一致；route 必须在 `_start_run` 前冻结 |
| Task 3 → Task 4 | 显式 `capability_resolution` DTO | UI 归一化、投影和详情 | 一致；UI 不透传整个 run config |
| Task 1/2 → Task 5 | package、announced tools、section IDs | behavior eval schema | 一致；评分区分公告、调用和最终回答 |
| Task 3/4 → Task 5 | API/UI route 展示 | 回归和执行台账 | 一致；本地证据与 CI/dev/真实页面分层记录 |
| Task 1 自洽性 | RED 矩阵、路由纯函数、GREEN | 同一文件集与签名 | 一致；不依赖 IO/LLM |
| Task 2 自洽性 | 错误契约翻转、原子过滤、条件 Prompt | 同一 call config | 一致；provider reasoning 仍在最终工具后适配 |
| Task 3 自洽性 | run config、event、summary、旧 Run | 显式 DTO 与白名单 | 一致；实际 Prompt 指纹与 bundle fingerprint 分开 |
| Task 4 自洽性 | wire 类型、validator、projection、presentation | targeted tests/build | 一致；脚本不存在 `lint`，计划已改为显式 `npx eslint` |
| Task 5 自洽性 | fixture schema、脚本、测试、台账 | 已存在 `test/test_agent_behavior_eval.py` | 一致；无在线模型调用 |

Ruling: 低置信不采用旧 `legacy_fallback` 全量工具 — 新 spec 的 revised recommendation 是绑定合同 — 代价是未知普通 MCP 只能在精确 alias 命中时公开。

Ruling: `bundle_fingerprint` 在 `_start_run` 前覆盖完整 resolution、announced tools、安全 MCP bindings、task/network/evidence policy 与 Prompt 模板版本；随后得到的实际 Prompt section/body 继续由独立 snapshot/fingerprint 证明 — 验收需同时查看能力包与实际 Prompt 两个指纹。

Ruling: continuation 每次创建新 Run 时按当前消息、当前授权和恢复的 task/plan policy 重新确定性路由；本期不恢复旧 route schema — spec 未要求 Skill/route 续跑冻结，且当前权限必须优先 — 代价是目录或 router 版本变化后新 continuation 可记录不同包，但旧 Run 不被改写。

Task 1: dispatched (BASE 7f4990af47b423113ff18e831a5acae669381c8b, agent /root/router_task1_impl)
Task 1: review failed (6 Important, 0 Critical/Minor) — knowledge-grounded 边界、Deep Research 必需工具、裸 A 到 B、外部公告摘要、verified 优先级、非出行城市误召回。
Task 1: fix round 1/5 dispatched to /root/router_task1_impl (FIX_BASE 6aff089f4616ca130198edf20b8150331df19767).
Task 1: fix round 1/5 (6 addressed, 2 open — 自然起终点三类漏召回；“发展路线”误触发；commits 6aff089..69d0762).
Task 1: fix round 2/5 dispatched to /root/router_task1_impl (FIX_BASE 69d0762ed38d0fc7322addc2b02ceedf1146a398).
Task 1: fix round 2/5 (2 addressed, 2 open — structured endpoint 组织关系误授权；大学端点真实路线被黑名单压掉；commits 69d0762..7e25d81).
Task 1: fix round 3/5 dispatched to /root/router_task1_impl (FIX_BASE 7e25d813fd922a2d6c43e3d92099ca8bf8934c75).
Task 1: fix round 3/5 (2 addressed, 2 open — 明确路线常见地点漏召回；“产品中心/研发中心”组织误授权；commits 7e25d81..2b7d285).
Task 1: fix round 4/5 dispatched to fresh implementer (FIX_BASE 2b7d285418c43e90f2244ec5bfb04a82083d1b5d).
Task 1: fix round 4/5 (2 addressed, 2 open — 抽象流程“怎么走”误授权；普通“的路线”漏召回；commits 2b7d285..d6a92cf).
Task 1: fix round 5/5 dispatched to fresh implementer (FIX_BASE d6a92cfdb99ef0089b310bbd1bb3b13540dfd0f8).
Task 1: fix round 5/5 (1 addressed, 2 open — 任意 bounded 抽象端点仍可被“怎么走/路线”提升；commits d6a92cf..1ada245; breaker tripped).
Task 1: Ruling: 抽象端点误授权是真实且下游依赖的 load-bearing finding；Task 2 在接入 Agent Loop 前必须把未知 BOUNDED 端点改为默认拒绝，只允许确认物理地点证据。代价：未识别 POI 会进入 clarification_only，产生可恢复的工具漏召回，而不是错误开放地图能力。
Task 1: complete (commits 7f4990a..1ada245, 5 rounds; 2 load-bearing trigger variants carried into Task 2 by ruling).

Task 2: dispatched (BASE 1ada245fe811e23668f867a695fce6d5ac362589, agent /root/router_task2_impl).
Task 2: review failed (2 Important — 空工具 plan_on schema 未禁止 planned_tools；disable/no-FC 精确 MCP 请求漏联网边界).
Task 2: fix round 1/5 dispatched to /root/router_task2_impl (FIX_BASE 5320e4d).
Task 2: fix round 1/5 (1 addressed, 1 open — MCP alias 分类元数据仍受 max_tools/schema-bytes 执行预算裁剪；commits 5320e4d..a14ae8c).
Task 2: fix round 2/5 dispatched to /root/router_task2_impl (FIX_BASE a14ae8c).
Task 2: fix round 2/5 (1 addressed, 0 open; commits a14ae8c..2658679).
Task 2: complete (commits 1ada245..2658679, review clean).

Task 3: dispatched (BASE 26586793924133594f79bf18aa2363557c4bc801, agent /root/router_task3_impl).
Task 3: review failed (1 Important — Trajectory resolution 仅形状校验，update_plan/非法包工具组合可进入实时与历史协议).
Task 3: fix round 1/5 dispatched to /root/router_task3_impl (FIX_BASE 92520fb).
Task 3: fix round 1/5 complete (1 Important addressed, 0 open; commits 92520fb..58b9b82; target 123 passed; adjacent 185 passed).
Task 4: complete (UI commit c239fbd; target 69 passed; Trajectory regression 303 passed; ESLint/build/diff check green; no local services, push, deploy, or browser acceptance).
Task 4: fix round 1/5 complete (2 Important addressed, 0 open; UI commits c239fbd..246430d; contract/panel target 141 passed; Trajectory regression 336 passed; ESLint/build/diff check green).
Task 3: fix round 2/5 complete (1 Important addressed, 0 open; commits 58b9b82..b664a93; target 127 passed / 73 subtests; adjacent 186 passed / 42 subtests; Ruff/format/diff check green).
Task 4: fix round 2/5 complete (1 Important addressed, 0 open; UI commits 246430d..5b603d5; target 151 passed; Trajectory regression 346 passed; ESLint/build/diff check green).
Task 5: final local implementation complete, release pending (START_BASE b664a93; COMMIT_BASE 7e49f5f after shared query-projection fix; behavior fixture 501 total / 491 route records; latest target 654 passed / 1065 subtests including production wiring, route unit 506 passed and real assembly 491 subtests; fresh authoritative API full 3489 passed / 2 skipped / 1895 subtests and static gates green; UI full 2430 passed, production build, ESLint and diff check green; Prompt/Trajectory independent review and final replacement adversarial review CLEAN; no services, CI, deploy, or browser acceptance yet).
Task 3: regression fix round 3/5 complete (1 authoritative full-regression failure addressed; commits b664a93..7e49f5f; failing test 1 passed; target 128 passed / 73 subtests; adjacent 186 passed / 42 subtests; qualified SQLite/Postgres JSON path compile checks and Ruff/format/diff green).
