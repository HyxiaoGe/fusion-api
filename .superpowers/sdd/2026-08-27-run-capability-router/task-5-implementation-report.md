# Task 5 实现报告：能力路由行为评测与 API 回归收口

## 元数据

- 完成时间（Asia/Shanghai）：2026-08-27 19:36 CST
- 开始 BASE：`b664a932b247683bbdb381556886e5ff32093691`
- 提交 BASE：`7e49f5f`（Task 5 全量发现的共享 Trajectory query projection 回归由上游先行收口）
- Task 5 HEAD：本报告所在提交；最终 SHA 以 `git log -1` 为准
- 工作区：`/Users/sean/code/fusion/.worktrees/run-capability-router-api`
- 范围：行为评测协议、路由/Prompt 真集成 fixture、旧 stream fixture 迁移、API 回归和本地证据文档
- 明确未做：未启动 API/UI/Docker，未访问网络，未改 Task 3 capability/Trajectory 生产契约，未改 UI、服务、计划或 Task 3+ 测试

## TDD RED 证据

### 1. 评测协议与矩阵 RED

先只增加协议校验、评分与矩阵测试，未实现脚本和 fixture：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_agent_behavior_eval.py -q
```

结果：`13 failed, 19 passed, 21 subtests passed, 5 warnings`。失败分别证明新增字段没有被验证或评分、缺失 observation 没有失败、dry-run 没有输出期望字段、fixture 不足 24+4。

随后为真实联结增加非空断言，单独运行：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q test/test_agent_behavior_eval.py::RunCapabilityBehaviorEvalIntegrationTests::test_route_fixture_runs_through_real_router_and_prompt_assembly
```

结果：`1 failed`，路由 fixture 数量为 0，未满足至少 29 条真实集成样本。

### 2. stream 旧假设 RED

基线命令：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q test/test_agent_behavior_eval.py test/test_stream_handler.py
```

结果：`7 failed, 41 passed, 21 subtests passed, 5 warnings`。七项失败共同证明旧 fixture 仍假设普通 `hi` 或任意 URL 会得到旧全量工具；模型返回的调用在新路由下属于未公告工具。

迁移 fixture 后首次精确回归为 `2 failed, 5 passed`：两个 URL 用例仍未声明 `searchCapable`，且“请看 URL”不属于显式读取动作。补齐真实能力并改为“请阅读 URL”后进入 GREEN；没有增加测试后门或恢复全工具行为。

## 实现内容

### 行为评测协议

- `scripts/agent_behavior_eval.py` 保留既有 V1 字段与评分逻辑，新增 8 个可选 Run capability 期望字段。
- package、resolution mode、plan mode、布尔值、字符串数组和重复项在加载阶段严格校验。
- 精确字段按原始顺序比较；reason codes 使用必需子集语义。样本声明期望字段而 observation 缺失时明确报告缺失，空值不能冒充通过。
- dry-run 继续不调用 LLM、搜索或浏览器，并输出新增期望字段供外部观测填充。

### 24 主样本与对抗矩阵

- fixture 总数 40：既有 V1 10 条、Run 路由 30 条。
- 30 条路由样本包含 24 条主样本和 6 条对抗记录；对抗记录形成 5 个独立变体：物理 A→B、组织语境、用户偏好越权、禁用工具联网边界、topic switch 两个独立 observation。
- 每条路由样本精确写出 package、announced tools、Prompt section IDs、resolution mode、必需 reason codes、effective plan mode 与 network boundary；可确定不会实际调用工具的样本另写 `expected_called_tools=[]`。

### 真实 router 与 Prompt assembly 联结

- 测试使用真实 `AMAP_PRODUCT_DEFINITIONS`、`FLYAI_TRAVEL_DEFINITIONS` 及配对 handler/binding 进入 `build_agent_loop_call_config()`。
- 同一个 call config 再交给 `prepare_agent_loop_messages(..., preprocess_user_input=False)`，因此 section IDs 来自生产 `assemble_system_prompt` 路径。
- 逐样本核对 resolution external tools、实际传给模型的 definitions、active handlers/bindings 对应的 announced tools、`final_tool_names`、日期标记与 Prompt sections；未复制第二套路由判断器，也没有网络调用。

### stream fixture 迁移

- 普通 stop 路径改为只断言 `app_identity`。
- 搜索、checkpoint、失败、max-steps 与 summary-timeout 用例改用明确最新信息搜索请求。
- URL run-start 和降级用例改用明确 URL 阅读/总结请求，并声明真实的 function calling + search 能力。
- URL run-start 只公告 `url_read`，不恢复旧 `web_search,url_read` 全量假设。

## 改动文件

- `scripts/agent_behavior_eval.py`
- `test/fixtures/agent_behavior_eval_samples.json`
- `test/test_agent_behavior_eval.py`
- `test/test_stream_handler.py`
- `docs/superpowers/specs/2026-08-27-run-capability-router.md`
- `docs/EXECUTION_LEDGER.md`
- `.superpowers/sdd/2026-08-27-run-capability-router/progress.md`
- `.superpowers/sdd/2026-08-27-run-capability-router/task-5-implementation-report.md`

## GREEN 与回归证据

评测覆盖文件：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q test/test_agent_behavior_eval.py
```

结果：`23 passed, 98 subtests passed, 5 warnings in 1.83s`。

精确七项 stream 迁移：`7 passed, 5 warnings in 11.67s`。完整 stream handler：`31 passed, 5 warnings in 12.02s`。

Task 5 指定目标集：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q test/test_agent_behavior_eval.py test/services/stream/test_run_capability_router.py test/services/stream/test_agent_loop_request_prep.py test/test_stream_handler.py
```

基于最终提交 BASE `7e49f5f` 的结果：`182 passed, 126 subtests passed, 5 warnings in 12.09s`。

扩大集：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q test/services/stream test/services/agent test/services/test_trajectory_query_service.py test/test_trajectory_api.py test/test_agent_behavior_eval.py test/test_stream_handler.py
```

基于最终提交 BASE `7e49f5f` 的结果：`1239 passed, 570 subtests passed, 8 warnings in 27.79s`。

仓库权威全量首次运行：`1 failed, 3056 passed, 2 skipped, 972 subtests passed, 9 warnings in 47.46s`。唯一失败为 Task 3 共享查询投影测试 `test_general_trajectory_and_other_node_details_do_not_read_system_prompt_body`：general trajectory SQL 读取了 `agent_sessions.config`。Task 5 没有编辑该共享路径；上游提交 `7e49f5f fix: 收窄轨迹能力配置读取` 后，先重跑原失败单项：`1 passed, 5 warnings in 1.96s`。

随后在最终提交 BASE 上重跑权威全量：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest -q --tb=short test/
```

结果：`3058 passed, 2 skipped, 972 subtests passed, 9 warnings in 46.00s`。

静态门禁初轮：`ruff check` 通过；`ruff format --check` 正确发现 `test/test_agent_behavior_eval.py` 需格式化。执行单文件 Ruff format 后，最终 `ruff check`、`ruff format --check` 与暂存差异检查均通过。

## 自审与剩余风险

- V1 样本继续可加载、评分和 dry-run；新字段全部可选，不强迫旧 observation 提供 Run route 数据。
- 路由 fixture 的期望值没有从实际结果运行时回填，错误 production route 会让固定 fixture 断言失败。
- definitions、handlers、bindings、announced/final tools 和 Prompt sections 均从一个真实 call config 派生；测试没有发送 schema 给模型，也没有执行外部 handler。
- 旧 stream fixture 只为真正需要工具的场景补充明确用户意图；普通直接回答保持空工具。
- 全量暴露的共享 query projection 回归已由独立上游提交关闭；Task 5 提交不夹带该生产修复，也没有把首次失败改写成 Task 5 GREEN。
- 本地测试未证明 CI、部署、多模型真实选择质量、实际 tool calls、UI 实时/刷新一致性或登录态浏览器体验。这些属于后续发布门禁，Task 5 没有伪造结论。

## 最终发布审查补强

最终多轮对抗审查在发布前继续拦截了显式内置/产品工具 deny 被自然语言重新放开、定义类稳定知识被 fresh/verified/product 词法抢占、定义尾部 `in ...` 吞掉真实新闻/价格查询、合法 URL query 被拆成公网搜索动作、未知交通方式回退全三工具、当前请求作用域和回指对象枚举不足、`never go online`/禁止访问网络等全局中英文联网禁用枚举不足、交通方式否定跨逗号/冒号/破折号子句、air pollution 误判以及产品子集筛选/否定疑问漏召回。修复保持 router version `2026-08-27.2`，以 hard deny、按名词类型区分的定义请求前置判定、URI/产品动作子句边界和未知方式 fail-closed 处理根因，不恢复低置信全工具包。

最终 fixture 共 501 条行为样本，其中 491 条通过真实 `build_agent_loop_call_config()` 与 `prepare_agent_loop_messages()` 校验 package、definitions、handlers、bindings、announced/final tools、Prompt sections、plan/date/boundary 与最多三工具边界。最新目标集（含生产 wiring 与 lifecycle 指纹契约）为 `654 passed + 1065 subtests`，其中路由单测 `506 passed`、真实组装 fixture `491 subtests`。API 权威全量 `3489 passed, 2 skipped, 1895 subtests`，Ruff、任务改动文件 format check 与 diff check 通过；能力包指纹覆盖完整 resolution、announced tools、安全 MCP bindings、task/network/evidence policy 与模板版本，实际 Prompt 仍由独立 snapshot/fingerprint 证明。UI 全量 `2430 passed` 且 production build、ESLint 与 diff check 通过；最终替换式对抗审查结论为 CLEAN。CI、dev 部署和真实登录态页面仍属于发布门禁。
