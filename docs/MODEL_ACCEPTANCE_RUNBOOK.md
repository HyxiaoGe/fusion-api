# Fusion 全模型验收 Runbook

## 目标

把“全模型验收/能力测试”固化为可复跑流程：先用脚本跑模型目录、流式对话、工具契约和质量风险，再用已登录 Chrome 做少量真实 UI 补证。该流程用于模型新增、下线、能力标注调整、Agent loop / 搜索链路 / SSE / 模型选择器等核心路径变更后的验收。

## 边界

- 自动脚本会真实调用 Fusion API 和模型供应商，可能产生费用。
- 自动脚本默认走 `stream` 链路，覆盖 `/api/chat/send` 的真实 SSE 产品路径。
- 自动脚本不登录浏览器、不检查 UI 布局、不上传真实文件。
- 真实 Chrome 回归只复用用户已打开且已登录的 Fusion 标签，禁止新开 Chrome/标签。
- 没有可复用 Chrome 标签时记录阻塞，不用本地服务、旧历史页或 mock 数据替代。

## 用例分层

| 层级 | 覆盖内容 | 执行方式 |
|---|---|---|
| A1 基础可用 | 全模型能否返回正常正文 | 自动脚本 |
| A2 工具守卫 | 简单问题不应触发搜索/读取/依据 | 自动脚本 |
| A3 实时问题 | 可联网模型应搜索并读取关键来源；不可联网模型应说明能力边界 | 自动脚本 |
| B1 视觉能力 | `vision=true` 的模型标签和上传入口 | Chrome 补证 |
| B2 深度思考 | 深度模型不泄露 `<think>` 标签，不误触工具 | 自动脚本 |
| R1 刷新恢复 | 搜索对话和普通对话刷新后恢复正确 | Chrome 补证 |

## 自动脚本

脚本路径：

```bash
scripts/model_catalog_eval_baseline.py
```

### 1. 查看当前将执行的矩阵

```bash
python scripts/model_catalog_eval_baseline.py \
  --base-url https://fusion.seanfield.org \
  --dry-run \
  --scenarios basic_chat,no_search_simple,autonomous_search,coding_reasoning
```

### 2. 执行真实验收

需要提供真实登录用户的 access token。不要把 token 写进仓库、报告或终端共享输出。

```bash
mkdir -p reports/model-acceptance
run_id=$(date +%Y%m%d-%H%M%S)

python scripts/model_catalog_eval_baseline.py \
  --base-url https://fusion.seanfield.org \
  --auth-token "$FUSION_AUTH_TOKEN" \
  --apply \
  --transport stream \
  --scenarios basic_chat,no_search_simple,autonomous_search,coding_reasoning \
  --output reports/model-acceptance/results-${run_id}.jsonl \
  --summary-output reports/model-acceptance/summary-${run_id}.json \
  --report-output reports/model-acceptance/report-${run_id}.md
```

### 3. 从已有 JSONL 重新生成报告

如果真实验收已经跑完，只需要重新生成 summary/report，不要重复调用模型：

```bash
python scripts/model_catalog_eval_baseline.py \
  --base-url https://fusion.seanfield.org \
  --from-jsonl reports/model-acceptance/results-20260702-120000.jsonl \
  --summary-output reports/model-acceptance/summary-20260702-120000.json \
  --report-output reports/model-acceptance/report-20260702-120000.md
```

## 自动报告内容

`--report-output` 会生成 Markdown 报告，包含：

- 总体通过率、失败数、工具契约不匹配数、质量风险数。
- 按场景统计。
- 按模型统计。
- 质量风险清单和处理建议。
- 每条模型/场景明细、耗时、工具调用、质量 flag、对话 URL。
- 真实 Chrome 回归补充记录模板。

## 质量 flag

| flag | 含义 | 默认处理 |
|---|---|---|
| `reasoning_tag_leak` | 正文泄露 `<think>` / `</think>` | 高风险，不应作为推荐模型 |
| `expected_search_without_agent_tools` | 实时问题命中不可联网模型 | 中风险，检查能力标注和产品说明 |
| `expected_search_without_read` | 搜索后没有深读关键来源 | 中风险，检查 Search / Read Planner |
| `slow_response` | 响应超过场景阈值 | 中风险，进入慢模型标注/路由评估 |

## 真实 Chrome 补证

只在自动脚本之后做 1-2 条代表路径补证。记录格式：

| 用例 | 输入/页面 | 预期 | 实际 | console error | 刷新后结果 | 结论 |
|---|---|---|---|---|---|---|
| 模型选择器 | `/chat/new` | 模型目录、能力标签、上传入口与 `/api/models/` 一致 |  |  |  |  |
| 实时搜索代表用例 | 新建真实对话 | 可联网模型展示搜索、读取、回答依据 |  |  |  |  |
| 非联网代表用例 | 新建真实对话 | 不展示工具过程，并说明实时能力边界 |  |  |  |  |
| 刷新恢复 | 已完成对话 URL | 正文、执行过程、回答依据按场景恢复 |  |  |  |  |

## 通过门槛

- 自动脚本失败数为 0。
- 工具契约不匹配数为 0。
- 不存在高风险质量 flag。
- 可联网模型在实时场景至少应出现搜索；要求证据的场景应出现 `url_read`。
- 不可联网模型不应展示工具过程，并应表达实时信息能力边界。
- Chrome 补证无 console error，刷新后状态恢复符合场景预期。

## 失败处理

- 供应商错误、超时、空回答：先确认是否单模型供应商波动，再决定降权、标注或下线。
- 简单问题误触工具：优先检查 Agent loop 的工具启用条件和提示词约束。
- 实时问题未搜索：优先检查能力标注、工具注册、搜索决策提示词和 Search / Read Planner。
- 搜索后未深读：优先检查 source selection / read planner，不要只靠 prompt 微调。
- Chrome 补证失败：先保留对话 URL、console error、刷新结果，再定位 UI/状态恢复链路。
