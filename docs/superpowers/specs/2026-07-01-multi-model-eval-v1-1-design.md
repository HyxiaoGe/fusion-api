# Multi Model Eval v1.1 Design

## 背景

现有 `scripts/model_catalog_eval_baseline.py` 只能对 `/api/models` 返回的模型执行单问题、非流式 smoke。它能证明模型路由大致可用，但不能覆盖 Fusion 用户实际使用的流式对话链路，也无法回答这些问题：

- 哪些模型在基础对话、中文事实问答、代码/推理、长回答等场景下可用。
- 哪些模型在需要实时信息的问题上会触发 agent loop 工具，哪些模型不会。
- 失败是 HTTP/认证/服务商/超时/流式协议/输出为空中的哪一类。
- 多模型测试结果如何复盘、比对和归档。

v1.1 的目标是把脚本从“单问题 smoke”增强为“多场景、多模型、可归档的产品链路测验 runner”。

## 范围

本轮只改 `fusion-api` 的后端脚本和测试，不改 UI，不引入数据库表，不启动本地服务。

包含：

- 内置场景矩阵。
- 支持按模型和场景筛选。
- 支持 `stream` 和 `nonstream` 两种 transport，默认使用 `stream`。
- 解析 SSE envelope，记录回答摘要、工具调用、错误、conversation/message id。
- 输出 JSONL 明细。
- 输出 summary JSON，按模型和场景汇总成功率、耗时、失败类型和工具期望命中情况。

不包含：

- 自动判定回答内容事实正确性。
- 浏览器 UI 回归。
- 自动把结果写入数据库或对象存储。
- 使用 LLM 当裁判。

## 场景矩阵

默认场景使用固定 id，便于后续历史结果对比。

| id | category | 目标 | 工具期望 |
| --- | --- | --- | --- |
| `basic_chat` | `basic` | 简单中文自我介绍，验证模型基础可回答 | `forbidden` |
| `cn_factual` | `factual` | 常识/稳定知识问答，验证中文表达和直接回答 | `forbidden` |
| `coding_reasoning` | `reasoning` | 简单代码/推理任务，验证基础推理输出 | `forbidden` |
| `autonomous_search` | `search` | 明确需要近期信息但不直接命令“联网搜索”，验证自主工具判断 | `expected` |
| `no_search_simple` | `search_guard` | 普通寒暄，验证不应误触发工具 | `forbidden` |
| `long_answer` | `long_form` | 要求稍长结构化回答，验证稳定输出 | `optional` |

工具期望只判断“是否触发工具”这一层，不判断搜索结果质量。内容质量评价后续单独做。

## 数据结构

每条 JSONL 结果代表一个 `(model, scenario)`。

关键字段：

- `model_id`, `provider`, `model_name`, `model_health`
- `scenario_id`, `scenario_category`, `question`, `expected_tool_use`
- `transport`
- `success`, `elapsed_ms`, `answer_preview`
- `conversation_id`, `message_id`
- `observed_tool_calls`, `observed_tool_names`, `tool_expectation_met`
- `quality_flags`
- `error`

summary JSON 包含：

- `total`, `success_count`, `failure_count`
- `success_rate`
- `by_model`
- `by_scenario`
- `failure_types`
- `quality_flags`
- `tool_expectation_mismatch_count`

### 输出质量标记

脚本的 `success` 只表示调用链路成功并返回了回答，不代表回答内容一定符合产品展示要求。为了避免多模型测验漏掉“可回答但体验不合格”的情况，JSONL 明细增加 `quality_flags` 字段，summary 同步聚合各类标记数量。

当前最小版先覆盖稳定、可解释的规则：

- `reasoning_tag_leak`：回答正文中出现 `<think>` 或 `</think>`，说明模型把内部思考标签暴露给用户。

后续可继续加入 markdown 破损、空洞回答、拒答、语言不匹配等质量规则，但不在本轮引入 LLM 裁判。

## 失败分型

脚本只做稳定、可解释的轻量分类：

- `http_error`
- `timeout`
- `stream_error`
- `empty_answer`
- `auth_error`
- `unknown_error`

分类只用于测验报告，不影响业务链路。

## CLI

默认 dry-run，只列出将测试的模型和场景组合。

示例：

```bash
.venv311/bin/python scripts/model_catalog_eval_baseline.py --dry-run
.venv311/bin/python scripts/model_catalog_eval_baseline.py --apply --auth-token "$TOKEN" --transport stream --output /tmp/fusion-model-eval.jsonl --summary-output /tmp/fusion-model-eval-summary.json
.venv311/bin/python scripts/model_catalog_eval_baseline.py --apply --auth-token "$TOKEN" --models deepseek-chat,mimo-v2.5-pro --scenarios basic_chat,autonomous_search
```

## 验收

- 单元测试覆盖场景选择、SSE 解析、stream 成功/失败结果、summary 汇总、工具期望命中判断。
- 单元测试覆盖 `quality_flags` 明细字段和 summary 聚合。
- 目标测试和后端全量测试通过。
- CI/CD 通过并部署 dev。
- 部署后至少执行 dry-run，确认脚本能从已部署 `/api/models` 拉取当前模型和场景矩阵。
