# Multi Model Eval v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把多模型基线脚本增强为多场景、多 transport、可汇总的 Fusion 产品链路测验 runner。

**Architecture:** 保持独立脚本形态，不改变业务 API。脚本从 `/api/models` 读取模型目录，生成模型和场景的笛卡尔积；`stream` transport 通过 `/api/chat/send` 解析 SSE envelope，`nonstream` transport 保留现有快速 smoke；结果写 JSONL，summary 由纯函数从结果聚合。

**Tech Stack:** Python 3.11, httpx, pytest/unittest, Fusion `/api/models`, Fusion `/api/chat/send`, SSE envelope.

---

## Files

- Modify: `scripts/model_catalog_eval_baseline.py`
  - 增加 `EvalScenario`, `EvalSummary`, stream SSE 解析、失败分型、summary 聚合和 CLI 参数。
- Modify: `test/test_model_catalog_eval_baseline.py`
  - 增加 v1.1 的场景矩阵、SSE 解析、stream 结果、summary 和兼容性测试。
- Create: `docs/superpowers/specs/2026-07-01-multi-model-eval-v1-1-design.md`
  - 记录设计边界。
- Create: `docs/superpowers/plans/2026-07-01-multi-model-eval-v1-1.md`
  - 记录实施计划。

## Requirements

- 默认 transport 为 `stream`，因为这是 Fusion 的真实产品链路。
- 保留 `nonstream` transport，便于快速判断模型基础可调用性。
- 默认场景矩阵至少包含基础对话、中文事实、代码/推理、自主搜索、不应搜索、长回答。
- dry-run 输出即将运行的模型和场景组合，不调用 `/api/chat/send`。
- apply 模式必须要求 `--auth-token`。
- JSONL 明细必须记录模型、场景、transport、成功状态、耗时、回答摘要、错误、工具调用观测、工具期望是否命中。
- summary 必须可由 JSONL 结果纯函数聚合，不依赖外部服务。
- 不在日志或输出中打印 auth token。

## Test Matrix

| Case | 输入 | 预期 |
| --- | --- | --- |
| EVAL11-01 | 未指定场景 | 使用默认场景矩阵 |
| EVAL11-02 | 指定场景 id | 只运行这些场景，保持用户输入顺序 |
| EVAL11-03 | 未知场景 id | 抛出明确错误 |
| EVAL11-04 | dry-run | 输出模型和场景组合 |
| EVAL11-05 | SSE 含 answering delta 和 agent tool events | 生成成功结果，记录工具名和回答摘要 |
| EVAL11-06 | SSE 含 error envelope | 生成失败结果，错误类型为 `stream_error` |
| EVAL11-07 | expected_tool_use=expected 但没有工具 | `tool_expectation_met=false` |
| EVAL11-08 | 结果列表 | summary 正确统计 success/failure、by_model、by_scenario、failure_types |
| EVAL11-09 | nonstream 成功 | 保持旧 JSONL 核心字段，并补齐新字段 |
| EVAL11-10 | 回答正文包含 `<think>` 标签 | JSONL 标记 `reasoning_tag_leak`，summary 聚合质量标记 |
| EVAL11-11 | 长批次逐项执行 | 每个条目完成后触发结果回调，CLI 可增量写 JSONL |

## Tasks

### Task 1: 写 v1.1 失败测试

- [ ] 在 `test/test_model_catalog_eval_baseline.py` 添加默认场景、场景筛选、未知场景、SSE 解析、summary 和 nonstream 兼容测试。
- [ ] 运行 `.venv311/bin/python -m pytest test/test_model_catalog_eval_baseline.py -q`，确认新增测试因缺少 v1.1 API 失败。

### Task 2: 实现场景矩阵和结果结构

- [ ] 在 `scripts/model_catalog_eval_baseline.py` 增加 `EvalScenario` 和默认场景矩阵。
- [ ] 扩展 `EvalResult` 字段，保持旧字段仍存在。
- [ ] 实现 `select_scenarios`、`tool_expectation_met`、失败分型函数。
- [ ] 运行目标测试确认相关用例通过。

### Task 3: 实现 stream transport

- [ ] 实现 SSE 行解析，把 `data: [DONE]` 作为终止。
- [ ] 从 `agent_event.tool_call_started/tool_call_completed` 统计工具名。
- [ ] 从 `answering` delta 聚合回答摘要。
- [ ] 从 `error` envelope 生成失败结果。
- [ ] 运行目标测试确认 stream 用例通过。

### Task 4: 实现 summary 和 CLI

- [ ] 实现 `build_summary` 和 `summary_to_json`。
- [ ] 增加 `--transport`, `--scenarios`, `--summary-output`。
- [ ] dry-run 输出模型和场景组合。
- [ ] apply 模式按模型和场景运行，写 JSONL 和 summary。

### Task 5: 验证和发布

- [ ] `.venv/bin/python -m ruff format --check scripts/model_catalog_eval_baseline.py test/test_model_catalog_eval_baseline.py`
- [ ] `.venv/bin/python -m ruff check .`
- [ ] `.venv311/bin/python -m pytest test/test_model_catalog_eval_baseline.py -q`
- [ ] `.venv311/bin/python -m pytest test/ -q`
- [ ] `.venv311/bin/python scripts/check_architecture.py`
- [ ] commit + push，监控 GitHub Actions 和 dev deploy。
- [ ] 部署后运行脚本 dry-run，确认已部署环境能列出模型和场景组合。

### Task 6: 增加输出质量标记

- [x] 为 `EvalResult` 增加 `quality_flags` 字段。
- [x] 标记回答正文中的 `<think>` / `</think>` 泄漏。
- [x] 在 summary 中聚合质量标记数量。
- [x] 补充 reasoning 标签泄漏和 summary 聚合测试。

### Task 7: 增加长批次保护

- [x] 为 `run_eval` 增加逐项 `on_result` 回调。
- [x] CLI apply 模式每完成一项就追加写 JSONL。
- [x] CLI apply 模式向 stderr 输出进度摘要。
- [x] 补充成功项和失败项都会触发回调的测试。
