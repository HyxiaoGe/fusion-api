# CI 构建可观测性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 fusion-api Windows build job 增加阶段耗时、失败分类、Job Summary 和失败日志 artifact。

**Architecture:** 新增 PowerShell 编排脚本，以单个 production 镜像和单个临时容器依次执行五个阶段，并输出 JSON 与 UTF-8 日志。workflow 保留现有 ACR 登录、推送与部署行为，只增加 push 结果记录、always 汇总和失败 artifact。

**Tech Stack:** GitHub Actions、Windows PowerShell 5.1、Docker CLI、Python unittest。

## Global Constraints

- 不修改 Dockerfile、依赖版本或 deploy-dev。
- 保持测试集合和超时上限不变。
- 日志不得包含 ACR 凭据。
- 失败日志 artifact 保留 7 天。
- 所有新增文本使用 UTF-8。

---

### Task 1: CI 编排脚本

**Files:**
- Create: `scripts/ci/run_windows_container_ci.ps1`
- Modify: `test/test_ci_container_contract.py`

- [ ] 先扩展契约测试，断言五个阶段、JSON 结果、UTF-8 日志、finally 容器清理和原退出码传播。
- [ ] 运行 `python -m unittest test.test_ci_container_contract -v`，确认因脚本缺失失败。
- [ ] 实现单镜像、单临时容器的阶段执行器。
- [ ] 重跑契约测试并通过。

### Task 2: Workflow 汇总与失败 artifact

**Files:**
- Modify: `.github/workflows/deploy.yml`
- Modify: `test/test_ci_container_contract.py`

- [ ] 先增加 workflow 契约：调用脚本、记录 image-push、always summary、失败 artifact、7 天保留。
- [ ] 运行契约测试，确认因 workflow 尚未接入失败。
- [ ] 接入脚本并增加 push 结果、summary、artifact；保持 deploy-dev 不变。
- [ ] 运行契约测试与 `python -m ruff check scripts test/test_ci_container_contract.py`。

### Task 3: 提交、PR 与真实 Runner 验证

**Files:**
- Modify: PR metadata only

- [ ] 提交实现并包含 Co-Authored-By。
- [ ] 推送 `ci/add-build-observability`。
- [ ] 创建 Draft PR，说明行为边界与测试。
- [ ] 跟踪 Windows Runner，确认 Job 成功、summary 生成且热缓存耗时无明显回退。
