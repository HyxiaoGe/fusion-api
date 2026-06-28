# 2026-06-28 agent_round helper 拆分计划

## 背景

`scripts/check_quality.py` 目前在 stream 目录里只剩 `app/services/stream/agent_round.py:run_agent_round()` 超过 50 行。该函数同时承担 LLM 调用、stream_round 参数转发、日志摘要和 usage 累计，虽然行为已经很薄，但仍是 agent-loop 后续优化的质量红线。

## 目标

1. 保持 `run_agent_round()` 对外契约不变。
2. 抽出普通 round 的 LLM 调用 + stream 消费 helper，使 `run_agent_round()` 退出超长函数列表。
3. 用回归测试锁定 helper 的调用顺序、参数透传和返回值。
4. 不启动本地 Fusion 服务；只做单元测试、质量脚本、CI/CD 和部署后真实 Chrome 回归。

## 验证

1. 先跑相关基线测试，确认当前行为稳定。
2. 先补会失败的 helper 测试，再实现。
3. 通过相关测试、全量测试、ruff、架构/质量脚本。
4. 提交并推送后监控 GitHub Actions 和 dev 部署。
