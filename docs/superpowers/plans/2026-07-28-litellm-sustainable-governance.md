# LiteLLM 可持续治理实施计划

## 阶段一：升级门禁

1. 定位并修复 dev 二级备份失败，保留修改前配置。
2. 对 LiteLLM 数据库备份执行完整性检查和隔离恢复演练。
3. 为 Compose 增加 readiness healthcheck、启动宽限期和失败重试。
4. 新增脱敏的升级前检查脚本，输出备份、数据库、模型目录、调度状态和容器健康结果。

## 阶段二：受控升级

1. 固定 `v1.93.0` tag 和镜像 digest。
2. 使用数据库副本启动隔离实例，验证累计 Prisma migration。
3. 对账 DB models、virtual key、allowlist 和成本记录。
4. 执行目录、SSE、reasoning、tool calling、vision 与 usage/cost 验收。
5. 只有隔离证据全部通过后才进入 dev 滚动升级。

## 阶段三：成本表同步

1. 先执行一次手动刷新并校验状态。
2. 配置每 6 小时刷新。
3. 同步保存官方成本表快照及 SHA-256、ETag、抓取时间和模型数。
4. 新增状态检查，异常时非零退出并进入现有监控。
5. 验证上游未知模型不会生成虚假价格。

## 阶段四：候选发现与准入

1. 新增 provider adapter 和只读候选报告。
2. 支持 Moonshot 与标准 OpenAI-compatible `/v1/models`。
3. 用官方成本表和受审 override 富化候选；证据不全时保持隔离。
4. 通过 LiteLLM wildcard 路由执行候选预准入文本、流式、工具和 usage/cost 验收。
5. 通过预准入后才允许生成 dry-run `/model/new` 和 allowlist 变更计划。
6. 实际注册后复用现有 Fusion 全模型验收脚本完成产品门禁。
7. 首版保持人工批准；积累稳定证据后再评估自动批准策略。

## 代码验证

- 目标单元测试；
- Ruff 检查与格式检查；
- `git diff --check`；
- 脱敏输出检查；
- dry-run 不产生网络写操作。

## 发布边界

当前授权允许实现、测试和 dev 主机上的可回滚准备工作；不包含生产部署、密钥轮换、敏感权限修改或绕过发布门禁。涉及实际 dev LiteLLM 容器升级时，先完成隔离迁移和备份恢复证据。
