# Fusion PromptHub 正式迁移实施计划

## Phase 1：恢复 PromptHub 基线

- [ ] 修复 Prompt 分享 API 直接 `db.flush()` 的架构违规并补测试。
- [ ] 为 PromptHub CI 增加 backend、SDK、ruff、架构和 Alembic 校验。
- [ ] 保持 dev 已移除 8200 暴露的状态，不覆盖用户备份文件。
- [ ] 部署 GitHub master，验证容器运行提交、health 和 smoke。

## Phase 2：PromptHub 消费者契约与安全

- [ ] 先写 published bundle 服务/API 失败测试。
- [ ] 实现 current published 版本批量读取、稳定 revision 和整包失败语义。
- [ ] 新增 project-bound、hash-only、`prompts:read` service token。
- [ ] 补 401、403、跨项目、写接口拒绝和 token 不泄漏测试。
- [ ] SDK 增加同步/异步 published bundle 读取与类型。
- [ ] 修正 publish 后 `Prompt.content/variables` 快照语义。

## Phase 3：迁入 Fusion 11 个 Prompt

- [ ] 导出 Fusion dev 实际 active Prompt、变量和 SHA-256。
- [ ] 编写 dry-run/apply、幂等导入脚本，通过 PromptHub 服务/API 创建 `fusion` 项目和 11 个 Prompt。
- [ ] 创建独立 Fusion service token，不复用 admin key。
- [ ] 核对 PromptHub bundle 与 Fusion active 内容字节级一致。

## Phase 4：Fusion shadow 同步

- [ ] 先写 client 的成功、超时、401、404、5xx 和坏 JSON 测试。
- [ ] 先写 bundle 完整性、变量、marker、checksum 和幂等同步测试。
- [ ] 实现轻量 httpx client、PromptSpec/Catalog、LKG bundle 和原子同步服务。
- [ ] 增加启动 best-effort 和周期同步；失败不影响健康启动。
- [ ] 增加 `disabled|shadow|apply` 门禁、同步诊断和版本观测。
- [ ] 部署 `shadow`，验证 11 项零漂移且聊天热路径无远程请求。

## Phase 5：apply 切换与收尾

- [ ] 切换 `apply`，验证 bundle 原子激活、相同 revision 幂等和旧版本可回滚。
- [ ] 限制 Fusion 本地 `prompt_template` 写入口，避免双事实源。
- [ ] 运行 PromptHub backend/SDK 全量测试和 Fusion 全量测试。
- [ ] 推送并跟踪两仓 CI/CD、迁移、health、smoke 和容器提交。
- [ ] 复用用户现有登录态 Fusion Chrome 标签执行验收矩阵。
- [ ] 更新 Fusion 双仓执行台账、架构文档和 PromptHub 接入文档。
