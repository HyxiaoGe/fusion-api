# 管理员审计中心 v1 Implementation Plan

## Goal

交付独立、只读、可审计的管理员内容观察面和压测汇总留存，保持普通用户聊天所有权、缓存和流式链路不变。

## Task 1：数据与安全骨架

- [x] 先补脱敏器、审计 fail-closed、性能导入校验和模型约束的失败测试。
- [x] 新增 `AdminAuditEvent`、`PerformanceRun` 模型与 Alembic 迁移。
- [x] 新增关键查询索引和非破坏 `agent_steps` FK 约束。
- [x] 实现递归脱敏、长度上限与邮箱遮蔽。

## Task 2：独立管理员查询层

- [x] 新增 `AdminAuditRepository`，实现用户/对话分页与聚合，禁止复用普通用户所有权接口。
- [x] 新增 `AdminAuditService`，编排脱敏、审计、404 和压测幂等导入。
- [x] 新增 `/api/admin/audit/*` 路由、Pydantic schemas、`no-store` 响应和独立 auditor 依赖。
- [x] 覆盖权限、分页、组合筛选、消息、工具、Agent、文件元数据、审计与性能导入 API 测试。

## Task 3：管理员前端协议与入口

- [x] 先补 API 编码、管理员菜单入口、AdminGuard 和只读约束的失败测试。
- [x] 新增 `adminAudit` types/API client。
- [x] 在管理员头像菜单增加“管理中心”；普通用户不展示。
- [x] 新增独立 `/admin` 页面和 noindex 元数据。

## Task 4：管理员审计中心 UI

- [x] 实现用户、对话、压测、访问审计四个 tab。
- [x] 实现组合筛选、稳定分页、加载/空/错误状态和旧请求取消。
- [x] 实现专用只读消息、Agent/tool 和文件元数据视图，不写普通聊天 Redux/Dexie。
- [x] 实现压测 JSON 导入与已留存结果详情。

## Task 5：验证与发布

- [x] 后端目标测试、全量 pytest、Ruff、架构检查和 Alembic 单 head。
- [x] 前端目标 Vitest、全量 Vitest、ESLint 和生产 build。
- [x] 安全复核响应不含凭据、路径、签名 token 和 raw provider payload。
- [x] 分仓中文提交并 push，监控 CI/CD、迁移、部署 health/smoke。
- [x] 复用已打开的真实 Fusion 管理员 Chrome 标签，以新建普通用户数据完成生产双轨验收。
- [x] 通过页面导入 2026-07-11 首轮生产压测基线，并验证清理后的聊天不存在但汇总可见。
