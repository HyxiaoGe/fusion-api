# 管理员审计中心 v1 生产验收

## 发布信息

- 后端：`1ec0f01`，GitHub Actions `29153889249` 成功。
- 前端：`f2517ca`，GitHub Actions `29154128135` 成功。
- hydration 修复：`5e47692`，GitHub Actions `29154607305` 成功。
- 环境：唯一生产环境 `https://fusion.seanfield.org`。
- 浏览器：复用用户已打开且已登录的 Fusion Chrome 标签；未创建新标签、窗口或隔离上下文。

## 回归记录

| case_id | 输入 | 页面 URL | 预期 | 实际 | network/API | console error | 刷新后结果 | 结论 | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| AA-01 | 已登录管理员访问管理中心 | `https://fusion.seanfield.org/admin` | 管理员可进入独立只读页面 | 用户、对话、压测、访问审计四个页签正常显示 | 管理员 users API 200，列表审计事件带 request ID | 初版发现 React #418，修复后重载无新增 error/warning | 修复后页面正常恢复 | 通过 | UI commit `5e47692`；Actions `29154607305` |
| AA-02 | 新建临时普通用户 `perf-20260711-132601-83513b8a` | `/admin` 用户页 | 管理员可检索新用户，列表邮箱脱敏，详情单独显示完整邮箱 | 列表显示 1 对话、2 消息、1961 tokens；详情显示完整测试邮箱 | `admin.audit.users.list` / `admin.audit.user.view` 写入审计 | 最终复验无新增错误 | 临时用户清理后不再存在 | 通过 | 测试用户 ID `bb945dd3-efbd-4175-9f5a-d2d8a02b77b2` |
| AA-03 | 新建真实 SSE 对话 `b68b762e-73f9-457d-a4ff-cf9209997139` | `/admin` 对话页 | 跨用户检索并显示消息、usage、模型和 Agent run/step | 2 条消息、`deepseek-chat`、输入 1936 / 输出 25 tokens、1 个 completed run/step 均可见 | 详情分区 API 均成功并写审计 request ID | 最终复验无新增错误 | 精确删除后搜索返回“没有匹配的对话” | 通过 | runner TTFT 1853.88 ms、总耗时 3789.81 ms、0 error frame |
| AA-04 | 既有工具对话 `cc462feb-d7c9-4207-83c4-84636fea82a4` | `/admin` 对话详情 | 展示 Agent 5 步与 4 次工具，工具只返回安全投影 | `web_search` / `url_read` 参数、指标、来源标题/URL/长度可见；无网页正文和 Provider 原始报文 | tool/agent API 成功并写审计 | 最终复验无新增错误 | 非临时数据，不做刷新变更 | 通过 | 工具结果只含字段白名单 |
| AA-05 | 既有文件对话 `b2cc8d1e-31a0-43e7-93f0-951f9f07687a` | `/admin` 对话详情 | 只显示文件元数据，不提供预览/下载 | `multi-agent.jpeg`、`image/jpeg`、570.3 KB、processed、1376×768 可见；无内容按钮 | files API 成功并写审计 | 最终复验无新增错误 | 非临时数据，不做刷新变更 | 通过 | UI 无预览、下载或签名 URL |
| AA-06 | 导入 `perf-20260711-093213-37c7d9cf` | `/admin` 压测页 | 脱敏基线可幂等留存，聊天清理不影响汇总 | 页面提示“压测结果已导入”，列表显示 completed / production | import 与 list API 成功；`admin.audit.performance_run.import` 带 request ID | 最终复验无新增错误 | 页面 reload 后仍可见；临时聊天删除后仍为 1 条 | 通过 | `docs/performance/2026-07-11-production-baseline-import.json` |
| AA-07 | 打开访问审计页 | `/admin` 访问审计页 | 管理员的列表、详情、工具、文件、压测读取都有留痕 | actor、action、resource、target、request ID、时间均显示 | 审计页自身读取也产生新事件 | 最终复验无新增错误 | 重载后历史事件仍在 | 通过 | 包含 import、conversation.view、messages/tool/agent/files.list |
| AA-08 | 普通用户访问 `/api/admin/audit/users` | API | 返回统一 403，不泄露内容 | 返回 `403 / FORBIDDEN / 需要会话审计员权限` | request ID `95de5d8009784b09adb1219dcef5b249` | 不适用 | 不适用 | 通过 | 使用 runner UA，排除 Cloudflare browser-signature 403 干扰 |
| AA-09 | 生产迁移和 FK 探针 | PostgreSQL | 新孤立 step 被拒绝，历史孤立数据保留 | head `b8d4f7a1c2e6`；约束 `convalidated=false`、`ON DELETE CASCADE`；11 条历史孤立 step 保留；新增孤立 insert 被 FK 拒绝 | CI migration、health、smoke 全绿 | 不适用 | 探针 ID 复查为 0 | 通过 | 新表 `admin_audit_events` / `performance_runs` 均存在 |
| AA-10 | `/admin` 响应头 | `https://fusion.seanfield.org/admin` | 禁止 iframe 点击劫持 | CSP 含 `frame-ancestors 'none'`，X-Frame-Options 为 `DENY` | HTTP/2 200 | 不适用 | 规则持续生效 | 通过 | 生产响应头实测 |

## 清理结果

- 临时会话删除 1 条，关联 Agent session / step 复查均为 0。
- Fusion 临时用户删除 2 条；auth-service 临时用户删除 4 条、登录日志删除 8 条。
- 对应 refresh token、登录日志、会话、Agent session / step 和测试用户最终均为 0 残留。
- 压测留存 `perf-20260711-093213-37c7d9cf` 保持 1 条，证明与聊天清理解耦。

## 已知边界

- auth-service 当前没有管理员升降权产品接口；管理员身份由 JWT `admin` scope 决定。若未来增加降权操作，必须在变更事务内同时写 per-user access-token revocation marker，不能只改 `users.is_superuser`。
- 历史 11 条孤立 `agent_steps` 按设计保留；清理和 `VALIDATE CONSTRAINT` 属于独立高风险运维任务。
