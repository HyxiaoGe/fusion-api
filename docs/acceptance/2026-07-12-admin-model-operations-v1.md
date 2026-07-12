# 管理员模型运营中心 v1 验收记录

## 发布信息

- 后端提交：`5624241`、`1a2b456`
- 前端提交：`3097c6b`
- 后端 Actions：`29190231438`（构建、测试、部署 smoke 全部通过）
- 前端 Actions：`29190141564`（构建、浏览器 smoke、部署全部通过）
- 环境：`https://fusion.seanfield.org`
- 验收时间：2026-07-12（Asia/Shanghai）

## 自动化验证

- 后端：`977 passed + 98 subtests`；Ruff、变更文件格式、架构检查通过。
- 前端：`1085 passed`；目标 ESLint、Next.js production build 通过。
- 独立对抗式审查：重点后端 `11/11`、前端模型/路由/对话/API `46/46`，无阻断问题。

## 生产真实登录态 Chrome 回归

| case_id | 输入 | 页面 URL | 预期 | 实际 | network/API | console error | 刷新后结果 | 结论 | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| AMO-PROD-01 | 打开模型 Tab | `https://fusion.seanfield.org/admin?tab=models` | 展示当前与历史模型、健康时间、能力和持久化统计 | 共 19 个模型；当前与历史状态分离；健康模型展示北京时间检测时间；历史模型为未记录/未知 | 页面真实加载模型列表数据；CI 部署 smoke 通过 | `0` | 列表可重新加载 | 通过 | DOM 显示 14 个当前模型、5 个历史模型及分页 `1/1` |
| AMO-PROD-02 | 查看历史模型 `mimo-v2-pro` | `https://fusion.seanfield.org/admin?tab=models&model_id=mimo-v2-pro` | URL 可恢复详情；历史字段不伪造当前目录信息 | 展示历史模型、未记录 provider/规格、18 个对话、3 位用户、22 条回复、8 次 Agent 运行 | 详情接口经真实页面加载 | `0` | 刷新后仍恢复同一详情和统计 | 通过 | DOM region `模型详情 mimo-v2-pro` |
| AMO-PROD-03 | 点击“查看该模型的对话” | `https://fusion.seanfield.org/admin?tab=conversations&model_id=mimo-v2-pro` | 对话 Tab 同步 model_id；浏览器返回恢复模型详情 | 筛选控件为 `mimo-v2-pro`，共 18 条；浏览器返回回到原模型详情 | 对话列表按 model_id 加载 | `0` | 返回后详情数据保持 | 通过 | DOM 显示 `共 18 条 · 第 1/1 页`，back 后 URL 恢复详情 |
| AMO-PROD-04 | 直达当前模型 `mimo-v2.5-pro-ultraspeed` | `https://fusion.seanfield.org/admin?tab=models&model_id=mimo-v2.5-pro-ultraspeed` | 展示当前状态、健康时间、能力、中文推荐场景和持久化统计；不展示价格 | 展示当前/健康、检测时间、深度思考/工具/联网/Agent、快速响应/Agent、49 个对话；无价格字段 | 详情接口经真实页面加载 | `0` | 直达可恢复 | 通过 | DOM region `模型详情 mimo-v2.5-pro-ultraspeed` |
| AMO-PROD-05 | 编码斜杠详情探测 | `/api/admin/audit/models/retired%2Fmodel-v1` | 确认生产 Nginx 对 `%2F` 的转发 | Browser Control 在直接 API 导航层返回 `ERR_BLOCKED_BY_CLIENT`，未获得 Nginx 响应 | FastAPI TestClient 与前端编码自动化均已验证斜杠 ID 200；生产无可用斜杠模型数据 | 无页面执行 | 不适用 | 环境限制，非阻断 | 不把该项表述为生产已验证；后续出现真实斜杠 alias 时补回归 |

## 已知非阻断 P2

- 模型统计仍对会话、消息和 Agent 表做全生命周期分组聚合。当前生产数据量响应正常；规模扩大后应转为汇总表、缓存或显式时间窗口。
- LiteLLM 目录刷新仍在同步锁内最多等待 5 秒；本次已增加 30 秒失败退避和 stale 保留，长期可改为 stale-while-revalidate。
- `%2F` 的生产 Nginx 行为受当前 Browser Control 直接 API 导航限制，尚未得到独立生产证据；应用层和前端自动化均已通过。
