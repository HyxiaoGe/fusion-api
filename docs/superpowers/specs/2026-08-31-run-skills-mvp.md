# Run 级 Skills MVP

## 目标

在现有 Run 级能力路由上增加第一版代码托管 Skills：每个 Run 在 `run_started` 之前完成 Skill 选择、读取、校验和全文冻结；同一冻结结果同时约束 Prompt、工具、Run 指纹与 Trajectory。首期仅把 `verified_web` 映射到 `verified-research@1.0.0`，不建设 PromptHub、Skill 管理后台、安装市场或运行中动态发现链路。

## DeerFlow 参考与取舍

保留 Agent Skills 的 `SKILL.md` 包结构、frontmatter、内容哈希、渐进加载、`allowed-tools` 和 fail closed 原则。Fusion 不复制 `skill_index → describe_skill → read_file`：现有路由已经能在首次 LLM 前高置信选择能力包，增加两轮模型工具调用会扩大延迟、竞态和权限面。

不实现用户 Skill、在线编辑、SkillScan、脚本执行、Secret 注入、沙箱投影、热重载、Slash 激活或线程级 `skill_context`。这些能力达到明确产品需求后再单独设计。

## 标准目录与运行时版本

首个 Skill 位于：

```text
app/ai/skills/verified-research/SKILL.md
```

文件使用受控 frontmatter：

```yaml
---
name: verified-research
description: 对需要官方原文与交叉来源的请求建立可核验证据链
metadata:
  version: "1.0.0"
allowed-tools: web_search url_read
---
```

- 只读取 `<skill-id>/SKILL.md`，不探测或兼容 `<skill-id>/<version>/SKILL.md`。
- `allowed-tools` 按 Agent Skills 标准为可选字段；省略时沿用能力路由已授权工具，声明时只能与路由授权集合一致，不能扩权。
- `metadata.version` 可选；省略或格式不适合账本时使用整份 `SKILL.md` 的 SHA-256 生成稳定运行时版本。
- 能力包发布映射同时固定 `skill_id + version + content_sha256`；磁盘正文与固定摘要不一致时按加载失败处理。
- 正文使用 UTF-8，包含 frontmatter 的完整文件不超过 32 KiB。
- 使用 PyYAML SafeLoader 解析标准 frontmatter，并拒绝危险标签、YAML 别名和重复键。

## Run 冻结与原子权限

`build_agent_loop_call_config()` 在返回前完成 Skill 快照：

```text
skill_id
version
description
content_sha256
allowed_tool_names
activation_source=capability_package
section_id=skill:verified-research@1.0.0
char_count
content
```

`content` 只存在于不可变的内存 `AgentLoopCallConfig` 和随后保存的 Prompt 快照，不能进入 Run config、SSE、账本、普通会话摘要或日志。

最终业务工具集合为：

```text
能力路由要求的工具 ∩ Skill allowed-tools ∩ 当前模型实际可用且已绑定的工具
```

该集合同时产生 LLM schemas、`announced_tools`、handlers、MCP bindings、`update_plan.planned_tools` 枚举和执行前 allowlist。Skill 不能新增能力包没有选择的工具。若 Skill 缺少能力包必需工具、文件缺失、frontmatter 非法或哈希/版本校验失败，则受控降级为：

- `package_id=tools_unavailable`
- `reason_codes=[required_skill_unavailable]`
- `external_tool_names=[]`
- `effective_plan_mode=off`
- `network_boundary_required=true`
- Skill 状态 `load_failed`

失败结果仍建立 Run 并进入 Trajectory，不在 `_start_run` 之前抛出导致状态丢失。

## Prompt 组装

已加载 Skill 以冻结内容生成一个 `SystemPromptSection`，section ID 为 `skill:<skill-id>@<version>`。`prepare_agent_loop_messages()` 只能消费冻结 section，不得重新读文件或重新选择。

`verified-research` 接管现有 `verified_research_plan` 的研究方法、来源核验与输出规则；旧文本段落停止注入，避免两套规则和版本漂移。服务端的 `web_search × 1`、`url_read × 2`、计划依赖及证据校验继续作为确定性门禁，不迁入 Skill。

每个 Run 重新选择 Skill。话题切换和重试不从上一 Run 隐式继承“当前活动 Skill”；continuation 按 owner-scoped previous Run 已持久化的 ID、版本和哈希校验当前标准目录文件，文件已经更新或当前路由无法继续承接时 fail closed。历史详情始终读取当次 Prompt snapshot，不从磁盘重建。

## 协议与 Trajectory

`RunCapabilityResolution` 升级为 schema v2，并增加安全 `skill_resolution`：

```typescript
{
  status: 'not_selected' | 'loaded' | 'load_failed';
  activation_source: 'capability_package';
  requested_skill_ids: string[];
  skills: Array<{
    skill_id: string;
    version: string;
    content_sha256: string;
    allowed_tool_names: string[];
    section_id: string;
    char_count: number;
  }>;
  duration_ms: number;
  error_code?: 'skill_load_failed' | null;
}
```

旧 schema v1 保持可读，按“历史未记录 Skill 状态”展示，不反推为 `not_selected`。Bundle fingerprint 增加安全 Skill resolution 中的稳定语义字段；Skill 版本、正文哈希、权限或状态变化必须改变指纹，加载耗时等观测字段不得改变指纹。路由选择语义变化同时提升 router version，Prompt section 变化提升 template version。

新增 `skills_resolved` 终态事件，每个新 Run 恰好一条：

- `loaded`：Skill 已冻结并进入组装；
- `not_selected`：本 Run 明确无需 Skill；
- `load_failed`：已选择但校验或加载失败。

事件只携带上述安全元数据和 `detail_status=available|degraded|null`，不包含正文、磁盘路径、原始异常或用户输入。同步本地加载不增加“准备中”。

Skill 正文继续复用 `agent_system_prompt_snapshots`：不增加表，不存第二份正文。新增 owner-scoped `node-detail/skills` 读取接口，依据 `skills_resolved` 元数据从当次 Prompt snapshot 提取 Skill sections，并交叉校验 ID、版本、section ID、字符数与 SHA-256；不得读取当前磁盘文件重建历史。非所属用户、会话/Run 不匹配统一 404，响应 `private, no-store`。

前端投影一个聚合 Skills Context 节点。默认展示真实状态；`loaded` 节点详情显示 ID、版本、哈希、激活来源、允许工具和完整正文，支持精确复制；切换 Run 必须清除旧正文并隔离迟到请求。`not_selected`、`load_failed`、旧 Run 未记录、详情保存失败和快照无效使用不同文案。

## 验收矩阵

| 场景 | Skill | 工具 | 必须验证 |
|---|---|---|---|
| `你好` | `not_selected` | 无 | Skills 节点存在，无正文/哈希 |
| 稳定知识 | `not_selected` | 无 | 不误激活 |
| 普通最新事实 | `not_selected` | `web_search` | 不把一次搜索升级为研究 Skill |
| 官方原文并交叉核验 | `verified-research@1.0.0` | `web_search,url_read` | Skill 正文只出现一次，版本/哈希/工具一致 |
| 不用核验，简单查一下 | `not_selected` | `web_search` | 否定语义不激活 Skill |
| 不要联网但询问最新公告 | `not_selected` | 无 | 网络边界准确，不声称核验 |
| 直接总结 URL | `not_selected` | `url_read` | URL 读取不等于研究 Skill |
| 研究后切换稳定知识 | 新 Run `not_selected` | 无 | 不继承旧 Skill/工具 |
| Skill 文件缺失/非法/未知工具 | `load_failed` | 无 | Run 正常建立、fail closed、正文不可用 |
| 伪造天气或 MCP 调用 | 仍为 loaded | 未授权调用拒绝 | handler 不执行，工具计数不增加 |
| 路由后文件变化 | 使用冻结正文 | 原冻结工具 | Prompt/哈希/公告无 TOCTOU 漂移 |
| Prompt 快照保存失败 | `loaded` | 正常 | 模型仍收到正文，详情 `degraded` |
| 刷新、历史、复制 | 原版本 | 原工具 | 正文与首次 Run 一致，不读当前文件 |
| 新版发布后查看旧 Run | 旧版本 | 原工具 | 旧正文、哈希保持不变 |
| 非所属用户访问 | 不暴露 | 无 | 统一 404 |

## 发布停止条件

出现任一项即停止发布：

1. Skill 在 `_start_run` 后才读取或选择。
2. Skill 正文进入 SSE、Run config、账本、Redux、Dexie、日志或普通会话响应。
3. Skill `allowed-tools` 扩大能力包，或 schemas、公告、handler、binding、计划枚举、执行 allowlist 不一致。
4. `verified_research_plan` 与 Skill 正文重复注入。
5. 普通问候、稳定知识、一次搜索或 URL 总结误激活 Skill。
6. Skill 加载失败仍保留业务工具或没有可见失败状态。
7. 刷新/历史按当前 `SKILL.md` 重建正文。
8. 旧 Run 被误标为本轮明确 `not_selected`。
9. 只根据最终回答判断通过，没有检查工具 schema、Prompt section、Skill 节点、正文、复制和刷新。

## 验证边界

本地测试与构建只证明代码协议。发布后必须复用现有已登录 Fusion Chrome 标签，使用真实新 Run 覆盖正常、未选择、否定、失败降级可模拟项、话题切换、正文复制和刷新恢复；同时检查 Prompt 正文只含一个 Skill section、实际工具调用和 console/network。未完成真实页面验收前不得称产品验收通过。
