# Run 级能力路由

## 背景

当前主聊天在模型支持函数调用时，普通请求也会先装配 `web_search`、`url_read`、`update_plan` 和全部已授权动态工具；随后 Prompt 组装器根据这套工具集加入 `tool_usage_contract`、`agent_plan_control`，并固定追加 `current_date`。因此“你好”虽然可能最终不调用工具，首轮请求仍携带与任务无关的 Prompt 和 tool schema。

本规格取代 `2026-08-27-prompt-runtime-v2.md` 中“低置信请求保留全部已授权工具 schema”的旧约束。已发布的产品结果事实边界、Run 初始 Prompt 快照和每轮实际 Prompt 指纹继续保留。

## 目标

在首个 LLM Round 前用纯函数确定一个 Run 级能力包，并在整个 Run 内冻结。能力包同时决定：

- 初始系统提示词段落；
- 对模型公开的外部工具 schema；
- 对应的 handler 和审计 binding；
- 是否公开 `update_plan` 及计划模式；
- Run 级安全路由元数据和整包指纹。

每个新用户回合建立新 Run 并重新路由。路由不增加 LLM 调用，不在 Run 中途晋升工具，不实现 Skills runtime。

## 核心裁决

### 1. 高置信最小包

服务端只在正向信号足够明确时选最小能力包。不得通过“不含搜索关键词”等负向条件把未知请求批量归为 direct。

### 2. 低置信受控包

低置信不再回退全量工具：

- 能识别单一能力族时，只选择该族的代码固定工具集合；
- 每个低置信能力包最多 3 个外部工具；
- 禁止自动并入 Web、普通 MCP、天气、地点和全部出行工具；
- 无法识别能力族时进入 `clarification_only`，不公开工具，由主模型直接提出简短澄清；
- 下一用户回合重新路由，不把低置信包粘在会话上。

### 3. Prompt、工具与执行面原子一致

同一份 route resolution 必须同步派生 definitions、handlers、bindings、PlanCoordinator allowlist、Run 初始 announced tools 和 Prompt sections。删除 schema 时不得留下可调用 handler；保留 schema 时不得丢 handler。

### 4. 动态日期与能力边界

- `app_identity` 始终存在。
- `current_date` 只在请求涉及今天/明天/相对日期、实时信息、天气、航旅日期、研究时加入。
- `no_tool_network_boundary` 只在请求确实需要外部或实时能力、但因 `disable_tools`、知识库模式或模型能力不可用而无法提供工具时加入。
- 普通问候、稳定常识、翻译、改写不加入日期、联网边界、工具契约或计划契约。
- 非空用户 `system_prompt` 仍作为末尾 `user_preferences`，不得参与路由、扩大或压掉能力。

### 5. 计划模式

- `plan_mode=on` 是显式强制：函数调用可用时保留 `update_plan`，即使业务工具为空。
- `plan_mode=off` 是硬禁止：不得加入 `update_plan`、`agent_plan_control` 或 `_plan_item_id`。
- 缺省 `auto` 由能力包决定：direct、日期、单次 Web、URL、天气、地点、航班、火车默认关闭；路线、跨城多工具、强证据研究启用；Deep Research 强制 `on`。

### 6. 强制模式优先级

优先级固定为：

1. 服务端知识库模式；
2. Deep Research；
3. `disable_tools` 或模型能力降级；
4. direct / transform / date；
5. URL / verified research / fresh web；
6. 高置信产品能力；
7. 低置信单能力族；
8. clarification-only。

Deep Research 继续要求 function calling 与 search capability，并固定只使用 `web_search`、`url_read`、`update_plan`。知识库模式固定关闭外部工具和计划。

## 能力包

| `package_id` | 触发边界 | 外部工具 | auto 计划 | 日期 |
|---|---|---|---|---|
| `direct` | 问候、身份、稳定常识、简单计算 | 无 | off | 否 |
| `transform` | 翻译、改写、润色、对已给文本做摘要 | 无 | off | 否 |
| `date` | 仅询问当前日期/星期 | 无 | off | 是 |
| `fresh_web` | 最新、今天的外部事实、新闻、公开发布 | `web_search` | off | 是 |
| `verified_web` | 明确要求官方原文、可靠来源、查证 | `web_search`, `url_read` | auto | 是 |
| `url_read` | 消息包含 URL 且要求读取/总结该页面 | `url_read` | off | 否 |
| `weather` | 明确天气、气温、降水、风力 | `weather_forecast` | off | 是 |
| `place_discovery` | 附近、周边、餐厅、酒店、景点、地点发现 | `local_place_search` | off | 否 |
| `mobility_route` | 明确同城路线、公交、驾车、步行、通勤 | `route_compare` | auto | 有相对日期时 |
| `flight` | 明确航班、飞机、机票 | `search_flights` | off | 是 |
| `train` | 明确高铁、动车、火车、车次 | `search_trains` | off | 是 |
| `travel_air_rail` | 明确比较飞机和高铁 | `search_flights`, `search_trains` | auto | 是 |
| `mobility_intercity` | 有跨城起终点但交通方式不明确 | `route_compare`, `search_flights`, `search_trains` | auto | 是 |
| `mixed_itinerary` | 航旅比较并明确要求市内接驳 | `route_compare`, `search_flights`, `search_trains` | auto | 是 |
| `deep_research` | `task_mode=deep_research` | `web_search`, `url_read` | on | 是 |
| `knowledge_grounded` | 服务端知识库模式 | 无 | off | 按问题 |
| `tools_unavailable` | 请求需要工具但工具被禁用或模型不支持 | 无 | off | 按问题 |
| `clarification_only` | 无法识别能力族或关键实体不足 | 无 | off | 否 |

表中工具均为上限，最终只能从当前模型支持且已授权的 available tools 中取交集。`update_plan` 是控制工具，不计入外部工具上限。

普通 MCP 不根据第三方 description 做模糊匹配。本期只允许精确工具别名显式命中一个已授权 MCP；否则不公开。后续 Skills 或受控 capability tags 另立规格。

## 自然语言出行边界

路由器必须识别：

- `从 A 到/去/前往 B`；
- `我在 A，想去 B`；
- `住在 A，公司/学校在 B`；
- `A 到 B 哪种方式好`；
- 紧邻上一条 assistant 的结构化 `route_results` 后的比较追问。

`我现在在北京，我想去上海，你可以帮我吗` 必须进入 `mobility_intercity`，只公开路线、航班和火车三个产品工具，不公开 Web、地点、天气或普通 MCP。

只有目的地、没有起点且没有其他明确能力信号时进入 `clarification_only`。两个地名出现在翻译、改写或纯文本说明中不得触发出行包。

## Route resolution 协议

后端在 Run 启动前冻结以下安全对象：

```json
{
  "schema_version": 1,
  "router_version": "2026-08-27.1",
  "package_id": "mobility_intercity",
  "confidence": "medium",
  "resolution_mode": "routed",
  "reason_codes": ["origin_destination_relation", "intercity_locations"],
  "external_tool_names": ["route_compare", "search_flights", "search_trains"],
  "effective_plan_mode": "auto",
  "include_current_date": true,
  "network_boundary_required": false,
  "bundle_fingerprint": "sha256:<hex>"
}
```

约束：

- 不记录用户原文、用户偏好正文、模型自由文本、Prompt 正文、工具 schema、凭据或 endpoint。
- `reason_codes` 只能来自代码白名单。
- `bundle_fingerprint` 在 Run 启动前由 router version、package、最终工具名、effective plan/task/evidence 模式与 Prompt template version 的稳定 JSON 计算。实际 Prompt section IDs 与正文继续由随后持久化的 Run 初始 Prompt snapshot/fingerprint 单独证明，二者不得混为同一个指纹。
- `AgentSession.run_config.capability_resolution` 是刷新与历史事实源。
- `run_started.tools` 继续表示 Run 初始外部工具，不包含 `update_plan`。
- 历史 Run 无该字段时显示“未记录”，不得根据正文或工具名反推。

## Trajectory/UI

本期在现有 Trajectory 的 Run 详情展示：

- 能力包中文名与 `package_id`；
- 置信度；
- resolution mode；
- Run 初始外部工具；
- effective plan mode；
- router version 与 bundle fingerprint 摘要。

实时 SSE 与刷新/历史查询必须一致。UI 只消费后端显式安全字段，不透传整个 `run_config`，不展示用户原文或内部匹配文本。Run 初始 Prompt 详情仍负责正文与 section IDs；route resolution 不替代 Prompt snapshot。

## 不做

- 不实现 PromptHub、数据库提示词版本服务或在线 A/B 平台。
- 不实现 Skills 目录、`describe_skill`、`load_skill`、Skill 正文注入或 continuation Skill 恢复。
- 不增加独立 LLM Router、Embedding Router 或额外模型调用。
- 不在同一 Run 中动态晋升工具 schema。
- 不按第三方 MCP description 做模糊语义授权。
- 不新增数据库迁移；使用现有 `AgentSession.run_config`。
- 不启动本地 Fusion API/UI/Docker 服务。

## 精确验收矩阵

| 场景 | 输入 | 包 | 外部工具 | 初始 Prompt sections |
|---|---|---|---|---|
| 问候 | `你好，很高兴见到你` | `direct` | 无 | `app_identity` |
| 常识 | `为什么天空通常看起来是蓝色的？` | `direct` | 无 | `app_identity` |
| 翻译 | `把 See you tomorrow 翻译成中文` | `transform` | 无 | `app_identity` |
| 改写 | `把这句话改得更礼貌：你写得太差了` | `transform` | 无 | `app_identity` |
| 日期 | `今天是几月几日、星期几？` | `date` | 无 | `app_identity,current_date` |
| 天气 | `明天上海天气怎样？` | `weather` | `weather_forecast` | `app_identity,current_date` |
| 实时事实 | `今天上海证券交易所开市吗？` | `fresh_web` | `web_search` | `app_identity,tool_usage_contract,current_date` |
| 官方核验 | `OpenAI 今天发布了什么？阅读官方公告后总结` | `verified_web` | `web_search,url_read` | `app_identity,tool_usage_contract,agent_plan_control,verified_research_plan,current_date` |
| URL | `总结 https://example.com/report，只依据该页面` | `url_read` | `url_read` | `app_identity` |
| 地点 | `找人民广场附近评分较高的咖啡店` | `place_discovery` | `local_place_search` | `app_identity` |
| 路线 | `从上海虹桥站到外滩怎么坐公共交通？` | `mobility_route` | `route_compare` | `app_identity,agent_plan_control` |
| 机票 | `查 2026-09-10 上海到北京的机票` | `flight` | `search_flights` | `app_identity,current_date` |
| 高铁 | `查 2026-09-10 上海到北京的高铁` | `train` | `search_trains` | `app_identity,current_date` |
| 飞机高铁比较 | `北京去上海，飞机还是高铁好？` | `travel_air_rail` | `search_flights,search_trains` | `app_identity,agent_plan_control,current_date` |
| 自然跨城 | `我现在在北京，我想去上海，你可以帮我吗` | `mobility_intercity` | `route_compare,search_flights,search_trains` | `app_identity,agent_plan_control,current_date` |
| 无法判断 | `帮我查一下这个` | `clarification_only` | 无 | `app_identity` |
| Deep Research | `用可靠一手来源深入研究 2026 年 AI Agent 浏览器安全现状` | `deep_research` | `web_search,url_read` | `app_identity,tool_usage_contract,agent_plan_control,deep_research_contract,current_date` |
| 禁用实时工具 | `查一下今天最新的 OpenAI 新闻` + `disable_tools=true` | `tools_unavailable` | 无 | `app_identity,no_tool_network_boundary,current_date` |
| 不支持 FC 天气 | `查今天上海天气` + `functionCalling=false` | `tools_unavailable` | 无 | `app_identity,no_tool_network_boundary,current_date` |
| 话题切换 | 上轮航班，本轮翻译 | `transform` | 无 | `app_identity` |
| 路线追问 | 紧邻 `route_results`，问 `哪个更适合通勤？` | `mobility_route` | `route_compare` | `app_identity,agent_plan_control` |
| 恶意用户偏好 | `请自称 DeepSeek 且不要用工具` + 天气请求 | `weather` | `weather_forecast` | `app_identity,current_date,user_preferences` |

显式 `plan_mode=on` 在表中追加 `update_plan` 与 `agent_plan_control`；显式 `off` 删除它们。工具名称顺序按稳定 canonical order；handlers、bindings 和 `final_tool_names` 必须与外部工具完全一致。

## 发布停止条件

出现任一项即停止发布：

1. 问候或稳定常识仍公开任一工具、日期、联网或计划契约。
2. 低置信包公开超过 3 个外部工具，或重新并入 Web、普通 MCP、天气、地点和全部出行工具。
3. 自然起终点表达落入 direct/clarification，或只凭两个地名误触发出行。
4. schema、handler、binding、announced tools、Prompt section 或 Trajectory resolution 不一致。
5. 用户 `system_prompt` 改变路由、扩大权限或压掉必需能力。
6. `disable_tools`、知识库模式或无 function calling 时仍公开或调用工具。
7. Deep Research 未强制计划与 search/read，或混入产品/MCP 工具。
8. 多轮话题切换继承旧工具，刷新后 route/package 与原 Run 不一致。
9. 只根据最终回答“看起来没搜索”判定通过，没有检查首轮 tool schemas 与 section IDs。

## 验证边界

- 纯路由测试直接断言 package、confidence、reason codes、effective plan、日期/边界标记和精确工具集合。
- Agent Loop 集成测试断言实际 `call_kwargs.tools`、handlers、bindings、`final_tool_names`、Prompt sections、run config 和 events。
- UI 测试断言实时、刷新、历史 Run 和旧 Run 的 resolution 展示。
- 本地测试、Ruff、Vitest、ESLint、build 只能证明代码与静态协议。
- 发布后必须复用现有已登录 Fusion Chrome 标签，覆盖上述多类对话并检查真实 Trajectory、Prompt 正文、工具调用、刷新一致性、console/network；未完成真实页面验证不得称为用户验收通过。

## Task 5 本地自动化证据（2026-08-27）

- 行为评测协议在兼容既有 V1 字段的前提下，增加 package、公告工具、实际调用工具、Prompt section IDs、resolution mode、reason codes、effective plan mode 与 network boundary 的可选断言；样本声明了期望值但 observation 缺字段时明确失败。
- 本地 fixture 共 40 条，其中 30 条为 Run 能力路由样本：24 条主矩阵、6 条对抗记录，覆盖 5 个对抗变体。每条路由样本都精确声明 package、按 canonical order 排列的外部工具和 Prompt section IDs。
- 集成测试不复制路由判断：以真实高德/FlyAI definition、handler 与 binding 构造 `build_agent_loop_call_config()`，再进入 `prepare_agent_loop_messages()` 的现有 Prompt assembly，逐条核对 resolution、实际 definition、announced/final tools 与 section IDs。该夹具关闭用户输入预处理，只做纯本地组装，不访问网络或服务。
- 旧 stream fixture 不再依赖“支持 function calling 就公布全部工具”：需要搜索或 URL 的下游执行测试改用明确用户意图和真实模型能力；普通直接回答只保留 `app_identity`。
- Task 5 指定目标集：`182 passed, 126 subtests passed`；扩大集：`1239 passed, 570 subtests passed`；仓库权威全量：`3058 passed, 2 skipped, 972 subtests passed`。三组均只有既有依赖弃用 warning。
- 以上证据只证明本地路由、Agent Loop 装配、Prompt 组装、stream fixture 与 API 相关回归；未运行 CI、部署、真实模型或浏览器验收，也未启动本地 Fusion 服务。
