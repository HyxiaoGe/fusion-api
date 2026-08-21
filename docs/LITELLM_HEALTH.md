# LiteLLM 全模型健康探测（/health）的成本治理

> 关联：fusion-api 与 ai-audio-assistant-web 两个服务共用同一个 LiteLLM Proxy，
> 历史上各自在 FastAPI 启动时起后台 asyncio 循环，每 30 分钟请求一次 LiteLLM
> `/health`。LiteLLM `/health` 不是普通进程存活检查，它会对允许访问的**每个模型**
> 发送真实 completion，且 qwen 等 reasoning 模型的 thinking token 无法通过
> `enable_thinking` / `max_tokens` 压掉——探得越频繁，服务商侧费用越高。

## 问题背景与修复目标

| 项 | 说明 |
|----|------|
| 现状（修复前） | fusion-api（1 worker）+ ai-audio-assistant-web（2 workers）各自起探测循环，每 30min 3 轮 × 3 模型（qwen3.7-max / qwen3.6-plus / qwen-vl-max）≈ 432 次 Qwen 推理/天，阿里云账单每天固定 1.5 元以上 |
| 根因 | `/health` 每次全模型真实 completion；进程内 `_by_alias` / `_refresh_task` 无法跨 worker、跨服务协调 |
| 修复目标 | ① 默认不再做全模型探活（止血）；② 需要探活时，跨 worker/跨服务每周期最多一轮；③ 健康结果进共享缓存，不只存在单个 worker 内存 |

## 行为设计

### 1. 总开关 `LITELLM_HEALTH_ENABLED`（默认 `false`）

- `false`（默认）：lifespan startup **不启动** `/health` 后台循环，进程内也不做任何
  LiteLLM `/health` 请求；`GET /api/models` 照常返回（健康字段回退 `unknown`，
  前端按可用处理，不灰显、不阻塞）；`record_success`（真实 LLM round 成功）仍会
  提升单个别名为 healthy。
- `true`：启动后台循环，但见下——用 Redis 协调，不会重复探测。

### 2. Redis round-claim：每周期全集群最多一轮（开启时）

两个服务共享同一 Redis（fusion 用 `REDIS_URL`，audio 用 `REDIS_URL`，均为同一
实例 db0），因此可以用同一组 key 协调：

- `litellm:health:probe:claim:v1`：round-claim 分布式锁。每轮探测前
  `SET ... NX EX <interval>`（TTL = 探测间隔，下限 300s），**只有抢到的实例**才
  真正请求 `/health`；其它实例本轮跳过 → 无论多少个 worker / 多少服务，每个周期
  全集群最多执行一轮 `/health`。
- `litellm:health:snapshot:v1`：健康快照（JSON：`{checked_at, by_alias}`，TTL 7 天）。
  探测成功后写入；所有 worker/服务读取健康状态时先限频（30s）从快照同步本地缓存，
  健康结果不再只存在单个 worker 内存。
- **Redis 不可用**（未配置 / 连接失败）：本轮跳过探测。宁可显示 `unknown`，也不能
  失去协调地重复探测烧钱。

> ⚠️ 两服务的 Redis key 与 JSON 格式必须一致，改动需两边同步
> （fusion：`app/ai/litellm_health.py`；audio：`app/core/litellm_health.py`）。

### 3. 保留的原行为

- 冷启动 / 未探测：`get_health(alias)` 返回 `{status: "unknown"}`，前端按可用处理。
- 探测失败（网络 / proxy 重启）：保留上一次结果（stale 数据），不清空。
- `/api/models` 只读共享快照 / 本地缓存，**不**在请求路径上同步触发 `/health`。
- 不改变正常聊天、摘要、Embedding、图片生成等业务调用（它们走 LiteLLM
  `/chat/completions` 等业务端点，与 `/health` 无关）。
- `/health/liveliness` 只判断 LiteLLM 进程存活，**不能**替代供应商可用性探测，
  本方案也没有用它来"等价替代"。

## 部署变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `LITELLM_HEALTH_ENABLED` | `false` | 全模型 `/health` 探测总开关 |
| `LITELLM_HEALTH_INTERVAL_SECONDS` | `1800` | 探测间隔（秒）；两服务应保持一致，改小会按最小的那个生效 |
| `LITELLM_HEALTH_REQUEST_TIMEOUT` | `90` | 单次 `/health` 请求超时（秒） |

已同步到 `.env.example`、`docker-compose.yml`、`.github/workflows/deploy.yml`
（均带 `:-false` 默认，dev / prod 缺省即关闭）。

## 迁移步骤

1. 合并本变更到两个仓库（fusion-api 与 ai-audio-assistant-web）。
2. **无需改任何环境变量**即可生效：默认 `LITELLM_HEALTH_ENABLED=false`，
   部署后启动即不再产生 `/health` 探测费用。
3. （可选）若要恢复模型健康灰度：在两个服务的运行环境设置
   `LITELLM_HEALTH_ENABLED=true`，并确认两服务指向同一 Redis 实例（同一 db），
   否则协调 key 不共享、退化为各探各的。
4. 验证：
   - 重启后观察服务日志，应出现
     `litellm_health: disabled (LITELLM_HEALTH_ENABLED=false), background refresh NOT started`；
   - `GET /api/models` 正常返回，`health.status` 为 `unknown`；
   - LiteLLM 侧日志不再出现 `["litellm-internal-health-check"]` 标记的请求。

## 回滚方法

- 临时恢复旧行为：设 `LITELLM_HEALTH_ENABLED=true` 并重启（注意此时每周期最多一轮，
  而非旧的 3 轮/周期）。
- 完全回退代码：`git revert` 本变更后重新部署；旧实现（每 worker 独立 30min 循环）
  即恢复。无数据库迁移，回滚无数据兼容风险。

## 费用估算

- 修复前（dev，全量探测）：3 worker/服务 × 每 30min 一轮 × 3 模型 ≈ **432 次 Qwen
  推理/天**（实测 AI Audio 258 次 + Fusion 129 次 = 387 次/天）。
- 修复后（默认关闭）：**0 次/天**，健康检查费用归零。
- 若显式开启：每周期 1 轮 × 3 模型 = **144 次/天**（比修复前降 67%，且只有单实例
  在执行）。
