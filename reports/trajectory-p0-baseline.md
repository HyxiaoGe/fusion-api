# Trajectory P0 本机基线

## 结论（2026-08-22，Asia/Shanghai）

- 本机 Mock fast path 的绝对门限通过：三轮中最差 p95 `0.0725ms`、p99 `0.0841ms`，低于 p95 `5ms` / p99 `15ms`。
- 本机 Mock 相对门限**不作为通过项，且数值不满足 1.2 倍**：无 trajectory 的空操作基线只有约 `0.0003–0.0005ms`，低于可用于工程比较的计时粒度；加入真实 `TrajectoryRecorder` 调度后的 p95 约 `0.071ms`。因此本报告不宣称 §9.1 相对回归已经通过。
- 真实 PostgreSQL `EXPLAIN/ANALYZE`、真实连接/锁等待、真实模型与客户端 SSE TTFT 均未验证，留给 Task 7 授权 dev 环境验收。P0 性能总门禁在这些证据补齐前保持未完成。

## 测试环境

- 系统：macOS 26.5.2（25F84），arm64
- Python：3.11.15
- 样本：每种路径每轮 500 次，共 3 轮；`time.perf_counter()`；串行调用
- Redis：确定性内存 stub，不访问网络
- progress sink：确定性同步空操作 stub，不访问数据库
- trajectory sink：真实 `TrajectoryRecorder` 的准入、executor、wait/shield 路径；事件落库函数替换为确定性空操作
- 未使用 prompt、完整用户内容、凭证或真实模型请求

## 本机样本（毫秒）

| 轮次 | 路径 | p50 | p95 | p99 |
|---:|---|---:|---:|---:|
| 1 | Redis + progress stub | 0.0003 | 0.0004 | 0.0005 |
| 1 | + TrajectoryRecorder fast path | 0.0626 | 0.0725 | 0.0841 |
| 2 | Redis + progress stub | 0.0003 | 0.0003 | 0.0004 |
| 2 | + TrajectoryRecorder fast path | 0.0612 | 0.0709 | 0.0761 |
| 3 | Redis + progress stub | 0.0003 | 0.0004 | 0.0004 |
| 3 | + TrajectoryRecorder fast path | 0.0619 | 0.0715 | 0.0769 |

相对值被接近零的空操作基线主导：以第一轮为例，p95 约为 `181.25x`、p99 约为 `168.2x`，明确不满足 `1.2x`。这不应被解释成真实生产回归；也不能用它替代改造前后同数据库、同模型、同 SSE 客户端的对照数据。

## 已验证边界

- `BoundedSemaphore(4)` 在 12 个并发未完成事件下同时 worker / 模拟 Session 最大值均为 4；其余 8 个直接 `admission_full`，结束后 permit 守恒。
- CompositeWriter 顺序为 Redis 可见写 → progress sink → trajectory sink；trajectory 等待超时后 Redis 仍已先完成，辅助失败不回滚可见流。
- SQLite 语义测试覆盖 pending 三种已知状态及未知状态、target complete 仍降级、reason allowlist、未知 intent version/status/reason、普通 stale 三断言、幂等、running 不处理、legacy/meta_missing 与缺失水位。
- PostgreSQL dialect 编译断言覆盖协调候选的稳定 `LIMIT` 与 `FOR UPDATE OF run_trajectory_meta, agent_sessions SKIP LOCKED`。
- 现有 Recorder 配置测试覆盖 PostgreSQL `statement_timeout=200ms`、`lock_timeout=100ms`、`connect_timeout=1s`；迁移测试覆盖 pending 部分索引。

## 未验证边界与 Task 7 门禁

- 未连接真实 PostgreSQL，未执行协调 SQL 的 `EXPLAIN/ANALYZE`，未观测真实连接数、锁等待或多 scheduler 进程并发。
- 未运行真实 Redis、真实 progress upsert、真实 agent event INSERT；Mock Session 上限不等于生产连接池证据。
- 未调用真实模型，未比较模型内部 `first_output_delta.ttft_ms` 与客户端首个 reasoning/content/tool SSE 到达时间。
- 未建立授权 dev 环境中改造前/改造后的同场景 p95/p99；因此不得宣称相对 `1.2x` 门限通过。
- 若 dev 环境真实 p95/p99 或端到端 TTFT 超过 §9.1 门限，必须停止 P0 性能验收并按 §5.4 切换异步有界队列方案，不能放宽测试阈值。
