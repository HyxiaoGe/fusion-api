# Fusion 生产压测工具

`runner.py` 提供公共 HTTP 与真实认证 SSE 两类阶梯压测。它不会读取浏览器存储；一次性邮箱、密码、访问令牌和刷新令牌只保留在进程内存。JSON 结果只暴露压测指标、`run_id`、邮箱指纹，以及清理孤立数据所需的非凭据 Agent run/trace ID。

## 安全边界

- 生产域名必须显式传入 `--confirm-production`。
- HTTP 默认阶梯为 `1,5,10,25,50`，SSE 默认阶梯为 `1,3,5`。
- HTTP 在至少 20 个样本后，错误率或超时率达到 5% 会停止；连续失败达到 10 次也会停止。
- SSE 任意 flow 失败或收到 error frame 后，不再提升并发。
- 每个 SSE flow 在发请求前登记唯一 `conversation_id`，`finally` 中按 ID 精确删除，并吊销注册和 Fusion 登录产生的两个刷新令牌。
- auth-service 暂无删除账号的公开接口，一次性账号会保留。可用结果中的 `run_id` 精确重建账号邮箱：`fusion-perf+<run_id>@seanfield.org`，由运维在压测后删除；日志和结果不会输出完整邮箱。
- runner 会从 `agent_event.run_started` 提取非凭据的 `agent_run_ids` / `agent_trace_ids`。error frame 只计数，不保存或输出 frame 内容。

## 压测后 SQL 清理顺序

runner 会先通过 Fusion API 按清单逐个删除 conversation；外键级联会清理 `agent_sessions` 和 `agent_progress_snapshots`。`agent_steps.trace_id` 当前没有指向 `agent_sessions` 的外键，因此需要在 API 清理成功后，用脱敏结果中的 `agent_trace_ids` 做第二步精确清理：

```sql
BEGIN;

SELECT trace_id, count(*)
FROM agent_steps
WHERE trace_id = ANY(:agent_trace_ids)
GROUP BY trace_id;

DELETE FROM agent_steps
WHERE trace_id = ANY(:agent_trace_ids);

COMMIT;
```

必须使用数据库驱动的数组参数绑定，不要把 ID 拼接进 SQL。删除后再次执行相同 `SELECT`，结果应为空。最后再到 auth-service 数据库按 `fusion-perf+<run_id>@seanfield.org` 精确删除一次性账号；不要使用宽泛的 `LIKE 'fusion-perf%'` 批量删除。

## 使用

仓库 `.env.example` 中的 `fusion-client` 已不是生产实际应用 ID。runner 默认使用当前公开生产应用 ID，也可通过 `FUSION_PERF_CLIENT_ID` 或 `--client-id` 覆盖：

```bash
python -m scripts.perf.runner \
  --mode all \
  --model-id '<低成本模型 ID>' \
  --confirm-production
```

只跑无认证 HTTP 基线：

```bash
python -m scripts.perf.runner --mode http --confirm-production
```

退出码 `2` 表示触发硬停止门禁、清理不完整或参数错误。不要把 stdout/stderr 重定向到包含环境变量或 shell trace 的日志中。
