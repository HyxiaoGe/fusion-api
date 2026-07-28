# LiteLLM v1.93.0 隔离升级证据

时间：2026-07-28（Asia/Shanghai）

## 生产影响边界

- 运行中的 `litellm-proxy:v1.88.0` 未重启、未切换、未修改数据库。
- 测试使用当日 PostgreSQL 备份恢复出的临时数据库。
- 临时 LiteLLM、PostgreSQL、Redis 只连接独立 Docker 网络，没有发布主机端口。
- 验证结束后临时容器和网络均已删除。

## 备份门禁

- restic 备份脚本精确排除了不可恢复且由容器重建的 `speedtest-data/log/logrotate.status`。
- 手动任务退出码：`0`。
- 新 daily 快照：`c17f85b9`。
- 日志结果：`=== 备份成功 ===`。
- 当日全库备份：`pg_20260728.sql.gz`，压缩包约 85 MB。
- marker 集成后的再次备份成功，完整 snapshot ID 为
  `aa48738456aabd7ca50c1c554647577d0714b3d7954afb7885dd22667e09878b`。
- marker 已由最新 `daily` snapshot 只读结果重新生成，包含 `completed_at` 和 `tag=daily`，文件权限为 `0600`；升级门禁按 snapshot 时间而不是 marker mtime 判断新鲜度。

## 恢复演练

- 使用生产同类 `postgres:15-cron-jieba` 镜像完整恢复 `pg_dumpall`。
- LiteLLM 恢复结果：
  - public tables：73；
  - DB models：17；
  - virtual keys：26。
- 标准 PostgreSQL 16 镜像因缺少其他业务库使用的 `pg_jieba` 扩展而失败，已确认不是 LiteLLM 数据损坏。

## v1.93.0 迁移

- 镜像：`litellm/litellm:v1.93.0`。
- digest：`sha256:a1745e629abfb17d434426ff48b115f54f4f4c4a0f5af241de569e93c63c411e`。
- readiness：healthy，DB connected。
- 迁移后 public tables：74。
- 迁移后 DB models：17。
- 迁移后 virtual keys：26。
- 迁移前后 17 个 DB model 业务别名完全一致。
- 临时 Redis 接入后启动日志无新增 ERROR/Traceback，容器 restart=0、OOM=false。

### DB 环境变量引用跨重启验证

使用 `scripts/validate_litellm_db_env_reference.sh` 在独立 Docker 网络中再次验证固定 digest：

- 通过 `/model/new` 注册 `api_key: os.environ/FAKE_PROVIDER_KEY`；
- 停止并重建 LiteLLM 容器，数据库和 salt 保持不变；
- 重启后调用本地 mock provider，mock 只接受预设假 Bearer key；
- completion 返回 `env-reference-ok`；
- 外部厂商请求数为 `0`，测试容器和网络在退出时清理。

这证明 v1.93.0 固定 digest 能在 DB model 跨重启加载时解析环境变量引用；实际准入仍需确认目标主机已注入对应环境变量。

## 成本表同步

- `POST /reload/model_cost_map`：HTTP 200。
- `POST /schedule/model_cost_map_reload?hours=6`：HTTP 200。
- 隔离状态：
  - scheduled：true；
  - interval_hours：6；
  - last_run：2026-07-28T14:21:50.080084；
  - next_run：2026-07-28T20:21:50.080084。

上述时间为 LiteLLM 返回的 UTC 时间；对应操作发生在 Asia/Shanghai 2026-07-28 22:21。

## 尚未覆盖

- 没有把 dev 运行容器切到 v1.93.0。
- 没有通过隔离 v1.93.0 发起收费模型 completion。
- 没有执行 Fusion 真实登录态 UI 回归。
- 没有启用 live 成本表调度。

这些项目必须在实际 dev 升级授权和门禁阶段继续完成，不能由本隔离证据替代。

## 真实只读候选发现

使用 dev 当前厂商凭据、live LiteLLM `/model/info` 和一次性 `v1.93.0` 工具容器运行候选协调器；没有调用模型 completion，也没有 LiteLLM 写操作。

| Provider | 请求状态 | new | existing | removed review | unknown |
|---|---:|---:|---:|---:|---:|
| Moonshot | 成功 | 4 | 1 | 0 | 7 |
| DeepSeek | 成功 | 0 | 2 | 6 | 0 |
| Qwen | 成功 | 59 | 1 | 2 | 171 |
| Xiaomi | 成功 | 5 | 2 | 0 | 0 |

Moonshot `new` 中明确包含 `kimi-k3`。Qwen 同时返回大量语音、图片、翻译等非聊天模型，证明“厂商发现成功”不能直接触发 Fusion 上架；后续必须增加 endpoint/capability 分类和真实能力验收。

`removed` 只表示厂商当前 `/v1/models` 与 LiteLLM 已知目录存在差异，状态保持 `retirement_review`，不会自动删除现有模型。

## Kimi K3 元数据证据边界

2026-07-28 再次核对当前官方资料：

- Moonshot 官方开放平台已列出 `kimi-k3`，中国区 API base 为 `https://api.moonshot.cn/v1`；
- 官方声明 K3 始终开启推理，支持 `reasoning_effort=low/high/max`、视觉输入、SSE 和 Tool Calling，上下文为 1,048,576 tokens；
- 官方中国区价格为每 1M tokens：缓存命中输入 ¥2、未命中输入 ¥20、输出 ¥100；
- 同时抓取的 LiteLLM 官方成本表共有 2,984 个条目，其中 63 个 key 包含 Moonshot，但没有 `kimi-k3`。

因此当前不写入 K3 override：Fusion metadata 目前以 USD 单位展示，LiteLLM 运行时成本表也尚未收录 K3；在没有受审汇率/自定义计费策略和真实 usage/cost 证据前，自动换算会制造错误计费。候选保持隔离，等待 LiteLLM 上游收录或后续明确批准的定价策略。

官方资料：

- <https://platform.kimi.com/docs/guide/kimi-k3-quickstart>
- <https://platform.kimi.com/docs/pricing/chat-k3>
