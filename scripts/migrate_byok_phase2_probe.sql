-- BYOK Phase 2: 加 providers.last_probe_at 字段，供自动探活任务使用
-- 探活任务每 30 min 跑，按 offline_reason 决定下次探活间隔（key_invalid/quota_exceeded/other 30min；tos_blocked 24h）

ALTER TABLE providers
    ADD COLUMN IF NOT EXISTS last_probe_at TIMESTAMP;
