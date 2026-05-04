-- Agent 状态机重设计 Phase 1 — DB 迁移
-- 1. agent_steps 增加 status 字段，默认 'completed' 兜底历史行
-- 2. agent_sessions.status 枚举扩展：终态新增 'interrupted'，
--    并新增 'running' 作为 run 启动时的占位值（finally 块更新为终态）
--    — 仅注释口径，无 DDL（VARCHAR(20) 已能容纳新枚举）
-- 注：agent_sessions 历史行无需 backfill，已写入的 status 已是
--    completed/limit_reached/error 之一，都在新枚举集合内。

BEGIN;

-- agent_steps.status 新字段
ALTER TABLE agent_steps
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed';

-- 索引：按 status 查 in-flight / failed step 用
CREATE INDEX IF NOT EXISTS idx_agent_steps_status ON agent_steps(status);

COMMIT;

-- 验证
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'agent_steps' AND column_name = 'status';
