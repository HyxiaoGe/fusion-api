-- Agent 可观测性迁移脚本
-- 新表 agent_sessions / agent_steps 由 create_all() 自动创建
-- 此脚本仅处理已有表 tool_call_logs 的字段扩展

-- 1. 给 tool_call_logs 添加 trace_id 和 step_number 字段
ALTER TABLE tool_call_logs ADD COLUMN IF NOT EXISTS trace_id VARCHAR;
ALTER TABLE tool_call_logs ADD COLUMN IF NOT EXISTS step_number INTEGER;

-- 2. 添加索引
CREATE INDEX IF NOT EXISTS ix_tool_call_logs_trace_id ON tool_call_logs (trace_id);
