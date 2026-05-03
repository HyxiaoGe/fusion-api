-- 用户级 system prompt 迁移脚本
-- 1. users 表新增 system_prompt 字段（用户自定义的 AI 个性化提示词）
-- 2. 删除已废弃的 memories 表（功能改由 system_prompt 承担）

-- 1. 新增 system_prompt 列
ALTER TABLE users ADD COLUMN IF NOT EXISTS system_prompt TEXT NOT NULL DEFAULT '';

-- 2. 删除 memories 表
DROP TABLE IF EXISTS memories;
