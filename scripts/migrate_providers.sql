-- scripts/migrate_providers.sql
-- Provider 表抽取迁移脚本
-- 执行前请先备份数据库

BEGIN;

-- 1. 创建 providers 表
CREATE TABLE IF NOT EXISTS providers (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    auth_config JSONB NOT NULL DEFAULT '{}',
    litellm_prefix VARCHAR NOT NULL,
    custom_base_url BOOLEAN DEFAULT FALSE,
    priority INTEGER DEFAULT 100,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 从现有数据和硬编码常量插入 provider 数据
-- auth_config 从 model_sources 中每个 provider 取第一条有效记录
INSERT INTO providers (id, name, auth_config, litellm_prefix, custom_base_url, priority, enabled)
SELECT DISTINCT ON (ms.provider)
    ms.provider AS id,
    CASE ms.provider
        WHEN 'openai' THEN 'OpenAI'
        WHEN 'anthropic' THEN 'Anthropic'
        WHEN 'google' THEN 'Google'
        WHEN 'xai' THEN 'xAI'
        WHEN 'deepseek' THEN 'DeepSeek'
        WHEN 'qwen' THEN '通义千问'
        WHEN 'volcengine' THEN '火山引擎'
        WHEN 'xiaomi' THEN '小米 MiMo'
        WHEN 'minimax' THEN 'MiniMax'
        WHEN 'moonshot' THEN '月之暗面'
        ELSE ms.provider
    END AS name,
    ms.auth_config::jsonb AS auth_config,
    CASE ms.provider
        WHEN 'openai' THEN 'openrouter/openai'
        WHEN 'anthropic' THEN 'openrouter/anthropic'
        WHEN 'google' THEN 'openrouter/google'
        WHEN 'xai' THEN 'openrouter/x-ai'
        WHEN 'deepseek' THEN 'deepseek'
        WHEN 'qwen' THEN 'openai'
        WHEN 'volcengine' THEN 'openai'
        WHEN 'xiaomi' THEN 'openai'
        WHEN 'minimax' THEN 'openai'
        WHEN 'moonshot' THEN 'openai'
        ELSE ms.provider
    END AS litellm_prefix,
    CASE WHEN ms.provider IN ('qwen', 'volcengine', 'xiaomi', 'minimax', 'moonshot')
        THEN TRUE ELSE FALSE
    END AS custom_base_url,
    MIN(ms.priority) AS priority,
    TRUE AS enabled
FROM model_sources ms
WHERE ms.auth_config IS NOT NULL AND ms.auth_config::text != '{}'
GROUP BY ms.provider, ms.auth_config
ORDER BY ms.provider, MIN(ms.priority);

-- 3. model_sources: JSON 列升级为 JSONB
ALTER TABLE model_sources
    ALTER COLUMN capabilities TYPE JSONB USING capabilities::jsonb,
    ALTER COLUMN pricing TYPE JSONB USING pricing::jsonb,
    ALTER COLUMN model_configuration TYPE JSONB USING model_configuration::jsonb;

-- 4. model_sources: 删除 auth_config 列
ALTER TABLE model_sources DROP COLUMN IF EXISTS auth_config;

-- 5. model_sources: provider 列加外键
ALTER TABLE model_sources
    ADD CONSTRAINT fk_model_sources_provider
    FOREIGN KEY (provider) REFERENCES providers(id);

-- 6. 验证
DO $$
DECLARE
    provider_count INTEGER;
    orphan_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO provider_count FROM providers;
    RAISE NOTICE '已创建 % 个 provider 记录', provider_count;

    SELECT COUNT(*) INTO orphan_count
    FROM model_sources ms
    WHERE NOT EXISTS (SELECT 1 FROM providers p WHERE p.id = ms.provider);
    IF orphan_count > 0 THEN
        RAISE EXCEPTION '存在 % 个 model_sources 无对应 provider', orphan_count;
    END IF;
END $$;

COMMIT;
