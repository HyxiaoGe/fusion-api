-- BYOK Phase A: 加表 / 加列（不删除任何东西）
-- 配套代码 commit 部署后再执行 Phase B（DROP TABLE model_credentials）

-- 1. user_credentials 新表
CREATE TABLE IF NOT EXISTS user_credentials (
    id                    SERIAL PRIMARY KEY,
    user_id               VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider_id           VARCHAR NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    api_key               TEXT NOT NULL,
    is_active             BOOLEAN NOT NULL DEFAULT TRUE,
    last_error_kind       VARCHAR,
    last_error_message    TEXT,
    last_failure_at       TIMESTAMP,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    created_at            TIMESTAMP NOT NULL DEFAULT now(),
    updated_at            TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (user_id, provider_id)
);
CREATE INDEX IF NOT EXISTS ix_user_credentials_user_id ON user_credentials(user_id);

-- 2. providers 加 health 字段
ALTER TABLE providers
    ADD COLUMN IF NOT EXISTS status VARCHAR NOT NULL DEFAULT 'ok',
    ADD COLUMN IF NOT EXISTS offline_reason VARCHAR,
    ADD COLUMN IF NOT EXISTS offline_message TEXT,
    ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_failure_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS last_failure_kind VARCHAR;

-- 3. users 加 is_superuser 镜像列（从 auth-service 同步）
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_superuser BOOLEAN NOT NULL DEFAULT FALSE;
