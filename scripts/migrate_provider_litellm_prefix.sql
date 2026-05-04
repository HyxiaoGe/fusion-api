-- 校准 providers.litellm_prefix
-- 背景：旧代码 LLMManager.resolve_model 直接用 provider.id 拼 litellm_model 字符串，
-- 没用 litellm_prefix 字段，导致这个字段写错也无影响。新代码改为用
-- provider_rel.litellm_prefix，所以必须把这 5 个 provider 的 prefix 校准到
-- 与 litellm-proxy model_list 注册的模式（或 OpenRouter 子路径）一致。

UPDATE providers SET litellm_prefix = 'qwen' WHERE id = 'qwen';
UPDATE providers SET litellm_prefix = 'moonshot' WHERE id = 'moonshot';
UPDATE providers SET litellm_prefix = 'xiaomi' WHERE id = 'xiaomi';
UPDATE providers SET litellm_prefix = 'minimax' WHERE id = 'minimax';
UPDATE providers SET litellm_prefix = 'doubao' WHERE id = 'volcengine';

-- 校准后预期：
--   deepseek          -> deepseek
--   qwen              -> qwen
--   moonshot          -> moonshot
--   xiaomi            -> xiaomi
--   minimax           -> minimax
--   volcengine        -> doubao
--   gemini provider 暂未在 fusion-api 注册（DB 没有 google 路径走 gemini，gemini 走 OpenRouter），无需改
--   anthropic         -> openrouter/anthropic（保持）
--   google            -> openrouter/google（保持）
--   openai            -> openrouter/openai（保持）
--   xai               -> openrouter/x-ai（保持）
