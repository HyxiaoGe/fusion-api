---
name: add-provider
description: Add a new LLM provider to fusion-api. Use when integrating a new AI model service.
argument-hint: <provider-name>
---

# 添加新 LLM 提供商

Fusion API 通过 LiteLLM 统一接口，添加新提供商只需配置映射，不需要写 adapter。

## 步骤

### 1. 添加 LiteLLM 前缀映射

编辑 `app/ai/llm_manager.py`：

```python
PROVIDER_LITELLM_PREFIX = {
    # ...现有映射...
    "{provider}": "{litellm_prefix}",  # 新增
}
```

LiteLLM 前缀说明：
- 原生支持的：用对应前缀（如 `anthropic`、`deepseek`、`gemini`、`xai`）
- OpenAI 兼容接口的：统一用 `openai`，通过 `api_base` 指定实际地址

### 2. 如需自定义 api_base

如果新提供商使用 OpenAI 兼容接口但有自己的 base URL，加入：

```python
CUSTOM_BASE_URL_PROVIDERS = {"qwen", "volcengine", "wenxin", "hunyuan", "{provider}"}
```

凭证的 `base_url` 字段会被自动读取并传给 LiteLLM。

### 3. 如支持 reasoning/thinking

如果模型支持推理/思考过程输出，加入 `StreamHandler`：

```python
class StreamHandler:
    REASONING_PROVIDERS = {"deepseek", "qwen", "xai", "volcengine", "{provider}"}
```

### 4. 添加环境变量

在 `.env.example` 和 `docker-compose.yml` 中添加：

```bash
{PROVIDER_UPPER}_API_KEY=
{PROVIDER_UPPER}_API_BASE=  # 如需自定义 base URL
```

### 5. 在数据库中注册模型

通过 `POST /api/models/` 端点或直接在 `model_sources` 表中插入模型定义：

```json
{
  "model_id": "{provider}-model-name",
  "name": "模型显示名称",
  "provider": "{provider}",
  "capabilities": {"deepThinking": true},
  "enabled": true
}
```

然后通过 `POST /api/models/{model_id}/credentials` 添加凭证。

## 当前支持的提供商

| 提供商 | 前缀 | 自定义 base | reasoning |
|--------|------|------------|-----------|
| openai | openai | - | - |
| anthropic | anthropic | - | - |
| deepseek | deepseek | - | ✓ |
| google | gemini | - | - |
| qwen | openai | ✓ | ✓ |
| volcengine | openai | ✓ | ✓ |
| wenxin | openai | ✓ | - |
| hunyuan | openai | ✓ | - |
| xai | xai | - | ✓ |

## 验证

```bash
# 用 curl 测试新提供商
TOKEN=$(curl -s -X POST http://localhost:8100/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"claude-test@fusion.dev","password":"test123456","client_id":"app_xxx"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -s -N -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:8002/api/chat/send \
  -d '{"model_id":"{provider}-model","message":"你好","stream":true}'
```
