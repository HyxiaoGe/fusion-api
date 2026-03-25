# Chat Core Data Flow

这份文档描述当前 `Fusion API` 的聊天核心运行面。

## Runtime Surface

默认启动入口是 [`main.py`](/Users/sean/code/fusion/fusion-api/main.py)。

当前默认暴露的主路由只有四组：

- `auth`
- `chat`
- `files`
- `models`

对应文件：

- [`app/api/auth.py`](/Users/sean/code/fusion/fusion-api/app/api/auth.py)
- [`app/api/chat.py`](/Users/sean/code/fusion/fusion-api/app/api/chat.py)
- [`app/api/files.py`](/Users/sean/code/fusion/fusion-api/app/api/files.py)
- [`app/api/models.py`](/Users/sean/code/fusion/fusion-api/app/api/models.py)

## Auth Flow

登录主线在 [`app/api/auth.py`](/Users/sean/code/fusion/fusion-api/app/api/auth.py)。

1. 前端打开 `/api/auth/login/{provider}`。
2. 后端基于 `SERVER_HOST` 生成 OAuth callback URL。
3. provider 回调 `/api/auth/callback/{provider}`。
4. 后端拉用户 profile，查或建 `users` / `social_accounts`。
5. 后端生成 JWT。
6. 后端重定向到 `FRONTEND_AUTH_CALLBACK_URL?token=...`。

JWT 校验在 [`app/core/security.py`](/Users/sean/code/fusion/fusion-api/app/core/security.py)。

## Chat Flow

聊天入口在 [`app/api/chat.py`](/Users/sean/code/fusion/fusion-api/app/api/chat.py)，主编排在 [`app/services/chat_service.py`](/Users/sean/code/fusion/fusion-api/app/services/chat_service.py)。

一次普通聊天的主线是：

1. 路由层接收 `provider / model / message / conversation_id / stream / options / file_ids`。
2. `ChatService` 校验用户、会话和模型。
3. 用户消息先写入会话历史。
4. 如果带 `file_ids`，会把已解析的文件内容注入提示上下文。
5. 根据 `stream` 和 `use_reasoning` 选择普通回复、推理流、或工具增强流。
6. assistant 回复写回 `messages`。
7. 前端随后可以再调用标题生成和推荐问题接口。

聊天持久化使用：

- [`app/services/memory_service.py`](/Users/sean/code/fusion/fusion-api/app/services/memory_service.py)
- [`app/db/repositories.py`](/Users/sean/code/fusion/fusion-api/app/db/repositories.py)

核心表是：

- `conversations`
- `messages`
- `files`
- `users`
- `social_accounts`
- `model_sources`
- `model_credentials`

## Streaming Flow

流式主逻辑在 [`app/services/stream_handler.py`](/Users/sean/code/fusion/fusion-api/app/services/stream_handler.py)。

当前有三种流：

- 普通流：直接输出 assistant 内容
- 推理流：分别输出 `reasoning_*` 和 `answering_*` 事件
- 工具增强流：先检测工具调用，再进入第二段模型回答

当前 SSE 事件的核心语义是：

- `reasoning_start`
- `reasoning_content`
- `reasoning_complete`
- `answering_start`
- `answering_content`
- `answering_complete`
- `done`
- `error`

推理流的占位消息会先落库，再在流结束后回写最终内容。

## File Flow

文件链路在 [`app/api/files.py`](/Users/sean/code/fusion/fusion-api/app/api/files.py) 和 [`app/services/file_service.py`](/Users/sean/code/fusion/fusion-api/app/services/file_service.py)。

主线是：

1. 上传文件
2. 写磁盘
3. 创建 `files` 记录
4. 关联 `conversation_files`
5. 异步解析文件
6. 成功时标记 `processed` 并写入 `parsed_content`
7. 失败时标记 `error`
8. 聊天时按 `file_ids` 把 `parsed_content` 注入上下文

底层解析器在 [`app/processor/file_processor.py`](/Users/sean/code/fusion/fusion-api/app/processor/file_processor.py)。

## Model Flow

模型主数据源在后端，不再以静态前端配置为真源。

主入口：

- [`app/api/models.py`](/Users/sean/code/fusion/fusion-api/app/api/models.py)
- [`app/ai/llm_manager.py`](/Users/sean/code/fusion/fusion-api/app/ai/llm_manager.py)

主线是：

1. 前端拉 `/api/models/`
2. 后端从 `model_sources` / `model_credentials` 读取可用模型和凭证
3. `llm_manager` 按 provider/model 构造实际客户端
4. 聊天、文件解析、标题生成都复用同一套模型来源

## Reading Order For New Engineers

如果只给 10 分钟，按这个顺序读：

1. [`main.py`](/Users/sean/code/fusion/fusion-api/main.py)
2. [`app/api/chat.py`](/Users/sean/code/fusion/fusion-api/app/api/chat.py)
3. [`app/services/chat_service.py`](/Users/sean/code/fusion/fusion-api/app/services/chat_service.py)
4. [`app/services/stream_handler.py`](/Users/sean/code/fusion/fusion-api/app/services/stream_handler.py)
5. [`app/services/file_service.py`](/Users/sean/code/fusion/fusion-api/app/services/file_service.py)
6. [`app/api/auth.py`](/Users/sean/code/fusion/fusion-api/app/api/auth.py)
7. [`app/api/models.py`](/Users/sean/code/fusion/fusion-api/app/api/models.py)

读完这几处，应该就能解释当前聊天主产品的数据流和边界。
