# 编码规范

## 代码风格

- 使用 4 个空格缩进，不使用 Tab
- 单个函数不超过 50 行，职责单一
- 优先使用 f-string 格式化，避免 `%` 格式化
- 文件路径使用 `pathlib.Path` 而不是 `os.path`
- 使用 ruff 做 lint 和格式化（配置见 `pyproject.toml`）

## 命名

- 变量和函数使用英文命名（snake_case）
- 注释和文档使用中文
- 函数名以动词开头

## 导入顺序

标准库 → 第三方库 → 本地模块，每组之间空一行。由 ruff isort 自动管理。

## 错误处理

- 优先使用 try-except，提供中文错误信息
- 正式代码使用 logging，不使用 print
- 日志使用中文，便于调试

## 配置管理

- 使用 `.env` 文件管理环境变量，敏感信息不写入代码
- `requirements.txt` 管理生产运行依赖，`requirements-ci.txt` 管理 CI 临时工具
- 本地开发和测试使用 `requirements-dev.txt` 汇总运行依赖与 CI 工具

## 测试

- 测试文件在 `test/` 目录，使用 pytest
- LLM 调用使用 mock，不依赖真实凭证
- Redis 测试使用 fakeredis
- 安装测试依赖：`pip install -r requirements-dev.txt`
- 运行测试：`python -m pytest test/`
