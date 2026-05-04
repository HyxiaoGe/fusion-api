---
name: add-endpoint
description: Add a new FastAPI API endpoint. Use when creating new REST API routes for the application.
argument-hint: <endpoint-name>
---

# 添加新 API 端点

按以下步骤添加新的 FastAPI 端点，以 `{endpoint_name}` 为例。

## 1. 定义 Pydantic Schema

在 `app/schemas/{endpoint_name}.py` 中定义请求/响应模型：

```python
from pydantic import BaseModel
from typing import Optional


class {Name}Request(BaseModel):
    """请求模型"""
    field: str


class {Name}Response(BaseModel):
    """响应模型"""
    id: str
    field: str
```

## 2. 创建 Service（如需业务逻辑）

在 `app/services/{endpoint_name}_service.py` 中实现业务逻辑：

```python
from sqlalchemy.orm import Session


class {Name}Service:
    def __init__(self, db: Session):
        self.db = db

    def create(self, data: dict) -> dict:
        # 业务逻辑
        pass
```

## 3. 创建 Router

在 `app/api/{endpoint_name}.py` 中定义路由：

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.{endpoint_name} import {Name}Request, {Name}Response

router = APIRouter()


@router.post("/", response_model={Name}Response)
async def create_{endpoint_name}(
    request: {Name}Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建{name}"""
    pass
```

## 4. 注册路由

在 `main.py` 中注册：

```python
from app.api.{endpoint_name} import router as {endpoint_name}_router

app.include_router({endpoint_name}_router, prefix="/api/{endpoint_name}", tags=["{endpoint_name}"])
```

## 5. 添加测试

在 `test/test_{endpoint_name}.py` 中编写测试。

## 注意事项

- 所有端点都需要 `get_current_user` 鉴权（除非明确公开）
- 遵循现有的错误处理模式：`HTTPException(status_code=xxx, detail="中文描述")`
- 代码注释使用中文
- 参考现有路由 `app/api/chat.py` 的写法
