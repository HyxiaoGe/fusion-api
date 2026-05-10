"""
集成测试：GET /api/chat/conversations/metadata 和 GET /api/chat/conversations/search

测试模式（与 test_core_surface.py 保持一致）：
- unittest.TestCase + setUpClass，DATABASE_URL 在模块顶部设置（env 先于 app 导入）
- TestClient(main.app)
- dependency_overrides 注入 get_current_user + get_db（提供内存 SQLite 会话）
- ChatService 通过正常依赖解析注入被覆盖的 db session 构建
- 响应格式：{"code": "SUCCESS", "data": {...}, "request_id": "..."}

重要：SQLAlchemy 相关导入必须放在 importlib.import_module("main") **之后**（即函数内部），
否则 SQLAlchemy 模块在 app/db/database.py 初始化前被导入，导致 pydantic FieldInfo deepcopy 失败。
"""

import importlib
import os
import sys
import unittest
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

# 必须在 import main 之前设置（与 test_core_surface.py 完全一致）
os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


def _build_in_memory_db():
    """创建隔离的内存 SQLite 会话，含全部表结构。

    注意：
    1. SQLAlchemy 导入放在函数内，确保 main 已被 setUpClass 导入后才调用。
    2. 使用命名内存数据库（sqlite:///file:uuid?mode=memory&cache=shared&uri=true）
       确保每次调用都得到完全隔离的数据库实例，同时同一 session 的所有连接共享同一个 DB。
    """
    import sqlite3

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.database import Base
    from app.db.models import (  # noqa: F401 — 确保 mapper 注册
        Conversation,
        File,
        Message,
        ModelSource,
        Provider,
        SocialAccount,
        User,
    )

    # 每次测试用唯一名称，避免跨测试数据污染
    db_name = f"test_{uuid.uuid4().hex}"
    # 通过 creator 固定同一个连接，确保 create_all 和后续 session 操作在同一个内存 DB 上
    conn = sqlite3.connect(f"file:{db_name}?mode=memory&cache=shared", uri=True, check_same_thread=False)
    engine = create_engine(
        "sqlite://",
        creator=lambda: conn,
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _make_user(db, user_id: str):
    """在 DB 中创建最小 User 记录。username 用 uuid4 保证唯一，避免同测试内碰撞。"""
    from app.db.models import User

    user = User(id=user_id, username=f"u_{uuid.uuid4().hex[:12]}")
    db.add(user)
    db.flush()
    return user


def _make_conversation(db, user_id: str, title: str, model_id: str = "gpt-4.1") -> str:
    """在 DB 中创建一条 Conversation 记录，返回 id。"""
    from app.db.models import Conversation

    conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, title=title, model_id=model_id)
    db.add(conv)
    db.flush()
    return conv.id


class ChatMetadataSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")

        cls.main = main
        cls.client = TestClient(main.app)

        # 提取路由依赖引用（与 test_core_surface.py 完全一致）
        cls._route_deps = {}
        for route in main.app.routes:
            if hasattr(route, "dependant"):
                for dep in route.dependant.dependencies:
                    name = dep.call.__qualname__
                    if name not in cls._route_deps:
                        cls._route_deps[name] = dep.call

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    # ------------------------------------------------------------------ helpers

    def _setup_request(self, user_id: str, db):
        """注入 get_current_user（返回 fake_user）+ get_db（返回内存 SQLite 会话）。
        ChatService 通过正常依赖解析使用被覆盖的 db 构建，避免注入不可 deepcopy 的对象。

        关键：override_db 不能有带默认值的参数，否则 FastAPI 会将其解析为查询参数，
        并尝试 deepcopy 默认值（SQLAlchemy session），导致 TypeError。
        """
        fake_user = SimpleNamespace(id=user_id)

        gcu = self._route_deps.get("get_current_user")
        if gcu:
            captured_user = fake_user

            def override_user():
                return captured_user

            self.main.app.dependency_overrides[gcu] = override_user

        gdb = self._route_deps.get("get_db")
        if gdb:
            captured_db = db

            def override_db():
                yield captured_db

            self.main.app.dependency_overrides[gdb] = override_db

    # ================================================================ metadata

    def test_metadata_empty_ids_returns_empty(self):
        """ids= 空字符串 → 200 + items: []"""
        db = _build_in_memory_db()
        _make_user(db, "user-meta-empty")
        db.commit()
        self._setup_request("user-meta-empty", db)

        response = self.client.get("/api/chat/conversations/metadata?ids=")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["items"], [])

    def test_metadata_returns_only_user_owned(self):
        """user_id 隔离：seed 当前用户 + 其他用户的对话，只返回当前用户的"""
        my_user_id = "user-meta-mine"
        other_user_id = "user-meta-other"

        db = _build_in_memory_db()
        _make_user(db, my_user_id)
        _make_user(db, other_user_id)

        my_conv_id = _make_conversation(db, my_user_id, "我的对话")
        other_conv_id = _make_conversation(db, other_user_id, "别人的对话")
        db.commit()

        self._setup_request(my_user_id, db)

        ids_param = f"{my_conv_id},{other_conv_id}"
        response = self.client.get(f"/api/chat/conversations/metadata?ids={ids_param}")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        items = body["data"]["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], my_conv_id)

    def test_metadata_too_many_ids_returns_400(self):
        """传 101 个 ID → 400"""
        db = _build_in_memory_db()
        _make_user(db, "user-meta-limit")
        db.commit()
        self._setup_request("user-meta-limit", db)

        ids_param = ",".join([str(uuid.uuid4()) for _ in range(101)])
        response = self.client.get(f"/api/chat/conversations/metadata?ids={ids_param}")

        self.assertEqual(response.status_code, 400)

    # ================================================================ search

    def test_search_returns_matching_titles(self):
        """seed 标题含 'Python' 的 2 条 + 不含的 3 条 → q=Python → 返回 2 条"""
        user_id = "user-search-title"
        db = _build_in_memory_db()
        _make_user(db, user_id)

        _make_conversation(db, user_id, "Python 入门教程")
        _make_conversation(db, user_id, "Python 进阶实践")
        _make_conversation(db, user_id, "JavaScript 基础")
        _make_conversation(db, user_id, "Go 语言并发")
        _make_conversation(db, user_id, "Rust 内存模型")
        db.commit()

        self._setup_request(user_id, db)

        response = self.client.get("/api/chat/conversations/search?q=Python")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        items = body["data"]["items"]
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertIn("Python", item["title"])

    def test_search_limit_enforced(self):
        """seed 60 条 'test' → q=test&limit=10 → 返回 10 条"""
        user_id = "user-search-limit"
        db = _build_in_memory_db()
        _make_user(db, user_id)

        for i in range(60):
            _make_conversation(db, user_id, f"test conversation {i:03d}")
        db.commit()

        self._setup_request(user_id, db)

        response = self.client.get("/api/chat/conversations/search?q=test&limit=10")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        items = body["data"]["items"]
        self.assertEqual(len(items), 10)

    def test_search_isolates_users(self):
        """user_id 隔离：当前用户 1 条 'shared' + 其他用户 2 条 'shared' → 只返回 1 条"""
        my_user_id = "user-srch-mine"
        other_user_id = "user-srch-other"

        db = _build_in_memory_db()
        _make_user(db, my_user_id)
        _make_user(db, other_user_id)

        _make_conversation(db, my_user_id, "shared topic")
        _make_conversation(db, other_user_id, "shared project")
        _make_conversation(db, other_user_id, "shared notes")
        db.commit()

        self._setup_request(my_user_id, db)

        response = self.client.get("/api/chat/conversations/search?q=shared")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        items = body["data"]["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "shared topic")

    def test_search_empty_query_rejected(self):
        """q= 空字符串 → 422（FastAPI Query min_length=1 校验）"""
        db = _build_in_memory_db()
        _make_user(db, "user-search-empty")
        db.commit()
        self._setup_request("user-search-empty", db)

        response = self.client.get("/api/chat/conversations/search?q=")

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
