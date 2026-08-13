"""使用 Milvus root 账号一次性创建知识库应用账号和专用数据库。"""

from __future__ import annotations

import os


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} 不能为空")
    return value


def _already_exists(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "already" in message or "duplicate" in message


def bootstrap() -> None:
    from pymilvus import MilvusClient

    uri = _required("MILVUS_URI")
    root_username = os.getenv("MILVUS_BOOTSTRAP_USERNAME", "root").strip()
    root_password = _required("MILVUS_BOOTSTRAP_PASSWORD")
    app_username = _required("MILVUS_USERNAME")
    app_password = _required("MILVUS_PASSWORD")
    database = _required("MILVUS_DATABASE")
    role = os.getenv("MILVUS_ROLE", f"{app_username}_role").strip()
    if app_username.casefold() == "root":
        raise ValueError("应用账号不能使用 root")

    admin = MilvusClient(uri=uri, user=root_username, password=root_password, db_name="default")
    if database not in admin.list_databases():
        admin.create_database(db_name=database)
    if app_username not in admin.list_users():
        admin.create_user(user_name=app_username, password=app_password)
    if role not in admin.list_roles():
        admin.create_role(role_name=role)
    described = admin.describe_user(user_name=app_username)
    assigned_roles = described.get("roles", []) if isinstance(described, dict) else []
    if isinstance(assigned_roles, str):
        assigned_roles = [assigned_roles]
    if role not in assigned_roles:
        admin.grant_role(user_name=app_username, role_name=role)
    # Milvus 的数据库级与 collection 级权限不会相互继承：前者负责发现、
    # 创建 collection，后者负责建索引和读写其中的数据，两者缺一不可。
    for privilege in ("DatabaseReadWrite", "CollectionReadWrite"):
        try:
            admin.grant_privilege_v2(
                role_name=role,
                privilege=privilege,
                collection_name="*",
                db_name=database,
            )
        except Exception as exc:
            if not _already_exists(exc):
                raise

    application = MilvusClient(
        uri=uri,
        user=app_username,
        password=app_password,
        db_name=database,
    )
    application.list_collections()


if __name__ == "__main__":
    bootstrap()
