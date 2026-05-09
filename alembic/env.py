"""Alembic 环境配置。

负责：
1. 从 app.core.config.settings 读 DATABASE_URL（不写死在 alembic.ini）
2. 把 app.db.database.Base.metadata 暴露给 autogenerate
3. 在线/离线两种模式（offline 输出 SQL；online 直接执行）

参考：https://alembic.sqlalchemy.org/en/latest/cookbook.html
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 把项目根加进 sys.path，让本文件能 import app.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 触发 SQLAlchemy 把所有 model 注册到 Base.metadata
import app.db.models  # noqa: F401, E402
from app.core.config import settings  # noqa: E402
from app.db.database import Base  # noqa: E402

# Alembic Config 对象
config = context.config

# 用 app 的 DATABASE_URL 覆盖 alembic.ini 默认值
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Python logging 配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 用的 metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：不连 DB，只输出 SQL。

    用法：alembic upgrade head --sql > out.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连 DB 直接执行。

    用法：alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 比较 server_default 时严格匹配（autogenerate 默认忽略，容易漏）
            compare_server_default=True,
            # 比较类型变更
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
