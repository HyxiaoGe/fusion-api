from app.db.database import engine, Base


def init_db():
    """初始化数据库表结构"""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        raise
