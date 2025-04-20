from app.db.repositories import PromptTemplateRepository
from app.db.database import get_db

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 加载提示词模板
    try:
        db = next(get_db())
        repo = PromptTemplateRepository(db)
        repo.load_to_prompt_manager()
        logging.info("提示词模板已加载到管理器")
    except Exception as e:
        logging.error(f"加载提示词模板失败: {e}") 