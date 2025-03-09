from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import chat, models, settings as settings_api
from app.core.logger import app_logger

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# 输出启动日志
app_logger.info(f"正在启动 {settings.APP_NAME} v{settings.APP_VERSION}")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境可以设置为"*"，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])

app_logger.info("所有路由已注册")

# main.py 的 if __name__ == "__main__" 部分
if __name__ == "__main__":
    import uvicorn
    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # 设置为debug级别以查看更多日志
        access_log=True     # 确保访问日志开启
    )