"""存储后端抽象层，支持 local 和 MinIO 两种存储方式"""

from typing import Optional

from app.core.config import settings
from app.services.storage.base import StorageBackend
from app.services.storage.local_storage import LocalStorageBackend

# 全局存储后端实例（在 lifespan 中初始化）
_storage_backend: Optional[StorageBackend] = None


async def init_storage() -> StorageBackend:
    """初始化存储后端，应用启动时调用"""
    global _storage_backend

    if settings.STORAGE_BACKEND == "minio":
        from app.services.storage.minio_storage import MinIOStorageBackend

        _storage_backend = MinIOStorageBackend(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            use_ssl=settings.MINIO_USE_SSL,
        )
        await _storage_backend.ensure_bucket()
    else:
        _storage_backend = LocalStorageBackend(
            base_path=settings.FILE_STORAGE_PATH,
            base_url_prefix="/api/files",
        )

    return _storage_backend


def get_storage() -> StorageBackend:
    """获取全局存储后端实例"""
    if _storage_backend is None:
        raise RuntimeError("存储后端未初始化，请先调用 init_storage()")
    return _storage_backend
