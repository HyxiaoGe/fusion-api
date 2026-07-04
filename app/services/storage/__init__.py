"""存储后端抽象层，支持 local、MinIO 和 OSS 存储方式"""

from typing import Optional

from app.core.config import settings
from app.services.storage.base import StorageBackend
from app.services.storage.local_storage import LocalStorageBackend

# 全局存储后端实例（在 lifespan 中初始化）
_storage_backend: Optional[StorageBackend] = None
_storage_backends: dict[str, StorageBackend] = {}


def _build_storage_backend(backend: str) -> StorageBackend:
    """按名称构造存储后端。"""
    if backend == "minio":
        from app.services.storage.minio_storage import MinIOStorageBackend

        return MinIOStorageBackend(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket=settings.MINIO_BUCKET,
            use_ssl=settings.MINIO_USE_SSL,
        )
    if backend == "oss":
        from app.services.storage.oss_storage import OSSStorageBackend

        return OSSStorageBackend(
            endpoint=settings.OSS_ENDPOINT,
            access_key_id=settings.OSS_ACCESS_KEY_ID,
            access_key_secret=settings.OSS_ACCESS_KEY_SECRET,
            bucket=settings.OSS_BUCKET,
            use_ssl=settings.OSS_USE_SSL,
        )
    if backend == "local":
        return LocalStorageBackend(
            base_path=settings.FILE_STORAGE_PATH,
            base_url_prefix="/api/files",
        )
    raise ValueError(f"不支持的存储后端: {backend}")


async def init_storage() -> StorageBackend:
    """初始化存储后端，应用启动时调用"""
    global _storage_backend

    _storage_backend = _build_storage_backend(settings.STORAGE_BACKEND)
    _storage_backends[settings.STORAGE_BACKEND] = _storage_backend
    if settings.STORAGE_BACKEND == "minio":
        await _storage_backend.ensure_bucket()

    return _storage_backend


def get_storage() -> StorageBackend:
    """获取全局存储后端实例"""
    if _storage_backend is None:
        raise RuntimeError("存储后端未初始化，请先调用 init_storage()")
    return _storage_backend


def get_storage_for_backend(backend: str | None) -> StorageBackend:
    """按文件记录里的 storage_backend 获取存储后端。"""
    # 旧文件记录可能没有 storage_backend 字段值；这些记录来自本地存储时代，按 local 读取。
    resolved_backend = backend or "local"
    if _storage_backend is not None and resolved_backend == settings.STORAGE_BACKEND:
        return _storage_backend

    cached = _storage_backends.get(resolved_backend)
    if cached is not None:
        return cached

    storage = _build_storage_backend(resolved_backend)
    _storage_backends[resolved_backend] = storage
    return storage
