import json
import os
import re
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI桌面聊天应用"
    APP_VERSION: str = "0.1.1"

    ENABLE_VECTOR_EMBEDDINGS: bool = False

    CHROMA_URL: str = os.getenv("CHROMA_URL", "http://localhost:8001")

    # 模型配置
    DEFAULT_MODEL: str = "qwen"  # 默认使用的模型
    LITELLM_GOVERNANCE_ROOT: str = os.getenv("LITELLM_GOVERNANCE_ROOT", "")
    LITELLM_GOVERNANCE_MAX_AGE_SECONDS: int = int(os.getenv("LITELLM_GOVERNANCE_MAX_AGE_SECONDS", "86400"))
    LITELLM_MODEL_MANAGEMENT_ENABLED: bool = os.getenv("LITELLM_MODEL_MANAGEMENT_ENABLED", "false").lower() == "true"
    LITELLM_MODEL_ADMISSION_WORKER_ENABLED: bool = (
        os.getenv("LITELLM_MODEL_ADMISSION_WORKER_ENABLED", "false").lower() == "true"
    )
    LITELLM_MODEL_ADMISSION_WORKER_TOKEN: str = os.getenv("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", "")
    LITELLM_MODEL_ADMISSION_LEASE_SECONDS: int = int(os.getenv("LITELLM_MODEL_ADMISSION_LEASE_SECONDS", "600"))
    LITELLM_PROXY_URL: str = os.getenv("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # Redis 配置
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_STREAM_TTL: int = 300  # 流状态缓存 TTL（秒）

    # 文件存储配置
    FILE_STORAGE_PATH: str = os.getenv("FILE_STORAGE_PATH", "./storage/files")
    FILE_STORAGE_KEY_PREFIX: str = os.getenv("FILE_STORAGE_KEY_PREFIX", "files/v1")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    FILE_UPLOAD_TIMEOUT_SECONDS: int = int(os.getenv("FILE_UPLOAD_TIMEOUT_SECONDS", "60"))
    DIRECT_UPLOAD_STALE_SECONDS: int = int(os.getenv("DIRECT_UPLOAD_STALE_SECONDS", "1800"))
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/heic",
        "image/heif",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/csv",
    ]

    # 存储后端配置（"local"、"minio" 或 "oss"）
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "fusion-files")
    MINIO_USE_SSL: bool = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
    MINIO_PRESIGN_EXPIRES: int = int(os.getenv("MINIO_PRESIGN_EXPIRES", "3600"))
    OSS_ENDPOINT: str = os.getenv("OSS_ENDPOINT", "")
    OSS_ACCESS_KEY_ID: str = os.getenv("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET: str = os.getenv("OSS_ACCESS_KEY_SECRET", "")
    OSS_BUCKET: str = os.getenv("OSS_BUCKET", "")
    OSS_USE_SSL: bool = os.getenv("OSS_USE_SSL", "true").lower() == "true"

    # 知识库 v1：API、独立 Worker、Embedding 与 Milvus 均通过配置显式启用。
    KNOWLEDGE_BASE_ENABLED: bool = os.getenv("KNOWLEDGE_BASE_ENABLED", "false").lower() == "true"
    KNOWLEDGE_MAX_BASES_PER_USER: int = int(os.getenv("KNOWLEDGE_MAX_BASES_PER_USER", "50"))
    KNOWLEDGE_MAX_DOCUMENTS_PER_BASE: int = int(os.getenv("KNOWLEDGE_MAX_DOCUMENTS_PER_BASE", "100"))
    KNOWLEDGE_MAX_FILE_SIZE: int = int(os.getenv("KNOWLEDGE_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
    KNOWLEDGE_ALLOWED_MIME_TYPES: str = os.getenv(
        "KNOWLEDGE_ALLOWED_MIME_TYPES",
        ",".join(
            (
                "text/plain",
                "text/markdown",
                "text/csv",
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ),
    )
    KNOWLEDGE_PARSER_VERSION: str = os.getenv("KNOWLEDGE_PARSER_VERSION", "parser-v1")
    KNOWLEDGE_CHUNKER_VERSION: str = os.getenv("KNOWLEDGE_CHUNKER_VERSION", "chunker-v1")
    KNOWLEDGE_CHUNK_SIZE: int = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "1200"))
    KNOWLEDGE_CHUNK_OVERLAP: int = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "200"))
    KNOWLEDGE_PARSE_TIMEOUT_SECONDS: int = int(os.getenv("KNOWLEDGE_PARSE_TIMEOUT_SECONDS", "60"))
    KNOWLEDGE_EMBEDDING_PROVIDER: str = os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "litellm")
    KNOWLEDGE_EMBEDDING_MODEL: str = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "")
    KNOWLEDGE_EMBEDDING_REVISION: str = os.getenv("KNOWLEDGE_EMBEDDING_REVISION", "")
    KNOWLEDGE_EMBEDDING_REVISION_ROUTES: str = os.getenv("KNOWLEDGE_EMBEDDING_REVISION_ROUTES", "{}")
    KNOWLEDGE_EMBEDDING_DIMENSION: int = int(os.getenv("KNOWLEDGE_EMBEDDING_DIMENSION", "1024"))
    KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS: str = os.getenv("KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS", "1024")
    KNOWLEDGE_EMBEDDING_BATCH_SIZE: int = int(os.getenv("KNOWLEDGE_EMBEDDING_BATCH_SIZE", "32"))
    KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS: float = float(os.getenv("KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", "30"))
    KNOWLEDGE_DISTANCE_METRIC: str = os.getenv("KNOWLEDGE_DISTANCE_METRIC", "COSINE").upper()
    KNOWLEDGE_WORKER_POLL_SECONDS: float = float(os.getenv("KNOWLEDGE_WORKER_POLL_SECONDS", "2"))
    KNOWLEDGE_WORKER_LEASE_SECONDS: int = int(os.getenv("KNOWLEDGE_WORKER_LEASE_SECONDS", "180"))
    KNOWLEDGE_WORKER_HEARTBEAT_SECONDS: int = int(os.getenv("KNOWLEDGE_WORKER_HEARTBEAT_SECONDS", "30"))
    KNOWLEDGE_WORKER_MAX_ATTEMPTS: int = int(os.getenv("KNOWLEDGE_WORKER_MAX_ATTEMPTS", "5"))
    KNOWLEDGE_WORKER_RETRY_BASE_SECONDS: int = int(os.getenv("KNOWLEDGE_WORKER_RETRY_BASE_SECONDS", "5"))
    KNOWLEDGE_WORKER_RETRY_MAX_SECONDS: int = int(os.getenv("KNOWLEDGE_WORKER_RETRY_MAX_SECONDS", "300"))
    KNOWLEDGE_WORKER_HEALTH_FILE: str = os.getenv(
        "KNOWLEDGE_WORKER_HEALTH_FILE",
        "/tmp/fusion-knowledge-worker-health.json",
    )
    MILVUS_URI: str = os.getenv("MILVUS_URI", "")
    MILVUS_USERNAME: str = os.getenv("MILVUS_USERNAME", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_DATABASE: str = os.getenv("MILVUS_DATABASE", "")
    MILVUS_COLLECTION_PREFIX: str = os.getenv("MILVUS_COLLECTION_PREFIX", "fusion_knowledge_chunks")
    MILVUS_TIMEOUT_SECONDS: float = float(os.getenv("MILVUS_TIMEOUT_SECONDS", "10"))

    @property
    def RESOLVED_KNOWLEDGE_ALLOWED_MIME_TYPES(self) -> List[str]:
        return list(
            dict.fromkeys(value.strip() for value in self.KNOWLEDGE_ALLOWED_MIME_TYPES.split(",") if value.strip())
        )

    @property
    def RESOLVED_KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS(self) -> List[int]:
        return list(
            dict.fromkeys(
                int(value.strip()) for value in self.KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS.split(",") if value.strip()
            )
        )

    @property
    def RESOLVED_KNOWLEDGE_EMBEDDING_REVISION_ROUTES(self) -> dict[str, str]:
        try:
            raw_routes = json.loads(self.KNOWLEDGE_EMBEDDING_REVISION_ROUTES)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 必须是 JSON 对象") from exc
        if not isinstance(raw_routes, dict):
            raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 必须是 JSON 对象")
        if not raw_routes:
            raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 不能为空")
        routes: dict[str, str] = {}
        for raw_key, raw_alias in raw_routes.items():
            if not isinstance(raw_key, str) or not isinstance(raw_alias, str):
                raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 的键和值必须是字符串")
            key = raw_key.strip()
            alias = raw_alias.strip()
            model, separator, revision = key.rpartition("@")
            if (
                raw_key != key
                or raw_alias != alias
                or not separator
                or re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", model) is None
                or re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", revision) is None
                or re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", alias) is None
                or alias.startswith("litellm_proxy/")
            ):
                raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 包含无效的 model@revision 或 alias")
            routes[key] = alias
        return routes

    def resolve_knowledge_embedding_route(self, model: str, revision: str | None) -> str:
        """把持久化 model+revision 解析为不可变 LiteLLM Proxy alias。"""
        if not model.strip() or not revision or not revision.strip():
            raise ValueError("Embedding profile 缺少 model 或 revision")
        route = self.RESOLVED_KNOWLEDGE_EMBEDDING_REVISION_ROUTES.get(f"{model}@{revision}")
        if route is None:
            raise ValueError("KNOWLEDGE_EMBEDDING_REVISION_ROUTES 缺少持久化 model@revision")
        return route

    def validate_knowledge_base_configuration(self) -> None:
        """知识库启用时执行集中、可复用的 fail-closed 配置校验。"""
        if not self.KNOWLEDGE_BASE_ENABLED:
            return
        errors: list[str] = []
        try:
            allowed_dimensions = self.RESOLVED_KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS
        except ValueError:
            allowed_dimensions = []
            errors.append("KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS 必须是整数列表")
        if self.KNOWLEDGE_EMBEDDING_PROVIDER != "litellm":
            errors.append("KNOWLEDGE_EMBEDDING_PROVIDER 必须为 litellm")
        if self.KNOWLEDGE_PARSER_VERSION != "parser-v1":
            errors.append("KNOWLEDGE_PARSER_VERSION 必须与当前解析器实现一致")
        if self.KNOWLEDGE_CHUNKER_VERSION != "chunker-v1":
            errors.append("KNOWLEDGE_CHUNKER_VERSION 必须与当前切片器实现一致")
        if not self.KNOWLEDGE_EMBEDDING_MODEL.strip():
            errors.append("KNOWLEDGE_EMBEDDING_MODEL 不能为空")
        if not self.KNOWLEDGE_EMBEDDING_REVISION.strip():
            errors.append("KNOWLEDGE_EMBEDDING_REVISION 不能为空")
        if self.KNOWLEDGE_EMBEDDING_MODEL != self.KNOWLEDGE_EMBEDDING_MODEL.strip():
            errors.append("KNOWLEDGE_EMBEDDING_MODEL 前后不能包含空白")
        if self.KNOWLEDGE_EMBEDDING_REVISION != self.KNOWLEDGE_EMBEDDING_REVISION.strip():
            errors.append("KNOWLEDGE_EMBEDDING_REVISION 前后不能包含空白")
        if self.KNOWLEDGE_EMBEDDING_MODEL.startswith("litellm_proxy/"):
            errors.append("KNOWLEDGE_EMBEDDING_MODEL 必须填写 Proxy 中的原始 alias")
        try:
            self.resolve_knowledge_embedding_route(
                self.KNOWLEDGE_EMBEDDING_MODEL.strip(),
                self.KNOWLEDGE_EMBEDDING_REVISION.strip(),
            )
        except ValueError as exc:
            errors.append(str(exc))
        litellm_proxy_url = urlparse(self.LITELLM_PROXY_URL)
        if litellm_proxy_url.scheme not in {"http", "https"} or not litellm_proxy_url.netloc:
            errors.append("LITELLM_PROXY_URL 必须是完整的 http(s) 地址")
        if not self.LITELLM_API_KEY:
            errors.append("LITELLM_API_KEY 不能为空")
        if self.KNOWLEDGE_EMBEDDING_DIMENSION <= 0 or self.KNOWLEDGE_EMBEDDING_DIMENSION not in allowed_dimensions:
            errors.append("KNOWLEDGE_EMBEDDING_DIMENSION 必须进入允许维度列表")
        if self.KNOWLEDGE_DISTANCE_METRIC != "COSINE":
            errors.append("KNOWLEDGE_DISTANCE_METRIC 必须为 COSINE")
        if not 100 <= self.KNOWLEDGE_CHUNK_SIZE <= 16_000:
            errors.append("KNOWLEDGE_CHUNK_SIZE 必须在 100 到 16000 之间")
        if (
            self.KNOWLEDGE_CHUNK_OVERLAP < 0
            or self.KNOWLEDGE_CHUNK_OVERLAP * 2 > self.KNOWLEDGE_CHUNK_SIZE
            or self.KNOWLEDGE_CHUNK_SIZE - self.KNOWLEDGE_CHUNK_OVERLAP < 100
        ):
            errors.append("KNOWLEDGE_CHUNK_OVERLAP 不得超过 chunk size 的一半，且切片最小步长为 100")
        if not 1 <= self.KNOWLEDGE_MAX_FILE_SIZE <= 50 * 1024 * 1024:
            errors.append("KNOWLEDGE_MAX_FILE_SIZE 必须在 1 到 52428800 字节之间")
        if not 1 <= self.KNOWLEDGE_PARSE_TIMEOUT_SECONDS <= 300:
            errors.append("KNOWLEDGE_PARSE_TIMEOUT_SECONDS 必须在 1 到 300 之间")
        if self.KNOWLEDGE_EMBEDDING_BATCH_SIZE <= 0:
            errors.append("KNOWLEDGE_EMBEDDING_BATCH_SIZE 必须大于 0")
        if not 1 <= self.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS <= 120:
            errors.append("KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS 必须在 1 到 120 秒之间")
        if not 0 < self.KNOWLEDGE_WORKER_POLL_SECONDS <= 60:
            errors.append("KNOWLEDGE_WORKER_POLL_SECONDS 必须大于 0 且不超过 60 秒")
        if self.KNOWLEDGE_WORKER_LEASE_SECONDS <= 0:
            errors.append("KNOWLEDGE_WORKER_LEASE_SECONDS 必须大于 0")
        if (
            self.KNOWLEDGE_WORKER_HEARTBEAT_SECONDS <= 0
            or self.KNOWLEDGE_WORKER_HEARTBEAT_SECONDS * 2 >= self.KNOWLEDGE_WORKER_LEASE_SECONDS
        ):
            errors.append("KNOWLEDGE_WORKER_HEARTBEAT_SECONDS 必须小于 lease 的一半")
        if self.KNOWLEDGE_WORKER_MAX_ATTEMPTS <= 0:
            errors.append("KNOWLEDGE_WORKER_MAX_ATTEMPTS 必须大于 0")
        allowed_mime_types = set(self.RESOLVED_KNOWLEDGE_ALLOWED_MIME_TYPES)
        supported_mime_types = {
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        if not allowed_mime_types or not allowed_mime_types.issubset(supported_mime_types):
            errors.append("KNOWLEDGE_ALLOWED_MIME_TYPES 包含 v1 解析器不支持的类型")
        milvus_uri = urlparse(self.MILVUS_URI)
        if milvus_uri.scheme not in {"http", "https"} or not milvus_uri.netloc:
            errors.append("MILVUS_URI 必须是完整的 http(s) 地址")
        if not self.MILVUS_USERNAME.strip() or not self.MILVUS_PASSWORD or not self.MILVUS_DATABASE.strip():
            errors.append("Milvus 应用账号和数据库配置不完整")
        if self.MILVUS_USERNAME.strip().casefold() == "root":
            errors.append("知识库禁止使用 Milvus root 账号")
        if not 1 <= self.MILVUS_TIMEOUT_SECONDS <= 60:
            errors.append("MILVUS_TIMEOUT_SECONDS 必须在 1 到 60 秒之间")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", self.MILVUS_COLLECTION_PREFIX):
            errors.append("MILVUS_COLLECTION_PREFIX 必须是长度不超过 40 的受控标识")
        if errors:
            raise ValueError("；".join(errors))

    # Github OAuth
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

    # Google OAuth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    # 前端基础URL，回调路径会自动拼接
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    FRONTEND_AUTH_CALLBACK_PATH: str = "/auth/callback"
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "")

    AUTH_SERVICE_BASE_URL: str = os.getenv("AUTH_SERVICE_BASE_URL", "http://localhost:8100")
    AUTH_SERVICE_CLIENT_ID: Optional[str] = os.getenv("AUTH_SERVICE_CLIENT_ID")
    AUTH_SERVICE_JWKS_URL: Optional[str] = os.getenv("AUTH_SERVICE_JWKS_URL")
    # 分布式认证节点允许极小、受控的墙钟偏差。默认 5 秒足以覆盖 NTP 正常漂移，
    # 上限 60 秒防止配置错误把 exp/nbf/iat 的安全窗口无限扩大。
    AUTH_SERVICE_JWT_LEEWAY_SECONDS: float = Field(default=5.0, ge=0.0, le=60.0)
    # 服务端内网直连地址（可选）。设置后 JWKS 抓取与 userinfo 走内网，绕开公网域名经
    # Cloudflare tunnel 的回环（实测 1-3s → 内网 3-8ms）。issuer/audience 仍用公网
    # AUTH_SERVICE_BASE_URL 校验，不受影响。
    AUTH_SERVICE_INTERNAL_BASE_URL: Optional[str] = os.getenv("AUTH_SERVICE_INTERNAL_BASE_URL")

    @property
    def FRONTEND_AUTH_CALLBACK_URL(self) -> str:
        """动态生成前端OAuth回调URL"""
        return f"{self.FRONTEND_URL.rstrip('/')}{self.FRONTEND_AUTH_CALLBACK_PATH}"

    @property
    def RESOLVED_CORS_ORIGINS(self) -> List[str]:
        configured = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        origins = configured or [self.FRONTEND_URL.rstrip("/")]
        # Keep order stable while de-duplicating.
        return list(dict.fromkeys(origins))

    @property
    def RESOLVED_AUTH_SERVICE_JWKS_URL(self) -> str:
        # 内网 base 优先（绕开公网域名的 CF tunnel 回环）；否则沿用显式 JWKS_URL；
        # 最后回退公网 base 拼接。
        if self.AUTH_SERVICE_INTERNAL_BASE_URL:
            return f"{self.AUTH_SERVICE_INTERNAL_BASE_URL.rstrip('/')}/.well-known/jwks.json"
        return self.AUTH_SERVICE_JWKS_URL or f"{self.AUTH_SERVICE_BASE_URL.rstrip('/')}/.well-known/jwks.json"

    @property
    def AUTH_SERVICE_USERINFO_URL(self) -> str:
        # userinfo 是服务端调用，优先内网直连；issuer 校验仍用公网 BASE_URL（见 security.py）。
        base = (self.AUTH_SERVICE_INTERNAL_BASE_URL or self.AUTH_SERVICE_BASE_URL).rstrip("/")
        return f"{base}/auth/userinfo"

    @property
    def RESOLVED_MCP_ALLOWED_HOSTS(self) -> List[str]:
        return list(dict.fromkeys(host.strip().lower() for host in self.MCP_ALLOWED_HOSTS.split(",") if host.strip()))

    @property
    def RESOLVED_MCP_ALLOWED_CREDENTIAL_REFS(self) -> List[str]:
        return list(dict.fromkeys(ref.strip() for ref in self.MCP_ALLOWED_CREDENTIAL_REFS.split(",") if ref.strip()))

    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_secret_key")
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str = "localhost"
    SERVER_HOST: str = os.getenv("SERVER_HOST", "http://localhost:8000")
    PROJECT_NAME: str = "Fusion API"
    SENTRY_DSN: Optional[str] = None

    # Moonshot (Kimi) — 用于 $web_search 生成动态示例问题
    MOONSHOT_API_KEY: Optional[str] = os.getenv("MOONSHOT_API_KEY")

    # 搜索服务地址（容器间通过 Docker 内网访问）
    SEARCH_SERVICE_URL: str = os.getenv("SEARCH_SERVICE_URL", "http://search-service:8080")

    # 网页读取服务地址
    READER_SERVICE_URL: str = os.getenv("READER_SERVICE_URL", "http://reader-service:8091")
    # reader-service 内部默认等待 Jina 10s；调用方保留更长余量，避免冷抓取或文档站读取提前断开。
    READER_SERVICE_TIMEOUT: float = float(os.getenv("READER_SERVICE_TIMEOUT", "20"))

    # 远程 MCP Client：仅允许精确配置的 HTTPS 主机和凭证环境变量引用。
    MCP_ALLOWED_HOSTS: str = os.getenv(
        "MCP_ALLOWED_HOSTS",
        "learn.microsoft.com,dashscope.aliyuncs.com,mcp.amap.com,mcp.context7.com",
    )
    MCP_ALLOWED_CREDENTIAL_REFS: str = os.getenv(
        "MCP_ALLOWED_CREDENTIAL_REFS",
        "DASHSCOPE_API_KEY,AMAP_MCP_API_KEY,CONTEXT7_API_KEY",
    )
    MCP_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "5"))
    MCP_CALL_TIMEOUT_SECONDS: float = float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "15"))
    MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS: float = float(os.getenv("MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS", "12"))
    MCP_ADMIN_OPERATION_TIMEOUT_SECONDS: int = int(os.getenv("MCP_ADMIN_OPERATION_TIMEOUT_SECONDS", "35"))
    MCP_MAX_DISCOVERY_PAGES: int = int(os.getenv("MCP_MAX_DISCOVERY_PAGES", "5"))
    MCP_MAX_DISCOVERED_TOOLS: int = int(os.getenv("MCP_MAX_DISCOVERED_TOOLS", "50"))
    MCP_MAX_TOOL_DESCRIPTION_CHARS: int = int(os.getenv("MCP_MAX_TOOL_DESCRIPTION_CHARS", "2000"))
    MCP_MAX_TOOL_SCHEMA_BYTES: int = int(os.getenv("MCP_MAX_TOOL_SCHEMA_BYTES", "32768"))
    MCP_MAX_RESPONSE_BYTES: int = int(os.getenv("MCP_MAX_RESPONSE_BYTES", "262144"))
    MCP_MAX_TOOL_CALLS_PER_SERVER_PER_RUN: int = int(os.getenv("MCP_MAX_TOOL_CALLS_PER_SERVER_PER_RUN", "8"))
    MCP_SERVER_CIRCUIT_FAILURE_THRESHOLD: int = int(os.getenv("MCP_SERVER_CIRCUIT_FAILURE_THRESHOLD", "3"))
    MCP_SERVER_CIRCUIT_COOLDOWN_SECONDS: float = float(os.getenv("MCP_SERVER_CIRCUIT_COOLDOWN_SECONDS", "30"))

    # FlyAI 出行产品工具通过私有 HTTP 适配器访问，不在 API 容器执行第三方 CLI。
    ENABLE_FLYAI_TRAVEL_TOOLS: bool = os.getenv("ENABLE_FLYAI_TRAVEL_TOOLS", "false").lower() == "true"
    FLYAI_ADAPTER_BASE_URL: str = os.getenv("FLYAI_ADAPTER_BASE_URL", "")
    FLYAI_ADAPTER_TOKEN: str = os.getenv("FLYAI_ADAPTER_TOKEN", "")
    FLYAI_TRAVEL_TOOL_TIMEOUT_SECONDS: float = float(os.getenv("FLYAI_TRAVEL_TOOL_TIMEOUT_SECONDS", "20"))
    FLYAI_TRAVEL_MAX_TOOL_CALLS_PER_RUN: int = int(os.getenv("FLYAI_TRAVEL_MAX_TOOL_CALLS_PER_RUN", "4"))

    # PromptHub bundle 后台同步；disabled 不发出请求，聊天热路径始终只读本地 LKG。
    PROMPTHUB_SYNC_MODE: str = os.getenv("PROMPTHUB_SYNC_MODE", "disabled").lower()
    PROMPTHUB_BASE_URL: str = os.getenv("PROMPTHUB_BASE_URL", "")
    PROMPTHUB_API_KEY: str = os.getenv("PROMPTHUB_API_KEY", "")
    PROMPTHUB_PROJECT_SLUG: str = os.getenv("PROMPTHUB_PROJECT_SLUG", "fusion")
    PROMPTHUB_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("PROMPTHUB_REQUEST_TIMEOUT_SECONDS", "3"))
    PROMPTHUB_SYNC_INTERVAL_SECONDS: int = int(os.getenv("PROMPTHUB_SYNC_INTERVAL_SECONDS", "300"))
    PROMPTHUB_SYNC_ON_STARTUP: bool = os.getenv("PROMPTHUB_SYNC_ON_STARTUP", "true").lower() == "true"

    # 文档站点开关（生产关掉减少攻击面）
    ENABLE_DOCS: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
