"""同源头像代理。

浏览器直连 ``lh3.googleusercontent.com`` / ``avatars.githubusercontent.com`` 等第三方头像源，
在国内常常很慢甚至被墙——登录数据早已返回，卡住的只是那张 ``<img>``。这里由后端按**严格
白名单**抓取头像图片并内存缓存，前端改走同源 URL 加载，把"国内访问 Google 图床"的慢/失败
从每个浏览器收敛到一次服务端抓取 + 浏览器强缓存。

安全：仅允许 https + 固定白名单 host，杜绝 SSRF（不得用本端点去探测内网/云元数据地址）；
不跟随重定向（避免被 30x 绕过白名单）；限制响应体大小与 image/* 类型。
"""

import time
from urllib.parse import urlparse

import httpx

# 仅代理这些已知头像源（OAuth provider：google / github）。新增 provider 时在此扩白名单。
ALLOWED_HOSTS = frozenset({"lh3.googleusercontent.com", "avatars.githubusercontent.com"})
CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 512
_FETCH_TIMEOUT = 10.0

# url -> (body, content_type, fetched_at)
_cache: dict[str, tuple[bytes, str, float]] = {}


class AvatarProxyError(Exception):
    """携带 (status_code, detail)，由路由层翻译成 HTTP 响应。"""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def is_allowed_avatar_url(url: str) -> bool:
    """仅 https + 白名单 host 通过；其余（含内网/元数据地址、非法 URL）一律拒绝。"""
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def _evict_if_needed() -> None:
    if len(_cache) <= MAX_ENTRIES:
        return
    # 超额时按写入时间 FIFO 淘汰最早的一批。
    overflow = len(_cache) - MAX_ENTRIES
    for key, _ in sorted(_cache.items(), key=lambda kv: kv[1][2])[:overflow]:
        _cache.pop(key, None)


def fetch_avatar(url: str, *, now: float | None = None) -> tuple[bytes, str]:
    """返回 ``(image_bytes, content_type)``；非法或上游失败抛 ``AvatarProxyError``。"""
    moment = time.time() if now is None else now
    if not is_allowed_avatar_url(url):
        raise AvatarProxyError(400, "Unsupported avatar host")

    cached = _cache.get(url)
    if cached is not None and moment - cached[2] < CACHE_TTL_SECONDS:
        return cached[0], cached[1]

    try:
        response = httpx.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=False)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AvatarProxyError(502, "Avatar fetch failed") from exc

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise AvatarProxyError(502, "Upstream is not an image")

    body = response.content
    if len(body) > MAX_BYTES:
        raise AvatarProxyError(502, "Avatar too large")

    _cache[url] = (body, content_type, moment)
    _evict_if_needed()
    return body, content_type
