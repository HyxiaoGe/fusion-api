"""PromptHub 已发布 Prompt bundle 的轻量只读客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class PromptHubBundleItem:
    id: str
    slug: str
    name: str
    version: str
    status: str
    content: str
    variables: tuple[str, ...]
    format: str
    template_engine: str
    published_at: str | None


@dataclass(frozen=True)
class PromptHubBundle:
    project_id: str
    project_slug: str
    revision: str
    prompts: tuple[PromptHubBundleItem, ...]


class PromptHubClientError(RuntimeError):
    """屏蔽底层异常和凭证的安全客户端错误。"""

    def __init__(self, kind: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class PromptHubPublishedBundleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        project_slug: str,
        timeout_seconds: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._project_slug = project_slug
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def fetch_published_bundle(self) -> PromptHubBundle:
        url = f"{self._base_url}/api/v1/projects/by-slug/{self._project_slug}/prompts/published"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise PromptHubClientError("timeout", "PromptHub 请求超时") from exc
        except httpx.HTTPStatusError as exc:
            raise PromptHubClientError(
                "http",
                f"PromptHub 返回 HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise PromptHubClientError("request", "PromptHub 请求失败") from exc

        try:
            envelope = response.json()
            return _parse_bundle_envelope(envelope)
        except (TypeError, ValueError, KeyError) as exc:
            raise PromptHubClientError("invalid_response", "PromptHub 返回无效 bundle") from exc


def _parse_bundle_envelope(envelope: Any) -> PromptHubBundle:
    if not isinstance(envelope, dict) or envelope.get("code") != 0:
        raise ValueError("响应 envelope 无效")
    data = envelope.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
        raise ValueError("bundle data 无效")

    prompts = tuple(_parse_bundle_item(item) for item in data["prompts"])
    return PromptHubBundle(
        project_id=_required_string(data, "project_id"),
        project_slug=_required_string(data, "project_slug"),
        revision=_required_string(data, "revision"),
        prompts=prompts,
    )


def _parse_bundle_item(item: Any) -> PromptHubBundleItem:
    if not isinstance(item, dict):
        raise ValueError("prompt item 无效")
    raw_variables = item.get("variables")
    if not isinstance(raw_variables, list):
        raise ValueError("prompt variables 无效")
    variables = tuple(_parse_variable_name(variable) for variable in raw_variables)
    published_at = item.get("published_at")
    if published_at is not None and not isinstance(published_at, str):
        raise ValueError("published_at 无效")
    return PromptHubBundleItem(
        id=_required_string(item, "id"),
        slug=_required_string(item, "slug"),
        name=_required_string(item, "name"),
        version=_required_string(item, "version"),
        status=_required_string(item, "status"),
        content=_required_string(item, "content", allow_blank=True),
        variables=variables,
        format=_required_string(item, "format"),
        template_engine=_required_string(item, "template_engine"),
        published_at=published_at,
    )


def _parse_variable_name(variable: Any) -> str:
    if isinstance(variable, str) and variable:
        return variable
    if isinstance(variable, dict):
        return _required_string(variable, "name")
    raise ValueError("variable 无效")


def _required_string(payload: dict[str, Any], key: str, *, allow_blank: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_blank and not value):
        raise ValueError(f"{key} 无效")
    return value
