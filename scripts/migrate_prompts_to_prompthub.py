"""将 Fusion 当前 effective Runtime Prompt 幂等迁入 PromptHub。

默认只做 dry-run。只有显式传入 ``--apply`` 才会创建项目、Prompt 或发布 patch。
管理 key 仅从 ``PROMPTHUB_ADMIN_API_KEY`` 读取，不写入配置、不输出到结果或日志。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.prompt_catalog import PROMPT_SPECS  # noqa: E402
from app.core.runtime_config import get_runtime_config_payload  # noqa: E402
from app.services.runtime_config_defaults import DEFAULT_PROMPT_TEMPLATES  # noqa: E402


@dataclass(frozen=True)
class MigrationPrompt:
    key: str
    slug: str
    name: str
    content: str
    variables: tuple[str, ...]
    content_sha256: str
    source: str
    source_version: str


class PromptHubAdminClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/projects", params={"page": 1, "page_size": 100})

    def create_project(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "/projects",
            json={
                "name": "Fusion",
                "slug": "fusion",
                "description": "Fusion 运行时 Prompt 发布事实源",
            },
        )

    def list_project_prompts(self, project_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/projects/{project_id}/prompts",
            params={"page": 1, "page_size": 100},
        )

    def get_prompt(self, prompt_id: str) -> dict[str, Any]:
        return self._request("GET", f"/prompts/{prompt_id}")

    def get_version(self, prompt_id: str, version: str) -> dict[str, Any]:
        return self._request("GET", f"/prompts/{prompt_id}/versions/{version}")

    def create_prompt(self, project_id: str, prompt: MigrationPrompt) -> dict[str, Any]:
        return self._request("POST", "/prompts", json=_prompt_payload(project_id, prompt))

    def update_prompt(self, prompt_id: str, project_id: str, prompt: MigrationPrompt) -> dict[str, Any]:
        payload = _prompt_payload(project_id, prompt)
        payload.pop("project_id")
        payload.pop("slug")
        return self._request("PUT", f"/prompts/{prompt_id}", json=payload)

    def publish_patch(self, prompt_id: str, prompt: MigrationPrompt) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/prompts/{prompt_id}/publish",
            json={
                "bump": "patch",
                "changelog": "同步 Fusion effective Runtime Prompt",
                "content": prompt.content,
                "variables": _variable_definitions(prompt.variables),
            },
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            envelope = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"PromptHub admin API 请求失败: {method} {path}") from exc
        if not isinstance(envelope, dict) or envelope.get("code") != 0 or "data" not in envelope:
            raise RuntimeError(f"PromptHub admin API 响应无效: {method} {path}")
        return envelope["data"]


def load_effective_prompt_manifest(
    *,
    runtime_loader: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = get_runtime_config_payload,
) -> list[MigrationPrompt]:
    manifest: list[MigrationPrompt] = []
    for spec in PROMPT_SPECS:
        payload, meta = runtime_loader(
            "prompt_template",
            spec.key,
            {"template": DEFAULT_PROMPT_TEMPLATES[spec.key]},
            use_cache=False,
        )
        content = payload.get("template")
        if not isinstance(content, str) or not content:
            raise RuntimeError(f"effective Prompt 为空: {spec.key}")
        manifest.append(
            MigrationPrompt(
                key=spec.key,
                slug=spec.slug,
                name=spec.name,
                content=content,
                variables=spec.variables,
                content_sha256=_sha256(content),
                source=str(meta.get("source", "unknown")),
                source_version=str(meta.get("version", "unknown")),
            )
        )
    return manifest


def migrate_prompts(
    client: PromptHubAdminClient,
    manifest: list[MigrationPrompt],
    *,
    apply: bool,
) -> dict[str, Any]:
    projects = client.list_projects()
    project = next((item for item in projects if item.get("slug") == "fusion"), None)
    result: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "create_project": project is None,
        "prompts": [],
    }
    if project is None and not apply:
        result["prompts"] = [_planned_result(prompt, "create") for prompt in manifest]
        return result
    if project is None:
        project = client.create_project()

    project_id = _required_id(project, "project")
    existing_prompts = client.list_project_prompts(project_id)
    existing_by_slug = {item.get("slug"): item for item in existing_prompts}
    expected_slugs = {prompt.slug for prompt in manifest}
    extra_slugs = sorted(slug for slug in existing_by_slug if isinstance(slug, str) and slug not in expected_slugs)
    if extra_slugs:
        raise RuntimeError(f"fusion 项目存在额外 Prompt slug: {', '.join(extra_slugs)}")
    for prompt in manifest:
        existing = existing_by_slug.get(prompt.slug)
        if existing is None:
            result["prompts"].append(_create_prompt(client, project_id, prompt, apply=apply))
        else:
            result["prompts"].append(_sync_existing_prompt(client, project_id, existing, prompt, apply=apply))
    return result


def _create_prompt(
    client: PromptHubAdminClient,
    project_id: str,
    prompt: MigrationPrompt,
    *,
    apply: bool,
) -> dict[str, Any]:
    if not apply:
        return _planned_result(prompt, "create")
    created = client.create_prompt(project_id, prompt)
    remote_hash = _sha256(_required_content(created))
    return _verified_result(prompt, "create", remote_hash)


def _sync_existing_prompt(
    client: PromptHubAdminClient,
    project_id: str,
    summary: dict[str, Any],
    prompt: MigrationPrompt,
    *,
    apply: bool,
) -> dict[str, Any]:
    prompt_id = _required_id(summary, "prompt")
    detail = client.get_prompt(prompt_id)
    current_version = detail.get("current_version")
    if not isinstance(current_version, str) or not current_version:
        raise RuntimeError(f"Prompt 当前版本无效: {prompt.slug}")
    published = client.get_version(prompt_id, current_version)
    remote_content = _required_content(published)
    remote_hash = _sha256(remote_content)
    remote_variables = _variable_names(published.get("variables"))
    needs_patch = (
        remote_hash != prompt.content_sha256
        or remote_variables != prompt.variables
        or published.get("status") != "published"
        or published.get("format") != "text"
        or published.get("template_engine") != "none"
        or set(detail.get("tags") or []) != {"fusion", "runtime-config"}
        or detail.get("is_shared") is not False
    )
    if not needs_patch:
        return _verified_result(prompt, "noop", remote_hash)
    if not apply:
        return {
            **_planned_result(prompt, "publish_patch"),
            "remote_sha256": remote_hash,
        }

    client.update_prompt(prompt_id, project_id, prompt)
    published_patch = client.publish_patch(prompt_id, prompt)
    verified_hash = _sha256(_required_content(published_patch))
    return _verified_result(prompt, "publish_patch", verified_hash)


def _prompt_payload(project_id: str, prompt: MigrationPrompt) -> dict[str, Any]:
    return {
        "name": prompt.name,
        "slug": prompt.slug,
        "description": f"Fusion Runtime Prompt: {prompt.key}",
        "content": prompt.content,
        "format": "text",
        "template_engine": "none",
        "variables": _variable_definitions(prompt.variables),
        "tags": ["fusion", "runtime-config"],
        "project_id": project_id,
        "is_shared": False,
    }


def _variable_definitions(variables: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "string",
            "required": True,
            "default": None,
            "description": None,
            "enum_values": None,
        }
        for name in variables
    ]


def _variable_names(raw_variables: Any) -> tuple[str, ...]:
    if not isinstance(raw_variables, list):
        raise RuntimeError("Prompt variables 响应无效")
    names: list[str] = []
    for item in raw_variables:
        name = item if isinstance(item, str) else item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name:
            raise RuntimeError("Prompt variable name 响应无效")
        names.append(name)
    return tuple(names)


def _planned_result(prompt: MigrationPrompt, action: str) -> dict[str, Any]:
    return {
        "key": prompt.key,
        "slug": prompt.slug,
        "action": action,
        "status": "planned",
        "local_sha256": prompt.content_sha256,
    }


def _verified_result(prompt: MigrationPrompt, action: str, remote_hash: str) -> dict[str, Any]:
    if remote_hash != prompt.content_sha256:
        raise RuntimeError(f"Prompt hash 核对失败: {prompt.slug}")
    return {
        "key": prompt.key,
        "slug": prompt.slug,
        "action": action,
        "status": "verified",
        "local_sha256": prompt.content_sha256,
        "remote_sha256": remote_hash,
    }


def _required_id(payload: Any, kind: str) -> str:
    value = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"PromptHub {kind} id 响应无效")
    return value


def _required_content(payload: Any) -> str:
    value = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(value, str):
        raise RuntimeError("PromptHub content 响应无效")
    return value


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移 Fusion Runtime Prompt 到 PromptHub")
    parser.add_argument("--apply", action="store_true", help="执行写入；默认仅 dry-run")
    args = parser.parse_args()
    base_url = os.getenv("PROMPTHUB_ADMIN_BASE_URL") or os.getenv("PROMPTHUB_BASE_URL", "")
    api_key = os.getenv("PROMPTHUB_ADMIN_API_KEY", "")
    if not base_url or not api_key:
        parser.error("需要 PROMPTHUB_ADMIN_BASE_URL（或 PROMPTHUB_BASE_URL）和 PROMPTHUB_ADMIN_API_KEY")

    client = PromptHubAdminClient(base_url=base_url, api_key=api_key)
    try:
        result = migrate_prompts(client, load_effective_prompt_manifest(), apply=args.apply)
    finally:
        client.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
