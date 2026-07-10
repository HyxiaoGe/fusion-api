"""Prompt bundle 校验、LKG 读取与旧 Runtime Config 降级。"""

from __future__ import annotations

import copy
import hashlib
import re
import string
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.core.prompt_catalog import PROMPT_SPEC_BY_KEY, PROMPT_SPEC_BY_SLUG, PROMPT_SPECS
from app.core.runtime_config import get_runtime_config_payload
from app.db.database import SessionLocal
from app.db.models import RuntimeConfigEntry

LegacyLoader = Callable[..., tuple[dict[str, Any], dict[str, Any]]]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_CACHE_TTL_SECONDS = 60.0
_BUNDLE_CACHE: tuple[float, dict[str, Any] | None] | None = None


class PromptBundleValidationError(ValueError):
    pass


def clear_prompt_bundle_cache() -> None:
    global _BUNDLE_CACHE
    _BUNDLE_CACHE = None


def validate_published_bundle(bundle: Any) -> dict[str, Any]:
    """将 PromptHub bundle 校验为可持久化的完整 LKG payload。"""

    issues: list[str] = []
    if getattr(bundle, "project_slug", None) != settings.PROMPTHUB_PROJECT_SLUG:
        issues.append("project_slug 不匹配")
    revision = getattr(bundle, "revision", None)
    if not isinstance(revision, str) or _SHA256_PATTERN.fullmatch(revision) is None:
        issues.append("revision 必须是 64 位 SHA-256")

    raw_prompts = tuple(getattr(bundle, "prompts", ()))
    slugs = [getattr(prompt, "slug", None) for prompt in raw_prompts]
    if len(slugs) != len(set(slugs)):
        issues.append("Prompt slug 不能重复")
    expected_slugs = set(PROMPT_SPEC_BY_SLUG)
    actual_slugs = {slug for slug in slugs if isinstance(slug, str)}
    if actual_slugs != expected_slugs or len(raw_prompts) != len(PROMPT_SPECS):
        issues.append("bundle 必须恰好包含 11 个约定 Prompt")

    validated_prompts: dict[str, dict[str, Any]] = {}
    for prompt in raw_prompts:
        slug = getattr(prompt, "slug", None)
        spec = PROMPT_SPEC_BY_SLUG.get(slug)
        if spec is None:
            continue
        prompt_issues, serialized = _validate_prompt_item(prompt, spec)
        issues.extend(prompt_issues)
        if serialized is not None:
            validated_prompts[spec.key] = serialized

    if issues:
        raise PromptBundleValidationError("; ".join(issues))
    return {
        "schema_version": 1,
        "project_slug": bundle.project_slug,
        "revision": revision,
        "prompts": validated_prompts,
    }


def validate_stored_bundle_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != 1 or payload.get("project_slug") != settings.PROMPTHUB_PROJECT_SLUG:
        return False
    revision = payload.get("revision")
    prompts = payload.get("prompts")
    if not isinstance(revision, str) or _SHA256_PATTERN.fullmatch(revision) is None:
        return False
    if not isinstance(prompts, dict) or set(prompts) != set(PROMPT_SPEC_BY_KEY):
        return False
    return all(_stored_prompt_is_valid(key, prompts.get(key)) for key in PROMPT_SPEC_BY_KEY)


def resolve_prompt_template(
    name: str,
    fallback: str,
    *,
    legacy_loader: LegacyLoader = get_runtime_config_payload,
) -> str:
    """按 active bundle -> per-key Runtime Config -> 代码默认值解析 Prompt。"""

    template, _metadata = resolve_prompt_template_with_metadata(
        name,
        fallback,
        legacy_loader=legacy_loader,
    )
    return template


def resolve_prompt_template_with_metadata(
    name: str,
    fallback: str,
    *,
    legacy_loader: LegacyLoader = get_runtime_config_payload,
) -> tuple[str, dict[str, str | None]]:
    """解析 Prompt，并返回可安全写入 LLM 观测字段的版本信息。"""

    if settings.PROMPTHUB_SYNC_MODE == "apply":
        bundle = _load_active_bundle_payload()
        resolved = _template_from_bundle_with_metadata(bundle, name)
        if resolved is not None:
            return resolved

    payload, meta = legacy_loader(
        "prompt_template",
        name,
        {"template": fallback},
    )
    template = payload.get("template")
    effective = template if isinstance(template, str) and template else fallback
    spec = PROMPT_SPEC_BY_KEY.get(name)
    return effective, {
        "source": str(meta.get("source", "code-default")),
        "prompt_slug": spec.slug if spec is not None else name,
        "prompt_version": str(meta.get("version", "code-default")),
        "prompt_revision": None,
    }


def get_active_prompt_bundle_revision() -> str | None:
    """返回当前实际参与热路径解析的 bundle revision。"""

    if settings.PROMPTHUB_SYNC_MODE != "apply":
        return None
    bundle = _load_active_bundle_payload()
    if not isinstance(bundle, dict) or not validate_stored_bundle_payload(bundle):
        return None
    revision = bundle.get("revision")
    return revision if isinstance(revision, str) else None


def get_active_prompt_bundle_payload() -> dict[str, Any] | None:
    """返回当前 apply 热路径使用的 bundle 副本，供只读治理诊断使用。"""

    if settings.PROMPTHUB_SYNC_MODE != "apply":
        return None
    bundle = _load_active_bundle_payload()
    if not isinstance(bundle, dict) or not validate_stored_bundle_payload(bundle):
        return None
    return copy.deepcopy(bundle)


def _validate_prompt_item(prompt: Any, spec: Any) -> tuple[list[str], dict[str, Any] | None]:
    issues: list[str] = []
    prefix = spec.slug
    content = getattr(prompt, "content", None)
    variables = getattr(prompt, "variables", None)
    if getattr(prompt, "status", None) != "published":
        issues.append(f"{prefix}: status 必须为 published")
    if getattr(prompt, "format", None) != "text":
        issues.append(f"{prefix}: format 必须为 text")
    if getattr(prompt, "template_engine", None) != "none":
        issues.append(f"{prefix}: template_engine 必须为 none")
    if not isinstance(content, str) or not content.strip():
        issues.append(f"{prefix}: content 必须是非空字符串")
    elif spec.marker not in content:
        issues.append(f"{prefix}: 缺少固定 marker")
    elif spec.variables and not _format_contract_is_valid(content, spec.variables):
        issues.append(f"{prefix}: content 占位符与 variables 不匹配")
    if (
        not isinstance(variables, tuple)
        or len(variables) != len(set(variables))
        or set(variables) != set(spec.variables)
    ):
        issues.append(f"{prefix}: variables 不匹配")
    version = getattr(prompt, "version", None)
    if not isinstance(version, str) or not version:
        issues.append(f"{prefix}: version 无效")
    if issues:
        return issues, None
    return [], {
        "slug": spec.slug,
        "version": version,
        "content": content,
        "variables": list(spec.variables),
        "content_sha256": _sha256(content),
        "published_at": getattr(prompt, "published_at", None),
    }


def _stored_prompt_is_valid(key: str, prompt: Any) -> bool:
    spec = PROMPT_SPEC_BY_KEY[key]
    if not isinstance(prompt, dict):
        return False
    content = prompt.get("content")
    variables = prompt.get("variables")
    checksum = prompt.get("content_sha256")
    return (
        prompt.get("slug") == spec.slug
        and isinstance(prompt.get("version"), str)
        and bool(prompt.get("version"))
        and isinstance(content, str)
        and bool(content.strip())
        and spec.marker in content
        and (not spec.variables or _format_contract_is_valid(content, spec.variables))
        and isinstance(variables, list)
        and set(variables) == set(spec.variables)
        and isinstance(checksum, str)
        and checksum == _sha256(content)
    )


def _template_from_bundle(bundle: Any, name: str) -> str | None:
    resolved = _template_from_bundle_with_metadata(bundle, name)
    return resolved[0] if resolved is not None else None


def _template_from_bundle_with_metadata(
    bundle: Any,
    name: str,
) -> tuple[str, dict[str, str | None]] | None:
    if not isinstance(bundle, dict):
        return None
    prompts = bundle.get("prompts")
    if not isinstance(prompts, dict):
        return None
    prompt = prompts.get(name)
    if not isinstance(prompt, dict):
        return None
    content = prompt.get("content")
    checksum = prompt.get("content_sha256")
    if not isinstance(content, str) or not content or checksum != _sha256(content):
        return None
    slug = prompt.get("slug")
    version = prompt.get("version")
    revision = bundle.get("revision")
    if not all(isinstance(value, str) and value for value in (slug, version, revision)):
        return None
    return content, {
        "source": "prompthub",
        "prompt_slug": slug,
        "prompt_version": version,
        "prompt_revision": revision,
    }


def _load_active_bundle_payload(
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    global _BUNDLE_CACHE
    now = time.monotonic()
    if use_cache and _BUNDLE_CACHE is not None and now - _BUNDLE_CACHE[0] < _BUNDLE_CACHE_TTL_SECONDS:
        return _BUNDLE_CACHE[1]

    session: Session | None = None
    try:
        session = session_factory()
        rows = (
            session.query(RuntimeConfigEntry)
            .filter(
                RuntimeConfigEntry.namespace == "prompt_bundle",
                RuntimeConfigEntry.key == "fusion",
                RuntimeConfigEntry.is_active.is_(True),
            )
            .order_by(RuntimeConfigEntry.updated_at.desc(), RuntimeConfigEntry.created_at.desc())
            .limit(10)
            .all()
        )
        payload = next((row.payload for row in rows if validate_stored_bundle_payload(row.payload)), None)
    except Exception as exc:
        logger.warning(f"prompt_bundle: 读取 active LKG 失败: {exc}")
        payload = None
    finally:
        if session is not None:
            session.close()
    if use_cache:
        _BUNDLE_CACHE = (now, payload)
    return payload


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_format_fields(content: str) -> set[str] | None:
    try:
        fields = {
            field_name
            for _literal, field_name, _spec, _conversion in string.Formatter().parse(content)
            if field_name is not None
        }
    except ValueError:
        return None
    return fields


def _format_contract_is_valid(content: str, variables: tuple[str, ...]) -> bool:
    if _extract_format_fields(content) != set(variables):
        return False
    try:
        content.format(**{name: "x" for name in variables})
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    return True
