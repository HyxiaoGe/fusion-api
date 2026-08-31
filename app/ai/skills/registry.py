"""将本地 Agent Skills 标准目录适配为 Run 级冻结快照。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

from app.ai.skills.document import MAX_SKILL_FILE_BYTES, SkillDocument, parse_skill_document
from app.utils.run_capability_contract import (
    CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER,
    CAPABILITY_CONTROL_TOOL_NAMES,
)

SkillLoadStatus = Literal["not_selected", "loaded", "load_failed"]
SkillActivationSource = Literal["capability_package"]
SkillLoadErrorCode = Literal["skill_load_failed"]

_SKILLS_ROOT = Path(__file__).resolve().parent
_PACKAGE_SKILLS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "verified_web": (
        (
            "verified-research",
            "1.0.0",
            "5c93abf51e64321ad42968ab8d01d3a9429bcd4ea90cb514b6fd0822c8842cdb",
        ),
    ),
}
_KNOWN_TOOL_NAMES = frozenset(CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER)
_SKILL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SkillReleasePin:
    skill_id: str
    version: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    skill_id: str
    version: str
    description: str
    content_sha256: str
    allowed_tool_names: tuple[str, ...]
    activation_source: SkillActivationSource
    section_id: str
    char_count: int


@dataclass(frozen=True, slots=True)
class LoadedSkillSnapshot:
    metadata: SkillMetadata
    content: str


@dataclass(frozen=True, slots=True)
class RunSkillResolution:
    status: SkillLoadStatus
    activation_source: SkillActivationSource
    requested_skill_ids: tuple[str, ...]
    skills: tuple[SkillMetadata, ...]
    duration_ms: int
    error_code: SkillLoadErrorCode | None = None


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    resolution: RunSkillResolution
    loaded_skills: tuple[LoadedSkillSnapshot, ...]


def load_skills_for_package(
    package_id: str,
    routed_tool_names: Sequence[str],
    *,
    skills_root: Path | None = None,
    release_pins: Sequence[SkillReleasePin] | None = None,
) -> SkillLoadResult:
    """加载代码固定 Skill；已选择 Skill 的任何异常都返回受控失败。"""

    started = perf_counter()
    configured_releases = _PACKAGE_SKILLS.get(package_id, ())
    requested_skill_ids = tuple(skill_id for skill_id, _version, _content_sha256 in configured_releases)
    if not configured_releases:
        return SkillLoadResult(
            resolution=_resolution(
                status="not_selected",
                requested_skill_ids=(),
                started=started,
            ),
            loaded_skills=(),
        )

    try:
        selected_releases = _resolve_release_pins(configured_releases, release_pins)
        routed_tools = _validate_routed_tool_names(routed_tool_names)
        root = (skills_root or _SKILLS_ROOT).resolve()
        loaded_skills = tuple(
            _load_skill(
                root=root,
                skill_id=skill_id,
                version=version,
                expected_content_sha256=content_sha256,
                routed_tool_names=routed_tools,
            )
            for skill_id, version, content_sha256 in selected_releases
        )
    except (OSError, UnicodeError, ValueError):
        return SkillLoadResult(
            resolution=_resolution(
                status="load_failed",
                requested_skill_ids=requested_skill_ids,
                started=started,
                error_code="skill_load_failed",
            ),
            loaded_skills=(),
        )

    return SkillLoadResult(
        resolution=_resolution(
            status="loaded",
            requested_skill_ids=requested_skill_ids,
            skills=tuple(snapshot.metadata for snapshot in loaded_skills),
            started=started,
        ),
        loaded_skills=loaded_skills,
    )


def _load_skill(
    *,
    root: Path,
    skill_id: str,
    version: str,
    expected_content_sha256: str,
    routed_tool_names: tuple[str, ...],
) -> LoadedSkillSnapshot:
    parsed = _read_selected_skill(root=root, skill_id=skill_id, version=version)
    allowed_tool_names = _validate_skill_metadata(parsed, routed_tool_names=routed_tool_names)
    content_sha256 = hashlib.sha256(parsed.body.encode("utf-8")).hexdigest()
    if content_sha256 != expected_content_sha256:
        raise ValueError("Skill 正文摘要与发布版本不一致")
    metadata = SkillMetadata(
        skill_id=parsed.name,
        version=parsed.resolved_version,
        description=parsed.description,
        content_sha256=content_sha256,
        allowed_tool_names=allowed_tool_names,
        activation_source="capability_package",
        section_id=f"skill:{parsed.name}@{parsed.resolved_version}",
        char_count=len(parsed.body),
    )
    return LoadedSkillSnapshot(metadata=metadata, content=parsed.body)


def _resolve_release_pins(
    configured_releases: tuple[tuple[str, str, str], ...],
    release_pins: Sequence[SkillReleasePin] | None,
) -> tuple[tuple[str, str, str], ...]:
    if release_pins is None:
        return configured_releases
    pins = tuple(release_pins)
    configured_skill_ids = tuple(skill_id for skill_id, _version, _content_sha256 in configured_releases)
    if tuple(pin.skill_id for pin in pins) != configured_skill_ids:
        raise ValueError("延续 Run 的 Skill 集合与能力包不一致")
    releases = []
    for pin in pins:
        if _SKILL_ID_RE.fullmatch(pin.skill_id) is None:
            raise ValueError("延续 Run 的 Skill ID 非法")
        if _VERSION_RE.fullmatch(pin.version) is None:
            raise ValueError("延续 Run 的 Skill 版本非法")
        if _SHA256_RE.fullmatch(pin.content_sha256) is None:
            raise ValueError("延续 Run 的 Skill 摘要非法")
        releases.append((pin.skill_id, pin.version, pin.content_sha256))
    return tuple(releases)


def _read_selected_skill(*, root: Path, skill_id: str, version: str) -> SkillDocument:
    candidates = (
        root / skill_id / "SKILL.md",
        root / skill_id / version / "SKILL.md",
    )
    found_version_mismatch = False
    for skill_path in candidates:
        if not skill_path.exists():
            continue
        resolved_path = skill_path.resolve(strict=True)
        if not resolved_path.is_relative_to(root):
            raise ValueError("Skill 路径越界")
        current = skill_path
        while current != root:
            if current.is_symlink():
                raise ValueError("Skill 路径不得使用符号链接")
            current = current.parent

        raw = resolved_path.read_bytes()
        if len(raw) > MAX_SKILL_FILE_BYTES:
            raise ValueError("Skill 文件超过大小上限")
        document = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        parsed = parse_skill_document(document, expected_skill_id=skill_id)
        if parsed.resolved_version != version:
            found_version_mismatch = True
            continue
        return parsed
    if found_version_mismatch:
        raise ValueError("Skill version 与发布版本不一致")
    raise FileNotFoundError("Skill 文件不存在")


def _validate_skill_metadata(
    parsed: SkillDocument,
    *,
    routed_tool_names: tuple[str, ...],
) -> tuple[str, ...]:
    declared_tools = parsed.declared_allowed_tools
    if not declared_tools:
        return routed_tool_names
    if CAPABILITY_CONTROL_TOOL_NAMES.intersection(declared_tools):
        raise ValueError("Skill 不得声明控制工具")
    if not set(declared_tools).issubset(_KNOWN_TOOL_NAMES):
        raise ValueError("Skill 声明了未知工具")
    if declared_tools != routed_tool_names:
        raise ValueError("Skill allowed-tools 与能力路由工具不一致")
    return routed_tool_names


def _validate_routed_tool_names(routed_tool_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(routed_tool_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("能力路由工具非法")
    if len(names) != len(set(names)):
        raise ValueError("能力路由工具不得重复")
    return names


def _resolution(
    *,
    status: SkillLoadStatus,
    requested_skill_ids: tuple[str, ...],
    started: float,
    skills: tuple[SkillMetadata, ...] = (),
    error_code: SkillLoadErrorCode | None = None,
) -> RunSkillResolution:
    return RunSkillResolution(
        status=status,
        activation_source="capability_package",
        requested_skill_ids=requested_skill_ids,
        skills=skills,
        duration_ms=(0 if status == "not_selected" else max(0, int((perf_counter() - started) * 1000))),
        error_code=error_code,
    )
