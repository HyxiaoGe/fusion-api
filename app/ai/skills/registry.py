"""从代码固定目录加载并冻结 Run 级 Skill 快照。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

from app.utils.run_capability_contract import (
    CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER,
    CAPABILITY_CONTROL_TOOL_NAMES,
)

SkillLoadStatus = Literal["not_selected", "loaded", "load_failed"]
SkillActivationSource = Literal["capability_package"]
SkillLoadErrorCode = Literal["skill_load_failed"]

MAX_SKILL_FILE_BYTES = 32 * 1024
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
_FRONTMATTER_FIELDS = frozenset({"name", "version", "description", "allowed-tools"})
_KNOWN_TOOL_NAMES = frozenset(CAPABILITY_CANONICAL_EXTERNAL_TOOL_ORDER)
_SKILL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
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


@dataclass(frozen=True, slots=True)
class _ParsedFrontmatter:
    name: str
    version: str
    description: str
    allowed_tool_names: tuple[str, ...]
    body: str


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
    skill_path = root / skill_id / version / "SKILL.md"
    resolved_path = skill_path.resolve(strict=True)
    if not resolved_path.is_relative_to(root):
        raise ValueError("Skill 路径越界")
    if any(path.is_symlink() for path in (skill_path, skill_path.parent, skill_path.parent.parent)):
        raise ValueError("Skill 路径不得使用符号链接")

    raw = resolved_path.read_bytes()
    if len(raw) > MAX_SKILL_FILE_BYTES:
        raise ValueError("Skill 文件超过大小上限")
    document = raw.decode("utf-8")
    parsed = _parse_skill_document(document)
    _validate_skill_metadata(
        parsed,
        expected_skill_id=skill_id,
        expected_version=version,
        routed_tool_names=routed_tool_names,
    )
    content_sha256 = hashlib.sha256(parsed.body.encode("utf-8")).hexdigest()
    if content_sha256 != expected_content_sha256:
        raise ValueError("Skill 正文摘要与发布版本不一致")
    metadata = SkillMetadata(
        skill_id=parsed.name,
        version=parsed.version,
        description=parsed.description,
        content_sha256=content_sha256,
        allowed_tool_names=parsed.allowed_tool_names,
        activation_source="capability_package",
        section_id=f"skill:{parsed.name}@{parsed.version}",
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


def _parse_skill_document(document: str) -> _ParsedFrontmatter:
    lines = document.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("Skill 缺少 frontmatter")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("Skill frontmatter 未闭合")

    metadata_lines = [line.rstrip("\r\n") for line in lines[1:closing_index]]
    body = "".join(lines[closing_index + 1 :])
    if not body.strip():
        raise ValueError("Skill 正文不能为空")

    scalars: dict[str, str] = {}
    allowed_tools: list[str] | None = None
    active_list: str | None = None
    for line in metadata_lines:
        if not line.strip():
            continue
        if line.startswith("  - "):
            if active_list != "allowed-tools" or allowed_tools is None:
                raise ValueError("Skill frontmatter 列表结构非法")
            tool_name = line[4:].strip()
            if not tool_name:
                raise ValueError("Skill 工具名不能为空")
            allowed_tools.append(tool_name)
            continue
        if line[:1].isspace() or ":" not in line:
            raise ValueError("Skill frontmatter 结构非法")
        field, value = line.split(":", 1)
        field = field.strip()
        value = value.strip()
        if (
            field not in _FRONTMATTER_FIELDS
            or field in scalars
            or (field == "allowed-tools" and allowed_tools is not None)
        ):
            raise ValueError("Skill frontmatter 字段非法")
        active_list = None
        if field == "allowed-tools":
            if value:
                raise ValueError("allowed-tools 必须使用受控列表格式")
            allowed_tools = []
            active_list = field
        else:
            if not value:
                raise ValueError("Skill frontmatter 标量不能为空")
            scalars[field] = value

    required_scalars = {"name", "version", "description"}
    if set(scalars) != required_scalars or allowed_tools is None:
        raise ValueError("Skill frontmatter 缺少必填字段")
    return _ParsedFrontmatter(
        name=scalars["name"],
        version=scalars["version"],
        description=scalars["description"],
        allowed_tool_names=tuple(allowed_tools),
        body=body,
    )


def _validate_skill_metadata(
    parsed: _ParsedFrontmatter,
    *,
    expected_skill_id: str,
    expected_version: str,
    routed_tool_names: tuple[str, ...],
) -> None:
    if _SKILL_ID_RE.fullmatch(parsed.name) is None or parsed.name != expected_skill_id:
        raise ValueError("Skill name 与受控目录不一致")
    if _VERSION_RE.fullmatch(parsed.version) is None or parsed.version != expected_version:
        raise ValueError("Skill version 与版本目录不一致")
    if len(parsed.description) > 1024:
        raise ValueError("Skill description 超过大小上限")
    if not parsed.allowed_tool_names or len(parsed.allowed_tool_names) != len(set(parsed.allowed_tool_names)):
        raise ValueError("Skill allowed-tools 不能为空或重复")
    if CAPABILITY_CONTROL_TOOL_NAMES.intersection(parsed.allowed_tool_names):
        raise ValueError("Skill 不得声明控制工具")
    if not set(parsed.allowed_tool_names).issubset(_KNOWN_TOOL_NAMES):
        raise ValueError("Skill 声明了未知工具")
    if parsed.allowed_tool_names != routed_tool_names:
        raise ValueError("Skill allowed-tools 与能力路由工具不一致")


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
