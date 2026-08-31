from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.ai.skills.registry import SkillReleasePin, load_skills_for_package

VALID_BODY = """# 可核验研究

先搜索候选来源，再读取原文并交叉核验关键事实。
"""


def _skill_document(
    *,
    name: str = "verified-research",
    version: str = "1.0.0",
    description: str = "对需要官方原文与交叉来源的请求建立可核验证据链",
    allowed_tools: tuple[str, ...] | None = ("web_search", "url_read"),
    body: str = VALID_BODY,
    extra_frontmatter: str = "",
) -> bytes:
    lines = [
        "---",
        f"name: {name}",
        f"version: {version}",
        f"description: {description}",
    ]
    if allowed_tools is not None:
        lines.append("allowed-tools:")
        lines.extend(f"  - {tool_name}" for tool_name in allowed_tools)
    if extra_frontmatter:
        lines.append(extra_frontmatter)
    lines.extend(["---", body])
    return "\n".join(lines).encode("utf-8")


def _write_skill(
    root: Path,
    *,
    version_directory: str = "1.0.0",
    payload: bytes | None = None,
) -> Path:
    skill_path = root / "verified-research" / version_directory / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_bytes(payload if payload is not None else _skill_document())
    return skill_path


def test_bundled_verified_research_skill_loads_as_frozen_snapshot() -> None:
    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
    )

    assert result.resolution.status == "loaded"
    assert result.resolution.activation_source == "capability_package"
    assert result.resolution.requested_skill_ids == ("verified-research",)
    assert result.resolution.error_code is None
    assert result.resolution.duration_ms >= 0
    assert len(result.resolution.skills) == 1
    assert len(result.loaded_skills) == 1

    loaded = result.loaded_skills[0]
    assert loaded.metadata == result.resolution.skills[0]
    assert loaded.metadata.skill_id == "verified-research"
    assert loaded.metadata.version == "1.0.0"
    assert loaded.metadata.description == "对需要官方原文与交叉来源的请求建立可核验证据链"
    assert loaded.metadata.allowed_tool_names == ("web_search", "url_read")
    assert loaded.metadata.activation_source == "capability_package"
    assert loaded.metadata.section_id == "skill:verified-research@1.0.0"
    assert loaded.metadata.char_count == len(loaded.content)
    assert loaded.metadata.content_sha256 == hashlib.sha256(loaded.content.encode("utf-8")).hexdigest()
    assert "官方" in loaded.content
    assert "交叉" in loaded.content

    serialized_resolution = asdict(result.resolution)
    assert "content" not in serialized_resolution
    assert all("content" not in skill for skill in serialized_resolution["skills"])
    serialized = json.dumps(serialized_resolution, ensure_ascii=False)
    assert loaded.content not in serialized


@pytest.mark.parametrize("line_ending", (b"\r\n", b"\r"))
def test_platform_line_endings_keep_published_body_digest_stable(
    tmp_path: Path,
    line_ending: bytes,
) -> None:
    bundled_path = Path(__file__).parents[3] / "app/ai/skills/verified-research/1.0.0/SKILL.md"
    bundled = bundled_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    _write_skill(tmp_path, payload=bundled.replace(b"\n", line_ending))

    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
    )

    assert result.resolution.status == "loaded"
    assert result.resolution.skills[0].content_sha256 == (
        "5c93abf51e64321ad42968ab8d01d3a9429bcd4ea90cb514b6fd0822c8842cdb"
    )
    assert "\r" not in result.loaded_skills[0].content


def test_published_version_rejects_body_changed_without_version_bump(tmp_path: Path) -> None:
    _write_skill(tmp_path, payload=_skill_document(body="# 被原地修改\n\n这不应继续冒充 1.0.0。\n"))

    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
    )

    assert result.resolution.status == "load_failed"
    assert result.resolution.error_code == "skill_load_failed"
    assert result.loaded_skills == ()


def test_continuation_release_pin_restores_exact_version_and_digest(tmp_path: Path) -> None:
    body = "# 旧版可核验研究\n\n继续同一业务尝试时必须恢复这份正文。\n"
    _write_skill(
        tmp_path,
        version_directory="0.9.0",
        payload=_skill_document(version="0.9.0", body=body),
    )
    pin = SkillReleasePin(
        skill_id="verified-research",
        version="0.9.0",
        content_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )

    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
        release_pins=(pin,),
    )

    assert result.resolution.status == "loaded"
    assert result.resolution.skills[0].version == "0.9.0"
    assert result.resolution.skills[0].content_sha256 == pin.content_sha256


def test_continuation_release_pin_fails_closed_when_old_version_is_missing(tmp_path: Path) -> None:
    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
        release_pins=(
            SkillReleasePin(
                skill_id="verified-research",
                version="0.9.0",
                content_sha256="a" * 64,
            ),
        ),
    )

    assert result.resolution.status == "load_failed"
    assert result.resolution.error_code == "skill_load_failed"
    assert result.loaded_skills == ()


@pytest.mark.parametrize("package_id", ["direct", "fresh_web", "url_read", "deep_research", "unknown"])
def test_unmapped_package_does_not_touch_skill_files(package_id: str, tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"

    result = load_skills_for_package(
        package_id,
        (),
        skills_root=missing_root,
    )

    assert result.resolution.status == "not_selected"
    assert result.resolution.activation_source == "capability_package"
    assert result.resolution.requested_skill_ids == ()
    assert result.resolution.skills == ()
    assert result.resolution.error_code is None
    assert result.loaded_skills == ()
    assert not missing_root.exists()


@pytest.mark.parametrize(
    ("case_name", "payload", "version_directory", "routed_tool_names"),
    [
        ("invalid_utf8", b"---\nname: verified-research\n---\n\xff", "1.0.0", ("web_search", "url_read")),
        ("missing_frontmatter", b"# no frontmatter", "1.0.0", ("web_search", "url_read")),
        (
            "unknown_frontmatter_field",
            _skill_document(extra_frontmatter="license: proprietary"),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "version_directory_mismatch",
            _skill_document(version="1.0.1"),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "invalid_version",
            _skill_document(version="v1"),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "missing_allowed_tools",
            _skill_document(allowed_tools=None),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "empty_allowed_tools",
            _skill_document(allowed_tools=()),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "duplicate_allowed_tool",
            _skill_document(allowed_tools=("web_search", "web_search")),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "control_tool",
            _skill_document(allowed_tools=("web_search", "url_read", "update_plan")),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "unknown_tool",
            _skill_document(allowed_tools=("web_search", "url_read", "shell_exec")),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "expands_routed_tools",
            _skill_document(),
            "1.0.0",
            ("web_search",),
        ),
        (
            "missing_required_tool",
            _skill_document(allowed_tools=("web_search",)),
            "1.0.0",
            ("web_search", "url_read"),
        ),
        (
            "oversized_file",
            _skill_document(body="x" * (32 * 1024)),
            "1.0.0",
            ("web_search", "url_read"),
        ),
    ],
)
def test_invalid_selected_skill_fails_closed(
    case_name: str,
    payload: bytes,
    version_directory: str,
    routed_tool_names: tuple[str, ...],
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, version_directory=version_directory, payload=payload)

    result = load_skills_for_package(
        "verified_web",
        routed_tool_names,
        skills_root=tmp_path,
    )

    assert result.resolution.status == "load_failed", case_name
    assert result.resolution.activation_source == "capability_package"
    assert result.resolution.requested_skill_ids == ("verified-research",)
    assert result.resolution.skills == ()
    assert result.resolution.error_code == "skill_load_failed"
    assert result.resolution.duration_ms >= 0
    assert result.loaded_skills == ()


def test_missing_selected_skill_fails_closed(tmp_path: Path) -> None:
    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
    )

    assert result.resolution.status == "load_failed"
    assert result.resolution.requested_skill_ids == ("verified-research",)
    assert result.resolution.error_code == "skill_load_failed"
    assert result.loaded_skills == ()


def test_selected_skill_symlink_cannot_escape_registry_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-skill.md"
    outside.write_bytes(_skill_document())
    skill_path = tmp_path / "verified-research" / "1.0.0" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.symlink_to(outside)

    result = load_skills_for_package(
        "verified_web",
        ("web_search", "url_read"),
        skills_root=tmp_path,
    )

    assert result.resolution.status == "load_failed"
    assert result.resolution.error_code == "skill_load_failed"
    assert result.loaded_skills == ()
