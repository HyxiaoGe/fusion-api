"""从 previous Run 的安全配置恢复 continuation 所需 Skill 发布版本。"""

from __future__ import annotations

from pydantic import ValidationError

from app.ai.skills.registry import SkillReleasePin
from app.db.trajectory_repository import TrajectoryRepository
from app.schemas.trajectory import TrajectoryCapabilityResolution

UNRESTORABLE_SKILL_RELEASE_PINS = (
    SkillReleasePin(
        skill_id="verified-research",
        version="0.0.0",
        content_sha256="0" * 64,
    ),
)


def load_previous_run_skill_release_pins(
    db,
    *,
    conversation_id: str,
    user_id: str,
    previous_run_id: str,
) -> tuple[SkillReleasePin, ...]:
    """只恢复 owner-scoped previous Run 已持久化的 loaded Skill 精确版本。"""

    row = TrajectoryRepository(db).get_run(conversation_id, previous_run_id, user_id)
    if row is None or row.capability_resolution is None:
        return UNRESTORABLE_SKILL_RELEASE_PINS
    try:
        resolution = TrajectoryCapabilityResolution.model_validate(row.capability_resolution)
    except ValidationError:
        return UNRESTORABLE_SKILL_RELEASE_PINS
    skill_resolution = resolution.skill_resolution
    if resolution.schema_version != 2 or skill_resolution is None:
        return UNRESTORABLE_SKILL_RELEASE_PINS
    if skill_resolution.status == "not_selected":
        return ()
    if skill_resolution.status != "loaded":
        return UNRESTORABLE_SKILL_RELEASE_PINS
    return tuple(
        SkillReleasePin(
            skill_id=skill.skill_id,
            version=skill.version,
            content_sha256=skill.content_sha256,
        )
        for skill in skill_resolution.skills
    )
