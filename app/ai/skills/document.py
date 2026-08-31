"""解析 Agent Skills 标准的 ``SKILL.md`` 文档。"""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

MAX_SKILL_FILE_BYTES = 32 * 1024

_SKILL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RUNTIME_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_STANDARD_FIELDS = frozenset(
    {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
    }
)
_LEGACY_FIELDS = frozenset({"version"})


class _StrictSafeLoader(yaml.SafeLoader):
    """禁用别名并拒绝重复键的安全 YAML loader。"""

    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            raise ValueError("Skill frontmatter 不允许 YAML 别名")
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader: _StrictSafeLoader, node: MappingNode, deep: bool = False) -> dict:
    if not isinstance(node, MappingNode):
        raise ValueError("Skill frontmatter 映射结构非法")
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("Skill frontmatter 字段名非法") from exc
        if duplicate:
            raise ValueError("Skill frontmatter 包含重复字段")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """与具体 Agent Runtime 解耦的 SKILL.md 解析结果。"""

    name: str
    description: str
    body: str
    license: str | None
    compatibility: str | None
    metadata: tuple[tuple[str, str], ...]
    declared_version: str | None
    declared_allowed_tools: tuple[str, ...]
    extension_field_names: tuple[str, ...]
    document_sha256: str

    @property
    def resolved_version(self) -> str:
        """返回 Run 账本可冻结的版本；无声明版本时使用整份文档摘要。"""

        if self.declared_version and _RUNTIME_VERSION_RE.fullmatch(self.declared_version):
            return self.declared_version
        return f"sha256-{self.document_sha256[:16]}"


def parse_skill_document(document: str, *, expected_skill_id: str | None = None) -> SkillDocument:
    """按 Agent Skills 标准解析文档，同时兼容既有 Fusion/DeerFlow 字段形态。"""

    if not isinstance(document, str):
        raise ValueError("Skill 文档必须是字符串")
    normalized = document.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized.encode("utf-8")) > MAX_SKILL_FILE_BYTES:
        raise ValueError("Skill 文件超过大小上限")

    frontmatter_text, body = _split_document(normalized)
    try:
        frontmatter = yaml.load(frontmatter_text, Loader=_StrictSafeLoader)
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError("Skill frontmatter YAML 非法") from exc
    if not isinstance(frontmatter, dict) or any(not isinstance(key, str) for key in frontmatter):
        raise ValueError("Skill frontmatter 必须是字符串键映射")

    name = _required_string(frontmatter, "name")
    description = _required_string(frontmatter, "description")
    if len(name) > 64 or _SKILL_ID_RE.fullmatch(name) is None:
        raise ValueError("Skill name 不符合 Agent Skills 标准")
    if expected_skill_id is not None and name != expected_skill_id:
        raise ValueError("Skill name 与目录身份不一致")
    if len(description) > 1024:
        raise ValueError("Skill description 超过大小上限")

    license_name = _optional_string(frontmatter, "license")
    compatibility = _optional_string(frontmatter, "compatibility")
    if compatibility is not None and len(compatibility) > 500:
        raise ValueError("Skill compatibility 超过大小上限")

    metadata = _parse_metadata(frontmatter.get("metadata"))
    metadata_map = dict(metadata)
    legacy_version = _optional_string(frontmatter, "version")
    metadata_version = metadata_map.get("version")
    if legacy_version is not None and metadata_version is not None and legacy_version != metadata_version:
        raise ValueError("Skill 版本声明冲突")
    declared_version = metadata_version or legacy_version
    declared_allowed_tools = _parse_allowed_tools(frontmatter.get("allowed-tools"))
    extension_field_names = tuple(
        key for key in frontmatter if key not in _STANDARD_FIELDS and key not in _LEGACY_FIELDS
    )

    return SkillDocument(
        name=name,
        description=description,
        body=body,
        license=license_name,
        compatibility=compatibility,
        metadata=metadata,
        declared_version=declared_version,
        declared_allowed_tools=declared_allowed_tools,
        extension_field_names=extension_field_names,
        document_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def _split_document(document: str) -> tuple[str, str]:
    lines = document.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != "---":
        raise ValueError("Skill 缺少 frontmatter")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\n") == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("Skill frontmatter 未闭合")
    body = "".join(lines[closing_index + 1 :])
    if not body.strip():
        raise ValueError("Skill 正文不能为空")
    return "".join(lines[1:closing_index]), body


def _required_string(frontmatter: dict, field: str) -> str:
    value = frontmatter.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Skill {field} 必须是非空字符串")
    return value.strip()


def _optional_string(frontmatter: dict, field: str) -> str | None:
    if field not in frontmatter:
        return None
    value = frontmatter[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Skill {field} 必须是非空字符串")
    return value.strip()


def _parse_metadata(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or any(not isinstance(key, str) or not key for key in value):
        raise ValueError("Skill metadata 必须是字符串键映射")
    normalized: list[tuple[str, str]] = []
    for key, item in value.items():
        if isinstance(item, str):
            normalized.append((key, item))
        elif isinstance(item, bool):
            normalized.append((key, "true" if item else "false"))
        elif isinstance(item, (int, float)):
            normalized.append((key, str(item)))
        else:
            raise ValueError("Skill metadata 值必须是安全标量")
    return tuple(normalized)


def _parse_allowed_tools(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            tools = tuple(shlex.split(value))
        except ValueError as exc:
            raise ValueError("Skill allowed-tools 字符串非法") from exc
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        tools = tuple(item.strip() for item in value)
    else:
        raise ValueError("Skill allowed-tools 必须是空格分隔字符串或字符串列表")
    if not tools or any(not tool for tool in tools) or len(tools) != len(set(tools)):
        raise ValueError("Skill allowed-tools 不能为空或重复")
    return tools
