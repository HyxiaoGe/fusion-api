from __future__ import annotations

import textwrap

import pytest

from app.ai.skills.document import parse_skill_document


def test_agent_skills_minimal_document_only_requires_name_and_description() -> None:
    parsed = parse_skill_document(
        textwrap.dedent(
            """\
            ---
            name: template-skill
            description: Replace with description of the skill and when the agent should use it.
            ---

            # Insert instructions below
            """
        ),
        expected_skill_id="template-skill",
    )

    assert parsed.name == "template-skill"
    assert parsed.description.startswith("Replace with description")
    assert parsed.license is None
    assert parsed.compatibility is None
    assert parsed.metadata == ()
    assert parsed.declared_version is None
    assert parsed.declared_allowed_tools == ()
    assert parsed.body == "\n# Insert instructions below\n"


def test_agent_skills_optional_fields_and_metadata_are_preserved() -> None:
    parsed = parse_skill_document(
        textwrap.dedent(
            """\
            ---
            name: vercel-react-best-practices
            description: React and Next.js performance optimization guidelines.
            license: MIT
            compatibility: Requires Node.js 22+ and network access
            metadata:
              author: vercel
              version: "1.0.0"
            ---

            # Vercel React Best Practices
            """
        ),
        expected_skill_id="vercel-react-best-practices",
    )

    assert parsed.license == "MIT"
    assert parsed.compatibility == "Requires Node.js 22+ and network access"
    assert parsed.metadata == (("author", "vercel"), ("version", "1.0.0"))
    assert parsed.declared_version == "1.0.0"


def test_community_metadata_scalars_are_normalized_without_becoming_runtime_permissions() -> None:
    parsed = parse_skill_document(
        """---
name: internal-preview
description: A community skill that uses scalar metadata extensions.
metadata:
  internal: true
  priority: 3
---
# Preview
""",
        expected_skill_id="internal-preview",
    )

    assert parsed.metadata == (("internal", "true"), ("priority", "3"))
    assert parsed.declared_allowed_tools == ()


def test_official_allowed_tools_string_and_legacy_list_are_both_normalized() -> None:
    official = parse_skill_document(
        """---
name: shell-review
description: Reviews a repository with pre-approved shell and read tools.
allowed-tools: Bash(git:*) Bash(jq:*) Read
---
# Review
""",
        expected_skill_id="shell-review",
    )
    legacy = parse_skill_document(
        """---
name: verified-research
version: 1.0.0
description: 对需要官方原文与交叉来源的请求建立可核验证据链
allowed-tools:
  - web_search
  - url_read
---
# 可核验研究
""",
        expected_skill_id="verified-research",
    )

    assert official.declared_allowed_tools == ("Bash(git:*)", "Bash(jq:*)", "Read")
    assert official.declared_version is None
    assert legacy.declared_allowed_tools == ("web_search", "url_read")
    assert legacy.declared_version == "1.0.0"


@pytest.mark.parametrize(
    ("case_name", "document"),
    [
        (
            "duplicate_key",
            """---
name: duplicated
name: replaced
description: Duplicate keys must not be silently overwritten.
---
# Body
""",
        ),
        (
            "unsafe_yaml_tag",
            """---
name: unsafe-tag
description: !!python/object/apply:os.system [echo unsafe]
---
# Body
""",
        ),
        (
            "metadata_nested_value",
            """---
name: nested-metadata
description: Metadata follows the Agent Skills string map contract.
metadata:
  nested:
    enabled: true
---
# Body
""",
        ),
        (
            "empty_body",
            """---
name: empty-body
description: Empty instructions are invalid.
---
""",
        ),
    ],
)
def test_invalid_or_unsafe_frontmatter_is_rejected(case_name: str, document: str) -> None:
    with pytest.raises(ValueError, match="Skill"):
        parse_skill_document(document)


def test_name_must_match_parent_directory_identity() -> None:
    with pytest.raises(ValueError, match="目录"):
        parse_skill_document(
            """---
name: actual-name
description: Name must match the directory selected by the registry.
---
# Body
""",
            expected_skill_id="different-name",
        )
