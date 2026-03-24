"""Unit tests for agentji.skill_translator."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agentji.skill_translator import (
    SkillParseError,
    translate_skill,
    translate_skills,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_skill_dir(tmp_path: Path, skill_md: str, name: str = "my-skill") -> Path:
    """Create a minimal skill directory with the given SKILL.md content."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(skill_md), encoding="utf-8")
    return skill_dir


VALID_SKILL_MD = """
    ---
    name: sql-query
    description: Execute SQL queries against a SQLite database.
    parameters:
      query:
        type: string
        description: The SQL query to execute.
        required: true
      database:
        type: string
        description: Path to the database file.
        required: false
        default: "./data/main.db"
    scripts:
      execute: scripts/run_query.py
    ---

    # SQL Query Skill

    Always validate SQL before executing.
"""


# ── Valid skill parsing ────────────────────────────────────────────────────────

class TestTranslateSkillValid:
    def test_returns_openai_tool_structure(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir)
        assert tool["type"] == "function"
        assert "function" in tool
        fn = tool["function"]
        assert fn["name"] == "sql-query"
        assert "Execute SQL" in fn["description"]
        assert fn["parameters"]["type"] == "object"

    def test_required_param_in_required_list(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir)
        params = tool["function"]["parameters"]
        assert "query" in params["required"]
        assert "database" not in params.get("required", [])

    def test_optional_param_has_default(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir)
        props = tool["function"]["parameters"]["properties"]
        assert props["database"]["default"] == "./data/main.db"

    def test_scripts_metadata_attached(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir)
        assert tool["_scripts"]["execute"] == "scripts/run_query.py"
        assert "_skill_dir" in tool

    def test_skill_dir_path_recorded(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir)
        assert str(skill_dir) == tool["_skill_dir"]

    def test_accept_path_to_skill_md_directly(self, tmp_path: Path) -> None:
        skill_dir = make_skill_dir(tmp_path, VALID_SKILL_MD)
        tool = translate_skill(skill_dir / "SKILL.md")
        assert tool["function"]["name"] == "sql-query"

    def test_no_scripts_execute_is_prompt_skill(self, tmp_path: Path) -> None:
        """A skill without scripts.execute is a prompt-only skill (Anthropic format)."""
        md = """
            ---
            name: ping
            description: A simple ping tool with no parameters.
            ---

            # Ping

            This is the skill body injected into the system prompt.
        """
        skill_dir = make_skill_dir(tmp_path, md)
        tool = translate_skill(skill_dir)
        assert tool["_prompt_only"] is True
        assert tool["function"]["name"] == "ping"
        assert "parameters" not in tool["function"]
        assert "Ping" in tool["_body"]
        assert "_skill_dir" in tool

    def test_tool_skill_with_no_parameters_has_empty_schema(self, tmp_path: Path) -> None:
        """A tool skill (has scripts.execute) with no parameters gets empty schema."""
        md = """
            ---
            name: ping
            description: A simple ping tool with no parameters.
            scripts:
              execute: scripts/run.py
            ---

            # Ping
        """
        skill_dir = make_skill_dir(tmp_path, md)
        tool = translate_skill(skill_dir)
        assert tool.get("_prompt_only") is not True
        params = tool["function"]["parameters"]
        assert params["properties"] == {}
        assert "required" not in params

    def test_enum_constraint_preserved(self, tmp_path: Path) -> None:
        md = """
            ---
            name: greet
            description: Greet in a language.
            parameters:
              language:
                type: string
                description: Language choice.
                required: true
                enum: ["english", "chinese"]
            scripts:
              execute: scripts/run.py
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        tool = translate_skill(skill_dir)
        props = tool["function"]["parameters"]["properties"]
        assert props["language"]["enum"] == ["english", "chinese"]

    def test_integer_param_type(self, tmp_path: Path) -> None:
        md = """
            ---
            name: counter
            description: Count things.
            parameters:
              count:
                type: integer
                description: How many.
                required: true
            scripts:
              execute: scripts/run.py
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        tool = translate_skill(skill_dir)
        props = tool["function"]["parameters"]["properties"]
        assert props["count"]["type"] == "integer"

    def test_agentji_bundled_skill(self) -> None:
        """Integration test against the real bundled agentji skill."""
        skill_path = Path(__file__).parent.parent / "skills" / "agentji"
        tool = translate_skill(skill_path)
        assert tool["function"]["name"] == "agentji"
        props = tool["function"]["parameters"]["properties"]
        assert "action" in props
        assert "config" in props


# ── Error cases ────────────────────────────────────────────────────────────────

class TestTranslateSkillErrors:
    def test_missing_skill_md_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            translate_skill(skill_dir)

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        md = """
            ---
            description: A skill without a name.
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        with pytest.raises(SkillParseError, match="name"):
            translate_skill(skill_dir)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        md = """
            ---
            name: my-skill
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        with pytest.raises(SkillParseError, match="description"):
            translate_skill(skill_dir)

    def test_no_frontmatter_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# No frontmatter here\n", encoding="utf-8")
        with pytest.raises(SkillParseError, match="frontmatter"):
            translate_skill(skill_dir)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: [unclosed bracket\n---\n", encoding="utf-8"
        )
        with pytest.raises(SkillParseError, match="YAML"):
            translate_skill(skill_dir)

    def test_param_missing_type_raises(self, tmp_path: Path) -> None:
        md = """
            ---
            name: my-skill
            description: A skill with a bad parameter.
            parameters:
              bad_param:
                description: No type defined.
                required: true
            scripts:
              execute: scripts/run.py
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        with pytest.raises(SkillParseError, match="type"):
            translate_skill(skill_dir)

    def test_param_unsupported_type_raises(self, tmp_path: Path) -> None:
        md = """
            ---
            name: my-skill
            description: Bad param type.
            parameters:
              bad:
                type: datetime
                description: Not a JSON Schema type.
                required: true
            scripts:
              execute: scripts/run.py
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        with pytest.raises(SkillParseError, match="datetime"):
            translate_skill(skill_dir)

    def test_param_not_a_mapping_raises(self, tmp_path: Path) -> None:
        md = """
            ---
            name: my-skill
            description: Param defined as a string.
            parameters:
              bad_param: "just a string"
            scripts:
              execute: scripts/run.py
            ---
        """
        skill_dir = make_skill_dir(tmp_path, md)
        with pytest.raises(SkillParseError, match="mapping"):
            translate_skill(skill_dir)


# ── translate_skills (list) ────────────────────────────────────────────────────

class TestTranslateSkills:
    def test_multiple_skills(self, tmp_path: Path) -> None:
        # Use tmp_path skills so the test is self-contained
        def _make(name: str, description: str) -> Path:
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {description}\n---\n"
            )
            return skill_dir

        tools = translate_skills([
            _make("skill-a", "First skill"),
            _make("skill-b", "Second skill"),
        ])
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"skill-a", "skill-b"}

    def test_empty_list_returns_empty(self) -> None:
        assert translate_skills([]) == []
