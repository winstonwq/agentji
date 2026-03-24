"""Unit tests for agentji.improver — skill improvement extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentji.improver import extract_and_save, _parse_json_array, _build_user_prompt


# ── _parse_json_array ─────────────────────────────────────────────────────────

class TestParseJsonArray:
    def test_plain_array(self):
        raw = '[{"type": "correction", "skill": "sql-query", "learning": "foo", "context": "bar"}]'
        result = _parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["type"] == "correction"

    def test_empty_array(self):
        assert _parse_json_array("[]") == []

    def test_fenced_json_block(self):
        raw = '```json\n[{"type": "hint", "skill": "general", "learning": "x", "context": "y"}]\n```'
        result = _parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["skill"] == "general"

    def test_fenced_no_lang(self):
        raw = '```\n[{"type": "affirmation", "skill": "s", "learning": "l", "context": "c"}]\n```'
        result = _parse_json_array(raw)
        assert len(result) == 1

    def test_array_embedded_in_text(self):
        raw = 'Here are the findings:\n[{"type": "correction", "skill": "x", "learning": "y", "context": "z"}]\nDone.'
        result = _parse_json_array(raw)
        assert len(result) == 1

    def test_non_array_returns_empty(self):
        # A JSON object (not array) is valid JSON but not the expected format — returns []
        result = _parse_json_array('{"type": "correction"}')
        assert result == []

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json_array("not json at all")

    def test_array_with_non_dict_items_filtered(self):
        raw = '[{"type": "correction", "skill": "x", "learning": "y", "context": "z"}, "string_item", 42]'
        result = _parse_json_array(raw)
        assert len(result) == 1


# ── _build_user_prompt ────────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_includes_skill_names(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = _build_user_prompt(msgs, ["sql-query", "data-analysis"])
        assert "sql-query" in result
        assert "data-analysis" in result

    def test_includes_messages(self):
        msgs = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "response here"},
        ]
        result = _build_user_prompt(msgs, [])
        assert "hello world" in result
        assert "response here" in result

    def test_role_uppercased(self):
        msgs = [{"role": "user", "content": "msg"}]
        result = _build_user_prompt(msgs, [])
        assert "USER:" in result

    def test_no_skills(self):
        msgs = [{"role": "user", "content": "x"}]
        result = _build_user_prompt(msgs, [])
        assert "none" in result


# ── extract_and_save ──────────────────────────────────────────────────────────

_GOOD_RESPONSE = json.dumps([
    {
        "type": "correction",
        "skill": "sql-query",
        "learning": "Always alias columns in GROUP BY.",
        "context": "User: the column is ambiguous.",
    },
    {
        "type": "affirmation",
        "skill": "general",
        "learning": "Percentage trends preferred over raw values.",
        "context": "User: yes, exactly.",
    },
])

_MOCK_LITELLM_RESP = MagicMock()
_MOCK_LITELLM_RESP.choices = [MagicMock(message=MagicMock(content=_GOOD_RESPONSE))]


class TestExtractAndSave:
    def _skill_refs(self, tmp_path: Path) -> list[dict]:
        skill_dir = tmp_path / "skills" / "sql-query"
        skill_dir.mkdir(parents=True)
        return [{"name": "sql-query", "path": str(skill_dir)}]

    def test_returns_extracted_improvements(self, tmp_path):
        messages = [
            {"role": "user", "content": "Query fails."},
            {"role": "assistant", "content": "Try adding an alias."},
            {"role": "user", "content": "Yes, that worked!"},
        ]
        skill_refs = self._skill_refs(tmp_path)

        with patch("litellm.completion", return_value=_MOCK_LITELLM_RESP):
            result = extract_and_save(
                messages=messages,
                session_id="sess-001",
                skill_refs=skill_refs,
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=[],
                fallback_improvements_path=tmp_path / "improvements.jsonl",
            )

        assert len(result) == 2
        assert result[0]["type"] == "correction"
        assert result[0]["skill"] == "sql-query"
        assert result[1]["skill"] == "general"

    def test_saves_to_skill_file(self, tmp_path):
        messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        skill_refs = self._skill_refs(tmp_path)

        with patch("litellm.completion", return_value=_MOCK_LITELLM_RESP):
            extract_and_save(
                messages=messages,
                session_id="sess-002",
                skill_refs=skill_refs,
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=[],
                fallback_improvements_path=tmp_path / "improvements.jsonl",
            )

        skill_file = tmp_path / "skills" / "sql-query" / "improvements.jsonl"
        assert skill_file.exists()
        lines = skill_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["session_id"] == "sess-002"
        assert entry["type"] == "correction"
        assert entry["skill"] == "sql-query"
        assert "ts" in entry

    def test_general_skill_uses_fallback_path(self, tmp_path):
        messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        fallback = tmp_path / "general_improvements.jsonl"

        with patch("litellm.completion", return_value=_MOCK_LITELLM_RESP):
            extract_and_save(
                messages=messages,
                session_id="sess-003",
                skill_refs=[],   # no skills registered — "general" must use fallback
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=[],
                fallback_improvements_path=fallback,
            )

        assert fallback.exists()
        lines = fallback.read_text().strip().splitlines()
        assert any(json.loads(l)["skill"] == "general" for l in lines)

    def test_empty_messages_returns_empty(self, tmp_path):
        result = extract_and_save(
            messages=[],
            session_id="sess-004",
            skill_refs=[],
            model="openai/gpt-4o-mini",
            litellm_kwargs={"api_key": "sk-test"},
            target_skills=[],
        )
        assert result == []

    def test_litellm_error_returns_empty(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        with patch("litellm.completion", side_effect=Exception("API error")):
            result = extract_and_save(
                messages=messages,
                session_id="sess-005",
                skill_refs=[],
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=[],
            )
        assert result == []

    def test_malformed_llm_response_returns_empty(self, tmp_path):
        messages = [{"role": "user", "content": "hello"}]
        bad_resp = MagicMock()
        bad_resp.choices = [MagicMock(message=MagicMock(content="this is not JSON at all"))]
        with patch("litellm.completion", return_value=bad_resp):
            result = extract_and_save(
                messages=messages,
                session_id="sess-006",
                skill_refs=[],
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=[],
            )
        assert result == []

    def test_target_skills_filters_skill_names_in_prompt(self, tmp_path):
        """Only targeted skill names appear in the extraction prompt."""
        messages = [{"role": "user", "content": "x"}]
        skill_refs = [
            {"name": "sql-query", "path": str(tmp_path / "s1")},
            {"name": "data-analysis", "path": str(tmp_path / "s2")},
        ]
        captured_prompts = []

        def fake_completion(model, messages, **kw):
            captured_prompts.append(messages[-1]["content"])
            return MagicMock(choices=[MagicMock(message=MagicMock(content="[]"))])

        with patch("litellm.completion", side_effect=fake_completion):
            extract_and_save(
                messages=messages,
                session_id="sess-007",
                skill_refs=skill_refs,
                model="openai/gpt-4o-mini",
                litellm_kwargs={"api_key": "sk-test"},
                target_skills=["sql-query"],  # only sql-query
            )

        assert "sql-query" in captured_prompts[0]
        assert "data-analysis" not in captured_prompts[0]

    def test_multiple_sessions_appended_to_same_file(self, tmp_path):
        """Multiple sessions append to the same improvements.jsonl, not overwrite."""
        skill_refs = self._skill_refs(tmp_path)

        for i in range(3):
            with patch("litellm.completion", return_value=_MOCK_LITELLM_RESP):
                extract_and_save(
                    messages=[{"role": "user", "content": f"session {i}"}],
                    session_id=f"sess-{i:03d}",
                    skill_refs=skill_refs,
                    model="openai/gpt-4o-mini",
                    litellm_kwargs={"api_key": "sk-test"},
                    target_skills=[],
                    fallback_improvements_path=tmp_path / "improvements.jsonl",
                )

        skill_file = tmp_path / "skills" / "sql-query" / "improvements.jsonl"
        lines = skill_file.read_text().strip().splitlines()
        assert len(lines) == 3   # one correction per session
