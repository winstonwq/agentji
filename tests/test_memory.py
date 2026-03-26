"""Unit tests for agentji.memory — sliding window compression and LTM."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentji.memory import (
    MemoryBackend,
    _format_as_transcript,
    _get_context_window,
    _SUMMARY_TAG,
)
from agentji.config import MemoryConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path, **overrides) -> MemoryConfig:
    defaults = dict(backend="local", user_id="testuser", ltm_path=str(tmp_path / "ltm"))
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _make_messages(n: int, system: str = "You are helpful.") -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
    return msgs


FAKE_LITELLM_KWARGS = {"model": "openai/gpt-4o", "api_key": "sk-test"}


# ── MemoryBackend disabled ─────────────────────────────────────────────────────

class TestMemoryBackendDisabled:
    def test_no_config_is_disabled(self):
        m = MemoryBackend(None)
        assert not m.enabled

    def test_inject_noop(self):
        m = MemoryBackend(None)
        assert m.inject("sys", "user") == "sys"

    def test_maybe_compress_noop_returns_same_list(self):
        m = MemoryBackend(None)
        msgs = _make_messages(50)
        assert m.maybe_compress(msgs, {}) is msgs

    def test_remember_noop_no_crash(self, tmp_path):
        m = MemoryBackend(None)
        m.remember("run1", "summary", {})  # must not raise


# ── LTM inject ─────────────────────────────────────────────────────────────────

class TestInject:
    def test_no_ltm_file_returns_unchanged(self, tmp_path):
        cfg = _make_config(tmp_path)
        m = MemoryBackend(cfg)
        result = m.inject("Base prompt.", "hello")
        assert result == "Base prompt."

    def test_prepends_facts_from_ltm(self, tmp_path):
        cfg = _make_config(tmp_path)
        ltm_dir = Path(cfg.ltm_path)
        ltm_dir.mkdir(parents=True)
        (ltm_dir / "testuser.jsonl").write_text(json.dumps({
            "ts": "2026-03-01T00:00:00+00:00",
            "run_id": "abc",
            "facts": ["User prefers metric units", "Works on data pipelines"],
        }) + "\n", encoding="utf-8")
        m = MemoryBackend(cfg)
        result = m.inject("Base prompt.", "hello")
        assert "Base prompt." in result
        assert "User prefers metric units" in result
        assert "Works on data pipelines" in result

    def test_inject_limit_respected(self, tmp_path):
        cfg = _make_config(tmp_path, inject_limit=2)
        ltm_dir = Path(cfg.ltm_path)
        ltm_dir.mkdir(parents=True)
        # Write 5 entries
        f = ltm_dir / "testuser.jsonl"
        for i in range(5):
            f.open("a").write(json.dumps({
                "ts": f"2026-03-0{i+1}T00:00:00+00:00",
                "run_id": f"r{i}",
                "facts": [f"fact-entry-{i}"],
            }) + "\n")
        m = MemoryBackend(cfg)
        result = m.inject("prompt", "hi")
        # Only last 2 entries injected
        assert result.count("fact-entry-") == 2

    def test_empty_ltm_file_returns_unchanged(self, tmp_path):
        cfg = _make_config(tmp_path)
        ltm_dir = Path(cfg.ltm_path)
        ltm_dir.mkdir(parents=True)
        (ltm_dir / "testuser.jsonl").write_text("", encoding="utf-8")
        m = MemoryBackend(cfg)
        result = m.inject("original", "hi")
        assert result == "original"


# ── Sliding window compression ─────────────────────────────────────────────────

class TestMaybeCompress:
    def test_off_never_compresses(self, tmp_path):
        cfg = _make_config(tmp_path, compression="off")
        m = MemoryBackend(cfg)
        msgs = _make_messages(100)
        result = m.maybe_compress(msgs, FAKE_LITELLM_KWARGS)
        assert result is msgs  # exact same object, untouched

    def test_auto_within_fallback_budget_no_change(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        msgs = _make_messages(10)  # well under fallback of 40
        with patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")):
            result = m.maybe_compress(msgs, {"model": "unknown/x"})
        assert result == msgs

    def test_auto_triggers_over_fallback_budget(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        msgs = _make_messages(50)  # >40 non-system triggers auto compression
        with (
            patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")),
            patch("agentji.memory._summarize", return_value="COMPRESSED SUMMARY"),
        ):
            result = m.maybe_compress(msgs, {"model": "unknown/x"})
        assert len(result) < len(msgs)
        assert result[0] == msgs[0]  # system prompt preserved
        summary_msgs = [r for r in result if _SUMMARY_TAG in r.get("content", "")]
        assert len(summary_msgs) == 1
        assert "COMPRESSED SUMMARY" in summary_msgs[0]["content"]

    def test_aggressive_triggers_at_lower_threshold(self, tmp_path):
        cfg = _make_config(tmp_path, compression="aggressive")
        m = MemoryBackend(cfg)
        msgs = _make_messages(25)  # >20 fallback for aggressive
        with (
            patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")),
            patch("agentji.memory._summarize", return_value="SUMMARY"),
        ):
            result = m.maybe_compress(msgs, {"model": "unknown/x"})
        assert len(result) < len(msgs)

    def test_aggressive_not_triggered_under_threshold(self, tmp_path):
        cfg = _make_config(tmp_path, compression="aggressive")
        m = MemoryBackend(cfg)
        msgs = _make_messages(15)  # <20, no trigger
        with patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")):
            result = m.maybe_compress(msgs, {"model": "unknown/x"})
        assert result == msgs

    def test_token_based_triggers_when_context_known(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        # 20 non-system messages; context=100 tokens; total=80 (80%>75% → trigger)
        # tail budget = 100*0.40=40 tokens; each msg=5 tokens → keep 8 as tail
        msgs = _make_messages(20)
        with (
            patch("agentji.memory.litellm.get_model_info", return_value={"max_input_tokens": 100}),
            patch("agentji.memory.litellm.token_counter", side_effect=[80] + [5] * 30),
            patch("agentji.memory._summarize", return_value="TOKEN SUMMARY"),
        ):
            result = m.maybe_compress(msgs, {"model": "openai/gpt-4o"})
        summary_msgs = [r for r in result if _SUMMARY_TAG in r.get("content", "")]
        assert len(summary_msgs) == 1

    def test_token_based_no_trigger_under_threshold(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        msgs = _make_messages(5)
        # Simulate: context=1000 tokens, current=500 → 50% < 75% threshold
        with (
            patch("agentji.memory.litellm.get_model_info", return_value={"max_input_tokens": 1000}),
            patch("agentji.memory.litellm.token_counter", return_value=500),
        ):
            result = m.maybe_compress(msgs, {"model": "openai/gpt-4o"})
        assert result == msgs

    def test_compression_failure_returns_original(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        msgs = _make_messages(50)
        with (
            patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")),
            patch("agentji.memory._summarize", side_effect=Exception("LLM down")),
        ):
            result = m.maybe_compress(msgs, {"model": "unknown/x"})
        # Best-effort: returns original on error
        assert result == msgs

    def test_system_prompt_always_at_index_0(self, tmp_path):
        cfg = _make_config(tmp_path, compression="auto")
        m = MemoryBackend(cfg)
        system_content = "You are a specialized agent."
        msgs = [{"role": "system", "content": system_content}] + [
            {"role": "user", "content": f"m{i}"} for i in range(50)
        ]
        with (
            patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")),
            patch("agentji.memory._summarize", return_value="summary"),
        ):
            result = m.maybe_compress(msgs, {"model": "x"})
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_content


# ── LTM remember ───────────────────────────────────────────────────────────────

class TestRemember:
    def test_writes_jsonl_entry(self, tmp_path):
        cfg = _make_config(tmp_path)
        m = MemoryBackend(cfg)
        with patch("agentji.memory._extract_facts", return_value=["fact A", "fact B"]):
            m.remember("run123", "Agent completed the task.", FAKE_LITELLM_KWARGS)
        ltm_file = Path(cfg.ltm_path) / "testuser.jsonl"
        assert ltm_file.exists()
        entry = json.loads(ltm_file.read_text().strip())
        assert entry["run_id"] == "run123"
        assert "fact A" in entry["facts"]
        assert "fact B" in entry["facts"]

    def test_no_op_when_auto_remember_false(self, tmp_path):
        cfg = _make_config(tmp_path, auto_remember=False)
        m = MemoryBackend(cfg)
        m.remember("run1", "summary", {})
        assert not (Path(cfg.ltm_path) / "testuser.jsonl").exists()

    def test_extraction_failure_does_not_crash(self, tmp_path):
        cfg = _make_config(tmp_path)
        m = MemoryBackend(cfg)
        with patch("agentji.memory._extract_facts", side_effect=Exception("LLM error")):
            m.remember("run1", "summary", FAKE_LITELLM_KWARGS)  # must not raise

    def test_empty_facts_not_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        m = MemoryBackend(cfg)
        with patch("agentji.memory._extract_facts", return_value=[]):
            m.remember("run1", "summary", FAKE_LITELLM_KWARGS)
        assert not (Path(cfg.ltm_path) / "testuser.jsonl").exists()

    def test_appends_multiple_entries(self, tmp_path):
        cfg = _make_config(tmp_path)
        m = MemoryBackend(cfg)
        with patch("agentji.memory._extract_facts", return_value=["fact X"]):
            m.remember("run1", "summary 1", FAKE_LITELLM_KWARGS)
            m.remember("run2", "summary 2", FAKE_LITELLM_KWARGS)
        ltm_file = Path(cfg.ltm_path) / "testuser.jsonl"
        lines = ltm_file.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "run1"
        assert json.loads(lines[1])["run_id"] == "run2"


# ── _format_as_transcript ──────────────────────────────────────────────────────

class TestFormatAsTranscript:
    def test_user_and_assistant_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = _format_as_transcript(msgs)
        assert "[user]: hello" in result
        assert "[assistant]: world" in result

    def test_tool_result_truncated_at_600(self):
        long = "x" * 700
        msgs = [{"role": "tool", "content": long}]
        result = _format_as_transcript(msgs)
        assert "truncated" in result
        assert len(result) < 750

    def test_prior_summary_passed_through(self):
        msgs = [{"role": "system", "content": f"{_SUMMARY_TAG}\nOld summary here."}]
        result = _format_as_transcript(msgs)
        assert "Old summary here." in result

    def test_plain_system_messages_skipped(self):
        msgs = [{"role": "system", "content": "system boilerplate"}]
        result = _format_as_transcript(msgs)
        assert "system boilerplate" not in result

    def test_assistant_with_tool_calls_shows_names(self):
        msgs = [{"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "bash"}},
            {"function": {"name": "read_file"}},
        ]}]
        result = _format_as_transcript(msgs)
        assert "bash" in result
        assert "read_file" in result

    def test_assistant_content_and_tool_calls_both_shown(self):
        msgs = [{"role": "assistant", "content": "thinking...", "tool_calls": [
            {"function": {"name": "bash"}},
        ]}]
        result = _format_as_transcript(msgs)
        assert "thinking..." in result
        assert "bash" in result


# ── _get_context_window ────────────────────────────────────────────────────────

class TestGetContextWindow:
    def test_returns_max_input_tokens(self):
        with patch("agentji.memory.litellm.get_model_info", return_value={"max_input_tokens": 128000}):
            assert _get_context_window("openai/gpt-4o") == 128000

    def test_falls_back_to_max_tokens(self):
        with patch("agentji.memory.litellm.get_model_info", return_value={"max_tokens": 8192}):
            assert _get_context_window("small/model") == 8192

    def test_returns_none_on_exception(self):
        with patch("agentji.memory.litellm.get_model_info", side_effect=Exception("unknown")):
            assert _get_context_window("custom/private-model") is None

    def test_returns_none_when_no_known_field(self):
        with patch("agentji.memory.litellm.get_model_info", return_value={}):
            assert _get_context_window("mystery/model") is None
