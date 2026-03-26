"""Tests for agentji.run_context.RunContext."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentji.run_context import RunContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ctx(tmp_path: Path, threshold: int = 8000, logger=None) -> RunContext:
    return RunContext(
        run_id="test-run",
        scratch_dir=tmp_path / "scratch",
        size_threshold=threshold,
        logger=logger,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_set_small_value_stays_in_memory(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    result = ctx.set("key1", "hello world", agent="analyst")
    assert result == "hello world"
    assert ctx.get("key1") == "hello world"


def test_set_large_value_offloads_to_disk(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, threshold=10)
    big_value = "x" * 11
    result = ctx.set("findings", big_value, agent="analyst")

    # result should be a file path string
    assert result.endswith("findings.md")
    path = Path(result)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == big_value


def test_get_offloaded_returns_path_not_content(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, threshold=10)
    big_value = "y" * 20
    stored = ctx.set("output", big_value, agent="analyst")
    retrieved = ctx.get("output")
    # get() returns the path string, not the file contents
    assert retrieved == stored
    assert Path(retrieved).read_text(encoding="utf-8") == big_value


def test_get_unknown_key_returns_none(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    assert ctx.get("nonexistent") is None


def test_summary_contains_metadata(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, threshold=10)
    ctx.set("small", "abc", agent="analyst")
    ctx.set("big", "z" * 20, agent="analyst")

    summary = ctx.summary()
    assert "small" in summary
    assert summary["small"]["offloaded"] is False
    assert summary["small"]["size"] == 3
    assert summary["small"]["agent"] == "analyst"

    assert "big" in summary
    assert summary["big"]["offloaded"] is True
    assert summary["big"]["size"] == 20


def test_scratch_dir_created_on_init(tmp_path: Path) -> None:
    scratch = tmp_path / "deep" / "nested" / "scratch"
    assert not scratch.exists()
    RunContext(run_id="r", scratch_dir=scratch)
    assert scratch.is_dir()


def test_logger_context_write_called(tmp_path: Path) -> None:
    logger = MagicMock()
    ctx = make_ctx(tmp_path, threshold=10, logger=logger)

    # small value — not offloaded
    ctx.set("small", "hi", agent="reporter")
    logger.context_write.assert_called_once_with(
        agent="reporter",
        key="small",
        size=2,
        offloaded=False,
        path=None,
    )

    logger.reset_mock()

    # large value — offloaded; path arg should be a string ending in .md
    ctx.set("big", "a" * 20, agent="reporter")
    call_kwargs = logger.context_write.call_args.kwargs
    assert call_kwargs["key"] == "big"
    assert call_kwargs["offloaded"] is True
    assert call_kwargs["path"] is not None
    assert call_kwargs["path"].endswith("big.md")


def test_concurrent_set_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent set() from multiple threads must not corrupt the store."""
    ctx = make_ctx(tmp_path)
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            ctx.set(f"key_{i}", f"value_{i}", agent="agent")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(ctx._store) == 30
    for i in range(30):
        assert ctx.get(f"key_{i}") == f"value_{i}"


# ── set_file ───────────────────────────────────────────────────────────────────

def test_set_file_stores_path_as_is(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    img = tmp_path / "output.png"
    img.write_bytes(b"\x89PNG")
    result = ctx.set_file("portrait", str(img), agent="painter")
    assert result == str(img)
    assert ctx.get("portrait") == str(img)


def test_set_file_marked_as_offloaded(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG")
    ctx.set_file("img", str(img), agent="painter")
    summary = ctx.summary()
    assert summary["img"]["offloaded"] is True
    assert summary["img"]["agent"] == "painter"
    assert summary["img"]["size"] == 0


def test_set_file_calls_logger(tmp_path: Path) -> None:
    logger = MagicMock()
    ctx = make_ctx(tmp_path, logger=logger)
    img = tmp_path / "result.png"
    img.write_bytes(b"\x89PNG")
    ctx.set_file("output", str(img), agent="imager")
    logger.context_write.assert_called_once()
    kwargs = logger.context_write.call_args.kwargs
    assert kwargs["key"] == "output"
    assert kwargs["offloaded"] is True
    assert kwargs["path"] == str(img)


def test_set_file_retrievable_after_set(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"JFIF")
    ctx.set_file("thumb", str(img), agent="gen")
    assert ctx.get("thumb") == str(img)
