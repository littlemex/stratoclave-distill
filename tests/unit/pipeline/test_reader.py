"""Unit tests for :class:`JsonlSessionReader`.

These cover the contract that downstream Stage B components rely on:

- happy-path JSONL deserializes into :class:`NormalizedTurn` records
- malformed lines are skipped with structured reasons (or raise in strict mode)
- ``seq`` falls back to file position when the field is absent
- optional fields (``tool_name`` / ``tool_input``) accept ``None`` or absence
- iteration order matches file order
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratoclave_distill.core.errors import IngestError
from stratoclave_distill.core.types import NormalizedTurn
from stratoclave_distill.pipeline import JsonlSessionReader, SkippedLine


def _line(record: dict[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reader_yields_turns_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            _line(
                {
                    "turn_id": "t-0",
                    "session_id": "s-1",
                    "seq": 0,
                    "role": "user",
                    "text_content": "hello",
                    "tool_name": None,
                    "tool_input": None,
                    "occurred_at": "2026-05-22T00:00:00Z",
                }
            ),
            _line(
                {
                    "turn_id": "t-1",
                    "session_id": "s-1",
                    "seq": 1,
                    "role": "assistant",
                    "text_content": "hi",
                    "tool_name": None,
                    "tool_input": None,
                    "occurred_at": "2026-05-22T00:00:01Z",
                }
            ),
        ],
    )

    reader = JsonlSessionReader(path)
    turns = list(reader.read())

    assert [t.turn_id for t in turns] == ["t-0", "t-1"]
    assert all(isinstance(t, NormalizedTurn) for t in turns)
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert reader.skipped == ()


def test_reader_preserves_raw_line_when_absent(tmp_path: Path) -> None:
    """If the JSONL omits ``raw_line``, the reader records the original line."""

    path = tmp_path / "session.jsonl"
    record = {
        "turn_id": "t-0",
        "session_id": "s-1",
        "seq": 0,
        "role": "user",
        "text_content": "hello",
        "tool_name": None,
        "tool_input": None,
        "occurred_at": "2026-05-22T00:00:00Z",
    }
    raw = _line(record)
    _write_jsonl(path, [raw])

    [turn] = JsonlSessionReader(path).read()
    assert turn.raw_line == raw


def test_reader_uses_explicit_raw_line_when_provided(tmp_path: Path) -> None:
    """When the producer set ``raw_line`` explicitly, we keep it verbatim."""

    path = tmp_path / "session.jsonl"
    explicit_raw = '{"original": "shape"}'
    record = {
        "turn_id": "t-0",
        "session_id": "s-1",
        "seq": 0,
        "role": "user",
        "text_content": "hello",
        "tool_name": None,
        "tool_input": None,
        "occurred_at": "2026-05-22T00:00:00Z",
        "raw_line": explicit_raw,
    }
    _write_jsonl(path, [_line(record)])

    [turn] = JsonlSessionReader(path).read()
    assert turn.raw_line == explicit_raw


def test_reader_falls_back_to_line_position_when_seq_missing(tmp_path: Path) -> None:
    """``seq`` is recoverable from file position when the field is absent."""

    path = tmp_path / "session.jsonl"
    base = {
        "turn_id": "t",
        "session_id": "s",
        "role": "user",
        "text_content": "x",
        "tool_name": None,
        "tool_input": None,
        "occurred_at": "2026-05-22T00:00:00Z",
    }
    _write_jsonl(path, [_line(base), _line(base), _line(base)])

    seqs = [t.seq for t in JsonlSessionReader(path).read()]
    assert seqs == [0, 1, 2]


def test_reader_carries_tool_call_blocks(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    record = {
        "turn_id": "t-tool",
        "session_id": "s-1",
        "seq": 0,
        "role": "tool_use",
        "text_content": "",
        "tool_name": "shell.run",
        "tool_input": {"cmd": "ls"},
        "occurred_at": "2026-05-22T00:00:00Z",
    }
    _write_jsonl(path, [_line(record)])

    [turn] = JsonlSessionReader(path).read()
    assert turn.tool_name == "shell.run"
    assert turn.tool_input == {"cmd": "ls"}


def test_reader_skips_blank_lines_silently(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    valid = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": 0,
            "role": "user",
            "text_content": "x",
            "tool_name": None,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    path.write_text(f"\n   \n{valid}\n\n", encoding="utf-8")

    reader = JsonlSessionReader(path)
    turns = list(reader.read())
    assert len(turns) == 1
    assert reader.skipped == ()


def test_reader_records_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    valid = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": 0,
            "role": "user",
            "text_content": "x",
            "tool_name": None,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, ["{not json", valid])

    reader = JsonlSessionReader(path)
    turns = list(reader.read())
    assert len(turns) == 1
    assert len(reader.skipped) == 1
    skipped = reader.skipped[0]
    assert skipped.line_no == 1
    assert "invalid JSON" in skipped.reason


def test_reader_records_non_object_top_level(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, ["[1, 2, 3]"])

    reader = JsonlSessionReader(path)
    turns = list(reader.read())
    assert turns == []
    assert reader.skipped[0].reason == "top-level value is not a JSON object"


def test_reader_records_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    incomplete = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "role": "user",
            # text_content + occurred_at deliberately missing
        }
    )
    _write_jsonl(path, [incomplete])

    reader = JsonlSessionReader(path)
    list(reader.read())
    assert reader.skipped[0].reason.startswith("missing required field")


def test_reader_rejects_non_int_seq(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    bad = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": "0",
            "role": "user",
            "text_content": "x",
            "tool_name": None,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, [bad])

    reader = JsonlSessionReader(path)
    list(reader.read())
    assert reader.skipped[0].reason.startswith("'seq' must be an int")


def test_reader_rejects_bool_seq(tmp_path: Path) -> None:
    """``True`` is technically an int subclass; we explicitly reject it."""

    path = tmp_path / "session.jsonl"
    bad = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": True,
            "role": "user",
            "text_content": "x",
            "tool_name": None,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, [bad])
    reader = JsonlSessionReader(path)
    list(reader.read())
    assert "must be an int" in reader.skipped[0].reason


def test_reader_rejects_blank_role(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    bad = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": 0,
            "role": "",
            "text_content": "x",
            "tool_name": None,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, [bad])
    reader = JsonlSessionReader(path)
    list(reader.read())
    assert "non-empty string" in reader.skipped[0].reason


def test_reader_rejects_non_string_tool_name(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    bad = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": 0,
            "role": "tool_use",
            "text_content": "",
            "tool_name": 42,
            "tool_input": None,
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, [bad])
    reader = JsonlSessionReader(path)
    list(reader.read())
    assert "tool_name" in reader.skipped[0].reason


def test_reader_rejects_non_object_tool_input(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    bad = _line(
        {
            "turn_id": "t-0",
            "session_id": "s-1",
            "seq": 0,
            "role": "tool_use",
            "text_content": "",
            "tool_name": "shell.run",
            "tool_input": ["ls"],
            "occurred_at": "2026-05-22T00:00:00Z",
        }
    )
    _write_jsonl(path, [bad])
    reader = JsonlSessionReader(path)
    list(reader.read())
    assert "tool_input" in reader.skipped[0].reason


def test_reader_strict_mode_raises_on_first_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, ["{not json"])

    reader = JsonlSessionReader(path, strict=True)
    with pytest.raises(IngestError, match=r"line 1: invalid JSON"):
        list(reader.read())


def test_reader_raises_when_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    reader = JsonlSessionReader(missing)
    with pytest.raises(IngestError, match=r"JSONL file not found"):
        list(reader.read())


def test_reader_skipped_resets_on_each_read(tmp_path: Path) -> None:
    """Consumers may call ``read()`` more than once; ``skipped`` resets."""

    path = tmp_path / "session.jsonl"
    _write_jsonl(path, ["{not json"])
    reader = JsonlSessionReader(path)
    list(reader.read())
    assert len(reader.skipped) == 1
    list(reader.read())
    assert len(reader.skipped) == 1


def test_skipped_line_is_frozen() -> None:
    sl = SkippedLine(line_no=1, reason="x", raw="y")
    with pytest.raises(AttributeError):
        sl.line_no = 2  # type: ignore[misc]


def test_reader_path_property_exposes_input(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text("", encoding="utf-8")
    reader = JsonlSessionReader(str(path))
    assert reader.path == path


def test_reader_handles_utf8_content(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    record = {
        "turn_id": "t-0",
        "session_id": "s-1",
        "seq": 0,
        "role": "user",
        "text_content": "日本語コンテンツ",
        "tool_name": None,
        "tool_input": None,
        "occurred_at": "2026-05-22T00:00:00Z",
    }
    _write_jsonl(path, [_line(record)])

    [turn] = JsonlSessionReader(path).read()
    assert turn.text_content == "日本語コンテンツ"
