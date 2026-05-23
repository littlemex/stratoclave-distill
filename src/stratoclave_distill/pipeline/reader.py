"""JSONL session reader.

Stage B's entry point. Reads a JSONL file where each line is the JSON
serialization of a :class:`stratoclave_distill.core.types.NormalizedTurn`
(typically produced by ``stratoclave_loom``'s adapter ``normalize()``) and
yields :class:`NormalizedTurn` instances in file order.

The reader is intentionally tolerant: malformed lines are skipped and
recorded with a structured reason so callers can surface them, but a
single bad line never aborts the whole ingest. ``seq`` is taken from the
record when present, otherwise auto-assigned by file position so a
contributor's hand-edited fixture still works.

The reader does *not* depend on ``stratoclave_loom``; the adapter that
produced the file is responsible for normalization. distill only knows
how to deserialize the public :class:`NormalizedTurn` shape.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from stratoclave_distill.core.errors import IngestError
from stratoclave_distill.core.types import NormalizedTurn


@dataclass(frozen=True, slots=True)
class SkippedLine:
    """A JSONL line we could not decode into a :class:`NormalizedTurn`.

    Recorded so tests and the CLI can report ingest hygiene without
    raising. ``line_no`` is 1-indexed to match editor conventions.
    """

    line_no: int
    reason: str
    raw: str


_REQUIRED_FIELDS = (
    "turn_id",
    "session_id",
    "role",
    "text_content",
    "occurred_at",
)


def _coerce_turn(obj: Mapping[str, object], default_seq: int, raw_line: str) -> NormalizedTurn:
    """Build a :class:`NormalizedTurn` from a decoded JSON object.

    Raises :class:`ValueError` with a human-readable message when a
    required field is missing or has the wrong type. Optional fields
    (``tool_name``, ``tool_input``) accept ``None`` or absence equally.
    """

    for key in _REQUIRED_FIELDS:
        if key not in obj:
            raise ValueError(f"missing required field {key!r}")

    seq_raw = obj.get("seq", default_seq)
    if not isinstance(seq_raw, int) or isinstance(seq_raw, bool):
        raise ValueError(f"'seq' must be an int, got {type(seq_raw).__name__}")

    role = obj["role"]
    if not isinstance(role, str) or not role:
        raise ValueError("'role' must be a non-empty string")

    tool_name_raw = obj.get("tool_name")
    if tool_name_raw is not None and not isinstance(tool_name_raw, str):
        raise ValueError("'tool_name' must be a string or null")

    tool_input_raw = obj.get("tool_input")
    if tool_input_raw is not None and not isinstance(tool_input_raw, dict):
        raise ValueError("'tool_input' must be an object or null")

    return NormalizedTurn(
        turn_id=str(obj["turn_id"]),
        session_id=str(obj["session_id"]),
        seq=seq_raw,
        role=role,
        text_content=str(obj["text_content"]),
        tool_name=tool_name_raw,
        tool_input=tool_input_raw,
        occurred_at=str(obj["occurred_at"]),
        raw_line=str(obj.get("raw_line", raw_line)),
    )


class JsonlSessionReader:
    """Read a JSONL transcript and yield :class:`NormalizedTurn` records.

    Parameters
    ----------
    path:
        Filesystem path to the JSONL file. Pass ``"-"`` only via the
        :meth:`from_stream` constructor; the path-based constructor
        always opens the file in text mode with UTF-8 decoding.
    strict:
        If ``True``, the first malformed line raises
        :class:`IngestError` instead of being skipped. Useful in tests
        to assert that fixtures are well-formed; the production CLI
        defaults to ``False`` so a single corrupt line cannot stall
        an ingest.

    Notes
    -----
    Skipped lines are accumulated in :attr:`skipped` so the CLI can
    log a summary after iteration completes.
    """

    __slots__ = ("_path", "_skipped", "_strict")

    def __init__(self, path: str | Path, *, strict: bool = False) -> None:
        self._path = Path(path)
        self._strict = strict
        self._skipped: list[SkippedLine] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def skipped(self) -> tuple[SkippedLine, ...]:
        """Lines that could not be decoded, in encounter order."""

        return tuple(self._skipped)

    def read(self) -> Iterator[NormalizedTurn]:
        """Yield turns in file order. Resets :attr:`skipped` per call."""

        self._skipped = []
        if not self._path.exists():
            raise IngestError(f"JSONL file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as fh:
            yield from self._iter_stream(fh)

    def _iter_stream(self, fh: IO[str]) -> Iterator[NormalizedTurn]:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                self._reject(line_no, f"invalid JSON: {exc.msg}", stripped)
                continue
            if not isinstance(obj, dict):
                self._reject(line_no, "top-level value is not a JSON object", stripped)
                continue
            try:
                turn = _coerce_turn(obj, default_seq=line_no - 1, raw_line=stripped)
            except ValueError as exc:
                self._reject(line_no, str(exc), stripped)
                continue
            yield turn

    def _reject(self, line_no: int, reason: str, raw: str) -> None:
        if self._strict:
            raise IngestError(f"line {line_no}: {reason}")
        self._skipped.append(SkippedLine(line_no=line_no, reason=reason, raw=raw))


__all__ = ["JsonlSessionReader", "SkippedLine"]
