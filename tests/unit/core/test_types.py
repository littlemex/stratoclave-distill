"""Unit tests for the public dataclasses.

These tests exist to lock in the *invariants* the rest of the codebase
relies on:

- All public types are frozen so they are safe to share across coroutines
  and use as cache keys.
- Every type uses ``__slots__`` so a typo in attribute access raises
  AttributeError instead of silently creating a stray instance attribute.
- :meth:`ContextPack.to_markdown` returns the rendered ``markdown`` field
  so callers can switch between attribute access and method call freely.
"""

from __future__ import annotations

import dataclasses

import pytest

from stratoclave_distill import (
    ContextPack,
    ContextPackItem,
    EmbeddingRecord,
    GroupLearning,
    Learning,
    NormalizedTurn,
    SessionDigest,
    SessionPurpose,
)

ALL_PUBLIC_TYPES = (
    NormalizedTurn,
    SessionPurpose,
    SessionDigest,
    Learning,
    GroupLearning,
    EmbeddingRecord,
    ContextPackItem,
    ContextPack,
)


@pytest.mark.parametrize("klass", ALL_PUBLIC_TYPES)
def test_public_types_are_frozen_dataclasses(klass: type) -> None:
    assert dataclasses.is_dataclass(klass), f"{klass.__name__} must be a dataclass"
    params = klass.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen, f"{klass.__name__} must be frozen"


@pytest.mark.parametrize("klass", ALL_PUBLIC_TYPES)
def test_public_types_use_slots(klass: type) -> None:
    assert hasattr(klass, "__slots__"), f"{klass.__name__} must declare __slots__"


def test_normalized_turn_round_trips_minimal_fields() -> None:
    turn = NormalizedTurn(
        turn_id="t-1",
        session_id="s-1",
        seq=0,
        role="user",
        text_content="hello",
        tool_name=None,
        tool_input=None,
        occurred_at="2026-05-22T00:00:00Z",
        raw_line='{"raw":true}',
    )
    assert turn.session_id == "s-1"
    assert turn.tool_name is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.seq = 99  # type: ignore[misc]


def test_session_purpose_defaults_are_safe() -> None:
    purpose = SessionPurpose(session_id="s-1", purpose="explore loom adapter")
    assert purpose.domain_tags == ()
    assert purpose.success_score is None
    assert purpose.polluted is False


def test_learning_evidence_count_starts_at_one() -> None:
    learning = Learning(
        learning_id="l-1",
        scope="project",
        rule="set PATH",
        why="OnNodeConfigured cannot find sbatch",
    )
    assert learning.evidence_count == 1
    assert learning.confidence == pytest.approx(0.5)
    assert learning.archived_at is None


def test_context_pack_to_markdown_returns_markdown_attribute() -> None:
    items = (ContextPackItem(kind="learning", source_id="l-1", text="rule", score=0.9, tokens=10),)
    pack = ContextPack(markdown="## ctx\nrule\n", items=items, total_tokens=10)
    assert pack.to_markdown() == "## ctx\nrule\n"
    assert pack.items[0].source_id == "l-1"
