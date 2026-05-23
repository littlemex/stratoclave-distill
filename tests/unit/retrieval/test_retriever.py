"""Unit tests for :class:`Retriever`.

The Retriever's contract:

- one query embedding, two ``search_hybrid`` calls (one per lane);
- ``canonical`` only includes rows passing the lane predicate
  (evidence_count >= threshold AND age >= threshold AND scope != experiment);
- ``emerging`` is exactly the complement of canonical among active rows;
- ``conflicts`` and ``gaps`` are populated only when their stores are wired;
- empty / invalid query texts are rejected at the boundary;
- thresholds are forwarded faithfully so the same row can shift lanes
  when ``canonical_min_evidence`` / ``canonical_min_age_days`` change.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.core.types import Learning, LearningConflict, SessionGap
from stratoclave_distill.db.memory import (
    InMemoryConflictStore,
    InMemoryGapStore,
    InMemoryLearningStore,
)
from stratoclave_distill.retrieval import (
    RetrievalResult,
    Retriever,
    hits_for_learning,
    learning_ids,
)


class _FixedEmbedder:
    """Stub :class:`EmbeddingProvider` returning a configured vector."""

    def __init__(self, vector: Sequence[float], *, dimension: int = 2) -> None:
        self._vector = list(vector)
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self._vector) for _ in texts]


class _EmptyEmbedder:
    """Returns no vectors — exercises the embedder-failure branch."""

    @property
    def model(self) -> str:
        return "empty"

    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        return []


def _learning(
    learning_id: str,
    *,
    rule: str,
    bm25_text: str,
    scope: str = "session",
    evidence_count: int = 1,
    created_offset_days: int = 30,
) -> Learning:
    """Build a Learning with ``created_at`` ``offset`` days in the past."""

    created = datetime.now(UTC) - timedelta(days=created_offset_days)
    return Learning(
        learning_id=learning_id,
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why="because tests",
        bm25_text=bm25_text,
        evidence_count=evidence_count,
        created_at=created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        updated_at=created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# --------------------------------------------------------------------------
# Construction validation
# --------------------------------------------------------------------------


def test_retriever_rejects_zero_top_k_canonical() -> None:
    store = InMemoryLearningStore()
    embedder = _FixedEmbedder([1.0, 0.0])
    with pytest.raises(ValueError, match=r"top_k_canonical must be >= 1"):
        Retriever(store, embedder, top_k_canonical=0)


def test_retriever_rejects_zero_top_k_emerging() -> None:
    store = InMemoryLearningStore()
    embedder = _FixedEmbedder([1.0, 0.0])
    with pytest.raises(ValueError, match=r"top_k_emerging must be >= 1"):
        Retriever(store, embedder, top_k_emerging=0)


def test_retriever_rejects_zero_min_evidence() -> None:
    store = InMemoryLearningStore()
    embedder = _FixedEmbedder([1.0, 0.0])
    with pytest.raises(ValueError, match=r"canonical_min_evidence must be >= 1"):
        Retriever(store, embedder, canonical_min_evidence=0)


def test_retriever_rejects_negative_min_age() -> None:
    store = InMemoryLearningStore()
    embedder = _FixedEmbedder([1.0, 0.0])
    with pytest.raises(ValueError, match=r"canonical_min_age_days must be >= 0"):
        Retriever(store, embedder, canonical_min_age_days=-1)


# --------------------------------------------------------------------------
# Empty query / embedder failure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query() -> None:
    store = InMemoryLearningStore()
    embedder = _FixedEmbedder([1.0, 0.0])
    retriever = Retriever(store, embedder)
    with pytest.raises(ValueError, match=r"query_text"):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_retrieve_raises_when_embedder_returns_nothing() -> None:
    store = InMemoryLearningStore()
    retriever = Retriever(store, _EmptyEmbedder())
    with pytest.raises(EmbeddingError):
        await retriever.retrieve("hello")


# --------------------------------------------------------------------------
# Lane separation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_splits_canonical_and_emerging() -> None:
    store = InMemoryLearningStore()
    # Canonical: 3 evidence, 30 days old, scope=session
    canonical_row = _learning(
        "lid-canonical",
        rule="prefer SQL views",
        bm25_text="prefer sql views",
        evidence_count=3,
        created_offset_days=30,
    )
    # Emerging: only 1 evidence
    emerging_low_evidence = _learning(
        "lid-fresh",
        rule="prefer SQL views",
        bm25_text="prefer sql views",
        evidence_count=1,
        created_offset_days=30,
    )
    # Emerging: 3 evidence but very fresh (3 days old)
    emerging_fresh = _learning(
        "lid-young",
        rule="prefer SQL views",
        bm25_text="prefer sql views",
        evidence_count=3,
        created_offset_days=3,
    )
    # Emerging: experiment scope, regardless of evidence/age
    emerging_experiment = _learning(
        "lid-exp",
        rule="prefer SQL views",
        bm25_text="prefer sql views",
        scope="experiment",
        evidence_count=10,
        created_offset_days=60,
    )

    for row in (canonical_row, emerging_low_evidence, emerging_fresh, emerging_experiment):
        await store.insert(row, embedding=[1.0, 0.0])

    embedder = _FixedEmbedder([1.0, 0.0])
    retriever = Retriever(
        store,
        embedder,
        top_k_canonical=10,
        top_k_emerging=10,
        canonical_min_evidence=3,
        canonical_min_age_days=14,
    )
    result = await retriever.retrieve("prefer sql views")

    assert isinstance(result, RetrievalResult)
    assert result.query_text == "prefer sql views"
    canonical_ids = set(learning_ids(result.canonical))
    emerging_ids = set(learning_ids(result.emerging))
    assert canonical_ids == {"lid-canonical"}
    assert emerging_ids == {"lid-fresh", "lid-young", "lid-exp"}
    # Lanes never overlap
    assert canonical_ids.isdisjoint(emerging_ids)
    # all_hits concatenates with canonical first
    assert learning_ids(result.all_hits)[: len(result.canonical)] == learning_ids(result.canonical)
    # Embedder is called exactly once
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["prefer sql views"]


@pytest.mark.asyncio
async def test_thresholds_shift_lane_membership() -> None:
    """Same row can move between lanes as the predicate parameters change."""

    store = InMemoryLearningStore()
    row = _learning(
        "lid-edge",
        rule="rule",
        bm25_text="rule",
        evidence_count=2,
        created_offset_days=30,
    )
    await store.insert(row, embedding=[1.0, 0.0])

    embedder = _FixedEmbedder([1.0, 0.0])

    strict = Retriever(store, embedder, canonical_min_evidence=3)
    relaxed = Retriever(store, embedder, canonical_min_evidence=2)

    strict_result = await strict.retrieve("rule")
    relaxed_result = await relaxed.retrieve("rule")

    assert learning_ids(strict_result.canonical) == ()
    assert learning_ids(strict_result.emerging) == ("lid-edge",)
    assert learning_ids(relaxed_result.canonical) == ("lid-edge",)
    assert learning_ids(relaxed_result.emerging) == ()


@pytest.mark.asyncio
async def test_retrieve_honors_per_lane_top_k() -> None:
    store = InMemoryLearningStore()
    for i in range(5):
        await store.insert(
            _learning(
                f"can-{i}",
                rule="rule",
                bm25_text="rule",
                evidence_count=10,
                created_offset_days=60,
            ),
            embedding=[1.0, 0.0],
        )
    for i in range(4):
        await store.insert(
            _learning(
                f"em-{i}",
                rule="rule",
                bm25_text="rule",
                evidence_count=1,
                created_offset_days=60,
            ),
            embedding=[1.0, 0.0],
        )

    retriever = Retriever(
        store,
        _FixedEmbedder([1.0, 0.0]),
        top_k_canonical=2,
        top_k_emerging=3,
    )
    result = await retriever.retrieve("rule")
    assert len(result.canonical) == 2
    assert len(result.emerging) == 3


# --------------------------------------------------------------------------
# Conflicts / gaps sidecars
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conflicts_and_gaps_empty_when_no_stores() -> None:
    store = InMemoryLearningStore()
    await store.insert(
        _learning("lid-1", rule="rule", bm25_text="rule", evidence_count=10),
        embedding=[1.0, 0.0],
    )
    retriever = Retriever(store, _FixedEmbedder([1.0, 0.0]))
    result = await retriever.retrieve("rule")
    assert result.conflicts == ()
    assert result.gaps == ()


@pytest.mark.asyncio
async def test_conflicts_populated_only_open() -> None:
    store = InMemoryLearningStore()
    await store.insert(
        _learning("lid-1", rule="rule", bm25_text="rule", evidence_count=10),
        embedding=[1.0, 0.0],
    )
    conflicts = InMemoryConflictStore()
    await conflicts.insert(
        LearningConflict(
            conflict_id="c-open",
            from_id="lid-1",
            to_id="lid-2",
            reason="borderline",
            cosine_at_detection=0.85,
            detected_at="2026-05-22T00:00:00Z",
            resolution="open",
        )
    )
    await conflicts.insert(
        LearningConflict(
            conflict_id="c-resolved",
            from_id="lid-3",
            to_id="lid-4",
            reason="borderline",
            cosine_at_detection=0.85,
            detected_at="2026-05-21T00:00:00Z",
            resolution="kept_old",
        )
    )

    retriever = Retriever(store, _FixedEmbedder([1.0, 0.0]), conflict_store=conflicts)
    result = await retriever.retrieve("rule")
    assert len(result.conflicts) == 1
    assert result.conflicts[0].conflict_id == "c-open"


@pytest.mark.asyncio
async def test_gaps_filtered_by_session_id() -> None:
    store = InMemoryLearningStore()
    await store.insert(
        _learning("lid-1", rule="rule", bm25_text="rule", evidence_count=10),
        embedding=[1.0, 0.0],
    )
    gaps = InMemoryGapStore()
    await gaps.insert(
        SessionGap(
            gap_id="g-a",
            session_id="sess-A",
            topic="ef_search",
            why_unknown="",
            bm25_text="ef_search",
            detected_at="2026-05-22T00:00:00Z",
        )
    )
    await gaps.insert(
        SessionGap(
            gap_id="g-b",
            session_id="sess-B",
            topic="other",
            why_unknown="",
            bm25_text="other",
            detected_at="2026-05-22T00:00:00Z",
        )
    )
    await gaps.insert(
        SessionGap(
            gap_id="g-resolved",
            session_id="sess-A",
            topic="resolved",
            why_unknown="",
            bm25_text="resolved",
            detected_at="2026-05-21T00:00:00Z",
            resolved_at="2026-05-22T00:00:00Z",
            resolved_by_learning="lid-1",
        )
    )

    retriever = Retriever(store, _FixedEmbedder([1.0, 0.0]), gap_store=gaps)

    a_result = await retriever.retrieve("rule", gap_session_id="sess-A")
    assert {g.gap_id for g in a_result.gaps} == {"g-a"}

    all_result = await retriever.retrieve("rule")
    assert {g.gap_id for g in all_result.gaps} == {"g-a", "g-b"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hits_for_learning_locates_match() -> None:
    store = InMemoryLearningStore()
    await store.insert(
        _learning("target", rule="rule", bm25_text="rule", evidence_count=10),
        embedding=[1.0, 0.0],
    )
    retriever = Retriever(store, _FixedEmbedder([1.0, 0.0]))
    result = await retriever.retrieve("rule")
    hit = hits_for_learning(result.canonical, "target")
    assert hit is not None
    assert hit.learning.learning_id == "target"
    assert hits_for_learning(result.canonical, "missing") is None
