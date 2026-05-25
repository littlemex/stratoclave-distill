"""Unit tests for :class:`InMemoryGroupLearningStore`.

Group rollups are the Aggregator's output. The store keeps the contract
narrow: upsert by ``group_learning_id``, query by ``group_id``, and
expose hybrid search (cosine + BM25 RRF) the same way the per-row
:class:`LearningStore.search_hybrid` does.

These tests exercise:

- Protocol conformance at runtime (so the asyncpg variant cannot drift).
- Upsert preserves history per ``group_id`` and ``list_by_group`` honors
  ``latest_only``.
- ``search_hybrid`` returns the most recent rollup per group, ranks by
  RRF of cosine + BM25, and tolerates an empty store / zero-norm vector.
- A dimension mismatch on the query vector raises :class:`EmbeddingError`
  (the same contract :class:`InMemoryLearningStore` enforces).
"""

from __future__ import annotations

import pytest

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.core.types import GroupLearning
from stratoclave_distill.db import (
    GroupLearningStore,
    InMemoryGroupLearningStore,
)


def _group(
    group_learning_id: str,
    *,
    group_id: str,
    summary_md: str,
    bm25_text: str = "",
    contributing: tuple[str, ...] = (),
    created_at: str = "2026-05-25T00:00:00Z",
) -> GroupLearning:
    return GroupLearning(
        group_learning_id=group_learning_id,
        group_id=group_id,
        summary_md=summary_md,
        contributing_learnings=contributing,
        bm25_text=bm25_text or summary_md,
        created_at=created_at,
    )


def test_inmemory_group_learning_store_satisfies_protocol() -> None:
    assert isinstance(InMemoryGroupLearningStore(), GroupLearningStore)


@pytest.mark.asyncio
async def test_upsert_and_get_round_trip() -> None:
    store = InMemoryGroupLearningStore()
    g = _group("gl-1", group_id="g-A", summary_md="commit before deploy")
    await store.upsert(g, embedding=[1.0, 0.0])

    fetched = await store.get("gl-1")
    assert fetched == g


@pytest.mark.asyncio
async def test_get_missing_returns_none() -> None:
    store = InMemoryGroupLearningStore()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_list_by_group_returns_empty_when_no_match() -> None:
    store = InMemoryGroupLearningStore()
    await store.upsert(
        _group("gl-1", group_id="g-A", summary_md="x"),
        embedding=[1.0, 0.0],
    )
    assert await store.list_by_group("g-other") == ()


@pytest.mark.asyncio
async def test_list_by_group_latest_only_returns_most_recent_row() -> None:
    store = InMemoryGroupLearningStore()
    older = _group(
        "gl-old",
        group_id="g-A",
        summary_md="v1",
        created_at="2026-05-20T00:00:00Z",
    )
    newer = _group(
        "gl-new",
        group_id="g-A",
        summary_md="v2",
        created_at="2026-05-25T00:00:00Z",
    )
    await store.upsert(older, embedding=[1.0, 0.0])
    await store.upsert(newer, embedding=[1.0, 0.0])

    latest = await store.list_by_group("g-A", latest_only=True)
    assert latest == (newer,)


@pytest.mark.asyncio
async def test_list_latest_per_group_returns_one_row_per_group_id() -> None:
    store = InMemoryGroupLearningStore()
    await store.upsert(
        _group(
            "gl-A-old",
            group_id="g-A",
            summary_md="A v1",
            created_at="2026-05-20T00:00:00Z",
        ),
        embedding=[1.0, 0.0],
    )
    await store.upsert(
        _group(
            "gl-A-new",
            group_id="g-A",
            summary_md="A v2",
            created_at="2026-05-25T00:00:00Z",
        ),
        embedding=[1.0, 0.0],
    )
    await store.upsert(
        _group(
            "gl-B",
            group_id="g-B",
            summary_md="B only",
            created_at="2026-05-22T00:00:00Z",
        ),
        embedding=[0.0, 1.0],
    )

    rows = await store.list_latest_per_group()
    ids = tuple(r.group_learning_id for r in rows)
    assert set(ids) == {"gl-A-new", "gl-B"}
    # Sorted by created_at DESC: gl-A-new (2026-05-25) before gl-B (2026-05-22).
    assert ids == ("gl-A-new", "gl-B")


@pytest.mark.asyncio
async def test_list_latest_per_group_empty_store_returns_empty() -> None:
    store = InMemoryGroupLearningStore()
    assert await store.list_latest_per_group() == ()


@pytest.mark.asyncio
async def test_list_by_group_history_returns_newest_first() -> None:
    store = InMemoryGroupLearningStore()
    older = _group(
        "gl-old",
        group_id="g-A",
        summary_md="v1",
        created_at="2026-05-20T00:00:00Z",
    )
    newer = _group(
        "gl-new",
        group_id="g-A",
        summary_md="v2",
        created_at="2026-05-25T00:00:00Z",
    )
    await store.upsert(older, embedding=[1.0, 0.0])
    await store.upsert(newer, embedding=[1.0, 0.0])

    history = await store.list_by_group("g-A", latest_only=False)
    assert tuple(h.group_learning_id for h in history) == ("gl-new", "gl-old")


@pytest.mark.asyncio
async def test_search_hybrid_empty_store_returns_empty() -> None:
    store = InMemoryGroupLearningStore()
    hits = await store.search_hybrid(
        query_text="anything",
        query_vector=[1.0, 0.0],
        top_k=5,
    )
    assert hits == ()


@pytest.mark.asyncio
async def test_search_hybrid_ranks_vector_match_first() -> None:
    store = InMemoryGroupLearningStore()
    await store.upsert(
        _group("gl-near", group_id="g-A", summary_md="alpha rollup"),
        embedding=[1.0, 0.0],
    )
    await store.upsert(
        _group("gl-far", group_id="g-B", summary_md="beta rollup"),
        embedding=[0.0, 1.0],
    )

    hits = await store.search_hybrid(
        query_text="rollup",
        query_vector=[1.0, 0.0],
        top_k=2,
    )
    assert tuple(h.group.group_learning_id for h in hits) == ("gl-near", "gl-far")
    assert hits[0].cosine == pytest.approx(1.0)
    assert hits[0].vector_rank == 1


@pytest.mark.asyncio
async def test_search_hybrid_dedupes_to_latest_per_group() -> None:
    store = InMemoryGroupLearningStore()
    older = _group(
        "gl-old",
        group_id="g-A",
        summary_md="old summary",
        created_at="2026-05-20T00:00:00Z",
    )
    newer = _group(
        "gl-new",
        group_id="g-A",
        summary_md="new summary",
        created_at="2026-05-25T00:00:00Z",
    )
    await store.upsert(older, embedding=[1.0, 0.0])
    await store.upsert(newer, embedding=[1.0, 0.0])

    hits = await store.search_hybrid(
        query_text="summary",
        query_vector=[1.0, 0.0],
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0].group.group_learning_id == "gl-new"


@pytest.mark.asyncio
async def test_search_hybrid_top_k_bounds_results() -> None:
    store = InMemoryGroupLearningStore()
    for i in range(5):
        await store.upsert(
            _group(f"gl-{i}", group_id=f"g-{i}", summary_md=f"rollup {i}"),
            embedding=[1.0, 0.0],
        )
    hits = await store.search_hybrid(
        query_text="rollup",
        query_vector=[1.0, 0.0],
        top_k=2,
    )
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_search_hybrid_dimension_mismatch_raises() -> None:
    store = InMemoryGroupLearningStore()
    await store.upsert(
        _group("gl-1", group_id="g-A", summary_md="rollup"),
        embedding=[1.0, 0.0],
    )
    with pytest.raises(EmbeddingError):
        await store.search_hybrid(
            query_text="rollup",
            query_vector=[1.0],
            top_k=1,
        )


@pytest.mark.asyncio
async def test_search_hybrid_zero_norm_vector_yields_zero_cosine() -> None:
    store = InMemoryGroupLearningStore()
    await store.upsert(
        _group("gl-1", group_id="g-A", summary_md="rollup"),
        embedding=[1.0, 0.0],
    )
    hits = await store.search_hybrid(
        query_text="unrelated",
        query_vector=[0.0, 0.0],
        top_k=1,
    )
    assert len(hits) == 1
    assert hits[0].cosine == 0.0
