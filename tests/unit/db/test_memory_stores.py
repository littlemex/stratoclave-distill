"""Unit tests for the in-memory store implementations.

These cover the contract that the Distiller and Curator depend on:

- watermarks are monotonic per session, default to ``0`` for unknown ids;
- purpose / digest stores are upsert-only, single-row-per-session;
- learning lifecycle (insert / update / supersede) preserves audit trail
  and never deletes rows;
- ``search_hybrid`` ranks active learnings by RRF of cosine + BM25 and
  filters out archived rows / mismatched scope.

Every store also satisfies its :class:`typing.Protocol` at runtime so
that injecting the asyncpg variant later cannot drift away from the
contract without breaking these tests.
"""

from __future__ import annotations

import pytest

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.core.types import Learning, SessionDigest, SessionPurpose
from stratoclave_distill.db import (
    DigestStore,
    InMemoryDigestStore,
    InMemoryLearningStore,
    InMemoryPurposeStore,
    InMemoryWatermarkStore,
    LearningStore,
    PurposeStore,
    WatermarkStore,
)


def _learning(
    learning_id: str,
    *,
    rule: str,
    bm25_text: str,
    scope: str = "session",
    archived_at: str | None = None,
) -> Learning:
    return Learning(
        learning_id=learning_id,
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why="because tests",
        bm25_text=bm25_text,
        archived_at=archived_at,
        created_at="2026-05-22T00:00:00Z",
        updated_at="2026-05-22T00:00:00Z",
    )


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_inmemory_stores_satisfy_protocols() -> None:
    """``runtime_checkable`` Protocols catch contract drift early."""

    assert isinstance(InMemoryWatermarkStore(), WatermarkStore)
    assert isinstance(InMemoryPurposeStore(), PurposeStore)
    assert isinstance(InMemoryDigestStore(), DigestStore)
    assert isinstance(InMemoryLearningStore(), LearningStore)


# --------------------------------------------------------------------------
# WatermarkStore
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watermark_unknown_session_returns_zero() -> None:
    store = InMemoryWatermarkStore()
    assert await store.get("nope") == 0


@pytest.mark.asyncio
async def test_watermark_advance_is_monotonic() -> None:
    store = InMemoryWatermarkStore()
    await store.advance("s-1", to_seq=5, last_run_at="2026-05-22T00:00:00Z")
    assert await store.get("s-1") == 5

    # Lower seq must not roll back.
    await store.advance("s-1", to_seq=3, last_run_at="2026-05-22T00:00:00Z")
    assert await store.get("s-1") == 5

    # Equal seq is a no-op (still monotonic).
    await store.advance("s-1", to_seq=5, last_run_at="2026-05-22T00:00:00Z")
    assert await store.get("s-1") == 5

    await store.advance("s-1", to_seq=10, last_run_at="2026-05-22T00:00:00Z")
    assert await store.get("s-1") == 10


@pytest.mark.asyncio
async def test_watermark_independent_per_session() -> None:
    store = InMemoryWatermarkStore()
    await store.advance("a", to_seq=1, last_run_at="t")
    await store.advance("b", to_seq=99, last_run_at="t")
    assert await store.get("a") == 1
    assert await store.get("b") == 99
    assert await store.snapshot() == {"a": 1, "b": 99}


# --------------------------------------------------------------------------
# PurposeStore
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_purpose_upsert_replaces_previous_row() -> None:
    store = InMemoryPurposeStore()
    p1 = SessionPurpose(session_id="s-1", purpose="initial")
    p2 = SessionPurpose(session_id="s-1", purpose="updated")
    await store.upsert(p1)
    await store.upsert(p2)
    fetched = await store.get("s-1")
    assert fetched is not None
    assert fetched.purpose == "updated"


@pytest.mark.asyncio
async def test_purpose_get_missing_returns_none() -> None:
    store = InMemoryPurposeStore()
    assert await store.get("nope") is None


# --------------------------------------------------------------------------
# DigestStore
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_upsert_stores_embedding_alongside_row() -> None:
    store = InMemoryDigestStore()
    digest = SessionDigest(
        digest_id="d-1",
        session_id="s-1",
        version_id="v1",
        summary_md="# summary",
        bm25_text="summary",
    )
    await store.upsert(digest, embedding=[0.1, 0.2, 0.3])
    fetched = await store.get("s-1")
    assert fetched == digest
    assert await store.get_embedding("s-1") == (0.1, 0.2, 0.3)


@pytest.mark.asyncio
async def test_digest_upsert_overwrites_previous() -> None:
    store = InMemoryDigestStore()
    d1 = SessionDigest(
        digest_id="d-1", session_id="s-1", version_id="v1", summary_md="a", bm25_text="a"
    )
    d2 = SessionDigest(
        digest_id="d-2", session_id="s-1", version_id="v2", summary_md="b", bm25_text="b"
    )
    await store.upsert(d1, embedding=[0.0])
    await store.upsert(d2, embedding=[1.0])
    fetched = await store.get("s-1")
    assert fetched is not None
    assert fetched.digest_id == "d-2"
    assert await store.get_embedding("s-1") == (1.0,)


@pytest.mark.asyncio
async def test_digest_get_missing_returns_none() -> None:
    store = InMemoryDigestStore()
    assert await store.get("nope") is None
    assert await store.get_embedding("nope") is None


# --------------------------------------------------------------------------
# LearningStore basic lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learning_insert_and_get() -> None:
    store = InMemoryLearningStore()
    learning = _learning("l-1", rule="prefer SQL views", bm25_text="prefer sql views")
    await store.insert(learning, embedding=[1.0, 0.0])
    fetched = await store.get("l-1")
    assert fetched == learning


@pytest.mark.asyncio
async def test_learning_update_rule_preserves_immutable_fields() -> None:
    store = InMemoryLearningStore()
    original = _learning("l-1", rule="A", bm25_text="A")
    await store.insert(original, embedding=[1.0, 0.0])

    await store.update_rule(
        "l-1",
        rule="B",
        why="because evidence grew",
        evidence_count=3,
        bm25_text="B B B",
        updated_at="2026-05-22T01:00:00Z",
        embedding=[0.0, 1.0],
    )
    fetched = await store.get("l-1")
    assert fetched is not None
    assert fetched.rule == "B"
    assert fetched.evidence_count == 3
    assert fetched.created_at == original.created_at  # immutable
    assert fetched.updated_at == "2026-05-22T01:00:00Z"
    assert fetched.bm25_text == "B B B"


@pytest.mark.asyncio
async def test_learning_update_rule_on_missing_row_is_noop() -> None:
    store = InMemoryLearningStore()
    await store.update_rule(
        "missing",
        rule="x",
        why="x",
        evidence_count=1,
        bm25_text="x",
        updated_at="t",
        embedding=[0.0],
    )
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_learning_supersede_marks_old_archived_not_deleted() -> None:
    store = InMemoryLearningStore()
    old = _learning("l-old", rule="O", bm25_text="O")
    new = _learning("l-new", rule="N", bm25_text="N")
    await store.insert(old, embedding=[1.0, 0.0])
    await store.insert(new, embedding=[0.0, 1.0])

    await store.supersede(old_id="l-old", new_id="l-new", archived_at="2026-05-22T02:00:00Z")

    fetched_old = await store.get("l-old")
    assert fetched_old is not None
    assert fetched_old.superseded_by == "l-new"
    assert fetched_old.archived_at == "2026-05-22T02:00:00Z"
    # The new row is unaffected.
    fetched_new = await store.get("l-new")
    assert fetched_new is not None
    assert fetched_new.superseded_by is None


@pytest.mark.asyncio
async def test_learning_supersede_missing_old_is_noop() -> None:
    store = InMemoryLearningStore()
    await store.supersede(old_id="nope", new_id="x", archived_at="t")  # must not raise


@pytest.mark.asyncio
async def test_learning_list_active_filters_archived_and_scope() -> None:
    store = InMemoryLearningStore()
    a = _learning("a", rule="x", bm25_text="x", scope="session")
    b = _learning("b", rule="y", bm25_text="y", scope="project")
    c = _learning("c", rule="z", bm25_text="z", scope="session", archived_at="t")
    await store.insert(a, embedding=[1.0, 0.0])
    await store.insert(b, embedding=[0.0, 1.0])
    await store.insert(c, embedding=[0.5, 0.5])

    active_all = await store.list_active()
    assert {row.learning_id for row in active_all} == {"a", "b"}

    active_session = await store.list_active(scope="session")
    assert {row.learning_id for row in active_session} == {"a"}


# --------------------------------------------------------------------------
# LearningStore search_hybrid
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_hybrid_returns_empty_for_empty_store() -> None:
    store = InMemoryLearningStore()
    hits = await store.search_hybrid(query_text="anything", query_vector=[1.0, 0.0], top_k=5)
    assert hits == ()


@pytest.mark.asyncio
async def test_search_hybrid_ranks_vector_match_first() -> None:
    """When the query vector matches one row exactly, RRF must place it on top."""

    store = InMemoryLearningStore()
    target = _learning("target", rule="target", bm25_text="target")
    other = _learning("other", rule="other", bm25_text="other")
    await store.insert(target, embedding=[1.0, 0.0])
    await store.insert(other, embedding=[0.0, 1.0])

    hits = await store.search_hybrid(
        query_text="unrelated keywords",
        query_vector=[1.0, 0.0],
        top_k=2,
    )
    assert [h.learning.learning_id for h in hits] == ["target", "other"]
    assert hits[0].cosine == pytest.approx(1.0)
    assert hits[1].cosine == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_hybrid_uses_bm25_when_vector_is_tied() -> None:
    """If two rows have equal cosine, BM25 overlap must break the tie."""

    store = InMemoryLearningStore()
    bm = _learning("bm", rule="postgres tuning", bm25_text="postgres tuning")
    plain = _learning("plain", rule="other", bm25_text="other")
    # Both vectors are orthogonal to the query → cosine 0 for both.
    await store.insert(bm, embedding=[0.0, 1.0])
    await store.insert(plain, embedding=[0.0, 1.0])

    hits = await store.search_hybrid(
        query_text="postgres",
        query_vector=[1.0, 0.0],
        top_k=2,
    )
    assert hits[0].learning.learning_id == "bm"
    assert hits[0].bm25_rank == 1
    assert hits[1].bm25_rank is None


@pytest.mark.asyncio
async def test_search_hybrid_excludes_archived_learnings() -> None:
    store = InMemoryLearningStore()
    live = _learning("live", rule="live", bm25_text="live")
    dead = _learning("dead", rule="dead", bm25_text="dead", archived_at="t")
    await store.insert(live, embedding=[1.0, 0.0])
    await store.insert(dead, embedding=[1.0, 0.0])

    hits = await store.search_hybrid(query_text="x", query_vector=[1.0, 0.0], top_k=5)
    assert {h.learning.learning_id for h in hits} == {"live"}


@pytest.mark.asyncio
async def test_search_hybrid_filters_by_scope() -> None:
    store = InMemoryLearningStore()
    await store.insert(_learning("s", rule="x", bm25_text="x", scope="session"), embedding=[1.0])
    await store.insert(_learning("p", rule="x", bm25_text="x", scope="project"), embedding=[1.0])

    hits = await store.search_hybrid(query_text="x", query_vector=[1.0], top_k=5, scope="session")
    assert {h.learning.learning_id for h in hits} == {"s"}


@pytest.mark.asyncio
async def test_search_hybrid_top_k_bounds_results() -> None:
    store = InMemoryLearningStore()
    for i in range(5):
        await store.insert(
            _learning(f"l-{i}", rule=f"rule {i}", bm25_text=f"rule {i}"),
            embedding=[1.0, float(i)],
        )
    hits = await store.search_hybrid(query_text="rule", query_vector=[1.0, 0.0], top_k=2)
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_search_hybrid_dimension_mismatch_raises() -> None:
    store = InMemoryLearningStore()
    await store.insert(_learning("x", rule="x", bm25_text="x"), embedding=[1.0, 0.0])

    with pytest.raises(EmbeddingError, match=r"dimension mismatch"):
        await store.search_hybrid(query_text="x", query_vector=[1.0], top_k=1)


@pytest.mark.asyncio
async def test_search_hybrid_zero_norm_vector_yields_zero_cosine() -> None:
    store = InMemoryLearningStore()
    await store.insert(_learning("x", rule="x", bm25_text="x"), embedding=[0.0, 0.0])
    hits = await store.search_hybrid(query_text="unrelated", query_vector=[1.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].cosine == 0.0
