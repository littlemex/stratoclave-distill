"""Persistence interfaces for stratoclave-distill.

Stage B persists four kinds of state:

- per-session **watermarks** (the highest ``seq`` already distilled, used to
  drive incremental ingest);
- per-session **purposes** (role, tags, pollution flag);
- per-session **digests** (compact summaries with embeddings + BM25 text);
- a flat collection of **learnings** with vector + BM25 indexes plus
  conflict-resolution lifecycle (``insert`` / ``merge`` / ``supersede``).

All four are exposed as :class:`typing.Protocol` so callers (Distiller,
Curator) can plug in either the in-memory implementation used by unit tests
or the asyncpg-backed implementation used in production. Returning
:class:`stratoclave_distill.core.types.Learning` and friends instead of raw
mappings keeps the shape stable as the SQL evolves.

The stores are deliberately small: nothing about prompt construction,
retry, or fusion lives here. That logic is the pipeline's job; the stores
just hold rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from stratoclave_distill.core.types import (
    Learning,
    LearningScope,
    SessionDigest,
    SessionPurpose,
)


@dataclass(frozen=True, slots=True)
class LearningSearchHit:
    """One result from :meth:`LearningStore.search_hybrid`.

    Carries both the raw cosine similarity (the Curator's decision input)
    and the RRF-fused score (the Retriever's ranking input) so callers can
    pick whichever signal they need without re-running the search.

    ``vector_rank`` is always populated; ``bm25_rank`` is ``None`` when
    BM25 returned no match for this row, which lets the Curator distinguish
    "matched only by vector" from "matched by both" when the cosine score
    is borderline.
    """

    learning: Learning
    cosine: float
    vector_rank: int
    bm25_rank: int | None
    rrf_score: float


@runtime_checkable
class WatermarkStore(Protocol):
    """Tracks the highest ``seq`` already distilled per session.

    ``get`` returns ``0`` (not raises) for an unknown session so the
    Distiller can treat first-time sessions and re-runs uniformly: read
    the current watermark, ingest everything strictly above it, advance.
    """

    async def get(self, session_id: str) -> int: ...

    async def advance(self, session_id: str, *, to_seq: int, last_run_at: str) -> None: ...


@runtime_checkable
class PurposeStore(Protocol):
    """Stores at most one :class:`SessionPurpose` per session id.

    ``upsert`` is idempotent: re-distilling a session must not create a
    second row, and must update ``last_updated_at`` (the caller is
    expected to set the timestamp on the dataclass).
    """

    async def upsert(self, purpose: SessionPurpose) -> None: ...

    async def get(self, session_id: str) -> SessionPurpose | None: ...


@runtime_checkable
class DigestStore(Protocol):
    """Stores at most one :class:`SessionDigest` per session id.

    The embedding is passed alongside the digest because :class:`SessionDigest`
    is a public, JSON-friendly dataclass that does not carry vectors. The
    store keeps the two associated internally.
    """

    async def upsert(self, digest: SessionDigest, *, embedding: Sequence[float]) -> None: ...

    async def get(self, session_id: str) -> SessionDigest | None: ...


@runtime_checkable
class LearningStore(Protocol):
    """The richest store: insert, lifecycle transitions, and hybrid search.

    The lifecycle methods (``update_rule``, ``supersede``) never delete
    rows so that audit trails stay intact; ``supersede`` sets
    ``superseded_by`` and ``archived_at`` on the old row instead.
    ``list_active`` filters out archived rows.

    ``search_hybrid`` is the Curator's single dependency: it returns the
    top-K candidate Learnings ranked by Reciprocal Rank Fusion of the
    vector and BM25 modalities, with the raw cosine similarity preserved
    so the Curator can apply ``tau_merge`` / ``tau_conflict`` thresholds.
    """

    async def insert(self, learning: Learning, *, embedding: Sequence[float]) -> None: ...

    async def get(self, learning_id: str) -> Learning | None: ...

    async def update_rule(
        self,
        learning_id: str,
        *,
        rule: str,
        why: str,
        evidence_count: int,
        bm25_text: str,
        updated_at: str,
        embedding: Sequence[float],
    ) -> None: ...

    async def supersede(self, *, old_id: str, new_id: str, archived_at: str) -> None: ...

    async def list_active(self, *, scope: LearningScope | None = None) -> Sequence[Learning]: ...

    async def search_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        rrf_k: int = 60,
        scope: LearningScope | None = None,
    ) -> Sequence[LearningSearchHit]: ...


__all__ = [
    "DigestStore",
    "LearningSearchHit",
    "LearningStore",
    "PurposeStore",
    "WatermarkStore",
]
