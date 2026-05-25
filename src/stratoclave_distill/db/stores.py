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
from typing import Literal, Protocol, runtime_checkable

from stratoclave_distill.core.types import (
    BranchState,
    ConflictResolution,
    GroupLearning,
    Learning,
    LearningConflict,
    LearningScope,
    SessionDigest,
    SessionGap,
    SessionPurpose,
)

RetrievalLane = Literal["canonical", "emerging", "all"]
"""Stage B+ retrieval lane.

- ``canonical`` — long-lived, well-attested rules. The retriever filters to
  ``evidence_count >= canonical_min_evidence`` AND created at least
  ``canonical_min_age_days`` ago AND ``scope != 'experiment'``.
- ``emerging`` — everything else still active.
- ``all`` — preserves the Stage B behaviour for callers that have not opted
  into lane filtering yet.
"""


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


@dataclass(frozen=True, slots=True)
class GroupLearningSearchHit:
    """One result from :meth:`GroupLearningStore.search_hybrid`.

    Mirrors :class:`LearningSearchHit` so the Retriever / Packer can treat
    rollups uniformly with per-row learnings. Group rollups are always
    treated as canonical (no lane gating), so there is no equivalent of
    ``RetrievalLane`` here.
    """

    group: GroupLearning
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

    Stage B+ adds branching helpers (:meth:`set_branch_state`,
    :meth:`list_branches`) so the CLI can manage the open/closed/promoted
    lifecycle without round-tripping through ``upsert``.
    """

    async def upsert(self, purpose: SessionPurpose) -> None: ...

    async def get(self, session_id: str) -> SessionPurpose | None: ...

    async def set_branch_state(
        self,
        session_id: str,
        *,
        branch_state: BranchState,
        closed_at: str | None,
        last_updated_at: str,
    ) -> None: ...

    async def list_branches(
        self,
        *,
        parent_session_id: str | None = None,
    ) -> Sequence[SessionPurpose]: ...


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
        lane: RetrievalLane = "all",
        canonical_min_evidence: int = 3,
        canonical_min_age_days: int = 14,
    ) -> Sequence[LearningSearchHit]: ...


@runtime_checkable
class GroupLearningStore(Protocol):
    """Stores Aggregator-produced :class:`GroupLearning` rollups.

    A group rollup is a single LLM-rewritten summary of every active
    learning that shares a ``group_id``. The store keeps the rollup, its
    embedding, and the list of contributing ``learning_id``s so the
    Retriever can join back to the source rows.

    ``upsert`` is keyed on ``group_learning_id`` (one rollup per
    Aggregator run) — a fresh re-aggregation produces a new id and
    overwrites the previous row for the same ``group_id`` via
    :meth:`list_by_group` semantics (the caller deletes the old one or
    the store keeps both and the retriever reads the latest by
    ``created_at``). The asyncpg implementation chooses the latter so
    the audit trail is preserved.

    ``search_hybrid`` mirrors :meth:`LearningStore.search_hybrid` minus
    the lane / scope arguments: groups have no scope and are always
    canonical.
    """

    async def upsert(
        self,
        group_learning: GroupLearning,
        *,
        embedding: Sequence[float],
    ) -> None: ...

    async def get(self, group_learning_id: str) -> GroupLearning | None: ...

    async def list_by_group(
        self,
        group_id: str,
        *,
        latest_only: bool = True,
    ) -> Sequence[GroupLearning]: ...

    async def list_latest_per_group(self) -> Sequence[GroupLearning]: ...

    async def search_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> Sequence[GroupLearningSearchHit]: ...


@runtime_checkable
class ConflictStore(Protocol):
    """Stores rows of :class:`LearningConflict`.

    The Curator writes here whenever the ``CONFLICT_NOTED`` action fires
    (or when ``SUPERSEDE`` runs and we want to record *why* a row was
    archived). The retriever reads via :meth:`list_open` / :meth:`list_for`
    so it can flag contested rules cheaply.
    """

    async def insert(self, conflict: LearningConflict) -> None: ...

    async def get(self, conflict_id: str) -> LearningConflict | None: ...

    async def list_open(self) -> Sequence[LearningConflict]: ...

    async def list_for(self, learning_id: str) -> Sequence[LearningConflict]: ...

    async def resolve(
        self,
        conflict_id: str,
        *,
        resolution: ConflictResolution,
    ) -> None: ...


@runtime_checkable
class GapStore(Protocol):
    """Stores rows of :class:`SessionGap`.

    Gaps are unresolved questions a session noted but did not answer.
    The pipeline writes them via :meth:`insert`; the retriever reads via
    :meth:`list_unresolved` so prompts can reference open questions.
    """

    async def insert(self, gap: SessionGap) -> None: ...

    async def get(self, gap_id: str) -> SessionGap | None: ...

    async def list_unresolved(
        self,
        *,
        session_id: str | None = None,
    ) -> Sequence[SessionGap]: ...

    async def resolve(
        self,
        gap_id: str,
        *,
        resolved_at: str,
        resolved_by_learning: str | None,
    ) -> None: ...


__all__ = [
    "ConflictStore",
    "DigestStore",
    "GapStore",
    "GroupLearningSearchHit",
    "GroupLearningStore",
    "LearningSearchHit",
    "LearningStore",
    "PurposeStore",
    "RetrievalLane",
    "WatermarkStore",
]
