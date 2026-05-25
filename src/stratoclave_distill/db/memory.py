"""In-memory implementations of the Stage B stores.

Used for unit tests, the offline ``examples/quickstart.py`` flow, and any
caller that wants to exercise Distiller / Curator without standing up
Postgres. The implementations are deliberately simple: dict-backed,
single-process, and protected by an :class:`asyncio.Lock` so that the
same instance can be shared across coroutines without races.

The hybrid search implementation runs both modalities in pure Python:

- vector: cosine similarity against every stored embedding;
- BM25: tokenize on whitespace, count overlapping tokens with a TF
  weighting that mirrors what Postgres ``ts_rank_cd`` would surface for
  short fixtures.

The goal is not to replicate Postgres byte-for-byte; it is to give the
Curator's logic something realistic enough that the unit tests catch
*ordering* mistakes (e.g. forgetting to apply the cosine threshold
before invoking RRF). Production code uses the asyncpg implementation.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from stratoclave_distill.core.errors import EmbeddingError
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
from stratoclave_distill.db.stores import (
    GroupLearningSearchHit,
    LearningSearchHit,
    RetrievalLane,
)


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 ``...Z`` timestamp; return ``None`` for empty / None."""

    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Returns 0 when either side is zero."""

    if len(a) != len(b):
        raise EmbeddingError(f"cosine: dimension mismatch {len(a)} vs {len(b)}")
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_lite(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    """Count overlapping tokens with TF weighting. Returns 0 for no overlap.

    This is intentionally not a real BM25; it is a deterministic surrogate
    that ranks the same way for the small fixtures used by unit tests.
    """

    if not query_tokens or not doc_tokens:
        return 0.0
    doc_freq: dict[str, int] = {}
    for token in doc_tokens:
        doc_freq[token] = doc_freq.get(token, 0) + 1
    score = 0.0
    for q in query_tokens:
        if q in doc_freq:
            score += 1.0 + math.log(1.0 + doc_freq[q])
    return score


class InMemoryWatermarkStore:
    """Per-session watermark tracker.

    Returns ``0`` for unknown sessions so the Distiller can treat
    first-time and incremental ingests uniformly.
    """

    __slots__ = ("_lock", "_marks")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._marks: dict[str, int] = {}

    async def get(self, session_id: str) -> int:
        async with self._lock:
            return self._marks.get(session_id, 0)

    async def advance(self, session_id: str, *, to_seq: int, last_run_at: str) -> None:
        del last_run_at  # surfaces for the asyncpg variant; ignored here
        async with self._lock:
            current = self._marks.get(session_id, 0)
            if to_seq > current:
                self._marks[session_id] = to_seq

    async def snapshot(self) -> dict[str, int]:
        """Test helper: return a copy of all watermarks."""

        async with self._lock:
            return dict(self._marks)


class InMemoryPurposeStore:
    """Single-row-per-session :class:`SessionPurpose` store.

    Stage B+ exposes branch lifecycle helpers
    (:meth:`set_branch_state`, :meth:`list_branches`) so the CLI can
    transition open / closed / promoted without reissuing a full upsert.
    """

    __slots__ = ("_lock", "_rows")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, SessionPurpose] = {}

    async def upsert(self, purpose: SessionPurpose) -> None:
        async with self._lock:
            self._rows[purpose.session_id] = purpose

    async def get(self, session_id: str) -> SessionPurpose | None:
        async with self._lock:
            return self._rows.get(session_id)

    async def set_branch_state(
        self,
        session_id: str,
        *,
        branch_state: BranchState,
        closed_at: str | None,
        last_updated_at: str,
    ) -> None:
        async with self._lock:
            row = self._rows.get(session_id)
            if row is None:
                return
            self._rows[session_id] = SessionPurpose(
                session_id=row.session_id,
                purpose=row.purpose,
                domain_tags=row.domain_tags,
                success_score=row.success_score,
                polluted=row.polluted,
                pollution_reason=row.pollution_reason,
                derived_from_version=row.derived_from_version,
                derived_at=row.derived_at,
                last_updated_at=last_updated_at,
                parent_session_id=row.parent_session_id,
                branched_at_seq=row.branched_at_seq,
                branch_kind=row.branch_kind,
                branch_state=branch_state,
                closed_at=closed_at,
            )

    async def list_branches(
        self,
        *,
        parent_session_id: str | None = None,
    ) -> Sequence[SessionPurpose]:
        async with self._lock:
            if parent_session_id is None:
                return tuple(self._rows.values())
            return tuple(
                row for row in self._rows.values() if row.parent_session_id == parent_session_id
            )


class InMemoryDigestStore:
    """Single-row-per-session :class:`SessionDigest` store with embeddings."""

    __slots__ = ("_embeddings", "_lock", "_rows")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, SessionDigest] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}

    async def upsert(self, digest: SessionDigest, *, embedding: Sequence[float]) -> None:
        async with self._lock:
            self._rows[digest.session_id] = digest
            self._embeddings[digest.session_id] = tuple(embedding)

    async def get(self, session_id: str) -> SessionDigest | None:
        async with self._lock:
            return self._rows.get(session_id)

    async def get_embedding(self, session_id: str) -> tuple[float, ...] | None:
        """Test helper: expose the stored embedding vector."""

        async with self._lock:
            return self._embeddings.get(session_id)


class InMemoryLearningStore:
    """Hybrid in-memory :class:`LearningStore`.

    Holds learnings keyed by ``learning_id`` along with their embeddings
    and pre-tokenized BM25 text. Lifecycle methods preserve audit trail:
    :meth:`supersede` sets ``superseded_by`` / ``archived_at`` on the old
    row instead of deleting it.
    """

    __slots__ = ("_embeddings", "_lock", "_rows", "_tokens")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, Learning] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._tokens: dict[str, list[str]] = {}

    async def insert(self, learning: Learning, *, embedding: Sequence[float]) -> None:
        async with self._lock:
            self._rows[learning.learning_id] = learning
            self._embeddings[learning.learning_id] = tuple(embedding)
            self._tokens[learning.learning_id] = _tokenize(learning.bm25_text or learning.rule)

    async def get(self, learning_id: str) -> Learning | None:
        async with self._lock:
            return self._rows.get(learning_id)

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
    ) -> None:
        async with self._lock:
            row = self._rows.get(learning_id)
            if row is None:
                return
            new = Learning(
                learning_id=row.learning_id,
                scope=row.scope,
                rule=rule,
                why=why,
                triggers=row.triggers,
                project_key=row.project_key,
                group_id=row.group_id,
                source_session=row.source_session,
                source_version=row.source_version,
                evidence_count=evidence_count,
                confidence=row.confidence,
                archived_at=row.archived_at,
                superseded_by=row.superseded_by,
                bm25_text=bm25_text,
                created_at=row.created_at,
                updated_at=updated_at,
                claim_type=row.claim_type,
            )
            self._rows[learning_id] = new
            self._embeddings[learning_id] = tuple(embedding)
            self._tokens[learning_id] = _tokenize(bm25_text)

    async def supersede(self, *, old_id: str, new_id: str, archived_at: str) -> None:
        async with self._lock:
            old = self._rows.get(old_id)
            if old is None:
                return
            self._rows[old_id] = Learning(
                learning_id=old.learning_id,
                scope=old.scope,
                rule=old.rule,
                why=old.why,
                triggers=old.triggers,
                project_key=old.project_key,
                group_id=old.group_id,
                source_session=old.source_session,
                source_version=old.source_version,
                evidence_count=old.evidence_count,
                confidence=old.confidence,
                archived_at=archived_at,
                superseded_by=new_id,
                bm25_text=old.bm25_text,
                created_at=old.created_at,
                updated_at=archived_at,
                claim_type=old.claim_type,
            )

    async def list_active(self, *, scope: LearningScope | None = None) -> Sequence[Learning]:
        async with self._lock:
            return tuple(
                row
                for row in self._rows.values()
                if row.archived_at is None and (scope is None or row.scope == scope)
            )

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
    ) -> Sequence[LearningSearchHit]:
        cutoff = datetime.now(UTC) - timedelta(days=canonical_min_age_days)

        def _row_in_lane(row: Learning) -> bool:
            if lane == "all":
                return True
            created = _parse_iso(row.created_at)
            is_canonical = (
                row.evidence_count >= canonical_min_evidence
                and created is not None
                and created <= cutoff
                and row.scope != "experiment"
            )
            if lane == "canonical":
                return is_canonical
            return not is_canonical

        async with self._lock:
            active = [
                (lid, row)
                for lid, row in self._rows.items()
                if row.archived_at is None
                and (scope is None or row.scope == scope)
                and _row_in_lane(row)
            ]
            if not active:
                return ()

            cosines: dict[str, float] = {}
            for lid, _row in active:
                cosines[lid] = _cosine(query_vector, self._embeddings[lid])

            query_tokens = _tokenize(query_text)
            bm25: dict[str, float] = {}
            for lid, _row in active:
                bm25[lid] = _bm25_lite(query_tokens, self._tokens.get(lid, []))

        # Build per-modality ranks (descending score, lower rank = better).
        # 1-indexed ranks because RRF expects ``1 / (rrf_k + rank)``.
        vec_ranked = sorted(cosines.items(), key=lambda kv: kv[1], reverse=True)
        bm25_ranked = sorted(
            (kv for kv in bm25.items() if kv[1] > 0.0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        vec_rank: dict[str, int] = {lid: i + 1 for i, (lid, _) in enumerate(vec_ranked)}
        bm25_rank: dict[str, int] = {lid: i + 1 for i, (lid, _) in enumerate(bm25_ranked)}

        hits: list[LearningSearchHit] = []
        for lid, row in active:
            vr = vec_rank[lid]
            br = bm25_rank.get(lid)
            rrf = 1.0 / (rrf_k + vr)
            if br is not None:
                rrf += 1.0 / (rrf_k + br)
            hits.append(
                LearningSearchHit(
                    learning=row,
                    cosine=cosines[lid],
                    vector_rank=vr,
                    bm25_rank=br,
                    rrf_score=rrf,
                )
            )

        hits.sort(key=lambda h: h.rrf_score, reverse=True)
        return tuple(hits[:top_k])


class InMemoryGroupLearningStore:
    """In-memory :class:`GroupLearningStore` keyed by ``group_learning_id``.

    Stores embeddings and pre-tokenized BM25 text alongside each rollup so
    that :meth:`search_hybrid` can mirror the asyncpg shape: vector
    similarity + BM25 token overlap fused via Reciprocal Rank Fusion.

    ``list_by_group`` with ``latest_only=True`` returns only the most
    recently created rollup for a group_id, which is the shape the
    Retriever wants when packing a context bundle. ``latest_only=False``
    surfaces the full history for audit / debugging.
    """

    __slots__ = ("_embeddings", "_lock", "_rows", "_tokens")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, GroupLearning] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._tokens: dict[str, list[str]] = {}

    async def upsert(
        self,
        group_learning: GroupLearning,
        *,
        embedding: Sequence[float],
    ) -> None:
        async with self._lock:
            self._rows[group_learning.group_learning_id] = group_learning
            self._embeddings[group_learning.group_learning_id] = tuple(embedding)
            self._tokens[group_learning.group_learning_id] = _tokenize(
                group_learning.bm25_text or group_learning.summary_md
            )

    async def get(self, group_learning_id: str) -> GroupLearning | None:
        async with self._lock:
            return self._rows.get(group_learning_id)

    async def list_by_group(
        self,
        group_id: str,
        *,
        latest_only: bool = True,
    ) -> Sequence[GroupLearning]:
        async with self._lock:
            matches = [row for row in self._rows.values() if row.group_id == group_id]
        if not matches:
            return ()
        matches.sort(key=lambda r: r.created_at, reverse=True)
        if latest_only:
            return (matches[0],)
        return tuple(matches)

    async def list_latest_per_group(self) -> Sequence[GroupLearning]:
        """Return the newest rollup per ``group_id``, sorted by ``created_at`` DESC."""

        async with self._lock:
            latest: dict[str, GroupLearning] = {}
            for row in self._rows.values():
                cur = latest.get(row.group_id)
                if cur is None or row.created_at > cur.created_at:
                    latest[row.group_id] = row
        return tuple(sorted(latest.values(), key=lambda r: r.created_at, reverse=True))

    async def search_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> Sequence[GroupLearningSearchHit]:
        async with self._lock:
            if not self._rows:
                return ()
            # Group rollups have no archived state; every row participates.
            # When multiple rollups exist for the same group_id we keep
            # only the latest so the search surface mirrors the
            # ``latest_only=True`` retrieval default.
            latest_per_group: dict[str, GroupLearning] = {}
            for row in self._rows.values():
                cur = latest_per_group.get(row.group_id)
                if cur is None or row.created_at > cur.created_at:
                    latest_per_group[row.group_id] = row
            active = [(row.group_learning_id, row) for row in latest_per_group.values()]

            cosines: dict[str, float] = {}
            for gid, _row in active:
                cosines[gid] = _cosine(query_vector, self._embeddings[gid])

            query_tokens = _tokenize(query_text)
            bm25: dict[str, float] = {}
            for gid, _row in active:
                bm25[gid] = _bm25_lite(query_tokens, self._tokens.get(gid, []))

        vec_ranked = sorted(cosines.items(), key=lambda kv: kv[1], reverse=True)
        bm25_ranked = sorted(
            (kv for kv in bm25.items() if kv[1] > 0.0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        vec_rank: dict[str, int] = {gid: i + 1 for i, (gid, _) in enumerate(vec_ranked)}
        bm25_rank: dict[str, int] = {gid: i + 1 for i, (gid, _) in enumerate(bm25_ranked)}

        hits: list[GroupLearningSearchHit] = []
        for gid, row in active:
            vr = vec_rank[gid]
            br = bm25_rank.get(gid)
            rrf = 1.0 / (rrf_k + vr)
            if br is not None:
                rrf += 1.0 / (rrf_k + br)
            hits.append(
                GroupLearningSearchHit(
                    group=row,
                    cosine=cosines[gid],
                    vector_rank=vr,
                    bm25_rank=br,
                    rrf_score=rrf,
                )
            )

        hits.sort(key=lambda h: h.rrf_score, reverse=True)
        return tuple(hits[:top_k])


class InMemoryConflictStore:
    """In-memory :class:`ConflictStore` keyed by ``conflict_id``."""

    __slots__ = ("_lock", "_rows")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, LearningConflict] = {}

    async def insert(self, conflict: LearningConflict) -> None:
        async with self._lock:
            self._rows[conflict.conflict_id] = conflict

    async def get(self, conflict_id: str) -> LearningConflict | None:
        async with self._lock:
            return self._rows.get(conflict_id)

    async def list_open(self) -> Sequence[LearningConflict]:
        async with self._lock:
            return tuple(c for c in self._rows.values() if c.resolution == "open")

    async def list_for(self, learning_id: str) -> Sequence[LearningConflict]:
        async with self._lock:
            return tuple(
                c for c in self._rows.values() if c.from_id == learning_id or c.to_id == learning_id
            )

    async def resolve(
        self,
        conflict_id: str,
        *,
        resolution: ConflictResolution,
    ) -> None:
        async with self._lock:
            row = self._rows.get(conflict_id)
            if row is None:
                return
            self._rows[conflict_id] = LearningConflict(
                conflict_id=row.conflict_id,
                from_id=row.from_id,
                to_id=row.to_id,
                reason=row.reason,
                cosine_at_detection=row.cosine_at_detection,
                detected_at=row.detected_at,
                resolution=resolution,
            )


class InMemoryGapStore:
    """In-memory :class:`GapStore` keyed by ``gap_id``."""

    __slots__ = ("_lock", "_rows")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._rows: dict[str, SessionGap] = {}

    async def insert(self, gap: SessionGap) -> None:
        async with self._lock:
            self._rows[gap.gap_id] = gap

    async def get(self, gap_id: str) -> SessionGap | None:
        async with self._lock:
            return self._rows.get(gap_id)

    async def list_unresolved(
        self,
        *,
        session_id: str | None = None,
    ) -> Sequence[SessionGap]:
        async with self._lock:
            return tuple(
                g
                for g in self._rows.values()
                if g.resolved_at is None and (session_id is None or g.session_id == session_id)
            )

    async def resolve(
        self,
        gap_id: str,
        *,
        resolved_at: str,
        resolved_by_learning: str | None,
    ) -> None:
        async with self._lock:
            row = self._rows.get(gap_id)
            if row is None:
                return
            self._rows[gap_id] = SessionGap(
                gap_id=row.gap_id,
                session_id=row.session_id,
                topic=row.topic,
                why_unknown=row.why_unknown,
                bm25_text=row.bm25_text,
                detected_at=row.detected_at,
                resolved_at=resolved_at,
                resolved_by_learning=resolved_by_learning,
            )


__all__ = [
    "InMemoryConflictStore",
    "InMemoryDigestStore",
    "InMemoryGapStore",
    "InMemoryGroupLearningStore",
    "InMemoryLearningStore",
    "InMemoryPurposeStore",
    "InMemoryWatermarkStore",
]
