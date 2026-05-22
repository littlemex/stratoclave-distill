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

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.core.types import (
    Learning,
    LearningScope,
    SessionDigest,
    SessionPurpose,
)
from stratoclave_distill.db.stores import LearningSearchHit


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
    """Single-row-per-session :class:`SessionPurpose` store."""

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
    ) -> Sequence[LearningSearchHit]:
        async with self._lock:
            active = [
                (lid, row)
                for lid, row in self._rows.items()
                if row.archived_at is None and (scope is None or row.scope == scope)
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


__all__ = [
    "InMemoryDigestStore",
    "InMemoryLearningStore",
    "InMemoryPurposeStore",
    "InMemoryWatermarkStore",
]
