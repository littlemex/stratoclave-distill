"""Stage B+ retriever — splits hits into canonical / emerging lanes.

The retriever sits in front of :meth:`LearningStore.search_hybrid` and adds
two pieces of structure that Stage B did not have:

1. Lane separation. Stage B returned a single ranked list; Stage B+ splits
   it into ``canonical`` (well-attested, stable) and ``emerging`` (recent,
   under-attested, or experiment-scoped). The lane predicate lives on the
   store so Postgres can apply it inside the SQL filter; the retriever
   simply issues two queries with ``lane="canonical"`` and
   ``lane="emerging"``.
2. Optional sidecar reads for open conflicts and unresolved gaps. The
   ContextPacker (Stage C) formats these alongside the canonical /
   emerging hits so prompts can flag contested rules and open questions.

The retriever is intentionally thin: no fusion logic, no prompt formatting,
no truncation. The Stage C ContextPacker layers those concerns on top of
:class:`RetrievalResult`.

Embedding is delegated to an :class:`EmbeddingProvider` so the retriever
does not bake in a particular vendor; tests inject a stub provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.core.types import (
    LearningConflict,
    LearningScope,
    SessionGap,
)
from stratoclave_distill.db.stores import (
    ConflictStore,
    GapStore,
    GroupLearningSearchHit,
    GroupLearningStore,
    LearningSearchHit,
    LearningStore,
)
from stratoclave_distill.providers.embedding import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The structured payload a Stage C packer reads.

    ``canonical`` and ``emerging`` carry the same :class:`LearningSearchHit`
    shape; the split is the whole point of Stage B+. ``conflicts`` and
    ``gaps`` are sidecars: empty tuples when the retriever was constructed
    without the corresponding stores. ``groups`` carries the latest rollup
    per ``group_id`` from the Stage D Aggregator and is empty when no
    :class:`GroupLearningStore` is wired.
    """

    query_text: str
    canonical: tuple[LearningSearchHit, ...]
    emerging: tuple[LearningSearchHit, ...]
    conflicts: tuple[LearningConflict, ...]
    gaps: tuple[SessionGap, ...]
    groups: tuple[GroupLearningSearchHit, ...] = ()

    @property
    def all_hits(self) -> tuple[LearningSearchHit, ...]:
        """Both lanes concatenated, canonical first.

        Useful for callers that want the union (e.g. quick smoke tests).
        Stage C should prefer reading the lanes separately.
        """

        return self.canonical + self.emerging


class Retriever:
    """Embed a query, fan out to canonical / emerging lanes, return both.

    Parameters
    ----------
    store:
        :class:`LearningStore` whose ``search_hybrid`` honors the ``lane``
        argument. Both the in-memory and asyncpg implementations qualify.
    embedder:
        :class:`EmbeddingProvider` used to vectorize the query. The
        retriever does not cache; callers that want caching should wrap
        the provider.
    top_k_canonical / top_k_emerging:
        Per-lane limits. Defaults to 5 each so the ContextPacker has
        enough material without blowing the prompt budget.
    rrf_k:
        Forwarded to :meth:`LearningStore.search_hybrid`.
    canonical_min_evidence / canonical_min_age_days:
        Forwarded as the lane predicate parameters. The defaults match
        :class:`LearningStore.search_hybrid` defaults so callers do not
        need to know the threshold values.
    conflict_store / gap_store:
        Optional. When wired, :meth:`retrieve` populates ``conflicts`` and
        ``gaps`` on the result; otherwise those fields stay empty.
    group_learning_store / top_k_groups:
        Optional Stage D rollup store. When wired, :meth:`retrieve` issues
        a parallel hybrid search against the group rollups and surfaces
        the top ``top_k_groups`` hits on :attr:`RetrievalResult.groups`.
        Defaults to ``None`` so existing callers keep working untouched.
    """

    __slots__ = (
        "_canonical_min_age_days",
        "_canonical_min_evidence",
        "_conflict_store",
        "_embedder",
        "_gap_store",
        "_group_learning_store",
        "_rrf_k",
        "_store",
        "_top_k_canonical",
        "_top_k_emerging",
        "_top_k_groups",
    )

    def __init__(
        self,
        store: LearningStore,
        embedder: EmbeddingProvider,
        *,
        top_k_canonical: int = 5,
        top_k_emerging: int = 5,
        top_k_groups: int = 3,
        rrf_k: int = 60,
        canonical_min_evidence: int = 3,
        canonical_min_age_days: int = 14,
        conflict_store: ConflictStore | None = None,
        gap_store: GapStore | None = None,
        group_learning_store: GroupLearningStore | None = None,
    ) -> None:
        if top_k_canonical < 1:
            raise ValueError(f"top_k_canonical must be >= 1, got {top_k_canonical}")
        if top_k_emerging < 1:
            raise ValueError(f"top_k_emerging must be >= 1, got {top_k_emerging}")
        if top_k_groups < 1:
            raise ValueError(f"top_k_groups must be >= 1, got {top_k_groups}")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
        if canonical_min_evidence < 1:
            raise ValueError(f"canonical_min_evidence must be >= 1, got {canonical_min_evidence}")
        if canonical_min_age_days < 0:
            raise ValueError(f"canonical_min_age_days must be >= 0, got {canonical_min_age_days}")
        self._store = store
        self._embedder = embedder
        self._top_k_canonical = top_k_canonical
        self._top_k_emerging = top_k_emerging
        self._top_k_groups = top_k_groups
        self._rrf_k = rrf_k
        self._canonical_min_evidence = canonical_min_evidence
        self._canonical_min_age_days = canonical_min_age_days
        self._conflict_store = conflict_store
        self._gap_store = gap_store
        self._group_learning_store = group_learning_store

    async def retrieve(
        self,
        query_text: str,
        *,
        scope: LearningScope | None = None,
        gap_session_id: str | None = None,
        source_session_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Embed ``query_text`` and return canonical / emerging hits.

        Parameters
        ----------
        query_text:
            Text query that drives both BM25 and embedding lanes.
        scope:
            Optional scope filter forwarded to ``search_hybrid``. ``None``
            keeps the default Stage B behaviour of searching all scopes.
        gap_session_id:
            Optional session id passed to :meth:`GapStore.list_unresolved`
            so prompts can surface session-local open questions. ``None``
            asks for global unresolved gaps.
        source_session_ids:
            Optional restriction: only consider learnings whose
            ``source_session`` is in this list. Used by atelier's "ask
            another session" path so retrieval is scoped to the user's
            picked sessions.
        """

        if not query_text:
            raise ValueError("query_text must be a non-empty string")
        vectors = await self._embedder.embed([query_text])
        if not vectors:
            raise EmbeddingError("embedder returned no vectors for the query")
        query_vector = vectors[0]

        canonical = await self._store.search_hybrid(
            query_text=query_text,
            query_vector=query_vector,
            top_k=self._top_k_canonical,
            rrf_k=self._rrf_k,
            scope=scope,
            lane="canonical",
            canonical_min_evidence=self._canonical_min_evidence,
            canonical_min_age_days=self._canonical_min_age_days,
            source_session_ids=source_session_ids,
        )
        emerging = await self._store.search_hybrid(
            query_text=query_text,
            query_vector=query_vector,
            top_k=self._top_k_emerging,
            rrf_k=self._rrf_k,
            scope=scope,
            lane="emerging",
            canonical_min_evidence=self._canonical_min_evidence,
            canonical_min_age_days=self._canonical_min_age_days,
            source_session_ids=source_session_ids,
        )

        conflicts: tuple[LearningConflict, ...] = ()
        if self._conflict_store is not None:
            conflicts = tuple(await self._conflict_store.list_open())

        gaps: tuple[SessionGap, ...] = ()
        if self._gap_store is not None:
            gaps = tuple(await self._gap_store.list_unresolved(session_id=gap_session_id))

        groups: tuple[GroupLearningSearchHit, ...] = ()
        if self._group_learning_store is not None:
            group_hits = await self._group_learning_store.search_hybrid(
                query_text=query_text,
                query_vector=query_vector,
                top_k=self._top_k_groups,
                rrf_k=self._rrf_k,
            )
            groups = tuple(group_hits)

        return RetrievalResult(
            query_text=query_text,
            canonical=tuple(canonical),
            emerging=tuple(emerging),
            conflicts=conflicts,
            gaps=gaps,
            groups=groups,
        )


def hits_for_learning(
    hits: Sequence[LearningSearchHit],
    learning_id: str,
) -> LearningSearchHit | None:
    """Test helper: locate the hit referencing ``learning_id``.

    Returns ``None`` when the id is not in ``hits``. Kept here rather than
    in tests so the helper has a single source of truth across unit and
    integration suites.
    """

    for hit in hits:
        if hit.learning.learning_id == learning_id:
            return hit
    return None


def learning_ids(hits: Sequence[LearningSearchHit]) -> tuple[str, ...]:
    """Extract ``(learning_id, ...)`` from a hit sequence in order."""

    return tuple(hit.learning.learning_id for hit in hits)


__all__ = [
    "RetrievalResult",
    "Retriever",
    "hits_for_learning",
    "learning_ids",
]
