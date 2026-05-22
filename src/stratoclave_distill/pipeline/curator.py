"""Conflict resolution for candidate learnings.

The Curator is the second stage of Stage B: it takes the candidates
the :class:`Distiller` emitted and decides, for each one, whether to:

- ``MERGE`` it into a near-duplicate existing row (cosine >= ``tau_merge``),
  by calling :meth:`LearningStore.update_rule` with a bumped
  ``evidence_count`` and a refreshed ``why`` / ``bm25_text``;
- ``SUPERSEDE`` a similar-but-not-identical existing row (cosine in
  ``[tau_conflict, tau_merge)``), by inserting the candidate as a new
  row and calling :meth:`LearningStore.supersede` to point the old row
  at the new one (audit-trail preserved, never deleted);
- ``INSERT`` it as a fresh row (cosine < ``tau_conflict``).

The thresholds come from :class:`DistillerConfig` so a deployment can
tune them without touching pipeline code. ``search_hybrid`` is the
Curator's only read against the store; everything else is local
arithmetic.

The Curator does **not** mint new identities for MERGE outcomes. The
candidate's freshly minted ``learning_id`` is discarded when the
decision is MERGE; this keeps callers from accidentally persisting
both rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from stratoclave_distill.core.types import Learning, LearningScope
from stratoclave_distill.db.stores import LearningStore
from stratoclave_distill.pipeline.distiller import CandidateLearning

CurationAction = Literal["INSERT", "MERGE", "SUPERSEDE"]


@dataclass(frozen=True, slots=True)
class CuratorDecision:
    """One decision the Curator made about one candidate.

    Carries the inputs to :class:`LearningStore` calls that have already
    been issued, so callers can log / audit what happened without
    re-running the search. ``existing_id`` is set for MERGE / SUPERSEDE
    and ``None`` for INSERT.
    """

    action: CurationAction
    candidate: CandidateLearning
    existing_id: str | None
    cosine: float


@dataclass(frozen=True, slots=True)
class CurationOutcome:
    """The full result of curating a batch of candidates."""

    decisions: tuple[CuratorDecision, ...]

    def by_action(self, action: CurationAction) -> tuple[CuratorDecision, ...]:
        return tuple(d for d in self.decisions if d.action == action)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Curator:
    """Apply MERGE / SUPERSEDE / INSERT decisions to a batch of candidates.

    Parameters
    ----------
    store:
        The :class:`LearningStore` to read (via ``search_hybrid``) and
        write (via ``insert`` / ``update_rule`` / ``supersede``).
    tau_merge:
        Cosine threshold above which a candidate is treated as a
        near-duplicate of the top hit and merged in. Defaults to 0.95.
    tau_conflict:
        Cosine threshold above which a candidate is treated as a
        conflicting refinement of the top hit and supersedes it.
        Must satisfy ``0 <= tau_conflict <= tau_merge <= 1``. Defaults
        to 0.80.
    top_k:
        How many neighbors to retrieve per candidate. Only the top one
        is used for the threshold check; ``top_k`` > 1 is reserved for
        future logic (e.g. transitive merges).
    rrf_k:
        Forwarded to :meth:`LearningStore.search_hybrid`.
    clock:
        Returns the ISO-8601 UTC timestamp stamped on MERGE / SUPERSEDE
        outcomes. Injectable for deterministic tests.
    """

    __slots__ = ("_clock", "_rrf_k", "_store", "_tau_conflict", "_tau_merge", "_top_k")

    def __init__(
        self,
        store: LearningStore,
        *,
        tau_merge: float = 0.95,
        tau_conflict: float = 0.80,
        top_k: int = 5,
        rrf_k: int = 60,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if not 0.0 <= tau_conflict <= tau_merge <= 1.0:
            raise ValueError(
                "tau thresholds must satisfy 0 <= tau_conflict <= tau_merge <= 1, "
                f"got tau_conflict={tau_conflict}, tau_merge={tau_merge}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
        self._store = store
        self._tau_merge = tau_merge
        self._tau_conflict = tau_conflict
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._clock = clock

    @property
    def tau_merge(self) -> float:
        return self._tau_merge

    @property
    def tau_conflict(self) -> float:
        return self._tau_conflict

    async def curate(self, candidates: Sequence[CandidateLearning]) -> CurationOutcome:
        """Decide and execute MERGE / SUPERSEDE / INSERT for every candidate.

        Each candidate is processed in order and its decision is
        committed before the next is considered. This means a fresh
        INSERT can become the SUPERSEDE target for a later candidate
        in the same batch — that ordering is intentional, mirroring
        the per-row commit semantics of the asyncpg variant.
        """

        decisions: list[CuratorDecision] = []
        for candidate in candidates:
            decision = await self._curate_one(candidate)
            decisions.append(decision)
        return CurationOutcome(decisions=tuple(decisions))

    async def _curate_one(self, candidate: CandidateLearning) -> CuratorDecision:
        scope: LearningScope | None = candidate.learning.scope
        hits = await self._store.search_hybrid(
            query_text=candidate.learning.bm25_text or candidate.learning.rule,
            query_vector=list(candidate.embedding),
            top_k=self._top_k,
            rrf_k=self._rrf_k,
            scope=scope,
        )
        top = hits[0] if hits else None
        cosine = top.cosine if top is not None else 0.0

        if top is not None and cosine >= self._tau_merge:
            await self._merge(candidate, top.learning)
            return CuratorDecision(
                action="MERGE",
                candidate=candidate,
                existing_id=top.learning.learning_id,
                cosine=cosine,
            )

        if top is not None and cosine >= self._tau_conflict:
            await self._supersede(candidate, top.learning)
            return CuratorDecision(
                action="SUPERSEDE",
                candidate=candidate,
                existing_id=top.learning.learning_id,
                cosine=cosine,
            )

        await self._insert(candidate)
        return CuratorDecision(
            action="INSERT", candidate=candidate, existing_id=None, cosine=cosine
        )

    async def _insert(self, candidate: CandidateLearning) -> None:
        await self._store.insert(candidate.learning, embedding=list(candidate.embedding))

    async def _merge(self, candidate: CandidateLearning, existing: Learning) -> None:
        bumped = existing.evidence_count + candidate.learning.evidence_count
        await self._store.update_rule(
            existing.learning_id,
            rule=existing.rule,
            why=candidate.learning.why or existing.why,
            evidence_count=bumped,
            bm25_text=candidate.learning.bm25_text or existing.bm25_text,
            updated_at=self._clock(),
            embedding=list(candidate.embedding),
        )

    async def _supersede(self, candidate: CandidateLearning, existing: Learning) -> None:
        await self._store.insert(candidate.learning, embedding=list(candidate.embedding))
        await self._store.supersede(
            old_id=existing.learning_id,
            new_id=candidate.learning.learning_id,
            archived_at=self._clock(),
        )


__all__ = [
    "CurationAction",
    "CurationOutcome",
    "Curator",
    "CuratorDecision",
]
