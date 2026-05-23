"""Conflict resolution for candidate learnings.

The Curator is the second stage of Stage B: it takes the candidates
the :class:`Distiller` emitted and decides, for each one, whether to:

- ``MERGE`` it into a near-duplicate existing row (cosine >= ``tau_merge``),
  by calling :meth:`LearningStore.update_rule` with a bumped
  ``evidence_count`` and a refreshed ``why`` / ``bm25_text``;
- ``SUPERSEDE`` an old row (the new candidate replaces it; audit-trail
  preserved via :meth:`LearningStore.supersede`);
- ``CONFLICT_NOTED`` (Stage B+) record both rows alongside a row in
  ``learning_conflicts`` so callers see the disagreement;
- ``INSERT`` it as a fresh row (cosine < ``tau_conflict``).

For Stage B+, the cosine band ``[tau_conflict, tau_merge)`` is no longer
auto-superseded — instead, an injectable :class:`ConflictJudge` decides
between SUPERSEDE / CONFLICT_NOTED / MERGE so the pipeline does not
silently lose contradictions. Callers that want the old behavior can
pass a judge that always returns ``"supersede"``.

The thresholds come from :class:`DistillerConfig` so a deployment can
tune them without touching pipeline code. ``search_hybrid`` is the
Curator's only read against the learning store; conflict rows are
written through the optional :class:`ConflictStore` dependency.

The Curator does **not** mint new identities for MERGE outcomes. The
candidate's freshly minted ``learning_id`` is discarded when the
decision is MERGE; this keeps callers from accidentally persisting
both rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from stratoclave_distill.core.types import Learning, LearningConflict, LearningScope
from stratoclave_distill.db.stores import ConflictStore, LearningStore
from stratoclave_distill.pipeline.distiller import CandidateLearning

CurationAction = Literal["INSERT", "MERGE", "SUPERSEDE", "CONFLICT_NOTED"]
ConflictVerdict = Literal["merge", "supersede", "conflict"]


@runtime_checkable
class ConflictJudge(Protocol):
    """Decides whether a borderline-cosine candidate is a conflict.

    The Curator only consults the judge when cosine is in
    ``[tau_conflict, tau_merge)`` — outside that band the action is
    obvious (MERGE above, INSERT below). The judge should return
    ``"merge"`` when the two rules say the same thing in different
    words, ``"supersede"`` when the new one cleanly replaces the old,
    and ``"conflict"`` when they disagree and *both* should remain.

    Implementations may be LLM-backed in production; tests inject a
    deterministic stub.
    """

    async def adjudicate(
        self,
        *,
        candidate: Learning,
        existing: Learning,
        cosine: float,
    ) -> ConflictVerdict: ...


class _DefaultSupersedeJudge:
    """Backward-compatible judge that always returns ``"supersede"``.

    This preserves the Stage B behaviour for callers that did not pass an
    explicit judge — every borderline-cosine candidate replaces the top
    hit, with no conflict row.
    """

    __slots__ = ()

    async def adjudicate(
        self,
        *,
        candidate: Learning,
        existing: Learning,
        cosine: float,
    ) -> ConflictVerdict:
        del candidate, existing, cosine
        return "supersede"


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
    """Apply INSERT / MERGE / SUPERSEDE / CONFLICT_NOTED decisions.

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
        possible conflict / supersession with the top hit. Must satisfy
        ``0 <= tau_conflict <= tau_merge <= 1``. Defaults to 0.80.
    top_k:
        How many neighbors to retrieve per candidate. Only the top one
        is used for the threshold check; ``top_k`` > 1 is reserved for
        future logic (e.g. transitive merges).
    rrf_k:
        Forwarded to :meth:`LearningStore.search_hybrid`.
    judge:
        Optional :class:`ConflictJudge`. Consulted only when cosine is
        in ``[tau_conflict, tau_merge)`` to break the SUPERSEDE /
        CONFLICT_NOTED / MERGE tie. Defaults to a backward-compatible
        always-supersede judge so existing callers keep working.
    conflict_store:
        Optional :class:`ConflictStore`. Required when the judge can
        return ``"conflict"`` so the row gets written to
        ``learning_conflicts`` (Stage B+).
    clock:
        Returns the ISO-8601 UTC timestamp stamped on emitted rows.
        Injectable for deterministic tests.
    """

    __slots__ = (
        "_clock",
        "_conflict_store",
        "_judge",
        "_rrf_k",
        "_store",
        "_tau_conflict",
        "_tau_merge",
        "_top_k",
    )

    def __init__(
        self,
        store: LearningStore,
        *,
        tau_merge: float = 0.95,
        tau_conflict: float = 0.80,
        top_k: int = 5,
        rrf_k: int = 60,
        judge: ConflictJudge | None = None,
        conflict_store: ConflictStore | None = None,
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
        self._judge = judge if judge is not None else _DefaultSupersedeJudge()
        self._conflict_store = conflict_store
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
            verdict = await self._judge.adjudicate(
                candidate=candidate.learning,
                existing=top.learning,
                cosine=cosine,
            )
            if verdict == "merge":
                await self._merge(candidate, top.learning)
                return CuratorDecision(
                    action="MERGE",
                    candidate=candidate,
                    existing_id=top.learning.learning_id,
                    cosine=cosine,
                )
            if verdict == "conflict":
                await self._conflict_noted(candidate, top.learning, cosine=cosine)
                return CuratorDecision(
                    action="CONFLICT_NOTED",
                    candidate=candidate,
                    existing_id=top.learning.learning_id,
                    cosine=cosine,
                )
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

    async def _conflict_noted(
        self,
        candidate: CandidateLearning,
        existing: Learning,
        *,
        cosine: float,
    ) -> None:
        """Persist the candidate plus a row in ``learning_conflicts``.

        Both rows stay active so the retriever can flag the disagreement
        cheaply via the partial index on ``resolution = 'open'``.
        """

        await self._store.insert(candidate.learning, embedding=list(candidate.embedding))
        if self._conflict_store is None:
            return
        conflict = LearningConflict(
            conflict_id=str(uuid.uuid4()),
            from_id=existing.learning_id,
            to_id=candidate.learning.learning_id,
            reason=(f"borderline cosine {cosine:.4f} between candidate and existing rule"),
            cosine_at_detection=cosine,
            detected_at=self._clock(),
            resolution="open",
        )
        await self._conflict_store.insert(conflict)


__all__ = [
    "ConflictJudge",
    "ConflictVerdict",
    "CurationAction",
    "CurationOutcome",
    "Curator",
    "CuratorDecision",
]
