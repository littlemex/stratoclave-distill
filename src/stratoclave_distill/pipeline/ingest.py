"""End-to-end ingest orchestration: Reader -> Distiller -> Curator -> Stores.

The CLI's ``ingest`` subcommand and any embedding host (a long-running
worker, for instance) call into :class:`IngestRunner`. The runner is
the single place that knows the order of operations:

1. read a JSONL transcript with :class:`JsonlSessionReader`;
2. group turns by session;
3. for each session, look up the watermark, filter to ``seq > watermark``;
4. fetch the prior :class:`SessionPurpose` (if any) for prompt context;
5. call :meth:`Distiller.distill` to get purpose / digest / candidate
   learnings;
6. upsert purpose, upsert digest (with embedding), curate learnings;
7. advance the watermark to ``last_seq`` on success.

Any exception aborts that *one session* — the runner records it on
:class:`IngestReport` and continues with the next session, so a single
malformed transcript cannot stall a batch ingest. ``strict=True``
escalates the first failure into a raised exception (the CLI exposes
this via ``--strict``).

The runner is store-agnostic: it depends only on the four
:class:`typing.Protocol` types from :mod:`stratoclave_distill.db`, so
unit tests use the in-memory variant and the CLI wires in asyncpg.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from stratoclave_distill.core.errors import DistillError
from stratoclave_distill.core.types import BranchKind, NormalizedTurn
from stratoclave_distill.db.stores import (
    DigestStore,
    PurposeStore,
    WatermarkStore,
)
from stratoclave_distill.pipeline.curator import CurationOutcome, Curator
from stratoclave_distill.pipeline.distiller import Distiller
from stratoclave_distill.pipeline.reader import JsonlSessionReader, SkippedLine


@dataclass(frozen=True, slots=True)
class BranchPlan:
    """Stage B+ branch hint applied during the *first* ingest of a session.

    When a caller (typically the CLI's ``ingest --branch-from``) wants the
    runner to record that ``session_id`` is a branch off ``parent_session_id``
    starting at ``at_seq``, they construct a :class:`BranchPlan` and pass
    it to :meth:`IngestRunner.run_path`.

    The plan is applied only when the session is *new* (no purpose row in
    the store yet). Re-running ``ingest`` against the same session id is
    idempotent: the existing branching topology is preserved.

    Turns with ``seq <= at_seq`` are skipped, so the experiment file can
    legitimately replay the parent's earlier turns without re-distilling
    them.
    """

    session_id: str
    parent_session_id: str
    at_seq: int
    branch_kind: BranchKind = "experiment"

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("BranchPlan.session_id must be a non-empty string")
        if not self.parent_session_id:
            raise ValueError("BranchPlan.parent_session_id must be a non-empty string")
        if self.parent_session_id == self.session_id:
            raise ValueError("BranchPlan.parent_session_id must differ from session_id")
        if self.at_seq < 0:
            raise ValueError(f"BranchPlan.at_seq must be >= 0, got {self.at_seq}")


@dataclass(frozen=True, slots=True)
class SessionIngestResult:
    """The outcome of distilling exactly one session.

    ``error`` is set if the session aborted before it could advance the
    watermark; the other fields capture what *did* land. ``new_seq`` is
    the watermark value the runner committed (or the unchanged prior
    watermark if nothing was distilled).
    """

    session_id: str
    distilled: bool
    prior_seq: int
    new_seq: int
    candidate_count: int
    curation: CurationOutcome | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestReport:
    """The aggregate outcome of an ``ingest`` run."""

    sessions: tuple[SessionIngestResult, ...]
    skipped_lines: tuple[SkippedLine, ...]

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def distilled_count(self) -> int:
        return sum(1 for s in self.sessions if s.distilled)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.sessions if s.error is not None)


def _group_by_session(turns: Sequence[NormalizedTurn]) -> dict[str, list[NormalizedTurn]]:
    """Group turns by ``session_id`` while preserving file order."""

    out: dict[str, list[NormalizedTurn]] = {}
    for turn in turns:
        out.setdefault(turn.session_id, []).append(turn)
    return out


class IngestRunner:
    """Orchestrate Reader -> Distiller -> Curator -> Stores end-to-end.

    Parameters
    ----------
    distiller / curator:
        Pre-built collaborators. The runner does not construct them so
        that the same instance can be reused across many ingests, and
        so tests can pin down their dependencies.
    watermarks / purposes / digests:
        Three of the four persistence Protocols. The fourth
        (:class:`LearningStore`) is held by the :class:`Curator`, so the
        runner does not duplicate that reference. The runner reads
        :meth:`WatermarkStore.get` before every session and writes
        :meth:`WatermarkStore.advance` only on success.
    strict:
        If ``True``, the first session error re-raises instead of being
        recorded on the report.
    """

    __slots__ = (
        "_curator",
        "_digests",
        "_distiller",
        "_purposes",
        "_strict",
        "_watermarks",
    )

    def __init__(
        self,
        *,
        distiller: Distiller,
        curator: Curator,
        watermarks: WatermarkStore,
        purposes: PurposeStore,
        digests: DigestStore,
        strict: bool = False,
    ) -> None:
        self._distiller = distiller
        self._curator = curator
        self._watermarks = watermarks
        self._purposes = purposes
        self._digests = digests
        self._strict = strict

    async def run_path(
        self,
        path: str,
        *,
        branch_plan: BranchPlan | None = None,
    ) -> IngestReport:
        """Read a JSONL file and ingest every session it contains."""

        reader = JsonlSessionReader(path, strict=self._strict)
        turns = list(reader.read())
        return await self.run_turns(turns, skipped_lines=reader.skipped, branch_plan=branch_plan)

    async def run_turns(
        self,
        turns: Sequence[NormalizedTurn],
        *,
        skipped_lines: Sequence[SkippedLine] = (),
        branch_plan: BranchPlan | None = None,
    ) -> IngestReport:
        """Ingest a pre-decoded sequence of turns. Useful in tests."""

        groups = _group_by_session(turns)
        results: list[SessionIngestResult] = []
        for session_id, session_turns in groups.items():
            plan = branch_plan if branch_plan and branch_plan.session_id == session_id else None
            result = await self._run_session(session_id, session_turns, branch_plan=plan)
            results.append(result)
        return IngestReport(sessions=tuple(results), skipped_lines=tuple(skipped_lines))

    async def _run_session(
        self,
        session_id: str,
        turns: Sequence[NormalizedTurn],
        *,
        branch_plan: BranchPlan | None = None,
    ) -> SessionIngestResult:
        prior_seq = await self._watermarks.get(session_id)
        # A branch plan implicitly skips parent-shared turns.
        if branch_plan is not None:
            prior_seq = max(prior_seq, branch_plan.at_seq)
        fresh = [t for t in turns if t.seq > prior_seq]
        if not fresh:
            return SessionIngestResult(
                session_id=session_id,
                distilled=False,
                prior_seq=prior_seq,
                new_seq=prior_seq,
                candidate_count=0,
                curation=None,
            )

        try:
            prior_purpose = await self._purposes.get(session_id)
            result = await self._distiller.distill(
                fresh, session_id=session_id, prior_purpose=prior_purpose
            )
            purpose = result.purpose
            if prior_purpose is not None:
                # Distiller does not know about branching; preserve the
                # topology that was set when the session was first ingested.
                purpose = replace(
                    purpose,
                    parent_session_id=prior_purpose.parent_session_id,
                    branched_at_seq=prior_purpose.branched_at_seq,
                    branch_kind=prior_purpose.branch_kind,
                    branch_state=prior_purpose.branch_state,
                    closed_at=prior_purpose.closed_at,
                )
            elif branch_plan is not None:
                purpose = replace(
                    purpose,
                    parent_session_id=branch_plan.parent_session_id,
                    branched_at_seq=branch_plan.at_seq,
                    branch_kind=branch_plan.branch_kind,
                    branch_state="open",
                    closed_at=None,
                )
            await self._purposes.upsert(purpose)
            await self._digests.upsert(result.digest, embedding=list(result.digest_embedding))
            outcome = await self._curator.curate(result.candidate_learnings)
            await self._watermarks.advance(
                session_id, to_seq=result.last_seq, last_run_at=result.purpose.derived_at
            )
        except DistillError as exc:
            if self._strict:
                raise
            return SessionIngestResult(
                session_id=session_id,
                distilled=False,
                prior_seq=prior_seq,
                new_seq=prior_seq,
                candidate_count=0,
                curation=None,
                error=str(exc),
            )

        return SessionIngestResult(
            session_id=session_id,
            distilled=True,
            prior_seq=prior_seq,
            new_seq=result.last_seq,
            candidate_count=len(result.candidate_learnings),
            curation=outcome,
        )


__all__ = [
    "BranchPlan",
    "IngestReport",
    "IngestRunner",
    "SessionIngestResult",
]
