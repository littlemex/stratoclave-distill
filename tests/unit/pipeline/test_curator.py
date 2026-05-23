"""Unit tests for :class:`Curator`.

The contract Stage B's CLI relies on:

- ``cosine >= tau_merge`` -> MERGE: ``update_rule`` is called on the
  existing row with bumped ``evidence_count``; the candidate's freshly
  minted ``learning_id`` is *not* persisted as a separate row;
- ``tau_conflict <= cosine < tau_merge`` -> SUPERSEDE: the candidate is
  inserted, then ``supersede(old_id=...)`` archives the previous row;
- ``cosine < tau_conflict`` (or empty store) -> INSERT;
- ``search_hybrid`` is called with the candidate's own scope so cross-
  scope rows never match;
- decisions are committed in order, so a fresh INSERT can become the
  SUPERSEDE target for a later candidate in the same batch;
- threshold validation is enforced at construction time.

All tests use :class:`InMemoryLearningStore` to exercise the real
hybrid-search code path; we never mock the store.
"""

from __future__ import annotations

import math

import pytest

from stratoclave_distill.core.types import Learning
from stratoclave_distill.db.memory import InMemoryConflictStore, InMemoryLearningStore
from stratoclave_distill.pipeline import (
    CandidateLearning,
    ConflictVerdict,
    Curator,
)


def _learning(
    learning_id: str,
    *,
    rule: str,
    bm25_text: str,
    scope: str = "session",
    evidence_count: int = 1,
) -> Learning:
    return Learning(
        learning_id=learning_id,
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why="because tests",
        bm25_text=bm25_text,
        evidence_count=evidence_count,
        created_at="2026-05-22T00:00:00Z",
        updated_at="2026-05-22T00:00:00Z",
    )


def _candidate(
    learning_id: str,
    *,
    embedding: tuple[float, ...],
    rule: str = "candidate rule",
    why: str = "candidate why",
    bm25_text: str | None = None,
    scope: str = "session",
    evidence_count: int = 1,
) -> CandidateLearning:
    return CandidateLearning(
        learning=Learning(
            learning_id=learning_id,
            scope=scope,  # type: ignore[arg-type]
            rule=rule,
            why=why,
            bm25_text=bm25_text if bm25_text is not None else rule,
            evidence_count=evidence_count,
            created_at="2026-05-22T01:00:00Z",
            updated_at="2026-05-22T01:00:00Z",
        ),
        embedding=embedding,
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_curator_rejects_bad_thresholds() -> None:
    store = InMemoryLearningStore()
    with pytest.raises(ValueError, match=r"tau thresholds"):
        Curator(store, tau_merge=0.5, tau_conflict=0.8)


def test_curator_rejects_zero_top_k() -> None:
    store = InMemoryLearningStore()
    with pytest.raises(ValueError, match=r"top_k must be >= 1"):
        Curator(store, top_k=0)


def test_curator_rejects_zero_rrf_k() -> None:
    store = InMemoryLearningStore()
    with pytest.raises(ValueError, match=r"rrf_k must be >= 1"):
        Curator(store, rrf_k=0)


# --------------------------------------------------------------------------
# Empty store -> INSERT
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_inserts_into_empty_store() -> None:
    store = InMemoryLearningStore()
    curator = Curator(store, clock=lambda: "2026-05-22T02:00:00Z")
    cand = _candidate("cand-1", embedding=(1.0, 0.0))

    outcome = await curator.curate([cand])

    assert len(outcome.decisions) == 1
    assert outcome.decisions[0].action == "INSERT"
    assert outcome.decisions[0].existing_id is None
    assert outcome.decisions[0].cosine == 0.0
    persisted = await store.get("cand-1")
    assert persisted is not None
    assert persisted.rule == "candidate rule"


# --------------------------------------------------------------------------
# MERGE path (cosine >= tau_merge)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_merges_into_near_duplicate() -> None:
    """A candidate whose embedding matches an existing row exactly merges in."""

    store = InMemoryLearningStore()
    existing = _learning("exist-1", rule="prefer SQL views", bm25_text="prefer sql views")
    await store.insert(existing, embedding=[1.0, 0.0])

    curator = Curator(
        store,
        tau_merge=0.95,
        tau_conflict=0.80,
        clock=lambda: "2026-05-22T02:00:00Z",
    )
    cand = _candidate(
        "cand-1",
        embedding=(1.0, 0.0),  # cosine 1.0 with existing
        rule="prefer SQL views",
        why="more evidence",
        bm25_text="prefer sql views",
    )

    outcome = await curator.curate([cand])

    [decision] = outcome.decisions
    assert decision.action == "MERGE"
    assert decision.existing_id == "exist-1"
    assert decision.cosine == pytest.approx(1.0)

    # The candidate's id was discarded; only the existing row remains
    assert await store.get("cand-1") is None
    merged = await store.get("exist-1")
    assert merged is not None
    assert merged.evidence_count == 2  # 1 + 1
    assert merged.why == "more evidence"
    assert merged.updated_at == "2026-05-22T02:00:00Z"
    assert merged.created_at == existing.created_at  # immutable


@pytest.mark.asyncio
async def test_curate_merge_preserves_existing_why_when_candidate_blank() -> None:
    store = InMemoryLearningStore()
    existing = _learning("exist-1", rule="rule", bm25_text="rule")
    await store.insert(existing, embedding=[1.0, 0.0])

    curator = Curator(store, clock=lambda: "t")
    cand = _candidate(
        "cand-1",
        embedding=(1.0, 0.0),
        rule="rule",
        why="",
        bm25_text="rule",
    )

    await curator.curate([cand])
    merged = await store.get("exist-1")
    assert merged is not None
    assert merged.why == "because tests"  # fell back to the existing why


# --------------------------------------------------------------------------
# SUPERSEDE path (tau_conflict <= cosine < tau_merge)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_supersedes_when_in_conflict_band() -> None:
    """An embedding partially aligned with an existing row supersedes it."""

    store = InMemoryLearningStore()
    existing = _learning("exist-1", rule="old rule", bm25_text="old rule")
    await store.insert(existing, embedding=[1.0, 0.0])

    # Cosine of (1, 0) and (0.9, sqrt(1-0.81)) ~ 0.9 -> in [0.8, 0.95)
    curator = Curator(
        store, tau_merge=0.95, tau_conflict=0.80, clock=lambda: "2026-05-22T02:00:00Z"
    )
    angle = math.sqrt(1 - 0.9 * 0.9)
    cand = _candidate(
        "cand-1",
        embedding=(0.9, angle),
        rule="refined rule",
        bm25_text="refined rule",
    )

    [decision] = (await curator.curate([cand])).decisions
    assert decision.action == "SUPERSEDE"
    assert decision.existing_id == "exist-1"
    assert 0.80 <= decision.cosine < 0.95

    # New row inserted, old row archived (not deleted)
    new = await store.get("cand-1")
    assert new is not None
    assert new.rule == "refined rule"
    archived = await store.get("exist-1")
    assert archived is not None
    assert archived.archived_at == "2026-05-22T02:00:00Z"
    assert archived.superseded_by == "cand-1"


# --------------------------------------------------------------------------
# Scope filtering
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_does_not_match_across_scopes() -> None:
    """A project-scoped candidate must not merge with a session-scoped row."""

    store = InMemoryLearningStore()
    session_existing = _learning("exist-session", rule="rule", bm25_text="rule", scope="session")
    await store.insert(session_existing, embedding=[1.0, 0.0])

    curator = Curator(store, clock=lambda: "t")
    project_cand = _candidate(
        "cand-project",
        embedding=(1.0, 0.0),
        rule="rule",
        scope="project",
    )

    [decision] = (await curator.curate([project_cand])).decisions
    assert decision.action == "INSERT"  # different scope, no match
    persisted = await store.get("cand-project")
    assert persisted is not None
    assert persisted.scope == "project"


# --------------------------------------------------------------------------
# Batch ordering: a new INSERT can become the next SUPERSEDE target
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_batch_commits_each_decision_in_order() -> None:
    """The second candidate must see the first one's INSERT in the store."""

    store = InMemoryLearningStore()
    curator = Curator(store, clock=lambda: "t")

    first = _candidate("cand-1", embedding=(1.0, 0.0), rule="alpha", bm25_text="alpha")
    # cosine ~ 1.0 with first -> MERGE
    second = _candidate("cand-2", embedding=(1.0, 0.0), rule="alpha", why="more", bm25_text="alpha")

    outcome = await curator.curate([first, second])
    actions = [d.action for d in outcome.decisions]
    assert actions == ["INSERT", "MERGE"]
    # second was merged into first; cand-2 was never persisted as its own row
    assert await store.get("cand-2") is None
    merged = await store.get("cand-1")
    assert merged is not None
    assert merged.evidence_count == 2


@pytest.mark.asyncio
async def test_curation_outcome_by_action_filters() -> None:
    store = InMemoryLearningStore()
    curator = Curator(store, clock=lambda: "t")
    a = _candidate("a", embedding=(1.0, 0.0), rule="a", bm25_text="a")
    b = _candidate("b", embedding=(0.0, 1.0), rule="b", bm25_text="b")
    outcome = await curator.curate([a, b])
    assert len(outcome.by_action("INSERT")) == 2
    assert outcome.by_action("MERGE") == ()
    assert outcome.by_action("SUPERSEDE") == ()


# --------------------------------------------------------------------------
# Empty input is a no-op
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_curate_empty_batch_returns_no_decisions() -> None:
    store = InMemoryLearningStore()
    curator = Curator(store, clock=lambda: "t")
    outcome = await curator.curate([])
    assert outcome.decisions == ()


# --------------------------------------------------------------------------
# Stage B+ — ConflictJudge / CONFLICT_NOTED action
# --------------------------------------------------------------------------


class _StubJudge:
    """Records adjudicate calls and returns a configurable verdict."""

    def __init__(self, verdict: ConflictVerdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str, float]] = []

    async def adjudicate(
        self,
        *,
        candidate: Learning,
        existing: Learning,
        cosine: float,
    ) -> ConflictVerdict:
        self.calls.append((candidate.learning_id, existing.learning_id, cosine))
        return self.verdict


def _conflict_band_candidate(learning_id: str = "cand-1") -> CandidateLearning:
    """Embedding designed to land cosine ~0.9 against an existing (1,0) row."""

    angle = math.sqrt(1 - 0.9 * 0.9)
    return _candidate(
        learning_id,
        embedding=(0.9, angle),
        rule="refined rule",
        bm25_text="refined rule",
    )


@pytest.mark.asyncio
async def test_default_judge_preserves_legacy_supersede_behavior() -> None:
    """Without an injected judge, borderline-cosine candidates still SUPERSEDE."""

    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="old rule", bm25_text="old rule"),
        embedding=[1.0, 0.0],
    )

    curator = Curator(
        store, tau_merge=0.95, tau_conflict=0.80, clock=lambda: "2026-05-22T02:00:00Z"
    )
    [decision] = (await curator.curate([_conflict_band_candidate()])).decisions
    assert decision.action == "SUPERSEDE"


@pytest.mark.asyncio
async def test_judge_conflict_verdict_writes_conflict_row() -> None:
    """verdict='conflict' inserts the candidate AND writes a learning_conflicts row."""

    store = InMemoryLearningStore()
    conflicts = InMemoryConflictStore()
    await store.insert(
        _learning("exist-1", rule="old rule", bm25_text="old rule"),
        embedding=[1.0, 0.0],
    )

    judge = _StubJudge("conflict")
    curator = Curator(
        store,
        tau_merge=0.95,
        tau_conflict=0.80,
        judge=judge,
        conflict_store=conflicts,
        clock=lambda: "2026-05-22T02:00:00Z",
    )

    [decision] = (await curator.curate([_conflict_band_candidate()])).decisions
    assert decision.action == "CONFLICT_NOTED"
    assert decision.existing_id == "exist-1"
    assert judge.calls and judge.calls[0][0:2] == ("cand-1", "exist-1")

    # Both rows remain active
    new = await store.get("cand-1")
    old = await store.get("exist-1")
    assert new is not None and new.archived_at is None
    assert old is not None and old.archived_at is None

    open_conflicts = await conflicts.list_open()
    assert len(open_conflicts) == 1
    row = open_conflicts[0]
    assert row.from_id == "exist-1"
    assert row.to_id == "cand-1"
    assert row.resolution == "open"
    assert row.detected_at == "2026-05-22T02:00:00Z"
    assert 0.80 <= row.cosine_at_detection < 0.95


@pytest.mark.asyncio
async def test_judge_merge_verdict_overrides_into_merge() -> None:
    """verdict='merge' folds the candidate into the existing row even below tau_merge."""

    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="old rule", bm25_text="old rule", evidence_count=1),
        embedding=[1.0, 0.0],
    )
    judge = _StubJudge("merge")
    curator = Curator(
        store,
        tau_merge=0.95,
        tau_conflict=0.80,
        judge=judge,
        clock=lambda: "2026-05-22T02:00:00Z",
    )

    [decision] = (await curator.curate([_conflict_band_candidate()])).decisions
    assert decision.action == "MERGE"
    assert decision.existing_id == "exist-1"
    assert await store.get("cand-1") is None  # candidate id discarded
    merged = await store.get("exist-1")
    assert merged is not None
    assert merged.evidence_count == 2


@pytest.mark.asyncio
async def test_judge_supersede_verdict_matches_default() -> None:
    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="old rule", bm25_text="old rule"),
        embedding=[1.0, 0.0],
    )
    judge = _StubJudge("supersede")
    curator = Curator(
        store,
        tau_merge=0.95,
        tau_conflict=0.80,
        judge=judge,
        clock=lambda: "2026-05-22T02:00:00Z",
    )

    [decision] = (await curator.curate([_conflict_band_candidate()])).decisions
    assert decision.action == "SUPERSEDE"
    archived = await store.get("exist-1")
    assert archived is not None
    assert archived.superseded_by == "cand-1"


@pytest.mark.asyncio
async def test_judge_not_consulted_above_tau_merge() -> None:
    """Cosine >= tau_merge is an unambiguous MERGE — no judge call."""

    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="rule", bm25_text="rule"),
        embedding=[1.0, 0.0],
    )
    judge = _StubJudge("conflict")
    curator = Curator(store, judge=judge, clock=lambda: "t")
    cand = _candidate("cand-1", embedding=(1.0, 0.0), rule="rule", bm25_text="rule")

    [decision] = (await curator.curate([cand])).decisions
    assert decision.action == "MERGE"
    assert judge.calls == []


@pytest.mark.asyncio
async def test_judge_not_consulted_below_tau_conflict() -> None:
    """Cosine < tau_conflict is an unambiguous INSERT — no judge call."""

    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="rule", bm25_text="rule"),
        embedding=[1.0, 0.0],
    )
    judge = _StubJudge("conflict")
    curator = Curator(store, tau_merge=0.95, tau_conflict=0.80, judge=judge, clock=lambda: "t")
    # Cosine 0 -> below tau_conflict
    cand = _candidate("cand-1", embedding=(0.0, 1.0), rule="other", bm25_text="other")

    [decision] = (await curator.curate([cand])).decisions
    assert decision.action == "INSERT"
    assert judge.calls == []


@pytest.mark.asyncio
async def test_conflict_store_is_optional() -> None:
    """A 'conflict' verdict without a conflict_store still inserts the candidate."""

    store = InMemoryLearningStore()
    await store.insert(
        _learning("exist-1", rule="old rule", bm25_text="old rule"),
        embedding=[1.0, 0.0],
    )
    judge = _StubJudge("conflict")
    curator = Curator(
        store,
        tau_merge=0.95,
        tau_conflict=0.80,
        judge=judge,
        clock=lambda: "t",
    )

    [decision] = (await curator.curate([_conflict_band_candidate()])).decisions
    assert decision.action == "CONFLICT_NOTED"
    assert await store.get("cand-1") is not None
    assert await store.get("exist-1") is not None  # not archived
