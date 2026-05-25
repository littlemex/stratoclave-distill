"""End-to-end test of the asyncpg-backed stores against a live Postgres.

Skipped unless ``DISTILL_TEST_DATABASE_URL`` is set, mirroring
``test_migrations_against_postgres.py``. Run locally with::

    docker compose up -d
    DISTILL_TEST_DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
        pytest -m integration tests/integration/test_asyncpg_stores.py

The test:

1. Brings the schema up via ``alembic upgrade head``;
2. Truncates the relevant tables so the suite is order-independent;
3. Exercises every store method (watermark, purpose, digest, learning) and
   the hybrid search SQL against a real pgvector index;
4. Tears the schema back down via ``alembic downgrade base``.

This is the gate that proves the asyncpg implementation matches the
in-memory contract before the CLI ingest path runs against production
Postgres.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from stratoclave_distill.core.types import (
    GroupLearning,
    Learning,
    LearningConflict,
    SessionDigest,
    SessionGap,
    SessionPurpose,
)
from stratoclave_distill.db.asyncpg import (
    AsyncpgConflictStore,
    AsyncpgDigestStore,
    AsyncpgGapStore,
    AsyncpgGroupLearningStore,
    AsyncpgLearningStore,
    AsyncpgPurposeStore,
    AsyncpgWatermarkStore,
    pool_context,
)
from stratoclave_distill.pipeline.aggregator import Aggregator
from stratoclave_distill.providers.embedding import StubEmbedding
from stratoclave_distill.providers.llm import StubLLM

pytestmark = pytest.mark.integration


def _database_url() -> str | None:
    return os.environ.get("DISTILL_TEST_DATABASE_URL")


def _run_alembic(direction: str, db_url: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    subprocess.run(
        [
            "alembic",
            "upgrade" if direction == "up" else "downgrade",
            "head" if direction == "up" else "base",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )


@pytest.fixture(scope="module")
def db_url() -> str:
    url = _database_url()
    if not url:
        pytest.skip("DISTILL_TEST_DATABASE_URL not set; integration test skipped")
    return url


@pytest.fixture(scope="module", autouse=True)
def _migrated_schema(db_url: str) -> AsyncIterator[None]:
    """Bring the schema up once for the module, tear it down at the end."""

    _run_alembic("up", db_url)
    try:
        yield
    finally:
        _run_alembic("down", db_url)


@pytest.fixture
async def truncated_pool(db_url: str) -> AsyncIterator[object]:
    """Yield a fresh pool with the relevant tables emptied."""

    async with pool_context(db_url) as pool:
        async with pool.acquire() as conn:
            await conn.execute(
                "TRUNCATE learnings, session_digests, session_purposes, "
                "distill_watermarks, group_learnings, learning_conflicts, "
                "session_gaps RESTART IDENTITY CASCADE"
            )
        yield pool


def _new_id() -> str:
    return str(uuid.uuid4())


def _embedding_dim() -> int:
    """Match the dimension the migration was bound to.

    The migration reads ``DISTILL_EMBEDDING_DIM`` (default 1024); the asyncpg
    integration test must produce vectors with the same length or the
    pgvector cast will fail.
    """

    return int(os.environ.get("DISTILL_EMBEDDING_DIM", "1024"))


async def test_watermark_get_returns_zero_for_unknown_session(truncated_pool: object) -> None:
    store = AsyncpgWatermarkStore(truncated_pool)
    assert await store.get(_new_id()) == 0


async def test_watermark_advance_is_monotonic(truncated_pool: object) -> None:
    store = AsyncpgWatermarkStore(truncated_pool)
    sid = _new_id()
    await store.advance(sid, to_seq=10, last_run_at="2026-05-22T00:00:00Z")
    await store.advance(sid, to_seq=5, last_run_at="2026-05-22T00:00:01Z")
    assert await store.get(sid) == 10
    await store.advance(sid, to_seq=42, last_run_at="2026-05-22T00:00:02Z")
    assert await store.get(sid) == 42


async def test_purpose_upsert_is_idempotent(truncated_pool: object) -> None:
    store = AsyncpgPurposeStore(truncated_pool)
    sid = _new_id()
    purpose = SessionPurpose(
        session_id=sid,
        purpose="exploration",
        domain_tags=("python", "async"),
        success_score=0.7,
        polluted=False,
        pollution_reason=None,
        derived_from_version="v-test",
        derived_at="2026-05-22T00:00:00Z",
        last_updated_at="2026-05-22T00:00:00Z",
    )
    await store.upsert(purpose)
    await store.upsert(purpose)  # idempotent: must not raise
    fetched = await store.get(sid)
    assert fetched is not None
    assert fetched.purpose == "exploration"
    assert fetched.domain_tags == ("python", "async")
    assert fetched.success_score == pytest.approx(0.7)


async def test_digest_upsert_replaces_existing_row(truncated_pool: object) -> None:
    store = AsyncpgDigestStore(truncated_pool)
    sid = _new_id()
    dim = _embedding_dim()
    first = SessionDigest(
        digest_id=_new_id(),
        session_id=sid,
        version_id="v-1",
        summary_md="first",
        bm25_text="alpha beta",
        extracted_at="2026-05-22T00:00:00Z",
    )
    await store.upsert(first, embedding=[0.1] * dim)

    second = SessionDigest(
        digest_id=_new_id(),
        session_id=sid,
        version_id="v-2",
        summary_md="second",
        bm25_text="gamma delta",
        extracted_at="2026-05-22T00:00:01Z",
    )
    await store.upsert(second, embedding=[0.2] * dim)

    fetched = await store.get(sid)
    assert fetched is not None
    assert fetched.summary_md == "second"
    assert fetched.version_id == "v-2"

    # Confirm only one digest row exists for the session (delete-then-insert semantics).
    async with truncated_pool.acquire() as conn:  # type: ignore[attr-defined]
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM session_digests WHERE session_id = $1", sid
        )
    assert count == 1


def _learning(rule: str, *, scope: str = "session", bm25: str | None = None) -> Learning:
    return Learning(
        learning_id=_new_id(),
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why="because tests demand it",
        triggers={"tag": "x"},
        project_key=None,
        group_id=None,
        source_session=None,
        source_version="v-test",
        evidence_count=1,
        confidence=0.5,
        archived_at=None,
        superseded_by=None,
        bm25_text=bm25 or rule,
        created_at="2026-05-22T00:00:00Z",
        updated_at="2026-05-22T00:00:00Z",
    )


def _vec(seed: float, dim: int | None = None) -> list[float]:
    """Return a deterministic non-zero vector for cosine math."""
    d = dim if dim is not None else _embedding_dim()
    return [seed] + [0.0] * (d - 1)


async def test_learning_insert_get_and_list_active(truncated_pool: object) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    a = _learning("always pin asyncpg version")
    b = _learning("prefer slots dataclasses", scope="project")
    await store.insert(a, embedding=_vec(1.0))
    await store.insert(b, embedding=_vec(0.5))

    fetched = await store.get(a.learning_id)
    assert fetched is not None
    assert fetched.rule == a.rule
    assert fetched.triggers == {"tag": "x"}

    rows = await store.list_active()
    assert {r.learning_id for r in rows} == {a.learning_id, b.learning_id}

    project_only = await store.list_active(scope="project")
    assert [r.learning_id for r in project_only] == [b.learning_id]


async def test_learning_update_rule_replaces_text_and_vector(truncated_pool: object) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    row = _learning("v1 rule")
    await store.insert(row, embedding=_vec(1.0))
    await store.update_rule(
        row.learning_id,
        rule="v2 rule",
        why="updated reason",
        evidence_count=5,
        bm25_text="updated bm25",
        updated_at="2026-05-22T01:00:00Z",
        embedding=_vec(0.25),
    )
    fetched = await store.get(row.learning_id)
    assert fetched is not None
    assert fetched.rule == "v2 rule"
    assert fetched.why == "updated reason"
    assert fetched.evidence_count == 5
    assert fetched.bm25_text == "updated bm25"


async def test_learning_supersede_marks_old_and_keeps_history(
    truncated_pool: object,
) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    old = _learning("old rule")
    new = _learning("new rule")
    await store.insert(old, embedding=_vec(1.0))
    await store.insert(new, embedding=_vec(0.5))
    await store.supersede(
        old_id=old.learning_id, new_id=new.learning_id, archived_at="2026-05-22T02:00:00Z"
    )

    old_after = await store.get(old.learning_id)
    assert old_after is not None
    assert old_after.archived_at == "2026-05-22T02:00:00Z"
    assert old_after.superseded_by == new.learning_id

    active = await store.list_active()
    assert [r.learning_id for r in active] == [new.learning_id]


async def test_learning_search_hybrid_ranks_cosine_match_first(
    truncated_pool: object,
) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    near = _learning("near match", bm25="apple banana")
    far = _learning("far match", bm25="zebra cougar")
    await store.insert(near, embedding=_vec(1.0))
    await store.insert(far, embedding=_vec(-1.0))

    hits = await store.search_hybrid(
        query_text="apple",
        query_vector=_vec(1.0),
        top_k=2,
    )
    assert len(hits) == 2
    assert hits[0].learning.learning_id == near.learning_id
    assert hits[0].cosine == pytest.approx(1.0, abs=1e-6)
    assert hits[0].bm25_rank == 1
    # ``far`` has no BM25 overlap with "apple"; bm25_rank must be None.
    far_hit = next(h for h in hits if h.learning.learning_id == far.learning_id)
    assert far_hit.bm25_rank is None


async def test_learning_search_hybrid_excludes_archived(
    truncated_pool: object,
) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    keep = _learning("keep me", bm25="alpha")
    drop = _learning("drop me", bm25="alpha")
    await store.insert(keep, embedding=_vec(1.0))
    await store.insert(drop, embedding=_vec(0.99))
    await store.supersede(
        old_id=drop.learning_id,
        new_id=keep.learning_id,
        archived_at="2026-05-22T03:00:00Z",
    )

    hits = await store.search_hybrid(query_text="alpha", query_vector=_vec(1.0), top_k=10)
    assert [h.learning.learning_id for h in hits] == [keep.learning_id]


async def test_learning_search_hybrid_filters_by_scope(truncated_pool: object) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    s = _learning("session scope", scope="session", bm25="alpha")
    p = _learning("project scope", scope="project", bm25="alpha")
    await store.insert(s, embedding=_vec(1.0))
    await store.insert(p, embedding=_vec(1.0))

    project_only = await store.search_hybrid(
        query_text="alpha",
        query_vector=_vec(1.0),
        top_k=10,
        scope="project",
    )
    assert [h.learning.learning_id for h in project_only] == [p.learning_id]


# --------------------------------------------------------------------------
# Stage B+ — branching, claim_type, lanes, conflict / gap stores
# --------------------------------------------------------------------------


async def test_purpose_branching_round_trip(truncated_pool: object) -> None:
    store = AsyncpgPurposeStore(truncated_pool)
    parent_id = _new_id()
    child_id = _new_id()

    parent = SessionPurpose(
        session_id=parent_id,
        purpose="parent goal",
        derived_from_version="v",
        derived_at="2026-05-22T00:00:00Z",
        last_updated_at="2026-05-22T00:00:00Z",
    )
    child = SessionPurpose(
        session_id=child_id,
        purpose="experiment goal",
        derived_from_version="v",
        derived_at="2026-05-22T01:00:00Z",
        last_updated_at="2026-05-22T01:00:00Z",
        parent_session_id=parent_id,
        branched_at_seq=5,
        branch_kind="experiment",
        branch_state="open",
    )
    await store.upsert(parent)
    await store.upsert(child)

    fetched = await store.get(child_id)
    assert fetched is not None
    assert fetched.parent_session_id == parent_id
    assert fetched.branched_at_seq == 5
    assert fetched.branch_kind == "experiment"
    assert fetched.branch_state == "open"

    await store.set_branch_state(
        child_id,
        branch_state="closed",
        closed_at="2026-05-22T02:00:00Z",
        last_updated_at="2026-05-22T02:00:00Z",
    )
    closed = await store.get(child_id)
    assert closed is not None
    assert closed.branch_state == "closed"
    assert closed.closed_at == "2026-05-22T02:00:00Z"

    branches = await store.list_branches(parent_session_id=parent_id)
    assert {b.session_id for b in branches} == {child_id}


async def test_learning_claim_type_round_trip(truncated_pool: object) -> None:
    store = AsyncpgLearningStore(truncated_pool)
    row = Learning(
        learning_id=_new_id(),
        scope="session",
        rule="experiment finding",
        why="from a probe",
        bm25_text="probe finding",
        evidence_count=1,
        confidence=0.5,
        created_at="2026-05-22T00:00:00Z",
        updated_at="2026-05-22T00:00:00Z",
        claim_type="signal",
    )
    await store.insert(row, embedding=_vec(1.0))
    fetched = await store.get(row.learning_id)
    assert fetched is not None
    assert fetched.claim_type == "signal"


async def test_learning_search_hybrid_lane_filtering(truncated_pool: object) -> None:
    """canonical lane requires evidence + age + scope!='experiment'."""

    store = AsyncpgLearningStore(truncated_pool)

    canonical_row = Learning(
        learning_id=_new_id(),
        scope="session",
        rule="canonical rule",
        why="why",
        bm25_text="alpha canonical",
        evidence_count=5,
        confidence=0.8,
        # 30 days old so it passes canonical_min_age_days=14
        created_at="2026-04-22T00:00:00Z",
        updated_at="2026-04-22T00:00:00Z",
    )
    fresh_row = Learning(
        learning_id=_new_id(),
        scope="session",
        rule="fresh rule",
        why="why",
        bm25_text="alpha fresh",
        evidence_count=5,
        confidence=0.8,
        # 1 day old → emerging
        created_at="2026-05-21T00:00:00Z",
        updated_at="2026-05-21T00:00:00Z",
    )
    experiment_row = Learning(
        learning_id=_new_id(),
        scope="experiment",
        rule="experiment rule",
        why="why",
        bm25_text="alpha experiment",
        evidence_count=5,
        confidence=0.8,
        created_at="2026-04-22T00:00:00Z",
        updated_at="2026-04-22T00:00:00Z",
    )
    await store.insert(canonical_row, embedding=_vec(1.0))
    await store.insert(fresh_row, embedding=_vec(1.0))
    await store.insert(experiment_row, embedding=_vec(1.0))

    canonical_hits = await store.search_hybrid(
        query_text="alpha",
        query_vector=_vec(1.0),
        top_k=10,
        lane="canonical",
        canonical_min_evidence=3,
        canonical_min_age_days=14,
    )
    emerging_hits = await store.search_hybrid(
        query_text="alpha",
        query_vector=_vec(1.0),
        top_k=10,
        lane="emerging",
        canonical_min_evidence=3,
        canonical_min_age_days=14,
    )
    canonical_ids = {h.learning.learning_id for h in canonical_hits}
    emerging_ids = {h.learning.learning_id for h in emerging_hits}
    assert canonical_ids == {canonical_row.learning_id}
    assert emerging_ids == {fresh_row.learning_id, experiment_row.learning_id}


async def test_conflict_store_lifecycle(truncated_pool: object) -> None:
    learning_store = AsyncpgLearningStore(truncated_pool)
    store = AsyncpgConflictStore(truncated_pool)
    from_id = _new_id()
    to_id = _new_id()
    for lid in (from_id, to_id):
        await learning_store.insert(
            Learning(
                learning_id=lid,
                scope="session",
                rule=f"rule for {lid[:8]}",
                why="why",
                bm25_text="alpha",
                evidence_count=1,
                confidence=0.5,
                created_at="2026-05-21T00:00:00Z",
                updated_at="2026-05-21T00:00:00Z",
            ),
            embedding=_vec(1.0),
        )
    cid = _new_id()
    conflict = LearningConflict(
        conflict_id=cid,
        from_id=from_id,
        to_id=to_id,
        reason="borderline cosine",
        cosine_at_detection=0.85,
        detected_at="2026-05-22T00:00:00Z",
        resolution="open",
    )
    await store.insert(conflict)

    fetched = await store.get(cid)
    assert fetched is not None
    assert fetched.reason == "borderline cosine"

    open_rows = await store.list_open()
    assert any(c.conflict_id == cid for c in open_rows)

    await store.resolve(cid, resolution="superseded")
    resolved = await store.get(cid)
    assert resolved is not None
    assert resolved.resolution == "superseded"
    open_after = await store.list_open()
    assert all(c.conflict_id != cid for c in open_after)


async def test_gap_store_lifecycle(truncated_pool: object) -> None:
    purpose_store = AsyncpgPurposeStore(truncated_pool)
    learning_store = AsyncpgLearningStore(truncated_pool)
    store = AsyncpgGapStore(truncated_pool)
    sid = _new_id()
    await purpose_store.upsert(
        SessionPurpose(
            session_id=sid,
            purpose="alpha",
            derived_at="2026-05-22T00:00:00Z",
            last_updated_at="2026-05-22T00:00:00Z",
        )
    )
    gid = _new_id()
    gap = SessionGap(
        gap_id=gid,
        session_id=sid,
        topic="ef_search optimum",
        why_unknown="not measured",
        bm25_text="ef_search optimum tuning",
        detected_at="2026-05-22T00:00:00Z",
    )
    await store.insert(gap)

    fetched = await store.get(gid)
    assert fetched is not None
    assert fetched.topic == "ef_search optimum"

    unresolved = await store.list_unresolved(session_id=sid)
    assert {g.gap_id for g in unresolved} == {gid}

    resolver_id = _new_id()
    await learning_store.insert(
        Learning(
            learning_id=resolver_id,
            scope="session",
            rule="resolved by experiment",
            why="why",
            bm25_text="resolution",
            evidence_count=1,
            confidence=0.5,
            created_at="2026-05-22T00:00:00Z",
            updated_at="2026-05-22T00:00:00Z",
        ),
        embedding=_vec(1.0),
    )
    await store.resolve(gid, resolved_at="2026-05-23T00:00:00Z", resolved_by_learning=resolver_id)
    resolved = await store.get(gid)
    assert resolved is not None
    assert resolved.resolved_at == "2026-05-23T00:00:00Z"
    assert resolved.resolved_by_learning == resolver_id
    unresolved_after = await store.list_unresolved(session_id=sid)
    assert all(g.gap_id != gid for g in unresolved_after)


# --------------------------------------------------------------------------
# Stage D -- group rollups (Aggregator + AsyncpgGroupLearningStore)
# --------------------------------------------------------------------------


def _grouped_learning(
    *,
    group_id: str,
    rule: str,
    bm25: str | None = None,
    created_at: str = "2026-05-22T00:00:00Z",
) -> Learning:
    return Learning(
        learning_id=_new_id(),
        scope="group",  # type: ignore[arg-type]
        rule=rule,
        why="from a Stage D integration test",
        triggers={"tag": "x"},
        project_key=None,
        group_id=group_id,
        source_session=None,
        source_version="v-test",
        evidence_count=3,
        confidence=0.8,
        archived_at=None,
        superseded_by=None,
        bm25_text=bm25 or rule,
        created_at=created_at,
        updated_at=created_at,
        claim_type="norm",
    )


async def test_group_learning_store_round_trip(truncated_pool: object) -> None:
    store = AsyncpgGroupLearningStore(truncated_pool)
    dim = _embedding_dim()
    group_id = _new_id()
    rollup = GroupLearning(
        group_learning_id=_new_id(),
        group_id=group_id,
        summary_md="rollup body",
        contributing_learnings=("L1", "L2"),
        bm25_text="rollup body for bm25",
        created_at="2026-05-22T00:00:00Z",
    )
    await store.upsert(rollup, embedding=[0.1] * dim)

    fetched = await store.get(rollup.group_learning_id)
    assert fetched is not None
    assert fetched.group_id == group_id
    assert fetched.contributing_learnings == ("L1", "L2")
    assert fetched.summary_md == "rollup body"


async def test_group_learning_list_by_group_latest_only(truncated_pool: object) -> None:
    """Re-aggregation appends a new row; latest_only returns just the newest."""

    store = AsyncpgGroupLearningStore(truncated_pool)
    dim = _embedding_dim()
    group_id = _new_id()
    older = GroupLearning(
        group_learning_id=_new_id(),
        group_id=group_id,
        summary_md="v1",
        contributing_learnings=("L1",),
        bm25_text="v1 body",
        created_at="2026-05-20T00:00:00Z",
    )
    newer = GroupLearning(
        group_learning_id=_new_id(),
        group_id=group_id,
        summary_md="v2",
        contributing_learnings=("L1", "L2"),
        bm25_text="v2 body",
        created_at="2026-05-25T00:00:00Z",
    )
    await store.upsert(older, embedding=[0.1] * dim)
    await store.upsert(newer, embedding=[0.2] * dim)

    latest = await store.list_by_group(group_id, latest_only=True)
    assert tuple(g.group_learning_id for g in latest) == (newer.group_learning_id,)

    history = await store.list_by_group(group_id, latest_only=False)
    assert tuple(g.group_learning_id for g in history) == (
        newer.group_learning_id,
        older.group_learning_id,
    )

    # Older row survives -- audit trail preserved.
    audit = await store.get(older.group_learning_id)
    assert audit is not None


async def test_group_learning_list_latest_per_group(truncated_pool: object) -> None:
    store = AsyncpgGroupLearningStore(truncated_pool)
    dim = _embedding_dim()
    group_c = _new_id()
    group_d = _new_id()
    rows = [
        GroupLearning(
            group_learning_id=_new_id(),
            group_id=group_c,
            summary_md="C v1",
            contributing_learnings=("L1",),
            bm25_text="C v1",
            created_at="2026-05-20T00:00:00Z",
        ),
        GroupLearning(
            group_learning_id=_new_id(),
            group_id=group_c,
            summary_md="C v2",
            contributing_learnings=("L1", "L2"),
            bm25_text="C v2",
            created_at="2026-05-25T00:00:00Z",
        ),
        GroupLearning(
            group_learning_id=_new_id(),
            group_id=group_d,
            summary_md="D only",
            contributing_learnings=("L9",),
            bm25_text="D only",
            created_at="2026-05-22T00:00:00Z",
        ),
    ]
    for r in rows:
        await store.upsert(r, embedding=[0.1] * dim)

    latest = await store.list_latest_per_group()
    ids = {g.group_learning_id for g in latest}
    # Two groups -- the older C row must not appear.
    assert ids == {rows[1].group_learning_id, rows[2].group_learning_id}
    # Sorted by created_at DESC: C v2 (2026-05-25) before D only (2026-05-22).
    assert tuple(g.group_learning_id for g in latest) == (
        rows[1].group_learning_id,
        rows[2].group_learning_id,
    )


async def test_group_learning_search_hybrid_dedupes_to_latest_per_group(
    truncated_pool: object,
) -> None:
    store = AsyncpgGroupLearningStore(truncated_pool)
    group_id = _new_id()
    older = GroupLearning(
        group_learning_id=_new_id(),
        group_id=group_id,
        summary_md="old summary",
        contributing_learnings=("L1",),
        bm25_text="alpha old",
        created_at="2026-05-20T00:00:00Z",
    )
    newer = GroupLearning(
        group_learning_id=_new_id(),
        group_id=group_id,
        summary_md="new summary",
        contributing_learnings=("L1", "L2"),
        bm25_text="alpha new",
        created_at="2026-05-25T00:00:00Z",
    )
    await store.upsert(older, embedding=_vec(1.0))
    await store.upsert(newer, embedding=_vec(0.99))

    hits = await store.search_hybrid(
        query_text="alpha",
        query_vector=_vec(1.0),
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0].group.group_learning_id == newer.group_learning_id
    assert hits[0].vector_rank == 1


async def test_group_learning_search_hybrid_ranks_two_groups(truncated_pool: object) -> None:
    store = AsyncpgGroupLearningStore(truncated_pool)
    near = GroupLearning(
        group_learning_id=_new_id(),
        group_id=_new_id(),
        summary_md="alpha rollup",
        contributing_learnings=("L1",),
        bm25_text="alpha banana",
        created_at="2026-05-22T00:00:00Z",
    )
    far = GroupLearning(
        group_learning_id=_new_id(),
        group_id=_new_id(),
        summary_md="zeta rollup",
        contributing_learnings=("L2",),
        bm25_text="zeta cougar",
        created_at="2026-05-22T00:00:00Z",
    )
    await store.upsert(near, embedding=_vec(1.0))
    await store.upsert(far, embedding=_vec(-1.0))

    hits = await store.search_hybrid(
        query_text="alpha",
        query_vector=_vec(1.0),
        top_k=2,
    )
    assert hits[0].group.group_learning_id == near.group_learning_id
    assert hits[0].cosine == pytest.approx(1.0, abs=1e-6)
    assert hits[0].bm25_rank == 1


async def test_aggregator_persists_via_asyncpg_store(truncated_pool: object) -> None:
    """Aggregator -> AsyncpgGroupLearningStore.upsert end-to-end."""

    learning_store = AsyncpgLearningStore(truncated_pool)
    group_store = AsyncpgGroupLearningStore(truncated_pool)

    group_id = _new_id()
    inputs = [
        _grouped_learning(group_id=group_id, rule="commit before deploy"),
        _grouped_learning(group_id=group_id, rule="lock asyncpg version"),
    ]
    dim = _embedding_dim()
    for row in inputs:
        await learning_store.insert(row, embedding=[0.1] * dim)

    llm_response = json.dumps(
        {
            "summary_md": ("Group rollup\n\n- commit before deploy\n- lock asyncpg version"),
            "bm25_text": "commit before deploy lock asyncpg version",
        }
    )
    aggregator = Aggregator(
        llm=StubLLM(responses=[llm_response]),
        embedder=StubEmbedding(dimension=dim),
        clock=lambda: "2026-05-25T12:00:00Z",
    )
    result = await aggregator.run(inputs, group_id=group_id)
    await group_store.upsert(result.group_learning, embedding=result.embedding)

    fetched = await group_store.get(result.group_learning.group_learning_id)
    assert fetched is not None
    assert fetched.group_id == group_id
    assert fetched.contributing_learnings == tuple(r.learning_id for r in inputs)
    assert fetched.summary_md.startswith("Group rollup")

    # Round-tripping via search must surface the freshly persisted rollup.
    hits = await group_store.search_hybrid(
        query_text="asyncpg",
        query_vector=list(result.embedding),
        top_k=3,
    )
    assert len(hits) == 1
    assert hits[0].group.group_learning_id == result.group_learning.group_learning_id
