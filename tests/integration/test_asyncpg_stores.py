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

import os
import subprocess
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from stratoclave_distill.core.types import (
    Learning,
    SessionDigest,
    SessionPurpose,
)
from stratoclave_distill.db.asyncpg import (
    AsyncpgDigestStore,
    AsyncpgLearningStore,
    AsyncpgPurposeStore,
    AsyncpgWatermarkStore,
    pool_context,
)

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
                "distill_watermarks, group_learnings RESTART IDENTITY CASCADE"
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
