"""asyncpg-backed implementations of the Stage B persistence Protocols.

Production deployments use these against Postgres + pgvector. Unit tests
use :mod:`stratoclave_distill.db.memory` to avoid a live database, so this
module's coverage comes from the integration test that runs against a
docker-compose Postgres (see
``tests/integration/test_asyncpg_stores.py``).

The asyncpg SDK and ``pgvector`` are imported lazily inside
:func:`open_pool` so the module is importable even when those extras are
not installed; a CLI ``--dry-run`` flow does not need them.

Design choices that callers should know about:

- The pool registers the ``vector`` codec on every connection via
  ``init=_register_vector``, so binding ``list[float]`` works without
  per-call casts.
- ``search_hybrid`` runs a single SQL statement: it ranks active rows by
  cosine and by ``ts_rank_cd`` separately, fuses with Reciprocal Rank
  Fusion in a CTE, and orders by the fused score. ``cosine`` and the
  per-modality ranks are surfaced on every row so the Curator's
  thresholds work the same way they do against the in-memory store.
- ``insert`` / ``update_rule`` / ``supersede`` never touch ``bm25_tsv``
  directly; the schema declares it as a STORED generated column derived
  from ``bm25_text``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from stratoclave_distill.core.errors import ConfigError
from stratoclave_distill.core.types import (
    Learning,
    LearningScope,
    SessionDigest,
    SessionPurpose,
)
from stratoclave_distill.db.stores import LearningSearchHit


def _normalize_dsn(database_url: str) -> str:
    """Convert SQLAlchemy-style URLs to a plain asyncpg DSN.

    Alembic / SQLAlchemy callers configure ``DATABASE_URL`` like
    ``postgresql+psycopg://...`` or ``postgresql+asyncpg://...``. asyncpg
    itself only understands the bare ``postgresql://...`` form, so we
    strip the dialect suffix here. Anything that does not match the
    expected prefixes is left untouched and surfaces as an asyncpg error
    at connect time.
    """

    if not database_url:
        raise ConfigError("database_url must be a non-empty string")
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://"):
        if database_url.startswith(prefix):
            return "postgresql://" + database_url[len(prefix) :]
    return database_url


async def _register_vector(conn: Any) -> None:
    """asyncpg ``init`` hook: register the pgvector codec on the connection."""

    try:
        from pgvector.asyncpg import register_vector
    except ImportError as exc:  # pragma: no cover - exercised when extra is missing
        raise ConfigError(
            "pgvector is required for asyncpg stores. Install stratoclave-distill[postgres]."
        ) from exc
    await register_vector(conn)


async def open_pool(database_url: str, *, min_size: int = 1, max_size: int = 4) -> Any:
    """Open an asyncpg connection pool wired for pgvector.

    ``min_size`` / ``max_size`` mirror the asyncpg defaults but expose the
    knobs so the CLI can keep the pool small for one-shot ingests.
    """

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ConfigError(
            "asyncpg is required for production stores. Install stratoclave-distill[postgres]."
        ) from exc
    dsn = _normalize_dsn(database_url)
    pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        init=_register_vector,
    )
    return pool


@asynccontextmanager
async def pool_context(
    database_url: str, *, min_size: int = 1, max_size: int = 4
) -> AsyncIterator[Any]:
    """Async context manager wrapper around :func:`open_pool`."""

    pool = await open_pool(database_url, min_size=min_size, max_size=max_size)
    try:
        yield pool
    finally:
        await pool.close()


def _row_to_purpose(row: Any) -> SessionPurpose:
    return SessionPurpose(
        session_id=str(row["session_id"]),
        purpose=row["purpose"],
        domain_tags=tuple(row["domain_tags"] or ()),
        success_score=row["success_score"],
        polluted=bool(row["polluted"]),
        pollution_reason=row["pollution_reason"],
        derived_from_version=row["derived_from_version"],
        derived_at=row["derived_at"].isoformat() if row["derived_at"] else "",
        last_updated_at=row["last_updated_at"].isoformat() if row["last_updated_at"] else "",
    )


def _row_to_digest(row: Any) -> SessionDigest:
    return SessionDigest(
        digest_id=str(row["digest_id"]),
        session_id=str(row["session_id"]),
        version_id=row["version_id"],
        summary_md=row["summary_md"],
        bm25_text=row["bm25_text"],
        extracted_at=row["extracted_at"].isoformat() if row["extracted_at"] else "",
    )


def _row_to_learning(row: Any) -> Learning:
    triggers_raw = row["triggers"]
    triggers = json.loads(triggers_raw) if isinstance(triggers_raw, str) else triggers_raw or {}
    return Learning(
        learning_id=str(row["learning_id"]),
        scope=row["scope"],
        rule=row["rule"],
        why=row["why"],
        triggers=triggers,
        project_key=row["project_key"],
        group_id=str(row["group_id"]) if row["group_id"] is not None else None,
        source_session=(str(row["source_session"]) if row["source_session"] is not None else None),
        source_version=row["source_version"],
        evidence_count=row["evidence_count"],
        confidence=row["confidence"],
        archived_at=row["archived_at"].isoformat() if row["archived_at"] else None,
        superseded_by=(str(row["superseded_by"]) if row["superseded_by"] is not None else None),
        bm25_text=row["bm25_text"],
        created_at=row["created_at"].isoformat() if row["created_at"] else "",
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else "",
    )


class AsyncpgWatermarkStore:
    """asyncpg-backed :class:`WatermarkStore`."""

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def get(self, session_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT published_up_to FROM distill_watermarks WHERE session_id = $1",
                session_id,
            )
        if row is None:
            return 0
        return int(row["published_up_to"])

    async def advance(self, session_id: str, *, to_seq: int, last_run_at: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO distill_watermarks (session_id, published_up_to, last_run_at)
                VALUES ($1, $2, $3::timestamptz)
                ON CONFLICT (session_id) DO UPDATE
                SET published_up_to = GREATEST(distill_watermarks.published_up_to, EXCLUDED.published_up_to),
                    last_run_at = EXCLUDED.last_run_at
                """,
                session_id,
                to_seq,
                last_run_at,
            )


class AsyncpgPurposeStore:
    """asyncpg-backed :class:`PurposeStore`."""

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def upsert(self, purpose: SessionPurpose) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_purposes (
                    session_id, purpose, domain_tags, success_score, polluted,
                    pollution_reason, derived_from_version, derived_at, last_updated_at
                ) VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8::timestamptz, $9::timestamptz)
                ON CONFLICT (session_id) DO UPDATE SET
                    purpose = EXCLUDED.purpose,
                    domain_tags = EXCLUDED.domain_tags,
                    success_score = EXCLUDED.success_score,
                    polluted = EXCLUDED.polluted,
                    pollution_reason = EXCLUDED.pollution_reason,
                    derived_from_version = EXCLUDED.derived_from_version,
                    derived_at = EXCLUDED.derived_at,
                    last_updated_at = EXCLUDED.last_updated_at
                """,
                purpose.session_id,
                purpose.purpose,
                json.dumps(list(purpose.domain_tags)),
                purpose.success_score,
                purpose.polluted,
                purpose.pollution_reason,
                purpose.derived_from_version,
                purpose.derived_at or _now_iso(),
                purpose.last_updated_at or _now_iso(),
            )

    async def get(self, session_id: str) -> SessionPurpose | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_purposes WHERE session_id = $1",
                session_id,
            )
        return _row_to_purpose(row) if row is not None else None


class AsyncpgDigestStore:
    """asyncpg-backed :class:`DigestStore`.

    Postgres' ``session_digests`` does not have a uniqueness constraint on
    ``session_id`` (there can be many digests in a re-run scenario). To
    keep the Stage B contract simple — "at most one digest per session" —
    we delete any existing rows for the session before inserting the new
    one. The integration test asserts this behavior.
    """

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def upsert(self, digest: SessionDigest, *, embedding: Sequence[float]) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM session_digests WHERE session_id = $1",
                digest.session_id,
            )
            await conn.execute(
                """
                INSERT INTO session_digests (
                    digest_id, session_id, version_id, summary_md, bm25_text,
                    embedding, extracted_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz)
                """,
                digest.digest_id,
                digest.session_id,
                digest.version_id,
                digest.summary_md,
                digest.bm25_text,
                list(embedding),
                digest.extracted_at or _now_iso(),
            )

    async def get(self, session_id: str) -> SessionDigest | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_digests WHERE session_id = $1 "
                "ORDER BY extracted_at DESC LIMIT 1",
                session_id,
            )
        return _row_to_digest(row) if row is not None else None


class AsyncpgLearningStore:
    """asyncpg-backed :class:`LearningStore` with hybrid search."""

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def insert(self, learning: Learning, *, embedding: Sequence[float]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learnings (
                    learning_id, scope, project_key, group_id, rule, why, triggers,
                    source_session, source_version, evidence_count, confidence,
                    archived_at, superseded_by, bm25_text, embedding,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7::jsonb,
                    $8, $9, $10, $11,
                    $12::timestamptz, $13, $14, $15,
                    $16::timestamptz, $17::timestamptz
                )
                """,
                learning.learning_id,
                learning.scope,
                learning.project_key,
                learning.group_id,
                learning.rule,
                learning.why,
                json.dumps(dict(learning.triggers)),
                learning.source_session,
                learning.source_version,
                learning.evidence_count,
                learning.confidence,
                learning.archived_at,
                learning.superseded_by,
                learning.bm25_text,
                list(embedding),
                learning.created_at or _now_iso(),
                learning.updated_at or _now_iso(),
            )

    async def get(self, learning_id: str) -> Learning | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learnings WHERE learning_id = $1",
                learning_id,
            )
        return _row_to_learning(row) if row is not None else None

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
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE learnings SET
                    rule = $2,
                    why = $3,
                    evidence_count = $4,
                    bm25_text = $5,
                    updated_at = $6::timestamptz,
                    embedding = $7
                WHERE learning_id = $1
                """,
                learning_id,
                rule,
                why,
                evidence_count,
                bm25_text,
                updated_at,
                list(embedding),
            )

    async def supersede(self, *, old_id: str, new_id: str, archived_at: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE learnings SET
                    archived_at = $2::timestamptz,
                    superseded_by = $3,
                    updated_at = $2::timestamptz
                WHERE learning_id = $1
                """,
                old_id,
                archived_at,
                new_id,
            )

    async def list_active(self, *, scope: LearningScope | None = None) -> Sequence[Learning]:
        query = "SELECT * FROM learnings WHERE archived_at IS NULL"
        params: list[Any] = []
        if scope is not None:
            query += " AND scope = $1"
            params.append(scope)
        query += " ORDER BY created_at"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return tuple(_row_to_learning(r) for r in rows)

    async def search_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        rrf_k: int = 60,
        scope: LearningScope | None = None,
    ) -> Sequence[LearningSearchHit]:
        # We pass the vector as a list[float]; pgvector codec converts.
        # ``ts_rank_cd`` is positive when there is any token overlap;
        # rows with no BM25 hit get ``bm25_rank = NULL`` so the Curator
        # can distinguish "vector-only" matches.
        scope_filter = "AND scope = $4" if scope is not None else ""
        sql = f"""
        WITH active AS (
            SELECT learning_id, embedding, bm25_tsv, bm25_text
            FROM learnings
            WHERE archived_at IS NULL
            {scope_filter}
        ),
        vec AS (
            SELECT learning_id,
                   1 - (embedding <=> $1) AS cosine,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> $1 ASC) AS vrank
            FROM active
        ),
        bm AS (
            SELECT learning_id,
                   ts_rank_cd(bm25_tsv, plainto_tsquery('simple', $2)) AS score,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(bm25_tsv, plainto_tsquery('simple', $2)) DESC
                   ) AS brank
            FROM active
            WHERE bm25_tsv @@ plainto_tsquery('simple', $2)
        ),
        fused AS (
            SELECT v.learning_id,
                   v.cosine,
                   v.vrank,
                   b.brank,
                   (1.0 / ($3 + v.vrank)) +
                   COALESCE(1.0 / ($3 + b.brank), 0.0) AS rrf
            FROM vec v
            LEFT JOIN bm b USING (learning_id)
        )
        SELECT l.*, f.cosine, f.vrank, f.brank, f.rrf
        FROM fused f
        JOIN learnings l USING (learning_id)
        ORDER BY f.rrf DESC
        LIMIT {int(top_k)}
        """
        params: list[Any] = [list(query_vector), query_text, rrf_k]
        if scope is not None:
            params.append(scope)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        hits: list[LearningSearchHit] = []
        for row in rows:
            hits.append(
                LearningSearchHit(
                    learning=_row_to_learning(row),
                    cosine=float(row["cosine"]),
                    vector_rank=int(row["vrank"]),
                    bm25_rank=int(row["brank"]) if row["brank"] is not None else None,
                    rrf_score=float(row["rrf"]),
                )
            )
        return tuple(hits)


def _now_iso() -> str:
    """Fallback ISO-8601 UTC timestamp for rows that arrived without one."""

    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "AsyncpgDigestStore",
    "AsyncpgLearningStore",
    "AsyncpgPurposeStore",
    "AsyncpgWatermarkStore",
    "open_pool",
    "pool_context",
]
