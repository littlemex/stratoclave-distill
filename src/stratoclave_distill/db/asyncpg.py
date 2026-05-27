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
from datetime import UTC, datetime
from typing import Any

from stratoclave_distill.core.errors import ConfigError
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


def _to_datetime(value: str | datetime | None) -> datetime:
    """Coerce ISO-8601 strings to aware datetimes for asyncpg's binary codec.

    Public types carry timestamps as ISO strings, but asyncpg's ``timestamptz``
    encoder demands a ``datetime.datetime`` regardless of any SQL ``::timestamptz``
    cast (the cast applies after the binary parameter is parsed). We accept a
    ``datetime`` as-is and treat ``None`` / empty as "now (UTC)".
    """

    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if value is None or value == "":
        return datetime.now(UTC)
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _to_optional_datetime(value: str | datetime | None) -> datetime | None:
    """Like :func:`_to_datetime` but preserves ``None`` for nullable columns."""

    if value is None or value == "":
        return None
    return _to_datetime(value)


def _from_datetime(value: datetime | None) -> str:
    """Render a datetime as the ``...Z`` ISO-8601 form the public types use.

    Postgres ``timestamptz`` returns timezone-aware ``datetime`` whose
    ``isoformat()`` emits ``+00:00``. The Stage B contract — and the
    in-memory store — render the same instant as ``...Z``, so we normalize
    here to keep round-trips byte-equivalent.
    """

    if value is None:
        return ""
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    aware_utc = aware.astimezone(UTC)
    return aware_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_optional_datetime(value: datetime | None) -> str | None:
    """Like :func:`_from_datetime` but preserves ``None`` for nullable columns."""

    return None if value is None else _from_datetime(value)


def _row_to_purpose(row: Any) -> SessionPurpose:
    tags_raw = row["domain_tags"]
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else (tags_raw or [])
    parent_raw = row.get("parent_session_id", None)
    return SessionPurpose(
        session_id=str(row["session_id"]),
        purpose=row["purpose"],
        domain_tags=tuple(tags),
        success_score=row["success_score"],
        polluted=bool(row["polluted"]),
        pollution_reason=row["pollution_reason"],
        derived_from_version=row["derived_from_version"],
        derived_at=_from_datetime(row["derived_at"]),
        last_updated_at=_from_datetime(row["last_updated_at"]),
        parent_session_id=str(parent_raw) if parent_raw is not None else None,
        branched_at_seq=row.get("branched_at_seq", None),
        branch_kind=row.get("branch_kind", "main"),
        branch_state=row.get("branch_state", "open"),
        closed_at=_from_optional_datetime(row["closed_at"]) if "closed_at" in row else None,
    )


def _row_to_digest(row: Any) -> SessionDigest:
    return SessionDigest(
        digest_id=str(row["digest_id"]),
        session_id=str(row["session_id"]),
        version_id=row["version_id"],
        summary_md=row["summary_md"],
        bm25_text=row["bm25_text"],
        extracted_at=_from_datetime(row["extracted_at"]),
    )


def _row_to_learning(row: Any) -> Learning:
    triggers_raw = row["triggers"]
    triggers = json.loads(triggers_raw) if isinstance(triggers_raw, str) else triggers_raw or {}
    claim_type_raw = row.get("claim_type", None)
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
        archived_at=_from_optional_datetime(row["archived_at"]),
        superseded_by=(str(row["superseded_by"]) if row["superseded_by"] is not None else None),
        bm25_text=row["bm25_text"],
        created_at=_from_datetime(row["created_at"]),
        updated_at=_from_datetime(row["updated_at"]),
        claim_type=claim_type_raw,
    )


def _row_to_group_learning(row: Any) -> GroupLearning:
    contributing_raw = row["contributing_learnings"]
    contributing = (
        json.loads(contributing_raw)
        if isinstance(contributing_raw, str)
        else contributing_raw or []
    )
    return GroupLearning(
        group_learning_id=str(row["group_learning_id"]),
        group_id=str(row["group_id"]),
        summary_md=row["summary_md"],
        contributing_learnings=tuple(str(c) for c in contributing),
        bm25_text=row["bm25_text"],
        created_at=_from_datetime(row["created_at"]),
    )


def _row_to_conflict(row: Any) -> LearningConflict:
    return LearningConflict(
        conflict_id=str(row["conflict_id"]),
        from_id=str(row["from_id"]),
        to_id=str(row["to_id"]),
        reason=row["reason"],
        cosine_at_detection=float(row["cosine_at_detection"]),
        detected_at=_from_datetime(row["detected_at"]),
        resolution=row["resolution"],
    )


def _row_to_gap(row: Any) -> SessionGap:
    return SessionGap(
        gap_id=str(row["gap_id"]),
        session_id=str(row["session_id"]),
        topic=row["topic"],
        why_unknown=row["why_unknown"],
        bm25_text=row["bm25_text"],
        detected_at=_from_datetime(row["detected_at"]),
        resolved_at=_from_optional_datetime(row["resolved_at"]),
        resolved_by_learning=(
            str(row["resolved_by_learning"]) if row["resolved_by_learning"] is not None else None
        ),
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
                VALUES ($1, $2, $3)
                ON CONFLICT (session_id) DO UPDATE
                SET published_up_to = GREATEST(distill_watermarks.published_up_to, EXCLUDED.published_up_to),
                    last_run_at = EXCLUDED.last_run_at
                """,
                session_id,
                to_seq,
                _to_datetime(last_run_at),
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
                    pollution_reason, derived_from_version, derived_at, last_updated_at,
                    parent_session_id, branched_at_seq, branch_kind, branch_state, closed_at
                ) VALUES (
                    $1, $2, $3::jsonb, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12, $13, $14
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    purpose = EXCLUDED.purpose,
                    domain_tags = EXCLUDED.domain_tags,
                    success_score = EXCLUDED.success_score,
                    polluted = EXCLUDED.polluted,
                    pollution_reason = EXCLUDED.pollution_reason,
                    derived_from_version = EXCLUDED.derived_from_version,
                    derived_at = EXCLUDED.derived_at,
                    last_updated_at = EXCLUDED.last_updated_at,
                    parent_session_id = EXCLUDED.parent_session_id,
                    branched_at_seq = EXCLUDED.branched_at_seq,
                    branch_kind = EXCLUDED.branch_kind,
                    branch_state = EXCLUDED.branch_state,
                    closed_at = EXCLUDED.closed_at
                """,
                purpose.session_id,
                purpose.purpose,
                json.dumps(list(purpose.domain_tags)),
                purpose.success_score,
                purpose.polluted,
                purpose.pollution_reason,
                purpose.derived_from_version,
                _to_datetime(purpose.derived_at),
                _to_datetime(purpose.last_updated_at),
                purpose.parent_session_id,
                purpose.branched_at_seq,
                purpose.branch_kind,
                purpose.branch_state,
                _to_optional_datetime(purpose.closed_at),
            )

    async def get(self, session_id: str) -> SessionPurpose | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_purposes WHERE session_id = $1",
                session_id,
            )
        return _row_to_purpose(row) if row is not None else None

    async def set_branch_state(
        self,
        session_id: str,
        *,
        branch_state: BranchState,
        closed_at: str | None,
        last_updated_at: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE session_purposes
                SET branch_state = $2,
                    closed_at = $3,
                    last_updated_at = $4
                WHERE session_id = $1
                """,
                session_id,
                branch_state,
                _to_optional_datetime(closed_at),
                _to_datetime(last_updated_at),
            )

    async def list_branches(
        self,
        *,
        parent_session_id: str | None = None,
    ) -> Sequence[SessionPurpose]:
        async with self._pool.acquire() as conn:
            if parent_session_id is None:
                rows = await conn.fetch("SELECT * FROM session_purposes ORDER BY derived_at")
            else:
                rows = await conn.fetch(
                    "SELECT * FROM session_purposes "
                    "WHERE parent_session_id = $1 ORDER BY derived_at",
                    parent_session_id,
                )
        return tuple(_row_to_purpose(r) for r in rows)


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
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                digest.digest_id,
                digest.session_id,
                digest.version_id,
                digest.summary_md,
                digest.bm25_text,
                list(embedding),
                _to_datetime(digest.extracted_at),
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
                    created_at, updated_at, claim_type
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7::jsonb,
                    $8, $9, $10, $11,
                    $12, $13, $14, $15,
                    $16, $17, $18
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
                _to_optional_datetime(learning.archived_at),
                learning.superseded_by,
                learning.bm25_text,
                list(embedding),
                _to_datetime(learning.created_at),
                _to_datetime(learning.updated_at),
                learning.claim_type,
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
                    updated_at = $6,
                    embedding = $7
                WHERE learning_id = $1
                """,
                learning_id,
                rule,
                why,
                evidence_count,
                bm25_text,
                _to_datetime(updated_at),
                list(embedding),
            )

    async def supersede(self, *, old_id: str, new_id: str, archived_at: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE learnings SET
                    archived_at = $2,
                    superseded_by = $3,
                    updated_at = $2
                WHERE learning_id = $1
                """,
                old_id,
                _to_datetime(archived_at),
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
        lane: RetrievalLane = "all",
        canonical_min_evidence: int = 3,
        canonical_min_age_days: int = 14,
        source_session_ids: Sequence[str] | None = None,
    ) -> Sequence[LearningSearchHit]:
        # We pass the vector as a list[float]; pgvector codec converts.
        # ``ts_rank_cd`` is positive when there is any token overlap;
        # rows with no BM25 hit get ``bm25_rank = NULL`` so the Curator
        # can distinguish "vector-only" matches.
        if source_session_ids is not None and len(source_session_ids) == 0:
            return ()
        params: list[Any] = [list(query_vector), query_text, rrf_k]
        scope_filter = ""
        if scope is not None:
            params.append(scope)
            scope_filter = f"AND scope = ${len(params)}"

        lane_filter = ""
        if lane != "all":
            params.append(canonical_min_evidence)
            min_ev_idx = len(params)
            params.append(canonical_min_age_days)
            min_age_idx = len(params)
            canonical_pred = (
                f"(evidence_count >= ${min_ev_idx} "
                f"AND created_at <= now() - make_interval(days => ${min_age_idx}) "
                "AND scope <> 'experiment')"
            )
            lane_filter = (
                f"AND {canonical_pred}" if lane == "canonical" else f"AND NOT {canonical_pred}"
            )

        session_filter = ""
        if source_session_ids is not None:
            params.append(list(source_session_ids))
            session_filter = f"AND source_session = ANY(${len(params)}::text[])"

        sql = f"""
        WITH active AS (
            SELECT learning_id, embedding, bm25_tsv, bm25_text
            FROM learnings
            WHERE archived_at IS NULL
            {scope_filter}
            {lane_filter}
            {session_filter}
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


class AsyncpgGroupLearningStore:
    """asyncpg-backed :class:`GroupLearningStore`.

    The ``group_learnings`` table was reserved by migration 0001 with the
    columns this store needs (``group_learning_id`` PK, ``group_id``,
    ``summary_md``, ``contributing_learnings`` JSONB, ``embedding``,
    ``bm25_text`` + generated ``bm25_tsv``, ``created_at``). HNSW and GIN
    indexes are also already in place.

    A re-aggregation produces a *new* ``group_learning_id`` and inserts
    a fresh row; the previous row is not deleted so the history can be
    audited. :meth:`list_by_group` with ``latest_only=True`` (the default)
    returns just the most recent row per group.
    """

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def upsert(
        self,
        group_learning: GroupLearning,
        *,
        embedding: Sequence[float],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO group_learnings (
                    group_learning_id, group_id, summary_md,
                    contributing_learnings, embedding, bm25_text, created_at
                ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                ON CONFLICT (group_learning_id) DO UPDATE SET
                    group_id = EXCLUDED.group_id,
                    summary_md = EXCLUDED.summary_md,
                    contributing_learnings = EXCLUDED.contributing_learnings,
                    embedding = EXCLUDED.embedding,
                    bm25_text = EXCLUDED.bm25_text,
                    created_at = EXCLUDED.created_at
                """,
                group_learning.group_learning_id,
                group_learning.group_id,
                group_learning.summary_md,
                json.dumps(list(group_learning.contributing_learnings)),
                list(embedding),
                group_learning.bm25_text,
                _to_datetime(group_learning.created_at),
            )

    async def get(self, group_learning_id: str) -> GroupLearning | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM group_learnings WHERE group_learning_id = $1",
                group_learning_id,
            )
        return _row_to_group_learning(row) if row is not None else None

    async def list_by_group(
        self,
        group_id: str,
        *,
        latest_only: bool = True,
    ) -> Sequence[GroupLearning]:
        async with self._pool.acquire() as conn:
            if latest_only:
                rows = await conn.fetch(
                    "SELECT * FROM group_learnings WHERE group_id = $1 "
                    "ORDER BY created_at DESC LIMIT 1",
                    group_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM group_learnings WHERE group_id = $1 ORDER BY created_at DESC",
                    group_id,
                )
        return tuple(_row_to_group_learning(r) for r in rows)

    async def list_latest_per_group(self) -> Sequence[GroupLearning]:
        """Return the newest rollup per ``group_id`` across the whole table."""

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT ON (group_id) * FROM group_learnings "
                "ORDER BY group_id, created_at DESC"
            )
        result = [_row_to_group_learning(r) for r in rows]
        result.sort(key=lambda g: g.created_at, reverse=True)
        return tuple(result)

    async def search_hybrid(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> Sequence[GroupLearningSearchHit]:
        # Mirrors ``LearningStore.search_hybrid`` but without lane / scope.
        # ``latest`` deduplicates rollup history per group_id so the
        # retrieval surface aligns with ``list_by_group(latest_only=True)``.
        params: list[Any] = [list(query_vector), query_text, rrf_k]
        sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON (group_id) group_learning_id, group_id, embedding, bm25_tsv
            FROM group_learnings
            ORDER BY group_id, created_at DESC
        ),
        vec AS (
            SELECT group_learning_id,
                   1 - (embedding <=> $1) AS cosine,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> $1 ASC) AS vrank
            FROM latest
        ),
        bm AS (
            SELECT group_learning_id,
                   ts_rank_cd(bm25_tsv, plainto_tsquery('simple', $2)) AS score,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(bm25_tsv, plainto_tsquery('simple', $2)) DESC
                   ) AS brank
            FROM latest
            WHERE bm25_tsv @@ plainto_tsquery('simple', $2)
        ),
        fused AS (
            SELECT v.group_learning_id,
                   v.cosine,
                   v.vrank,
                   b.brank,
                   (1.0 / ($3 + v.vrank)) +
                   COALESCE(1.0 / ($3 + b.brank), 0.0) AS rrf
            FROM vec v
            LEFT JOIN bm b USING (group_learning_id)
        )
        SELECT g.*, f.cosine, f.vrank, f.brank, f.rrf
        FROM fused f
        JOIN group_learnings g USING (group_learning_id)
        ORDER BY f.rrf DESC
        LIMIT {int(top_k)}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        hits: list[GroupLearningSearchHit] = []
        for row in rows:
            hits.append(
                GroupLearningSearchHit(
                    group=_row_to_group_learning(row),
                    cosine=float(row["cosine"]),
                    vector_rank=int(row["vrank"]),
                    bm25_rank=int(row["brank"]) if row["brank"] is not None else None,
                    rrf_score=float(row["rrf"]),
                )
            )
        return tuple(hits)


class AsyncpgConflictStore:
    """asyncpg-backed :class:`ConflictStore`."""

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def insert(self, conflict: LearningConflict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learning_conflicts (
                    conflict_id, from_id, to_id, reason, cosine_at_detection,
                    detected_at, resolution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                conflict.conflict_id,
                conflict.from_id,
                conflict.to_id,
                conflict.reason,
                conflict.cosine_at_detection,
                _to_datetime(conflict.detected_at),
                conflict.resolution,
            )

    async def get(self, conflict_id: str) -> LearningConflict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learning_conflicts WHERE conflict_id = $1",
                conflict_id,
            )
        return _row_to_conflict(row) if row is not None else None

    async def list_open(self) -> Sequence[LearningConflict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM learning_conflicts WHERE resolution = 'open' ORDER BY detected_at"
            )
        return tuple(_row_to_conflict(r) for r in rows)

    async def list_for(self, learning_id: str) -> Sequence[LearningConflict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM learning_conflicts "
                "WHERE from_id = $1 OR to_id = $1 ORDER BY detected_at",
                learning_id,
            )
        return tuple(_row_to_conflict(r) for r in rows)

    async def resolve(
        self,
        conflict_id: str,
        *,
        resolution: ConflictResolution,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE learning_conflicts SET resolution = $2 WHERE conflict_id = $1",
                conflict_id,
                resolution,
            )


class AsyncpgGapStore:
    """asyncpg-backed :class:`GapStore`."""

    __slots__ = ("_pool",)

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def insert(self, gap: SessionGap) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_gaps (
                    gap_id, session_id, topic, why_unknown, bm25_text,
                    detected_at, resolved_at, resolved_by_learning
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                gap.gap_id,
                gap.session_id,
                gap.topic,
                gap.why_unknown,
                gap.bm25_text,
                _to_datetime(gap.detected_at),
                _to_optional_datetime(gap.resolved_at),
                gap.resolved_by_learning,
            )

    async def get(self, gap_id: str) -> SessionGap | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_gaps WHERE gap_id = $1",
                gap_id,
            )
        return _row_to_gap(row) if row is not None else None

    async def list_unresolved(
        self,
        *,
        session_id: str | None = None,
    ) -> Sequence[SessionGap]:
        async with self._pool.acquire() as conn:
            if session_id is None:
                rows = await conn.fetch(
                    "SELECT * FROM session_gaps WHERE resolved_at IS NULL ORDER BY detected_at"
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM session_gaps "
                    "WHERE resolved_at IS NULL AND session_id = $1 "
                    "ORDER BY detected_at",
                    session_id,
                )
        return tuple(_row_to_gap(r) for r in rows)

    async def resolve(
        self,
        gap_id: str,
        *,
        resolved_at: str,
        resolved_by_learning: str | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE session_gaps
                SET resolved_at = $2, resolved_by_learning = $3
                WHERE gap_id = $1
                """,
                gap_id,
                _to_datetime(resolved_at),
                resolved_by_learning,
            )


__all__ = [
    "AsyncpgConflictStore",
    "AsyncpgDigestStore",
    "AsyncpgGapStore",
    "AsyncpgGroupLearningStore",
    "AsyncpgLearningStore",
    "AsyncpgPurposeStore",
    "AsyncpgWatermarkStore",
    "open_pool",
    "pool_context",
]
