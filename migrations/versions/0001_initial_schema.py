"""initial schema: purposes, digests, learnings, watermarks, group_learnings

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-22 00:00:00

The embedding column dimension is read from ``DISTILL_EMBEDDING_DIM`` so
Voyage (1024), OpenAI text-embedding-3-small (1536), and other providers
can each be deployed without forking the migration. Defaults to 1024 to
match the v0.1 reference profile (Voyage voyage-3).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _embedding_dim() -> int:
    raw = os.environ.get("DISTILL_EMBEDDING_DIM", "1024")
    try:
        dim = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"DISTILL_EMBEDDING_DIM must be an integer, got {raw!r}") from exc
    if dim < 1:
        raise RuntimeError(f"DISTILL_EMBEDDING_DIM must be positive, got {dim}")
    return dim


def upgrade() -> None:
    dim = _embedding_dim()

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(
        """
        CREATE TABLE session_purposes (
            session_id           UUID PRIMARY KEY,
            purpose              TEXT NOT NULL,
            domain_tags          JSONB NOT NULL DEFAULT '[]'::jsonb,
            success_score        REAL,
            polluted             BOOLEAN NOT NULL DEFAULT false,
            pollution_reason     TEXT,
            derived_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            derived_from_version TEXT NOT NULL,
            last_updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_session_purposes_polluted ON session_purposes(polluted)")
    op.execute("CREATE INDEX idx_session_purposes_tags ON session_purposes USING GIN(domain_tags)")

    op.execute(
        f"""
        CREATE TABLE session_digests (
            digest_id     UUID PRIMARY KEY,
            session_id    UUID NOT NULL,
            version_id    TEXT NOT NULL,
            summary_md    TEXT NOT NULL,
            bm25_text     TEXT NOT NULL,
            bm25_tsv      tsvector GENERATED ALWAYS AS (to_tsvector('simple', bm25_text)) STORED,
            embedding     vector({dim}),
            extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_digests_session ON session_digests(session_id)")
    op.execute("CREATE INDEX idx_digests_bm25 ON session_digests USING GIN(bm25_tsv)")
    op.execute(
        "CREATE INDEX idx_digests_vec ON session_digests "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.execute(
        f"""
        CREATE TABLE learnings (
            learning_id     UUID PRIMARY KEY,
            scope           TEXT NOT NULL CHECK (scope IN ('session','project','group','shared')),
            project_key     TEXT,
            group_id        UUID,
            rule            TEXT NOT NULL,
            why             TEXT NOT NULL,
            triggers        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            source_session  UUID,
            source_version  TEXT,
            evidence_count  INTEGER NOT NULL DEFAULT 1,
            confidence      REAL NOT NULL DEFAULT 0.5,
            archived_at     TIMESTAMPTZ,
            superseded_by   UUID REFERENCES learnings(learning_id),
            bm25_text       TEXT NOT NULL,
            bm25_tsv        tsvector GENERATED ALWAYS AS (to_tsvector('simple', bm25_text)) STORED,
            embedding       vector({dim}),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_learnings_active ON learnings(scope, archived_at)")
    op.execute("CREATE INDEX idx_learnings_group ON learnings(group_id)")
    op.execute("CREATE INDEX idx_learnings_bm25 ON learnings USING GIN(bm25_tsv)")
    op.execute(
        "CREATE INDEX idx_learnings_vec ON learnings "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.execute(
        """
        CREATE TABLE distill_watermarks (
            session_id      UUID PRIMARY KEY,
            published_up_to BIGINT NOT NULL DEFAULT 0,
            last_run_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        f"""
        CREATE TABLE group_learnings (
            group_learning_id      UUID PRIMARY KEY,
            group_id               UUID NOT NULL,
            summary_md             TEXT NOT NULL,
            contributing_learnings JSONB NOT NULL,
            embedding              vector({dim}),
            bm25_text              TEXT NOT NULL,
            bm25_tsv               tsvector GENERATED ALWAYS AS (to_tsvector('simple', bm25_text)) STORED,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_group_learnings_vec ON group_learnings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX idx_group_learnings_bm25 ON group_learnings USING GIN(bm25_tsv)")


def downgrade() -> None:
    for table in (
        "group_learnings",
        "distill_watermarks",
        "learnings",
        "session_digests",
        "session_purposes",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
