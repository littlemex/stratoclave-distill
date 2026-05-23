"""End-to-end migration test against a live Postgres + pgvector.

Skipped unless ``DISTILL_TEST_DATABASE_URL`` is set, so the unit suite stays
fast and dependency-free. Run locally with::

    docker compose up -d
    DISTILL_TEST_DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
        pytest -m integration

The test runs ``alembic upgrade head`` then introspects the schema via
``information_schema`` to confirm every required table, column, and
pgvector / GIN index landed, then runs ``alembic downgrade base`` and
checks the schema is gone. This is the gate that proves both 0001 and
0002 actually apply cleanly before the pipeline starts using them.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


REQUIRED_TABLES = {
    "session_purposes",
    "session_digests",
    "learnings",
    "distill_watermarks",
    "group_learnings",
    "learning_conflicts",
    "session_gaps",
}

REQUIRED_INDEXES = {
    "idx_digests_vec",  # HNSW on session_digests.embedding
    "idx_digests_bm25",  # GIN on session_digests.bm25_tsv
    "idx_learnings_vec",
    "idx_learnings_bm25",
    "idx_group_learnings_vec",
    "idx_group_learnings_bm25",
    "idx_session_purposes_parent",
    "idx_session_purposes_branch_state",
    "learning_conflicts_from_idx",
    "learning_conflicts_to_idx",
    "learning_conflicts_unresolved",
    "session_gaps_session_idx",
    "session_gaps_unresolved",
    "session_gaps_bm25_idx",
}

REQUIRED_EXTENSIONS = {"vector", "pg_trgm"}

REQUIRED_NEW_COLUMNS = {
    "session_purposes": {
        "parent_session_id",
        "branched_at_seq",
        "branch_kind",
        "branch_state",
        "closed_at",
    },
    "learnings": {"claim_type"},
}


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


@pytest.fixture
def db_url() -> str:
    url = _database_url()
    if not url:
        pytest.skip("DISTILL_TEST_DATABASE_URL not set; integration test skipped")
    return url


def test_migration_creates_and_drops_all_tables(db_url: str) -> None:
    pytest.importorskip("psycopg")
    import psycopg  # type: ignore[import-not-found]

    sync_url = db_url.replace("+psycopg", "").replace("+asyncpg", "")
    _run_alembic("up", db_url)
    try:
        with psycopg.connect(sync_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = {row[0] for row in cur.fetchall()}
            for required in REQUIRED_TABLES:
                assert required in tables, f"missing table {required!r}"

            cur.execute("SELECT extname FROM pg_extension")
            extensions = {row[0] for row in cur.fetchall()}
            for required in REQUIRED_EXTENSIONS:
                assert required in extensions, f"missing extension {required!r}"

            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
            indexes = {row[0] for row in cur.fetchall()}
            for required in REQUIRED_INDEXES:
                assert required in indexes, f"missing index {required!r}"

            cur.execute(
                """
                SELECT amname FROM pg_class c
                JOIN pg_am a ON a.oid = c.relam
                WHERE c.relname = 'idx_digests_vec'
                """
            )
            row = cur.fetchone()
            assert row is not None and row[0] == "hnsw", (
                "vector index must use the HNSW access method"
            )

            for table, columns in REQUIRED_NEW_COLUMNS.items():
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    """,
                    (table,),
                )
                actual = {row[0] for row in cur.fetchall()}
                for col in columns:
                    assert col in actual, f"missing column {table}.{col}"

            cur.execute(
                """
                SELECT pg_get_constraintdef(c.oid)
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE t.relname = 'learnings' AND c.conname = 'learnings_scope_check'
                """
            )
            row = cur.fetchone()
            assert row is not None, "learnings_scope_check constraint missing"
            assert "experiment" in row[0], (
                "learnings_scope_check must include the 'experiment' scope after 0002"
            )
    finally:
        _run_alembic("down", db_url)
        with psycopg.connect(sync_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables_after = {row[0] for row in cur.fetchall()}
            for required in REQUIRED_TABLES:
                assert required not in tables_after, f"downgrade left {required!r} behind"
