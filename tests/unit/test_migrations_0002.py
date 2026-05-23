"""Static checks for the Stage B+ ``0002_branching_and_relations`` migration.

The full Postgres round-trip lives in ``tests/integration``. Here we only
parse the migration module and inspect the SQL strings so a typo in the
raw SQL (or a stale ``down_revision`` pointer) is caught by the unit
suite that runs without a database.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def stage_b_plus_migration() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    migration_path = repo_root / "migrations" / "versions" / "0002_branching_and_relations.py"
    assert migration_path.exists(), f"missing migration: {migration_path}"

    spec = importlib.util.spec_from_file_location("_distill_0002", migration_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_has_revision_metadata(stage_b_plus_migration: object) -> None:
    assert stage_b_plus_migration.revision == "0002_branching_and_relations"
    assert stage_b_plus_migration.down_revision == "0001_initial_schema"


def _migration_source() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "migrations" / "versions" / "0002_branching_and_relations.py").read_text(
        encoding="utf-8"
    )


def test_branching_columns_added_with_defaults() -> None:
    src = _migration_source()
    for fragment in (
        "ADD COLUMN parent_session_id UUID NULL",
        "ADD COLUMN branched_at_seq BIGINT NULL",
        "ADD COLUMN branch_kind TEXT NOT NULL DEFAULT 'main'",
        "ADD COLUMN branch_state TEXT NOT NULL DEFAULT 'open'",
        "ADD COLUMN closed_at TIMESTAMPTZ NULL",
    ):
        assert fragment in src, f"missing branching column SQL: {fragment!r}"


def test_branch_kind_and_state_are_constrained() -> None:
    src = _migration_source()
    assert "CHECK (branch_kind IN ('main', 'experiment'))" in src
    assert "CHECK (branch_state IN ('open', 'closed', 'promoted'))" in src


def test_claim_type_is_nullable_with_check() -> None:
    src = _migration_source()
    assert "ADD COLUMN claim_type TEXT NULL" in src
    assert (
        "claim_type IS NULL OR claim_type IN\n"
        "              ('observation', 'interpretation', 'signal', 'norm')" in src
    )


def test_scope_check_is_extended_to_include_experiment() -> None:
    src = _migration_source()
    assert "DROP CONSTRAINT IF EXISTS learnings_scope_check" in src
    assert "scope IN ('session', 'project', 'group', 'shared', 'experiment')" in src


def test_learning_conflicts_table_definition() -> None:
    src = _migration_source()
    for fragment in (
        "CREATE TABLE learning_conflicts",
        "from_id              UUID NOT NULL",
        "ON DELETE CASCADE",
        "cosine_at_detection  REAL NOT NULL",
        "resolution           TEXT NOT NULL DEFAULT 'open'",
        "CHECK (resolution IN ('open', 'merged', 'superseded', 'coexist'))",
        "learning_conflicts_unresolved",
        "WHERE resolution = 'open'",
    ):
        assert fragment in src, f"missing learning_conflicts SQL: {fragment!r}"


def test_session_gaps_table_definition() -> None:
    src = _migration_source()
    for fragment in (
        "CREATE TABLE session_gaps",
        "REFERENCES session_purposes(session_id) ON DELETE CASCADE",
        "resolved_by_learning  UUID NULL REFERENCES learnings(learning_id)",
        "GENERATED ALWAYS AS\n              (to_tsvector('simple', bm25_text)) STORED",
        "session_gaps_unresolved",
        "WHERE resolved_at IS NULL",
        "session_gaps_bm25_idx",
        "USING GIN(bm25_tsv)",
    ):
        assert fragment in src, f"missing session_gaps SQL: {fragment!r}"


def test_downgrade_refuses_promoted_or_experiment_rows() -> None:
    src = _migration_source()
    assert "RAISE EXCEPTION" in src
    assert "branch_state=''promoted''" in src
    assert "scope=''experiment''" in src


def test_downgrade_drops_in_reverse_order() -> None:
    src = _migration_source()
    upgrade_anchor = src.index("def upgrade()")
    downgrade_anchor = src.index("def downgrade()")
    upgrade_block = src[upgrade_anchor:downgrade_anchor]
    downgrade_block = src[downgrade_anchor:]

    assert "DROP TABLE IF EXISTS session_gaps" in downgrade_block
    assert "DROP TABLE IF EXISTS learning_conflicts" in downgrade_block
    assert "DROP COLUMN IF EXISTS claim_type" in downgrade_block
    assert "DROP COLUMN IF EXISTS parent_session_id" in downgrade_block

    assert "CREATE TABLE learning_conflicts" in upgrade_block
    assert "CREATE TABLE session_gaps" in upgrade_block
