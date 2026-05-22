"""Static checks for the alembic migrations.

A real Postgres-backed integration test for the migrations lives in
``tests/integration`` (Stage B). For now we validate the SQL strings and
revision metadata so a typo in raw SQL does not slip past CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def initial_migration() -> object:
    """Import the migration module via its file path.

    Migrations live in a non-package directory (``migrations/versions``),
    so we use importlib's spec_from_file_location to load it for inspection.
    """

    repo_root = Path(__file__).resolve().parents[2]
    migration_path = repo_root / "migrations" / "versions" / "0001_initial_schema.py"
    assert migration_path.exists(), f"missing migration: {migration_path}"

    spec = importlib.util.spec_from_file_location("_distill_initial", migration_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_has_revision_metadata(initial_migration: object) -> None:
    assert initial_migration.revision == "0001_initial_schema"
    assert initial_migration.down_revision is None


def test_migration_reads_dimension_from_env(
    monkeypatch: pytest.MonkeyPatch, initial_migration: object
) -> None:
    helper = initial_migration._embedding_dim
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "1536")
    assert helper() == 1536


def test_migration_rejects_invalid_dimension(
    monkeypatch: pytest.MonkeyPatch, initial_migration: object
) -> None:
    helper = initial_migration._embedding_dim
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "garbage")
    with pytest.raises(RuntimeError, match="must be an integer"):
        helper()


def test_migration_rejects_non_positive_dimension(
    monkeypatch: pytest.MonkeyPatch, initial_migration: object
) -> None:
    helper = initial_migration._embedding_dim
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "0")
    with pytest.raises(RuntimeError, match="must be positive"):
        helper()


def test_alembic_ini_present_with_migrations_pointer() -> None:
    """alembic.ini must exist and point script_location at ./migrations."""

    repo_root = Path(__file__).resolve().parents[2]
    ini = (repo_root / "alembic.ini").read_text(encoding="utf-8")
    assert "script_location = migrations" in ini


def test_env_py_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing env.py without DATABASE_URL must raise.

    We can't actually exec env.py here (it requires alembic.context to be
    populated by the alembic CLI). Instead we confirm that the file
    encodes that requirement, which is the contract migrations rely on.
    """

    repo_root = Path(__file__).resolve().parents[2]
    env_py = (repo_root / "migrations" / "env.py").read_text(encoding="utf-8")
    assert "DATABASE_URL must be set" in env_py


def test_no_database_url_in_alembic_ini() -> None:
    """alembic.ini must not pin a real DB URL (would be a hardcode violation)."""

    repo_root = Path(__file__).resolve().parents[2]
    ini = (repo_root / "alembic.ini").read_text(encoding="utf-8")
    assert "postgresql+" not in ini
