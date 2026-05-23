"""Unit tests for the asyncpg helpers that do not require a live database.

The store classes themselves are exercised by the integration test
(``tests/integration/test_asyncpg_stores.py``) against a real Postgres,
because their entire contract is "talks to Postgres correctly". The
DSN-normalization helper is the one piece worth covering in unit tests
because it has SDK-independent string logic and the CLI relies on it to
accept SQLAlchemy-style URLs as input.
"""

from __future__ import annotations

import pytest

from stratoclave_distill.core.errors import ConfigError
from stratoclave_distill.db.asyncpg import _normalize_dsn


def test_normalize_dsn_strips_psycopg_dialect() -> None:
    assert _normalize_dsn("postgresql+psycopg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"


def test_normalize_dsn_strips_asyncpg_dialect() -> None:
    assert _normalize_dsn("postgresql+asyncpg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"


def test_normalize_dsn_passes_plain_url_through() -> None:
    assert _normalize_dsn("postgresql://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"


def test_normalize_dsn_rejects_empty_string() -> None:
    with pytest.raises(ConfigError, match="non-empty"):
        _normalize_dsn("")
