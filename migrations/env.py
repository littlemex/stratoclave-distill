"""Alembic env.py for stratoclave-distill.

Reads ``DATABASE_URL`` from the environment so the same migrations can run
against local docker-compose Postgres, CI Postgres, and a managed instance
without editing ``alembic.ini``. The migrations are SQL-only (no SQLAlchemy
models) because the schema lives close to pgvector / tsvector specifics
that are easier to express directly.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    pass  # logging configured via alembic.ini at CLI invocation

_database_url = os.environ.get("DATABASE_URL")
if not _database_url:
    raise RuntimeError(
        "DATABASE_URL must be set to run migrations. "
        "Example: postgresql+psycopg://distill:distill@localhost:5432/distill"
    )

config.set_main_option("sqlalchemy.url", _database_url)


def run_migrations_offline() -> None:
    """Generate SQL without a live connection (used for `alembic upgrade --sql`)."""

    context.configure(url=_database_url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations using a real connection."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
