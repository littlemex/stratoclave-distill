"""Shared pytest fixtures for stratoclave-distill.

Stage A keeps the fixture set small: a no-database :class:`DistillerConfig`
and the deterministic stub providers. Stage B will add a Postgres fixture
behind the ``integration`` marker that spins up the docker-compose service
and runs alembic before yielding a connection pool.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from stratoclave_distill import DistillerConfig
from stratoclave_distill.providers import StubEmbedding, StubLLM


@pytest.fixture
def stub_env() -> dict[str, str]:
    """A minimal env mapping that lets ``DistillerConfig.from_env`` succeed."""

    return {
        "DATABASE_URL": "postgresql+asyncpg://distill:distill@localhost:5432/distill",
        "DISTILL_LLM_PROVIDER": "stub",
        "DISTILL_EMBEDDING_PROVIDER": "stub",
        "DISTILL_EMBEDDING_DIM": "8",
    }


@pytest.fixture
def cfg(stub_env: dict[str, str]) -> DistillerConfig:
    """A configured :class:`DistillerConfig` aimed at the stub providers."""

    return DistillerConfig.from_env(stub_env)


@pytest.fixture
def stub_llm() -> Iterator[StubLLM]:
    """A scripted :class:`StubLLM` with one canned response per test.

    Tests that need multiple responses should construct their own StubLLM;
    this fixture is just the simplest happy-path tool.
    """

    yield StubLLM(responses=["ok"])


@pytest.fixture
def stub_embedding() -> StubEmbedding:
    """An 8-dimensional :class:`StubEmbedding` consistent with ``cfg``."""

    return StubEmbedding(dimension=8)
