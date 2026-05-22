"""Tests for the embedding provider abstractions.

The deterministic :class:`StubEmbedding` is the most heavily exercised
fixture in the test suite, so we lock down its invariants:

- Vectors are unit-norm so cosine similarity equals dot product.
- Same input always yields the same vector (cache-key stable).
- Different inputs yield distinct vectors.
- Output dimension always matches the configured ``embedding_dim`` so
  the pipeline's pgvector inserts will not silently truncate.
"""

from __future__ import annotations

import math

import pytest

from stratoclave_distill import DistillerConfig
from stratoclave_distill.core.errors import ConfigError
from stratoclave_distill.providers import (
    OpenAIEmbedding,
    StubEmbedding,
    VoyageEmbedding,
    build_embedding_provider,
)


@pytest.mark.asyncio
async def test_stub_vectors_are_unit_norm() -> None:
    stub = StubEmbedding(dimension=8)
    [vec] = await stub.embed(["hello"])
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_stub_dimension_matches_config() -> None:
    stub = StubEmbedding(dimension=12)
    vectors = await stub.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 12 for v in vectors)


@pytest.mark.asyncio
async def test_stub_is_deterministic() -> None:
    stub = StubEmbedding(dimension=8)
    [v1] = await stub.embed(["same-input"])
    [v2] = await stub.embed(["same-input"])
    assert v1 == v2


@pytest.mark.asyncio
async def test_stub_distinguishes_inputs() -> None:
    stub = StubEmbedding(dimension=8)
    v1, v2 = await stub.embed(["alpha", "beta"])
    assert v1 != v2


@pytest.mark.asyncio
async def test_stub_records_call_history() -> None:
    stub = StubEmbedding(dimension=4)
    await stub.embed(["one", "two"])
    await stub.embed(["three"])
    assert stub.calls == (("one", "two"), ("three",))


def test_stub_rejects_zero_dimension() -> None:
    with pytest.raises(ConfigError):
        StubEmbedding(dimension=0)


def test_voyage_constructor_does_not_import_sdk() -> None:
    obj = VoyageEmbedding(model="voyage-3", dimension=1024, api_key=None)
    assert obj.model == "voyage-3"
    assert obj.dimension == 1024


def test_openai_embedding_constructor_does_not_import_sdk() -> None:
    obj = OpenAIEmbedding(model="text-embedding-3-small", dimension=1536, api_key=None)
    assert obj.dimension == 1536


def test_voyage_rejects_empty_model() -> None:
    with pytest.raises(ConfigError):
        VoyageEmbedding(model="", dimension=1024, api_key=None)


def test_voyage_rejects_zero_dimension() -> None:
    with pytest.raises(ConfigError):
        VoyageEmbedding(model="voyage-3", dimension=0, api_key=None)


def test_build_embedding_provider_dispatches_by_provider_name() -> None:
    cfg = DistillerConfig(database_url="x", embedding_provider="voyage")
    assert isinstance(build_embedding_provider(cfg), VoyageEmbedding)
    cfg2 = DistillerConfig(database_url="x", embedding_provider="openai")
    assert isinstance(build_embedding_provider(cfg2), OpenAIEmbedding)
    cfg3 = DistillerConfig(database_url="x", embedding_provider="stub", embedding_dim=8)
    assert isinstance(build_embedding_provider(cfg3), StubEmbedding)


def test_build_embedding_provider_rejects_unknown_provider() -> None:
    """Defensive branch when the provider string is mutated past validation."""

    cfg = DistillerConfig(database_url="x")
    object.__setattr__(cfg, "embedding_provider", "nope")
    with pytest.raises(ConfigError, match="unknown embedding_provider"):
        build_embedding_provider(cfg)


def test_openai_embedding_rejects_empty_model() -> None:
    with pytest.raises(ConfigError):
        OpenAIEmbedding(model="", dimension=128, api_key=None)


def test_openai_embedding_rejects_zero_dimension() -> None:
    with pytest.raises(ConfigError):
        OpenAIEmbedding(model="text-embedding-3-small", dimension=0, api_key=None)


def test_voyage_and_openai_expose_dimension_property() -> None:
    voyage = VoyageEmbedding(model="voyage-3", dimension=256, api_key=None)
    openai = OpenAIEmbedding(model="text-embedding-3-small", dimension=512, api_key=None)
    assert voyage.dimension == 256
    assert openai.dimension == 512


def test_stub_exposes_model_property() -> None:
    stub = StubEmbedding(dimension=4, model="my-stub")
    assert stub.model == "my-stub"


def test_stub_exposes_dimension_property() -> None:
    stub = StubEmbedding(dimension=11)
    assert stub.dimension == 11


def test_openai_embedding_exposes_model_property() -> None:
    emb = OpenAIEmbedding(model="text-embedding-3-large", dimension=3, api_key=None)
    assert emb.model == "text-embedding-3-large"
